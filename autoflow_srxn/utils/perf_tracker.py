"""Lightweight wall-time + CPU + memory profiler for AutoFlow-SRXN stages.

Usage (module-level singleton, recommended):
    from autoflow_srxn.utils.perf_tracker import set_perf_tracker, perf_stage, PerfTracker

    tracker = PerfTracker()
    set_perf_tracker(tracker)

    with perf_stage("slab_relax"):
        engine.relax(slab)

    tracker.write_report("results/perf_report.log")
    tracker.log_report(logger)   # also emit to the workflow logger

The ``perf_stage(name)`` helper returns a no-op context manager when no
tracker has been registered, so instrumented code is safe to call from
unit tests or scripts that never call ``set_perf_tracker``.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore
    _HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_TRACKER: Optional["PerfTracker"] = None


def get_perf_tracker() -> Optional["PerfTracker"]:
    """Return the currently registered PerfTracker, or *None*."""
    return _TRACKER


def set_perf_tracker(tracker: Optional["PerfTracker"]) -> None:
    """Register *tracker* as the module-level singleton."""
    global _TRACKER
    _TRACKER = tracker


# ---------------------------------------------------------------------------
# No-op context for when no tracker is registered
# ---------------------------------------------------------------------------

class _NoOpCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def perf_stage(name: str):
    """Return a context-manager timing *name*.

    If no tracker has been registered via :func:`set_perf_tracker` this
    returns a lightweight no-op so instrumented code is always safe to use.
    """
    t = get_perf_tracker()
    if t is None:
        return _NoOpCtx()
    return t.stage(name)


# ---------------------------------------------------------------------------
# Record dataclass (plain dict for simplicity)
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{int(h)}h{int(m):02d}m"


# ---------------------------------------------------------------------------
# PerfTracker
# ---------------------------------------------------------------------------

class PerfTracker:
    """Tracks wall time, mean CPU%, and peak RSS memory for named stages.

    A background thread polls ``psutil.Process`` metrics every
    *sample_interval* seconds while a stage is active.  All operations
    are thread-safe (though the tracker is designed for single-threaded
    sequential workflows).

    Parameters
    ----------
    sample_interval : float
        Seconds between CPU/memory samples (default 1.0).
    log_on_exit : bool
        If *True* (default) emit a one-liner to the AutoFlow-SRXN logger
        when each stage finishes.
    """

    def __init__(self, sample_interval: float = 1.0, log_on_exit: bool = True):
        self._sample_interval = sample_interval
        self._log_on_exit = log_on_exit
        self._records: List[Dict] = []
        self._lock = threading.Lock()
        self._proc = _psutil.Process() if _HAS_PSUTIL else None

        # Prime the cpu_percent baseline (first call always returns 0.0)
        if self._proc:
            self._proc.cpu_percent(interval=None)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def stage(self, name: str):
        """Return a context manager that times *name*."""
        return _StageCtx(self, name)

    # ------------------------------------------------------------------
    # Internal start/stop
    # ------------------------------------------------------------------

    def _begin(self, name: str) -> Dict:
        cpu_samples: List[float] = []
        mem_samples: List[float] = []
        stop_evt = threading.Event()

        # Prime per-stage baseline
        if self._proc:
            self._proc.cpu_percent(interval=None)

        def _sample():
            while not stop_evt.wait(self._sample_interval):
                if self._proc:
                    try:
                        cpu_samples.append(self._proc.cpu_percent(interval=None))
                        mem_samples.append(self._proc.memory_info().rss / 1024 ** 2)
                    except Exception:
                        pass

        thread = threading.Thread(target=_sample, daemon=True, name=f"perf-{name}")
        thread.start()

        return {
            "name": name,
            "t_start": time.monotonic(),
            "cpu_samples": cpu_samples,
            "mem_samples": mem_samples,
            "stop_evt": stop_evt,
            "thread": thread,
        }

    def _finish(self, ctx: Dict) -> Dict:
        wall_s = time.monotonic() - ctx["t_start"]
        ctx["stop_evt"].set()
        ctx["thread"].join(timeout=max(self._sample_interval * 2, 3.0))

        cpu = ctx["cpu_samples"]
        mem = ctx["mem_samples"]

        record = {
            "name": ctx["name"],
            "wall_s": wall_s,
            "mean_cpu_pct": (sum(cpu) / len(cpu)) if cpu else float("nan"),
            "peak_mem_mb": max(mem) if mem else float("nan"),
        }

        with self._lock:
            self._records.append(record)

        if self._log_on_exit:
            self._log_inline(record)

        return record

    def _log_inline(self, record: Dict) -> None:
        try:
            from .logger_utils import get_workflow_logger

            logger = get_workflow_logger()
            cpu_str = (
                f"{record['mean_cpu_pct']:.0f}% CPU"
                if not (record["mean_cpu_pct"] != record["mean_cpu_pct"])  # NaN check
                else "CPU N/A"
            )
            mem_str = (
                f"{record['peak_mem_mb']:.0f} MB peak"
                if not (record["peak_mem_mb"] != record["peak_mem_mb"])
                else "Mem N/A"
            )
            logger.info(
                f"  [PerfTracker] {record['name']:<35} "
                f"wall={_fmt_duration(record['wall_s'])}  "
                f"{cpu_str}  {mem_str}"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def write_report(self, path: str) -> None:
        """Write a formatted cost-breakdown table to *path*."""
        if not self._records:
            return

        lines = self._build_report_lines()
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def log_report(self, logger=None) -> None:
        """Emit the full cost table to *logger* (AutoFlow-SRXN logger if None)."""
        if not self._records:
            return

        if logger is None:
            try:
                from .logger_utils import get_workflow_logger

                logger = get_workflow_logger()
            except Exception:
                return

        for line in self._build_report_lines():
            logger.info(line)

    def _build_report_lines(self) -> List[str]:
        records = list(self._records)
        total_s = sum(r["wall_s"] for r in records)

        W = 96
        lines: List[str] = [
            "",
            "=" * W,
            " AutoFlow-SRXN :: Computation Cost Report",
            "-" * W,
            f"  {'Stage':<38} {'Wall time':>10}  {'% total':>7}  {'CPU% (mean)':>11}  {'Peak RAM':>10}",
            "-" * W,
        ]

        for r in records:
            pct = r["wall_s"] / total_s * 100 if total_s > 0 else 0.0
            cpu_str = f"{r['mean_cpu_pct']:>10.1f}%" if r["mean_cpu_pct"] == r["mean_cpu_pct"] else "       N/A "
            mem_str = f"{r['peak_mem_mb']:>7.0f} MB" if r["peak_mem_mb"] == r["peak_mem_mb"] else "     N/A"
            lines.append(
                f"  {r['name']:<38} {_fmt_duration(r['wall_s']):>10}  {pct:>6.1f}%  {cpu_str}  {mem_str}"
            )

        lines += [
            "-" * W,
            f"  {'TOTAL':<38} {_fmt_duration(total_s):>10}  {'100.0%':>7}",
            "=" * W,
        ]

        if not _HAS_PSUTIL:
            lines.append("")
            lines.append("  [Note] Install psutil for CPU% and memory stats: pip install psutil")

        return lines


# ---------------------------------------------------------------------------
# Internal context manager
# ---------------------------------------------------------------------------

class _StageCtx:
    def __init__(self, tracker: PerfTracker, name: str):
        self._tracker = tracker
        self._name = name
        self._ctx: Optional[Dict] = None

    def __enter__(self):
        self._ctx = self._tracker._begin(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is not None:
            self._tracker._finish(self._ctx)
        return False  # do not suppress exceptions
