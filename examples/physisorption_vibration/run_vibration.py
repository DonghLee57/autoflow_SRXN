"""
run_vibration.py  —  FHVA / PHVA calculation runner
=====================================================
Run from the examples/physisorption_vibration/ directory:

    python run_vibration.py               # runs both FHVA and PHVA
    python run_vibration.py --mode fhva   # FHVA only
    python run_vibration.py --mode phva   # PHVA only

Results are written to:
    results/fhva/qpoints.yaml
    results/phva/qpoints.yaml
"""

import argparse
import os
import sys
import time
import yaml
from ase.io import read

# Ensure the package root is on sys.path when run directly
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.analysis.vibrational_analyzer import VibrationalAnalyzer
from autoflow_srxn.utils.logger_utils import setup_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_structure(config: dict, config_path: str) -> str:
    """Resolve input_structure path relative to the config file location."""
    struct = config.get("paths", {}).get("input_structure")
    if struct is None:
        raise ValueError("'paths.input_structure' not specified in config")
    if not os.path.isabs(struct):
        struct = os.path.join(os.path.dirname(os.path.abspath(config_path)), struct)
    if not os.path.exists(struct):
        raise FileNotFoundError(f"Structure not found: {struct}")
    return struct


def run_one(config_path: str, overwrite: bool = False) -> None:
    """Run a single PHVA or FHVA calculation from a config file."""
    # ----------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config = _load_config(config_path)

    vib_cfg = config.get("analysis", {}).get("vibrational", {})
    name_base = vib_cfg.get("name", "results/vib_analysis")

    # Resolve name_base relative to config directory
    if not os.path.isabs(name_base):
        name_base = os.path.join(config_dir, name_base)
    os.makedirs(name_base, exist_ok=True)

    cache_name = os.path.join(name_base, "cache")
    log_path = os.path.join(name_base, "vibration.log")

    label = os.path.basename(name_base).upper()
    logger = setup_logger(log_path=log_path, verbose=True)

    # ----------------------------------------------------------------
    # Load structure
    # ----------------------------------------------------------------
    struct_path = _resolve_structure(config, config_path)
    atoms = read(struct_path)
    logger.info(f"[{label}] Config  : {os.path.relpath(config_path)}")
    logger.info(f"[{label}] Structure: {os.path.relpath(struct_path)} — {len(atoms)} atoms")

    # ----------------------------------------------------------------
    # Estimate active atoms before running
    # ----------------------------------------------------------------
    engine = SimulationEngine(config=config)
    disp = vib_cfg.get("displacement_ang", 0.01)

    analyzer = VibrationalAnalyzer(
        atoms=atoms,
        engine=engine,
        displacement=disp,
        name=cache_name,
    )

    n_active = len(analyzer.indices) if analyzer.indices else len(atoms)
    n_evals = 2 * 3 * n_active
    logger.info(
        f"[{label}] Active atoms : {n_active} / {len(atoms)}  "
        f"({n_evals} MACE evaluations)"
    )

    # ----------------------------------------------------------------
    # Run analysis
    # ----------------------------------------------------------------
    t0 = time.time()
    logger.info(f"[{label}] Starting vibrational analysis …")
    freqs, _ = analyzer.run_analysis(overwrite=overwrite)
    elapsed = time.time() - t0

    n_imag = sum(1 for f in freqs if f < -0.01)
    n_real = sum(1 for f in freqs if f > 0.01)
    qpoints_path = os.path.join(name_base, "qpoints.yaml")

    logger.info(
        f"[{label}] Done in {elapsed:.1f}s — "
        f"{n_real} real modes, {n_imag} imaginary modes"
    )
    logger.info(f"[{label}] Result: {os.path.relpath(qpoints_path)}")
    print(
        f"  [{label}] {n_real} real / {n_imag} imag modes  "
        f"({elapsed/60:.1f} min)  →  {os.path.relpath(qpoints_path)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run FHVA / PHVA vibrational analysis for DIPAS on SiO₂."
    )
    parser.add_argument(
        "--mode",
        choices=["fhva", "phva", "both"],
        default="both",
        help="Which calculation to run (default: both)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Discard existing cache and recompute from scratch",
    )
    args = parser.parse_args()

    # Run from the example directory so relative paths in configs resolve correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    configs = {
        "fhva": "config_fhva.yaml",
        "phva": "config_phva.yaml",
    }

    to_run = ["fhva", "phva"] if args.mode == "both" else [args.mode]

    for mode in to_run:
        print(f"\n{'='*60}")
        print(f"  Running {mode.upper()} …")
        print(f"{'='*60}")
        run_one(configs[mode], overwrite=args.overwrite)

    print("\nAll calculations complete.")
    print("Run  python analyze_vibration.py  to generate analysis figures.")


if __name__ == "__main__":
    main()
