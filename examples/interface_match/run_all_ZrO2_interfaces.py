"""
Batch runner: t-ZrO2 vs NbO / NbO2 / Nb2O5 / Ta2O5 interface screening.

Runs all four interface pairs sequentially. Each pair writes results to its
own sub-directory under examples/interface_match/<pair>/results/.

Usage
-----
    # From the repo root:
    python examples/interface_match/run_all_ZrO2_interfaces.py

    # With an explicit strain cutoff override:
    python examples/interface_match/run_all_ZrO2_interfaces.py --strain 0.06

Options
-------
    --strain FLOAT   Override strain_cutoff for all pairs (default: from each config.yaml)
    --top-k INT      Override build_top_k for all pairs
    --dry-run        Print resolved configs but skip the actual screening
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)

from autoflow_srxn.utils import load_yaml_config
from autoflow_srxn.interface import run_interface_screening

PAIRS = [
    ("ZrO2_t_NbO",   "t-ZrO2 / NbO"),
    ("ZrO2_t_NbO2",  "t-ZrO2 / NbO2"),
    ("ZrO2_t_Nb2O5", "t-ZrO2 / B-Nb2O5"),
    ("ZrO2_t_Ta2O5", "t-ZrO2 / B-Ta2O5"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch ZrO2 interface screening")
    p.add_argument("--strain", type=float, default=None, help="Override strain_cutoff")
    p.add_argument("--top-k",  type=int,   default=None, help="Override build_top_k")
    p.add_argument("--dry-run", action="store_true",     help="Print config only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0_global = time.time()

    print("=" * 70)
    print("  t-ZrO2 interface batch screening")
    print(f"  Pairs: {len(PAIRS)}")
    if args.strain is not None:
        print(f"  strain_cutoff override: {args.strain}")
    if args.top_k is not None:
        print(f"  build_top_k override:   {args.top_k}")
    if args.dry_run:
        print("  DRY RUN — no structures will be written")
    print("=" * 70)

    results_log: list[dict] = []

    for pair_dir, label in PAIRS:
        cfg_path = os.path.join(HERE, pair_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            print(f"\n[SKIP] {label}: config not found at {cfg_path}")
            continue

        cfg = load_yaml_config(cfg_path)

        # Apply overrides
        if args.strain is not None:
            cfg["strain_cutoff"] = args.strain
        if args.top_k is not None:
            cfg["build_top_k"] = args.top_k

        # Make output_dir relative to the pair directory
        rel_out = cfg.get("output_dir", "results")
        cfg["output_dir"] = os.path.join(HERE, pair_dir, rel_out)

        # Make structure paths absolute
        pair_root = os.path.join(HERE, pair_dir)
        for key in ("sub_path", "film_path"):
            p = cfg.get(key, "")
            if p and not os.path.isabs(p):
                cfg[key] = os.path.normpath(os.path.join(pair_root, p))

        print(f"\n{'─'*70}")
        print(f"  Pair : {label}")
        print(f"  Sub  : {cfg.get('sub_path')}")
        print(f"  Film : {cfg.get('film_path')}")
        print(f"  Out  : {cfg.get('output_dir')}")

        if args.dry_run:
            print("  [dry-run] Skipping execution.")
            continue

        t0 = time.time()
        try:
            run_interface_screening(cfg)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")
            results_log.append({"pair": label, "status": "OK", "elapsed": elapsed})
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  ERROR: {exc}")
            results_log.append({"pair": label, "status": f"ERROR: {exc}", "elapsed": elapsed})

    total = time.time() - t0_global
    print(f"\n{'='*70}")
    print(f"  Batch complete in {total:.1f}s")
    if results_log:
        for r in results_log:
            status = r["status"]
            print(f"    {r['pair']:30s}  {status:6s}  ({r['elapsed']:.1f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
