"""Phase 0 — Mode-following relaxation for all molecular precursors/inhibitor.

Runs mode-following on:
  - AllylCpNi  (eta3-allyl + eta5-Cp haptic ligands)
  - secret_inhibitor
  - Ni(PF3)4

Relaxed structures are saved to structures/<name>_relaxed.vasp
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Ensure package is importable
sys.path.insert(0, str(ROOT))

from autoflow_srxn.vibrational.mode_following import run_mode_following


MOLECULES = [
    {
        "name": "AllylCpNi",
        "config": ROOT / "phase0/molecules/AllylCpNi/config.yaml",
        "output_vasp": ROOT / "structures/AllylCpNi_relaxed.vasp",
    },
    {
        "name": "inhibitor",
        "config": ROOT / "phase0/molecules/inhibitor/config.yaml",
        "output_vasp": ROOT / "structures/inhibitor_relaxed.vasp",
    },
    {
        "name": "NiPF3_4",
        "config": ROOT / "phase0/molecules/NiPF3_4/config.yaml",
        "output_vasp": ROOT / "structures/NiPF3_4_relaxed.vasp",
    },
]


def run_all():
    for mol in MOLECULES:
        print("\n" + "=" * 70)
        print(f"  MODE-FOLLOWING RELAXATION: {mol['name']}")
        print("=" * 70)

        config_path = str(mol["config"])
        if not os.path.exists(config_path):
            print(f"  [Skip] Config not found: {config_path}")
            continue

        atoms = run_mode_following(config_path)

        # Copy final structure to structures/ with standardized name
        config_dir = os.path.dirname(os.path.abspath(config_path))
        final_vasp_candidates = [
            os.path.join(config_dir, "results", f"{mol['name']}_final.vasp"),
            os.path.join(config_dir, "results", f"results/{mol['name']}_final.vasp"),
        ]
        # run_mode_following saves to output_prefix + "_final.vasp"
        # output_prefix is "results/<name>" relative to config_dir
        final_vasp = os.path.join(config_dir, f"results/{mol['name']}_final.vasp")

        if os.path.exists(final_vasp):
            shutil.copy2(final_vasp, str(mol["output_vasp"]))
            print(f"  -> Copied to: {mol['output_vasp'].relative_to(ROOT)}")
        else:
            # Fallback: write directly from returned atoms object
            atoms.write(str(mol["output_vasp"]), vasp5=True)
            print(f"  -> Written directly to: {mol['output_vasp'].relative_to(ROOT)}")

        print(f"  [Done] {mol['name']}")


if __name__ == "__main__":
    # Allow running a single molecule: python relax_molecules.py AllylCpNi
    if len(sys.argv) > 1:
        target = sys.argv[1]
        MOLECULES[:] = [m for m in MOLECULES if m["name"] == target]
        if not MOLECULES:
            print(f"Unknown molecule: {target}. Options: AllylCpNi, inhibitor, NiPF3_4")
            sys.exit(1)
    run_all()
