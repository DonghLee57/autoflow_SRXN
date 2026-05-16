"""Phase 0 — Full cell relaxation (volume + shape + positions) for bulk structures.

Uses ASE ExpCellFilter with SevenNet-0 (7net-0). Equivalent to VASP ISIF=3.

Outputs:
  structures/Si_relaxed.vasp
  structures/SiO2_relaxed.vasp
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.filters import ExpCellFilter
from ase.io import read, write
from ase.optimize import FIRE

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.utils.logger_utils import setup_logger

BULKS = [
    {
        "name": "Si",
        "input": ROOT / "structures/Si_mp149.vasp",
        "output": ROOT / "structures/Si_relaxed.vasp",
    },
    {
        "name": "SiO2",
        "input": ROOT / "structures/SiO2_mp-546794.vasp",
        "output": ROOT / "structures/SiO2_relaxed.vasp",
    },
]

CONFIG = {
    "engine": {
        "potential": {
            "backend": "sevennet",
            "model": "7net-0",
            "device": "cpu",
            "dtype": "float32",
        }
    }
}

FMAX = 0.01    # eV/A  — tighter for bulk to converge cell params
STEPS = 600


def relax_bulk_cell(atoms, calc, fmax=FMAX, steps=STEPS, name="bulk"):
    """Full cell relaxation: positions + cell shape + volume (ExpCellFilter)."""
    atoms.calc = calc

    print(f"\n  Initial cell: {atoms.cell.lengths()}")
    print(f"  Initial volume: {atoms.get_volume():.3f} A^3")

    ecf = ExpCellFilter(atoms)
    opt = FIRE(ecf, logfile="-")
    opt.run(fmax=fmax, steps=steps)

    print(f"  Final cell:   {atoms.cell.lengths()}")
    print(f"  Final volume: {atoms.get_volume():.3f} A^3")
    return atoms


def run_all(targets=None):
    logger = setup_logger(log_path="phase0/bulk_relax.log", verbose=True)

    engine = SimulationEngine(config=CONFIG)
    calc = engine.get_calculator()

    for bulk in BULKS:
        if targets and bulk["name"] not in targets:
            continue

        print("\n" + "=" * 70)
        print(f"  BULK CELL RELAXATION: {bulk['name']}")
        print("=" * 70)

        if not bulk["input"].exists():
            print(f"  [Skip] Input not found: {bulk['input']}")
            continue

        atoms = read(str(bulk["input"]))
        print(f"  Loaded: {bulk['input'].name}  ({len(atoms)} atoms, {atoms.get_chemical_formula()})")
        print(f"  PBC: {atoms.pbc}")

        atoms = relax_bulk_cell(atoms, calc, name=bulk["name"])

        write(str(bulk["output"]), atoms, vasp5=True)
        print(f"  -> Saved: {bulk['output'].relative_to(ROOT)}")

        e_per_atom = atoms.get_potential_energy() / len(atoms)
        logger.info(
            f"{bulk['name']}: E/atom={e_per_atom:.4f} eV, "
            f"V={atoms.get_volume():.3f} A^3, "
            f"cell={atoms.cell.lengths()}"
        )


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    run_all(targets)
