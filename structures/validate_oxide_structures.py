"""
Quick sanity check for the four oxide bulk VASP files.

Checks:
  - File parses without error (via ASE)
  - Correct element types present
  - Expected atom count
  - No overlapping atoms (min interatomic distance > 1.5 Ang)
  - Lattice parameters match target values within tolerance

Usage:  python structures/validate_oxide_structures.py
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

TARGETS = {
    "NbO_bulk.vasp": {
        "elements": {"Nb", "O"},
        "n_atoms": 6,
        "a": 4.2099, "b": 4.2099, "c": 4.2099,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
    },
    "NbO2_bulk.vasp": {
        "elements": {"Nb", "O"},
        "n_atoms": 6,
        "a": 4.8360, "b": 4.8360, "c": 2.9900,
        "alpha": 90.0, "beta": 90.0, "gamma": 90.0,
    },
    "Nb2O5_B_bulk.vasp": {
        "elements": {"Nb", "O"},
        "n_atoms": 28,
        "a": 12.727, "b": 4.880, "c": None,  # c is tilted
    },
    "Ta2O5_B_bulk.vasp": {
        "elements": {"Ta", "O"},
        "n_atoms": 28,
        "a": 12.780, "b": 4.900, "c": None,
    },
}

TOL_DIST = 0.05   # Angstrom tolerance on lattice parameters
MIN_BOND = 1.50   # Angstrom minimum allowed interatomic distance


def check_file(fname: str, tgt: dict) -> list[str]:
    errors = []
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        return [f"MISSING: {path}"]

    try:
        from ase.io import read
        atoms = read(path, format="vasp")
    except Exception as e:
        return [f"PARSE ERROR: {e}"]

    # Element check
    actual_elements = set(atoms.get_chemical_symbols())
    if not tgt["elements"].issubset(actual_elements):
        errors.append(f"elements expected {tgt['elements']}, got {actual_elements}")

    # Atom count
    if len(atoms) != tgt["n_atoms"]:
        errors.append(f"n_atoms expected {tgt['n_atoms']}, got {len(atoms)}")

    # Lattice parameters
    cell = atoms.cell
    lengths = cell.lengths()
    a_val, b_val, c_val = lengths
    if abs(a_val - tgt["a"]) > TOL_DIST:
        errors.append(f"a={a_val:.4f} expected {tgt['a']:.4f}")
    if abs(b_val - tgt["b"]) > TOL_DIST:
        errors.append(f"b={b_val:.4f} expected {tgt['b']:.4f}")
    if tgt.get("c") is not None and abs(c_val - tgt["c"]) > TOL_DIST:
        errors.append(f"c={c_val:.4f} expected {tgt['c']:.4f}")

    # Min interatomic distance (PBC-aware, via neighbour list)
    try:
        from ase.neighborlist import neighbor_list
        i_idx, j_idx, dists = neighbor_list("ijd", atoms, cutoff=4.0)
        if len(dists) == 0:
            errors.append("no neighbours found within 4 Ang (cell too small?)")
        else:
            min_d = float(np.min(dists))
            if min_d < MIN_BOND:
                errors.append(f"min_interatomic_dist={min_d:.3f} < {MIN_BOND} Ang (atom overlap?)")
    except Exception as e:
        errors.append(f"distance check failed: {e}")

    return errors


def main() -> None:
    all_ok = True
    print("Oxide structure validation")
    print("=" * 60)
    for fname, tgt in TARGETS.items():
        errs = check_file(fname, tgt)
        if errs:
            all_ok = False
            print(f"  FAIL  {fname}")
            for e in errs:
                print(f"         → {e}")
        else:
            path = os.path.join(HERE, fname)
            from ase.io import read
            atoms = read(path, format="vasp")
            lengths = atoms.cell.lengths()
            angles  = atoms.cell.angles()
            print(f"  OK    {fname}  ({len(atoms)} atoms)  "
                  f"a={lengths[0]:.3f} b={lengths[1]:.3f} c={lengths[2]:.3f} Ang  "
                  f"α={angles[0]:.1f} β={angles[1]:.1f} γ={angles[2]:.1f}°")
    print("=" * 60)
    if all_ok:
        print("All structures OK.")
    else:
        print("Some checks FAILED - review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
