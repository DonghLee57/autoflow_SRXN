"""Sort molecular structure files by atomic number (ascending Z).

Sorts both original and relaxed VASP files for:
  - AllylCpNi (original + relaxed)
  - secret_inhibitor (original + relaxed)
  - NiPF3_4 (original + relaxed)

Atomic number reference: ase.data.atomic_numbers
Sorting: ascending Z (H=1, C=6, N=7, O=8, F=9, P=15, Ni=28, ...)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.data import atomic_numbers
from ase.io import read, write

MOLECULE_FILES = [
    ROOT / "structures/AllylCpNi.vasp",
    ROOT / "structures/AllylCpNi_relaxed.vasp",
    ROOT / "structures/secret_inhibitor.vasp",
    ROOT / "structures/inhibitor_relaxed.vasp",
    ROOT / "structures/NiPF3_4.vasp",
    ROOT / "structures/NiPF3_4_relaxed.vasp",
]


def sort_by_atomic_number(atoms):
    """Return a copy of atoms sorted by ascending atomic number (Z)."""
    z_vals = [atomic_numbers[s] for s in atoms.symbols]
    order  = np.argsort(z_vals, kind="stable")  # stable sort preserves relative order of same-Z atoms
    return atoms[order]


def formula_with_z(atoms):
    """Compact display showing element: Z for each unique species."""
    seen = {}
    for s in atoms.symbols:
        z = atomic_numbers[s]
        seen.setdefault(s, z)
    parts = [f"{s}(Z={z})" for s, z in sorted(seen.items(), key=lambda x: x[1])]
    return ", ".join(parts)


def main():
    print("=" * 60)
    print("SORT MOLECULAR STRUCTURES BY ATOMIC NUMBER (ascending Z)")
    print("=" * 60)

    for path in MOLECULE_FILES:
        if not path.exists():
            print(f"  [Skip] Not found: {path.name}")
            continue

        atoms = read(str(path))
        before = atoms.get_chemical_formula(mode="all")  # ordered formula
        z_before = [atomic_numbers[s] for s in atoms.symbols]

        sorted_atoms = sort_by_atomic_number(atoms)
        after = sorted_atoms.get_chemical_formula(mode="all")
        z_after = [atomic_numbers[s] for s in sorted_atoms.symbols]

        already_sorted = (z_before == z_after)

        print(f"\n  {path.name}")
        print(f"    Species present : {formula_with_z(atoms)}")
        print(f"    Before sort     : {before}")
        print(f"    After sort      : {after}")
        print(f"    Already sorted  : {already_sorted}")

        write(str(path), sorted_atoms, vasp5=True)
        print(f"    -> Written (sorted): {path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
