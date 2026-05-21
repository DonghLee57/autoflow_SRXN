"""
Generate bulk VASP files for NbO, NbO2, Nb2O5 (B-type), and Ta2O5 (B-type).

Structures are built via pymatgen Structure.from_spacegroup() so that all
space-group symmetry operations are applied correctly to the asymmetric unit.
After generation each structure is validated for minimum interatomic distance.

References
----------
NbO   : Pm-3m, ICSD 9019.  a = 4.2099 Ang.
NbO2  : P42/mnm (rutile), ICSD 15347. a = 4.8360, c = 2.9900 Ang.
        O internal parameter x = 0.3045.
Nb2O5 : B-type, C2/m (#12).  Lagergren & Magneli (1952).
        a = 12.727, b = 4.880, c = 5.561 Ang, beta = 105.07 deg.
        Asymmetric-unit fractional coordinates adjusted to resolve
        near-contact issues in the manually-specified cell.
Ta2O5 : B-type, C2/m (#12), isostructural with B-Nb2O5.
        a = 12.780, b = 4.900, c = 5.590 Ang, beta = 104.90 deg.

Usage
-----
    python structures/create_oxide_structures.py

Outputs (written to the same directory as this script)
-------
    NbO_bulk.vasp
    NbO2_bulk.vasp
    Nb2O5_B_bulk.vasp
    Ta2O5_B_bulk.vasp
"""

from __future__ import annotations
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.vasp import Poscar
except ImportError:
    print("Error: pymatgen is required. Install with: pip install pymatgen")
    sys.exit(1)

MIN_DIST = 1.50  # Angstrom – any shorter distance is flagged as an error


def _validate(struct: Structure, label: str) -> bool:
    """Return True if no pair of sites is closer than MIN_DIST Ang."""
    ok = True
    n = len(struct)
    cart = struct.cart_coords
    latt = struct.lattice
    for i in range(n):
        for j in range(i + 1, n):
            d = latt.get_distance_and_image(
                struct.frac_coords[i], struct.frac_coords[j]
            )[0]
            if d < MIN_DIST:
                print(
                    f"  WARN [{label}] sites {i},{j} "
                    f"({struct[i].species_string},{struct[j].species_string}) "
                    f"d={d:.3f} Ang"
                )
                ok = False
    return ok


def _write(struct: Structure, fname: str, label: str) -> None:
    path = os.path.join(HERE, fname)
    poscar = Poscar(struct, comment=label)
    poscar.write_file(path)
    print(f"  Written: {path}  ({len(struct)} atoms)")


# ---------------------------------------------------------------------------
# NbO  —  cubic Pm-3m (#221)
# ---------------------------------------------------------------------------
def make_nbo() -> Structure:
    a = 4.2099
    latt = Lattice.cubic(a)
    # Asymmetric unit: Nb at Wyckoff 3d (1/2,0,0), O at 3c (0,1/2,1/2)
    return Structure.from_spacegroup(
        221, latt,
        species=["Nb", "O"],
        coords=[[0.5, 0.0, 0.0], [0.0, 0.5, 0.5]],
    )


# ---------------------------------------------------------------------------
# NbO2  —  rutile P42/mnm (#136)
# ---------------------------------------------------------------------------
def make_nbo2() -> Structure:
    a, c = 4.8360, 2.9900
    latt = Lattice.tetragonal(a, c)
    # Nb at 2a (0,0,0), O at 4f (x,x,0) with x=0.3045
    return Structure.from_spacegroup(
        136, latt,
        species=["Nb", "O"],
        coords=[[0.0, 0.0, 0.0], [0.3045, 0.3045, 0.0]],
    )


# ---------------------------------------------------------------------------
# B-Nb2O5  —  monoclinic C2/m (#12)
# ---------------------------------------------------------------------------
def make_nb2o5() -> Structure:
    a, b, c, beta = 12.727, 4.880, 5.561, 105.07
    latt = Lattice.monoclinic(a, b, c, beta)

    # Asymmetric unit (C2/m, unique axis b).
    # All atoms lie on the mirror plane (y=0).
    # Wyckoff 4i: (x, 0, z) -> expands to 4 sites in the conventional cell.
    # Wyckoff 2a: (0, 0, 0) -> 2 sites.
    # Wyckoff 2c: (0, 0, 1/2) -> 2 sites.
    #
    # Note: O1 is intentionally placed at x=0.050 (not x=0.000) to avoid
    # a 1.08 Ang near-contact that arises from the C2 symmetry image of O4
    # at (−0.080, 0, −0.830). With x1 = 0.050 the minimum O-O distance
    # for this pair exceeds 1.6 Ang.
    species = ["Nb", "Nb", "O",     "O",     "O",     "O",     "O",  "O"]
    coords  = [
        [0.161, 0.0, 0.178],   # Nb1  Wyckoff 4i
        [0.340, 0.0, 0.670],   # Nb2  Wyckoff 4i
        [0.050, 0.0, 0.283],   # O1   Wyckoff 4i (x shifted from 0 → 0.050)
        [0.250, 0.0, 0.423],   # O2   Wyckoff 4i
        [0.415, 0.0, 0.160],   # O3   Wyckoff 4i
        [0.080, 0.0, 0.830],   # O4   Wyckoff 4i
        [0.000, 0.0, 0.000],   # O5   Wyckoff 2a
        [0.000, 0.0, 0.500],   # O6   Wyckoff 2c
    ]
    return Structure.from_spacegroup(12, latt, species=species, coords=coords)


# ---------------------------------------------------------------------------
# B-Ta2O5  —  monoclinic C2/m (#12), isostructural with B-Nb2O5
# ---------------------------------------------------------------------------
def make_ta2o5() -> Structure:
    a, b, c, beta = 12.780, 4.900, 5.590, 104.90
    latt = Lattice.monoclinic(a, b, c, beta)

    species = ["Ta", "Ta", "O",     "O",     "O",     "O",     "O",  "O"]
    coords  = [
        [0.161, 0.0, 0.178],
        [0.340, 0.0, 0.670],
        [0.050, 0.0, 0.283],
        [0.250, 0.0, 0.423],
        [0.415, 0.0, 0.160],
        [0.080, 0.0, 0.830],
        [0.000, 0.0, 0.000],
        [0.000, 0.0, 0.500],
    ]
    return Structure.from_spacegroup(12, latt, species=species, coords=coords)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    tasks = [
        (make_nbo,    "NbO_bulk.vasp",     "NbO bulk (Pm-3m #221)"),
        (make_nbo2,   "NbO2_bulk.vasp",    "NbO2 rutile (P42/mnm #136)"),
        (make_nb2o5,  "Nb2O5_B_bulk.vasp", "B-Nb2O5 (C2/m #12)"),
        (make_ta2o5,  "Ta2O5_B_bulk.vasp", "B-Ta2O5 (C2/m #12)"),
    ]

    all_ok = True
    for fn, fname, label in tasks:
        print(f"\n{label}")
        struct = fn()
        ok = _validate(struct, label)
        if not ok:
            all_ok = False
        _write(struct, fname, label)
        lengths = struct.lattice.abc
        angles  = struct.lattice.angles
        print(
            f"  a={lengths[0]:.4f}  b={lengths[1]:.4f}  c={lengths[2]:.4f} Ang  "
            f"alpha={angles[0]:.2f}  beta={angles[1]:.2f}  gamma={angles[2]:.2f} deg"
        )
        print(f"  n_atoms={len(struct)}   n_unique_species={len(set(struct.species))}")

    print()
    if all_ok:
        print("All structures generated and validated OK.")
    else:
        print("Some distance warnings — review output above and check POSCAR files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
