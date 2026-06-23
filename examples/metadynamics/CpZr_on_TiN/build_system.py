#!/usr/bin/env python
"""
Build the CpZr(NMe2)3 precursor on a TiN(100) slab for the metadynamics example.

Pure-geometry construction (no MLIP needed) — writes:
    inputs/CpZr_gas.xyz          gas-phase precursor
    inputs/CpZr_on_TiN.vasp      precursor placed above a surface N site

Tags: slab atoms = 1 (substrate), precursor atoms = 2 (adsorbate), so the
config CV selectors "Zr", "N@substrate", "N@adsorbate" resolve correctly.

The starting geometry is a reasonable physisorbed guess; run_metad.py performs
an MLIP relaxation (7net-0) before sampling.
"""
import os
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write

CC, CH, ZrCp, ZrN, NC = 1.42, 1.09, 2.25, 2.05, 1.46


def _orthonormal(b):
    b = b / np.linalg.norm(b)
    t = np.array([1.0, 0.0, 0.0]) if abs(b[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(b, t); e1 /= np.linalg.norm(e1)
    e2 = np.cross(b, e1)
    return e1, e2


def methyl(C, away_dir):
    """3 H around carbon C, splayed away from the parent (tetrahedral)."""
    b = away_dir / np.linalg.norm(away_dir)
    e1, e2 = _orthonormal(b)
    syms, pos = ["C"], [C]
    for k in range(3):
        psi = np.radians(120 * k)
        d = np.cos(np.radians(70.5)) * b + np.sin(np.radians(70.5)) * (np.cos(psi) * e1 + np.sin(psi) * e2)
        syms.append("H"); pos.append(C + CH * d)
    return syms, pos


def build_precursor():
    syms, pos = [], []
    # Cyclopentadienyl ring in the xy-plane (centroid at origin), H radial outward
    r = CC / (2 * np.sin(np.radians(36)))
    for k in range(5):
        th = np.radians(72 * k)
        c = np.array([r * np.cos(th), r * np.sin(th), 0.0])
        syms.append("C"); pos.append(c)
        syms.append("H"); pos.append(c + CH * np.array([np.cos(th), np.sin(th), 0.0]))
    # Zr below the ring centroid (ligand side = -z, Cp side = +z)
    Zr = np.array([0.0, 0.0, -ZrCp])
    syms.append("Zr"); pos.append(Zr)
    # Three -N(CH3)2 groups, tripod pointing down and out from Zr
    for j in range(3):
        phi = np.radians(120 * j)
        ndir = np.array([np.sin(np.radians(70)) * np.cos(phi),
                         np.sin(np.radians(70)) * np.sin(phi),
                         -np.cos(np.radians(70))])
        N = Zr + ZrN * ndir
        syms.append("N"); pos.append(N)
        a = (N - Zr) / np.linalg.norm(N - Zr)       # away from Zr
        e1, _ = _orthonormal(a)
        for sgn in (+1, -1):
            cdir = np.cos(np.radians(54.75)) * a + sgn * np.sin(np.radians(54.75)) * e1
            C = N + NC * cdir
            ms, mp = methyl(C, C - N)
            syms += ms; pos += mp
    return Atoms(syms, positions=np.array(pos))


def build_slab():
    slab = bulk("TiN", "rocksalt", a=4.235, cubic=True).repeat((3, 3, 2))
    slab.center(vacuum=9.0, axis=2)
    slab.set_tags([1] * len(slab))     # substrate
    return slab


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "inputs")
    os.makedirs(out, exist_ok=True)

    precursor = build_precursor()
    write(os.path.join(out, "CpZr_gas.xyz"), precursor)

    slab = build_slab()
    # pick a top-layer N atom as the target surface site
    z_top = slab.positions[:, 2].max()
    syms = np.array(slab.get_chemical_symbols())
    top = np.where((slab.positions[:, 2] > z_top - 0.5) & (syms == "N"))[0]
    site = slab.positions[top[len(top) // 2]]

    mol = precursor.copy()
    zr_idx = [a.index for a in mol if a.symbol == "Zr"][0]
    # align Zr laterally over the surface N site, then lift so the lowest
    # precursor atom clears the surface by 2.0 Å (clash-free physisorbed start)
    mol.translate([site[0] - mol.positions[zr_idx, 0],
                   site[1] - mol.positions[zr_idx, 1], 0.0])
    mol.translate([0.0, 0.0, (z_top + 2.0) - mol.positions[:, 2].min()])
    mol.set_tags([2] * len(mol))       # adsorbate

    system = slab + mol
    system.center(vacuum=9.0, axis=2)

    # report minimum interatomic distance (clash check)
    d = system.get_all_distances(mic=True)
    np.fill_diagonal(d, 9.9)
    print(f"slab={len(slab)} + precursor={len(mol)} = {len(system)} atoms; "
          f"min interatomic distance = {d.min():.2f} A")
    # extxyz preserves the substrate/adsorbate tags (VASP/POSCAR does not, and
    # also reorders atoms by element) — these tags drive the CV selectors.
    write(os.path.join(out, "CpZr_on_TiN.extxyz"), system)
    write(os.path.join(out, "CpZr_on_TiN.vasp"), system)   # for visualisation only
    print(f"wrote {out}/CpZr_on_TiN.extxyz (+ .vasp for viewing) and CpZr_gas.xyz")


if __name__ == "__main__":
    main()
