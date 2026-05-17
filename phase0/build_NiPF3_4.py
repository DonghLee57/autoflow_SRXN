"""Build a tetrahedral Ni(PF3)4 gas-phase structure and write to structures/NiPF3_4.vasp.

Geometry:
  - Ni at origin, Td symmetry
  - Ni-P bond: 2.05 A  (Ni(0)-P in d10 complex)
  - P-F bond:  1.57 A
  - Ni-P-F angle: 117 deg (F cone opening away from Ni)
  - F-P-F angle: ~99 deg (pyramidal PF3)
"""

import numpy as np
from pathlib import Path
from ase import Atoms
from ase.io import write


def _perp(u):
    """Return a unit vector perpendicular to u."""
    v = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v, u)) > 0.85:
        v = np.array([0.0, 1.0, 0.0])
    w = np.cross(u, v)
    return w / np.linalg.norm(w)


def build_NiPF3_4(ni_p=2.05, p_f=1.57, ni_p_f_deg=117.0):
    # Tetrahedral unit vectors (Td point group)
    tet = np.array([
        [ 1,  1,  1],
        [ 1, -1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
    ], dtype=float)
    tet /= np.linalg.norm(tet, axis=1, keepdims=True)

    theta = np.deg2rad(ni_p_f_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    symbols = ["Ni"]
    positions = [np.zeros(3)]

    for u in tet:
        p_pos = ni_p * u
        symbols.append("P")
        positions.append(p_pos)

        # Perpendicular basis around Ni-P axis
        e1 = _perp(u)
        e2 = np.cross(u, e1)
        # P->Ni direction = -u; F cone angle theta from -u
        for k in range(3):
            phi = 2 * np.pi * k / 3
            pf_dir = cos_t * (-u) + sin_t * (np.cos(phi) * e1 + np.sin(phi) * e2)
            pf_dir /= np.linalg.norm(pf_dir)
            symbols.append("F")
            positions.append(p_pos + p_f * pf_dir)

    mol = Atoms(symbols=symbols, positions=positions)
    mol.center(vacuum=10.0)
    return mol


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    out = root / "structures" / "NiPF3_4.vasp"
    mol = build_NiPF3_4()
    write(str(out), mol, vasp5=True)
    print(f"Written: {out}  ({len(mol)} atoms: {mol.get_chemical_formula()})")
    print(f"  Ni-P distances: {[f'{d:.3f}' for d in [np.linalg.norm(mol.positions[i] - mol.positions[0]) for i in range(1, 5)]]} A")
