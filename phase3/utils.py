"""Shared geometry / IO helpers for phase3 adsorption scripts."""
import csv
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.geometry import get_distances

from autoflow_srxn.surface.surface_utils import (
    standardize_vasp_atoms,
    get_pair_bond_cutoff,
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def center_molecule(mol):
    """Translate molecule so its geometric center is at origin."""
    mol = mol.copy()
    pos = mol.get_positions()
    mol.set_positions(pos - pos.mean(axis=0))
    return mol


def orient_atom_toward_surface(mol, atom_idx):
    """Rodrigues rotation: move atom_idx to lowest z (toward -z = toward surface)."""
    mol = mol.copy()
    pos = mol.get_positions()
    com = pos.mean(axis=0)
    pos -= com

    v = pos[atom_idx]
    nv = np.linalg.norm(v)
    if nv < 1e-6:
        mol.set_positions(pos)
        return mol

    v_hat = v / nv
    target = np.array([0.0, 0.0, -1.0])
    axis   = np.cross(v_hat, target)
    sin_a  = np.linalg.norm(axis)
    cos_a  = np.dot(v_hat, target)

    if sin_a < 1e-6:
        if cos_a > 0:
            mol.set_positions(pos)
        else:
            R = np.diag([-1.0, 1.0, -1.0])
            mol.set_positions(pos @ R.T)
        return mol

    axis /= sin_a
    K = np.array([[     0, -axis[2],  axis[1]],
                  [ axis[2],      0, -axis[0]],
                  [-axis[1],  axis[0],     0]])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    mol.set_positions(pos @ R.T)
    return mol


def spin_z(mol, deg):
    """In-plane rotation around z through origin."""
    mol = mol.copy()
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    mol.set_positions(mol.get_positions() @ Rz.T)
    return mol


def place_center_on_site(slab, mol, site_xyz, center_idx, height):
    """Flat-placement: translate mol so center_idx lands at (site_x, site_y, site_z+height).

    Does NOT orient any atom toward surface — uses current mol orientation.
    Returns combined slab+mol Atoms with molecule atoms tagged=2.
    """
    mol = mol.copy()
    pos = mol.get_positions()
    target = np.array([site_xyz[0], site_xyz[1], site_xyz[2] + height])
    mol.set_positions(pos + (target - pos[center_idx]))

    combined = slab.copy()
    tags = list(combined.get_tags())
    for a in mol:
        combined.append(a)
        tags.append(2)
    combined.set_tags(tags)
    return combined


def place_atom_on_site(slab, mol, site_xyz, atom_idx, height):
    """Chemisorption placement: mol already oriented, atom_idx at site_z+height."""
    return place_center_on_site(slab, mol, site_xyz, atom_idx, height)


# ---------------------------------------------------------------------------
# Interface analysis
# ---------------------------------------------------------------------------

def interface_analysis(atoms, bond_slack=0.25, max_cutoff=2.8):
    """Return (min_dist, min_pair_str, n_covalent_bonds) at mol-slab interface."""
    tags = atoms.get_tags()
    mol_idx = [i for i, t in enumerate(tags) if t >= 2]
    sub_idx = [i for i, t in enumerate(tags) if t <  2]
    if not mol_idx or not sub_idx:
        return 999.0, "--", 0

    mind, minpair, nb = 999.0, "--", 0
    for i in mol_idx:
        _, d = get_distances(
            atoms.positions[i], atoms.positions[sub_idx],
            cell=atoms.cell, pbc=atoms.pbc,
        )
        for k, j in enumerate(sub_idx):
            dd = float(d[0][k])
            si, sj = atoms.symbols[i], atoms.symbols[j]
            if dd < mind:
                mind = dd
                minpair = f"{si}-{sj}"
            try:
                cut = get_pair_bond_cutoff(si, sj, bond_slack=bond_slack,
                                           max_cutoff=max_cutoff)
            except Exception:
                cut = 2.3
            if dd < cut:
                nb += 1
    return mind, minpair, nb


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

def relax_and_score(atoms, engine, e_slab, e_gas,
                    frozen_z=5.5, steps=250, fmax=0.05):
    """standardize → relax → return (atoms_r, e_ads, mind, mpair, nb)."""
    atoms_r = standardize_vasp_atoms(atoms.copy(), z_min_offset=0.5)
    engine.relax(atoms_r, frozen_z_ang=frozen_z, steps=steps, fmax=fmax,
                 verbose=False)
    e_r   = float(atoms_r.get_potential_energy())
    e_ads = e_r - e_slab - e_gas
    mind, mpair, nb = interface_analysis(atoms_r)
    return atoms_r, e_ads, mind, mpair, nb


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

E_ADS_MAX = 8.0    # eV — reject unphysically HIGH energy (endothermic blow-up)
E_ADS_MIN = -500.0 # eV — reject catastrophic atomic overlap (ML potential divergence)


def _clean_info(atoms):
    """Remove non-serializable entries from atoms.info before extxyz write."""
    bad = []
    for k, v in list(atoms.info.items()):
        if isinstance(v, (Atoms, dict, list)):
            bad.append(k)
        else:
            try:
                json.dumps(v)
            except (TypeError, ValueError):
                bad.append(k)
    for k in bad:
        del atoms.info[k]
    return atoms


def write_results(sub_dir: Path, label: str, relaxed: list,
                  extra_fields: list = None):
    """Sort by E_ads, write VASP / extxyz / CSV.

    relaxed : list of (e_ads, atoms, mind, mpair, nb, info_dict)
    Returns  : list of csv_row dicts
    """
    relaxed.sort(key=lambda x: x[0])
    csv_rows      = []
    atoms_by_size = {}   # natoms -> [Atoms, ...]

    for rank, (e_ads, atoms_r, mind, mpair, nb, info) in enumerate(relaxed, 1):
        out_name = f"{label}_rank{rank:02d}.vasp"
        write(str(sub_dir / out_name), atoms_r, vasp5=True)

        physical = (E_ADS_MIN <= e_ads <= E_ADS_MAX)
        row = {
            "rank":            rank,
            "e_ads_eV":        f"{e_ads:.6f}",
            "min_dist_A":      f"{mind:.3f}",
            "min_pair":        mpair,
            "interface_bonds": nb,
            "physical":        ("Y" if physical else
                               ("N(low-E)" if e_ads < E_ADS_MIN else "N(high-E)")),
            "output":          str((sub_dir / out_name)),
        }
        row.update(info)
        csv_rows.append(row)

        if physical:
            n = len(atoms_r)
            atoms_by_size.setdefault(n, []).append(_clean_info(atoms_r.copy()))

    # extxyz grouped by atom count
    for n, group in atoms_by_size.items():
        write(str(sub_dir / f"{label}_n{n}.extxyz"), group)
    all_sizes = list(atoms_by_size.keys())
    if len(all_sizes) == 1:
        write(str(sub_dir / f"{label}_ranked.extxyz"), atoms_by_size[all_sizes[0]])
    elif len(all_sizes) > 1:
        print(f"  [Note] Mixed atom counts {all_sizes}: {len(all_sizes)} extxyz files.")

    # CSV
    base_keys = ["rank", "e_ads_eV", "min_dist_A", "min_pair",
                 "interface_bonds", "physical"]
    extra      = [k for k in csv_rows[0] if k not in base_keys + ["output"]]
    fieldnames = base_keys + extra + ["output"]
    with open(sub_dir / f"{label}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(csv_rows)

    return csv_rows


# ---------------------------------------------------------------------------
# Site map loader
# ---------------------------------------------------------------------------

def load_sites(csv_path):
    """Load site map CSV, return list of dicts with float x_A, y_A, z_A."""
    sites = []
    for row in csv.DictReader(open(csv_path)):
        row["x_A"] = float(row["x_A"])
        row["y_A"] = float(row["y_A"])
        row["z_A"] = float(row["z_A"])
        sites.append(row)
    return sites
