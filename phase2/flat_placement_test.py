"""Hardcoded flat-placement physisorption test.

For each candidate site on each substrate:
  1. PCA-align inhibitor so its flat face points toward the surface
     (PC3 = smallest-variance axis → +z; PC1,PC2 in xy-plane).
  2. Place center C atom (idx=13) at (site_x, site_y, site_z + HEIGHT_ANG).
  3. Try N_SPIN in-plane rotations (0, 90, 180, 270 deg around z) to avoid
     azimuthal bias while keeping the face-down orientation fixed.
  4. Relax with FIRE (frozen bottom 5.5 A, fmax=0.05 eV/A).
  5. Rank by E_ads and compare with the existing Fibonacci-sphere results.

Outputs: phase2/results/flat_placement/<substrate>/
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.geometry import get_distances

from autoflow_srxn.surface.surface_utils import (
    standardize_vasp_atoms,
    get_pair_bond_cutoff,
)
from autoflow_srxn.simulation.potentials import SimulationEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HEIGHT_ANG = 2.3          # center-C to site-z distance (A)
N_SPIN     = 4            # azimuthal rotations: 0, 90, 180, 270 deg
CENTER_IDX = 13           # 0-based index of center C in inhibitor
FROZEN_Z   = 5.5          # A
FMAX       = 0.05         # eV/A
RELAX_STEPS = 200

ENGINE_CONFIG = {
    "engine": {
        "potential": {
            "backend": "sevennet",
            "model": "7net-0",
            "device": "cpu",
            "dtype": "float32",
        }
    },
    "relaxation": {
        "fmax": FMAX,
        "steps": RELAX_STEPS,
        "optimizer": "FIRE",
        "frozen_z_ang": FROZEN_Z,
    },
}

OUT_DIR = ROOT / "phase2/results/flat_placement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSTRATES = [
    ("Si100",        ROOT / "structures/slabs/Si100_slab.vasp",
                     ROOT / "structures/slabs/site_maps/Si100_sites.csv"),
    ("SiO2_O_term",  ROOT / "structures/slabs/SiO2_O_term_slab.vasp",
                     ROOT / "structures/slabs/site_maps/SiO2_O_term_sites.csv"),
    ("SiO2_Si_term", ROOT / "structures/slabs/SiO2_Si_term_slab.vasp",
                     ROOT / "structures/slabs/site_maps/SiO2_Si_term_sites.csv"),
]

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def pca_flat_align(mol):
    """Return a copy of mol rotated so PC3 (smallest variance) -> +z.

    After rotation:
      - PC1 (largest spread) -> +x
      - PC2 -> +y
      - PC3 (thickness) -> +z
    COM is translated to origin.
    """
    mol = mol.copy()
    pos = mol.get_positions()
    com = pos.mean(axis=0)
    pos_c = pos - com

    cov = np.cov(pos_c.T)
    vals, vecs = np.linalg.eigh(cov)     # ascending eigenvalues
    # reorder: largest -> PC1(x), middle -> PC2(y), smallest -> PC3(z)
    order = np.argsort(vals)[::-1]
    R = vecs[:, order]                   # rotation matrix: cols = new axes in old frame

    # Ensure right-handedness
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    new_pos = pos_c @ R                  # rotate: each row is a position vector
    mol.set_positions(new_pos)
    mol.set_cell([50, 50, 50])
    mol.set_pbc([False, False, False])
    return mol


def spin_z(mol, deg):
    """Rotate mol in-plane (around z through its COM)."""
    mol = mol.copy()
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    Rz = np.array([[c, -s, 0],
                   [s,  c, 0],
                   [0,  0, 1]])
    mol.set_positions(mol.get_positions() @ Rz.T)
    return mol


def place_on_slab(slab, mol_aligned, site_xyz, center_idx):
    """Place mol so center atom is at (site_x, site_y, site_z + HEIGHT_ANG).

    Returns combined slab+mol Atoms with mol atoms tagged=2.
    """
    mol = mol_aligned.copy()
    pos = mol.get_positions()

    # translate so center atom lands at target
    target = np.array([site_xyz[0], site_xyz[1], site_xyz[2] + HEIGHT_ANG])
    shift = target - pos[center_idx]
    mol.set_positions(pos + shift)

    combined = slab.copy()
    combined_tags = list(combined.get_tags())
    for a in mol:
        combined.append(a)
        combined_tags.append(2)
    combined.set_tags(combined_tags)
    return combined


def interface_analysis(atoms):
    """(min_dist, min_pair_str, n_covalent_bonds)."""
    tags = atoms.get_tags()
    mol = [i for i, t in enumerate(tags) if t >= 2]
    sub = [i for i, t in enumerate(tags) if t < 2]
    if not mol or not sub:
        return 999.0, "--", 0

    mind, minpair, nb = 999.0, "--", 0
    for i in mol:
        D, d = get_distances(
            atoms.positions[i], atoms.positions[sub],
            cell=atoms.cell, pbc=atoms.pbc,
        )
        for k, j in enumerate(sub):
            dd = float(d[0][k])
            si, sj = atoms.symbols[i], atoms.symbols[j]
            if dd < mind:
                mind = dd
                minpair = f"{si}-{sj}"
            try:
                cut = get_pair_bond_cutoff(si, sj, bond_slack=0.25, max_cutoff=2.6)
            except Exception:
                cut = 2.2
            if dd < cut:
                nb += 1
    return mind, minpair, nb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    targets = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    engine = SimulationEngine(ENGINE_CONFIG)
    calc   = engine.get_calculator()

    # Gas-phase inhibitor energy
    inhibitor_raw = read(str(ROOT / "structures/inhibitor_relaxed.vasp"))
    gas = inhibitor_raw.copy()
    gas.center(vacuum=10.0)
    gas.calc = calc
    e_gas = float(gas.get_potential_energy())
    print(f"Gas inhibitor energy: {e_gas:.6f} eV")
    print(f"Formula: {gas.get_chemical_formula()}")
    print(f"Center atom (idx={CENTER_IDX}): {gas.get_chemical_symbols()[CENTER_IDX]}")

    # PCA-align inhibitor (COM at origin, PC3->z)
    inh_flat = pca_flat_align(inhibitor_raw)
    pos_flat = inh_flat.get_positions()
    pc3_extent = pos_flat[:, 2].max() - pos_flat[:, 2].min()
    print(f"\nFlat-aligned inhibitor: PC3-extent (thickness) = {pc3_extent:.2f} A")
    print(f"  Center C z in flat frame: {pos_flat[CENTER_IDX, 2]:.3f} A")

    all_summary = [
        "Phase 2 Flat-Placement Physisorption Test",
        "=" * 72,
        f"HEIGHT_ANG = {HEIGHT_ANG} A  (center-C to site_z)",
        f"N_SPIN     = {N_SPIN}  (0/90/180/270 deg in-plane rotations)",
        f"CENTER_IDX = {CENTER_IDX} ({gas.get_chemical_symbols()[CENTER_IDX]})",
        f"e_gas      = {e_gas:.6f} eV",
        "",
    ]

    for name, slab_path, csv_path in SUBSTRATES:
        if targets and name not in targets:
            continue

        sub_dir = OUT_DIR / name
        sub_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*72}")
        print(f"  Substrate: {name}")
        print(f"{'='*72}")

        slab = read(str(slab_path))
        slab.calc = calc
        e_slab = float(slab.get_potential_energy())
        print(f"  Slab energy: {e_slab:.6f} eV")

        # Load sites
        sites = list(csv.DictReader(open(csv_path)))
        print(f"  Sites: {len(sites)}")

        candidates = []   # (cand_id, site_id, spin_deg, atoms_combined)
        for site in sites:
            site_xyz = np.array([float(site["x_A"]),
                                  float(site["y_A"]),
                                  float(site["z_A"])])
            for spin_deg in range(0, 360, 360 // N_SPIN):
                mol = spin_z(inh_flat, spin_deg)
                combined = place_on_slab(slab, mol, site_xyz, CENTER_IDX)
                cid = len(candidates)
                candidates.append((cid, site["site_id"], spin_deg, combined))

        print(f"  Candidates generated: {len(candidates)} "
              f"({len(sites)} sites x {N_SPIN} spins)")

        # Single-point pre-screen
        screened = []
        for cid, site_id, spin_deg, atoms in candidates:
            atoms.calc = calc
            e0 = float(atoms.get_potential_energy())
            screened.append((e0 - e_slab - e_gas, cid, site_id, spin_deg, atoms, e0))
        screened.sort(key=lambda x: x[0])

        print(f"  Pre-screen top 5 (initial E_ads):")
        for row in screened[:5]:
            print(f"    site={row[2]}, spin={row[3]}deg, E_ads={row[0]:+.4f} eV")

        # Relax all (small N - manageable)
        relaxed = []
        for rank_i, (e_ads0, cid, site_id, spin_deg, atoms, e0) in enumerate(screened):
            print(f"\n  [{name}] Relaxing cand {rank_i+1}/{len(screened)}: "
                  f"site={site_id}, spin={spin_deg}deg, E_ads_init={e_ads0:+.4f} eV")
            atoms_r = standardize_vasp_atoms(atoms.copy(), z_min_offset=0.5)
            engine.relax(atoms_r, frozen_z_ang=FROZEN_Z, steps=RELAX_STEPS,
                         fmax=FMAX, verbose=False)
            e_r = float(atoms_r.get_potential_energy())
            e_ads_r = e_r - e_slab - e_gas
            mind, mpair, nb = interface_analysis(atoms_r)
            atoms_r.info.update({
                "substrate": name,
                "site_id": site_id,
                "spin_deg": spin_deg,
                "e_ads": e_ads_r,
                "min_interface_dist": mind,
                "interface_bonds": nb,
                "placement": "flat",
            })
            relaxed.append((e_ads_r, cid, site_id, spin_deg, atoms_r, mind, mpair, nb))
            print(f"    -> E_ads_relaxed={e_ads_r:+.4f} eV, "
                  f"min_dist={mind:.3f} A ({mpair}), cov_bonds={nb}")

        # Sort and save
        relaxed.sort(key=lambda x: x[0])
        csv_rows = []
        final_atoms = []
        for rank, (e_ads_r, cid, site_id, spin_deg, atoms_r,
                   mind, mpair, nb) in enumerate(relaxed, 1):
            out_name = f"{name}_flat_rank{rank:02d}_{site_id}_spin{spin_deg}.vasp"
            out_path = sub_dir / out_name
            write(str(out_path), atoms_r, vasp5=True)
            csv_rows.append({
                "rank": rank,
                "site_id": site_id,
                "spin_deg": spin_deg,
                "e_ads_eV": f"{e_ads_r:.6f}",
                "min_dist_A": f"{mind:.3f}",
                "min_pair": mpair,
                "interface_bonds": nb,
                "output": str(out_path.relative_to(ROOT)),
            })
            final_atoms.append(atoms_r)

        write(str(sub_dir / f"{name}_flat_ranked.extxyz"), final_atoms)

        with open(sub_dir / f"{name}_flat_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

        print(f"\n  [{name}] Final ranking:")
        print(f"  {'Rank':<5} {'Site':<5} {'Spin':<6} {'E_ads(eV)':<12} "
              f"{'MinDist':<9} {'Pair':<9} {'CovBonds'}")
        for row in csv_rows:
            flag = " <COVALENT" if int(row["interface_bonds"]) > 0 else ""
            print(f"  {row['rank']:<5} {row['site_id']:<5} "
                  f"{row['spin_deg']:<6} {row['e_ads_eV']:<12} "
                  f"{row['min_dist_A']:<9} {row['min_pair']:<9} "
                  f"{row['interface_bonds']}{flag}")

        top3 = csv_rows[:3]
        all_summary.append(
            f"{name}: slab_E={e_slab:.4f} eV, "
            f"candidates={len(candidates)}, relaxed={len(relaxed)}"
        )
        for row in top3:
            all_summary.append(
                f"  rank {row['rank']}: site={row['site_id']}, spin={row['spin_deg']}deg, "
                f"E_ads={row['e_ads_eV']} eV, min_dist={row['min_dist_A']} A "
                f"({row['min_pair']}), cov_bonds={row['interface_bonds']}"
            )
        all_summary.append("")

    summary_path = OUT_DIR / "flat_placement_summary.txt"
    summary_path.write_text("\n".join(all_summary), encoding="utf-8")
    print("\n" + "\n".join(all_summary))
    print(f"\nSummary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
