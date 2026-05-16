"""Phase 2 result analysis: compare old vs new Fibonacci sphere sampling."""

import sys
import csv
from pathlib import Path

import numpy as np
from ase.io import read
from ase.geometry import get_distances

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from autoflow_srxn.surface.surface_utils import get_pair_bond_cutoff

NEW_DIR = ROOT / "phase2/results/inhibitor_physisorption"
OLD_DIR = ROOT / "phase2/results/inhibitor_physisorption_old"

SUBSTRATE_MAP = {
    "Si100":       ("Si100",              "Si100"),
    "SiO2_O_term": ("SiO2_O_term_nograv", "SiO2_O_term_nograv"),
    "SiO2_Si_term":("SiO2_Si_term",       "SiO2_Si_term"),
}


def interface_analysis(atoms):
    """Return (min_dist, min_pair, n_covalent_bonds) for molecule-slab interface."""
    tags = atoms.get_tags()
    mol = np.where(tags >= 2)[0]
    sub = np.where(tags < 2)[0]
    if len(mol) == 0 or len(sub) == 0:
        return None, None, 0

    mind, minpair, nbonds = 999.0, None, 0
    for i in mol:
        D, d = get_distances(
            atoms.positions[i], atoms.positions[sub],
            cell=atoms.cell, pbc=atoms.pbc,
        )
        for k, j in enumerate(sub):
            dd = float(d[0][k])
            sym_i, sym_j = atoms.symbols[i], atoms.symbols[j]
            if dd < mind:
                mind, minpair = dd, (sym_i, sym_j)
            try:
                cutoff = get_pair_bond_cutoff(sym_i, sym_j, bond_slack=0.25, max_cutoff=2.6)
            except Exception:
                cutoff = 2.2
            if dd < cutoff:
                nbonds += 1
    return mind, minpair, nbonds


def read_csv_eads(csv_path):
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_relaxed(directory, label):
    xyz = directory / label / f"{label}_physi_relaxed_ranked.extxyz"
    if not xyz.exists():
        return []
    return read(str(xyz), ":")


def print_separator(char="=", width=72):
    print(char * width)


def main():
    print_separator()
    print("Phase 2 Inhibitor Physisorption - Result Analysis")
    print("New: fixed Fibonacci sphere (n_rot=8 true SO(3))")
    print("Old: n_polar=sqrt(n_rot)=2 polar × 4 azimuthal spins (polar grid)")
    print_separator()

    for name, (new_label, old_label) in SUBSTRATE_MAP.items():
        print(f"\n{'─'*72}")
        print(f"  Substrate: {name}")
        print(f"{'─'*72}")

        new_rows = read_csv_eads(NEW_DIR / new_label / f"{new_label}_physi_summary.csv")
        old_rows = read_csv_eads(OLD_DIR / old_label / f"{old_label}_physi_summary.csv")

        # candidates count from extxyz
        new_cand_file = NEW_DIR / new_label / f"{new_label}_physi_candidates.extxyz"
        old_cand_file = OLD_DIR / old_label / f"{old_label}_physi_candidates.extxyz"
        n_new_cand = len(read(str(new_cand_file), ":")) if new_cand_file.exists() else "?"
        n_old_cand = len(read(str(old_cand_file), ":")) if old_cand_file.exists() else "?"

        print(f"  Generated candidates  — old: {n_old_cand:>3}  new: {n_new_cand:>3}")
        print(f"  Relaxed              — old: {len(old_rows):>3}  new: {len(new_rows):>3}")
        print()

        # Top-3 E_ads comparison
        print(f"  {'Rank':<5} {'Old E_ads (eV)':<20} {'New E_ads (eV)':<20} {'Site (new)'}")
        print(f"  {'----':<5} {'------------------':<20} {'------------------':<20} {'----------'}")
        max_rank = max(len(old_rows), len(new_rows), 3)
        for rank in range(1, min(max_rank + 1, 9)):
            old_e = old_rows[rank-1]["e_ads_eV"] if rank <= len(old_rows) else "—"
            new_e = new_rows[rank-1]["e_ads_eV"] if rank <= len(new_rows) else "—"
            new_site = new_rows[rank-1].get("site_id", "") if rank <= len(new_rows) else ""
            print(f"  {rank:<5} {old_e:<20} {new_e:<20} {new_site}")

        # Interface bond analysis for new results
        new_atoms_list = load_relaxed(NEW_DIR, new_label)
        if new_atoms_list:
            print(f"\n  Interface analysis (new results):")
            print(f"  {'Rank':<5} {'E_ads(eV)':<12} {'Min dist(A)':<13} {'Pair':<10} {'Cov. bonds'}")
            for rank, atoms in enumerate(new_atoms_list[:8], 1):
                e_ads = atoms.info.get("e_ads", float("nan"))
                mind, pair, nb = interface_analysis(atoms)
                pair_str = f"{pair[0]}-{pair[1]}" if pair else "—"
                physi_flag = "" if nb == 0 else " ← COVALENT"
                print(f"  {rank:<5} {e_ads:<12.4f} {mind:<13.3f} {pair_str:<10} {nb}{physi_flag}")

    print(f"\n{'='*72}")
    print("Summary")
    print(f"{'='*72}")

    # Read new summary file
    summary_path = NEW_DIR / "physisorption_summary.txt"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    else:
        print("(summary file not found — run still in progress?)")


if __name__ == "__main__":
    main()
