"""Phase 2 inhibitor physisorption search on relaxed Phase 1 substrates.

This script intentionally runs physisorption only:
  1. Generate candidates with reduced site map symprec and n_rot=8.
  2. Single-point rank all generated candidates.
  3. Relax the best pre-screened candidates.
  4. Save ranked structures and summaries per substrate.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ase.io import read, write

from autoflow_srxn.surface.ads_workflow_mgr import AdsorptionWorkflowManager
from autoflow_srxn.surface.surface_utils import standardize_vasp_atoms
from autoflow_srxn.simulation.potentials import SimulationEngine


OUT_DIR = ROOT / "phase2/results/inhibitor_physisorption"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_Z = 5.5
FMAX = 0.05
RELAX_STEPS = 180
PRESELECT = 8
BRANCHING_LIMIT = 3

PHYSI_CFG = {
    "placement_height": 3.0,
    "n_rot": 8,
    "height_mode": "clearance",
    "gravity_pull": {"enabled": True},
}

CONFIG = {
    "workflow": {
        "candidate_relax": True,
        "md_equilibrate": False,
        "post_md_relax": False,
    },
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
    "reaction_search": {
        "symprec": 1.5,
        "candidate_filter": {"overlap_scale": 0.65},
        "mechanisms": {
            "inhibitor": {
                "enabled": True,
                "center": 13,
                "physisorption": PHYSI_CFG,
                "chemisorption": {"enabled": False},
                "branching_limit": BRANCHING_LIMIT,
            }
        },
    },
}

SUBSTRATES = [
    ("Si100", ROOT / "structures/slabs/Si100_slab.vasp"),
    ("SiO2_O_term", ROOT / "structures/slabs/SiO2_O_term_slab.vasp"),
    ("SiO2_Si_term", ROOT / "structures/slabs/SiO2_Si_term_slab.vasp"),
]

PHYSI_OVERRIDES = {
    # With gravity pull enabled, O-terminated SiO2 relaxes into O-H/C-O bonded
    # chemisorption-like structures.  For the physisorption-only branch, keep
    # the initial clearance placement and let relaxation find a nonbonded local
    # minimum.
    "SiO2_O_term": {"gravity_pull": {"enabled": False}, "output_label": "SiO2_O_term_nograv"},
}


def attach_calc(atoms, calc):
    atoms.calc = calc
    return atoms


def calc_energy(atoms, calc):
    attach_calc(atoms, calc)
    return float(atoms.get_potential_energy())


def relax_atoms(atoms, engine, *, frozen_z_ang=None, steps=RELAX_STEPS, fmax=FMAX):
    atoms = standardize_vasp_atoms(atoms, z_min_offset=0.5)
    engine.relax(atoms, frozen_z_ang=frozen_z_ang, steps=steps, fmax=fmax, verbose=False)
    return atoms, float(atoms.get_potential_energy())


def write_summary_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "candidate_id",
                "site_id",
                "e_initial_eV",
                "e_relaxed_eV",
                "e_ads_eV",
                "delta_relax_eV",
                "output",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    targets = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    engine = SimulationEngine(CONFIG)
    calc = engine.get_calculator()

    inhibitor = read(str(ROOT / "structures/inhibitor_relaxed.vasp"))
    inhibitor_symbols = inhibitor.get_chemical_symbols()
    center_idx = CONFIG["reaction_search"]["mechanisms"]["inhibitor"]["center"]
    center_note = (
        f"center index {center_idx} (0-based symbol={inhibitor_symbols[center_idx]})"
        if isinstance(center_idx, int) and center_idx < len(inhibitor_symbols)
        else f"center setting {center_idx}"
    )

    gas = inhibitor.copy()
    gas.center(vacuum=10.0)
    gas, e_gas = relax_atoms(gas, engine, frozen_z_ang=None, steps=120, fmax=0.03)
    write(str(OUT_DIR / "inhibitor_gas_relaxed.vasp"), gas, vasp5=True)

    summary_lines = [
        "Phase 2 Inhibitor Physisorption Summary",
        "=" * 70,
        f"inhibitor_formula = {inhibitor.get_chemical_formula()}",
        f"inhibitor_{center_note}",
        f"gas_energy_eV = {e_gas:.6f}",
        f"symprec = {CONFIG['reaction_search']['symprec']:.2f} A",
        f"placement_height = {PHYSI_CFG['placement_height']:.2f} A",
        f"n_rot = {PHYSI_CFG['n_rot']}",
        f"gravity_pull = {PHYSI_CFG['gravity_pull']['enabled']}",
        f"preselect = {PRESELECT}",
        "",
    ]

    for name, slab_path in SUBSTRATES:
        if targets and name not in targets:
            continue
        physi_cfg = dict(PHYSI_CFG)
        override = PHYSI_OVERRIDES.get(name, {})
        physi_cfg.update({k: v for k, v in override.items() if k != "output_label"})
        output_label = override.get("output_label", name)
        sub_dir = OUT_DIR / output_label
        sub_dir.mkdir(parents=True, exist_ok=True)

        slab = read(str(slab_path))
        e_slab = calc_energy(slab, calc)

        mgr = AdsorptionWorkflowManager(slab, config=CONFIG, symprec=CONFIG["reaction_search"]["symprec"], verbose=True)
        candidates = mgr.generate_physisorption_candidates(
            inhibitor,
            height=physi_cfg["placement_height"],
            n_rot=physi_cfg["n_rot"],
            rot_center="com",
            height_mode=physi_cfg["height_mode"],
            gravity_pull=physi_cfg["gravity_pull"],
            config=CONFIG,
            tag=2,
        )

        write(str(sub_dir / f"{output_label}_physi_candidates.extxyz"), candidates)

        screened = []
        for idx, cand in enumerate(candidates):
            e_initial = calc_energy(cand, calc)
            screened.append(
                {
                    "idx": idx,
                    "site_id": cand.info.get("site_id", ""),
                    "atoms": cand,
                    "e_initial": e_initial,
                    "e_ads_initial": e_initial - e_slab - e_gas,
                }
            )
        screened.sort(key=lambda row: row["e_ads_initial"])
        selected = screened[: min(PRESELECT, len(screened))]

        relaxed_rows = []
        relaxed_atoms = []
        for local_rank, row in enumerate(selected, start=1):
            atoms_relaxed, e_relaxed = relax_atoms(row["atoms"], engine, frozen_z_ang=FROZEN_Z)
            e_ads = e_relaxed - e_slab - e_gas
            atoms_relaxed.info.update(
                {
                    "substrate": name,
                    "candidate_id": row["idx"],
                    "site_id": row["site_id"],
                    "e_initial": row["e_initial"],
                    "e_relaxed": e_relaxed,
                    "e_ads": e_ads,
                    "reaction_type": "physisorption",
                    "mechanism": "physisorption",
                }
            )
            relaxed_atoms.append(atoms_relaxed)
            relaxed_rows.append(
                {
                    "rank": local_rank,
                    "candidate_id": row["idx"],
                    "site_id": row["site_id"],
                    "e_initial_eV": f"{row['e_initial']:.6f}",
                    "e_relaxed_eV": f"{e_relaxed:.6f}",
                    "e_ads_eV": f"{e_ads:.6f}",
                    "delta_relax_eV": f"{e_relaxed - row['e_initial']:.6f}",
                    "output": "",
                }
            )

        ranked = sorted(zip(relaxed_rows, relaxed_atoms), key=lambda item: float(item[0]["e_ads_eV"]))
        final_rows = []
        final_atoms = []
        for rank, (row, atoms_relaxed) in enumerate(ranked, start=1):
            out_name = f"{output_label}_inhibitor_physi_rank{rank:02d}.vasp"
            out_path = sub_dir / out_name
            write(str(out_path), atoms_relaxed, vasp5=True)
            row = dict(row)
            row["rank"] = rank
            row["output"] = str(out_path.relative_to(ROOT))
            final_rows.append(row)
            final_atoms.append(atoms_relaxed)

        write(str(sub_dir / f"{output_label}_physi_relaxed_ranked.extxyz"), final_atoms)
        write_summary_csv(sub_dir / f"{output_label}_physi_summary.csv", final_rows)

        top = final_rows[:BRANCHING_LIMIT]
        summary_lines.append(
            f"{name}: slab_E={e_slab:.6f} eV, generated={len(candidates)}, "
            f"relaxed={len(final_rows)}, gravity={physi_cfg['gravity_pull'].get('enabled', False)}"
        )
        for row in top:
            summary_lines.append(
                f"  rank {row['rank']}: E_ads={row['e_ads_eV']} eV, "
                f"candidate={row['candidate_id']}, site={row['site_id']}, "
                f"file={row['output']}"
            )
        summary_lines.append("")

    summary_path = OUT_DIR / "physisorption_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nSummary written to: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
