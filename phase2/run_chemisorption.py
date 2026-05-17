"""Phase 2 chemisorption search on Si100 and SiO2_Si_term.

Si100
-----
Uses the autoflow_srxn chemisorption builder (build_chemisorption_structures).
Two runs: center_target="C" (center C, idx=13) and center_target="N".
The surface dangling bonds are detected automatically from the reconstructed slab.

Mechanism routes attempted by the builder:
  1. Single-site: main fragment binds to one Si dangling bond.
  2. Dissociative: both fragments bind to a pair of Si dangling bonds.
  3. Protector exchange: inhibitor replaces a surface H (if passivated).

SiO2_Si_term
------------
Site-map guided approach with extended initial distances.
The default covalent-radii placement (~1.8 A for O-Si / N-Si) is augmented
by scanning over multiple heights (HEIGHT_RANGE) starting from DIST_MIN to
DIST_MAX in steps of DIST_STEP.  For each (site, reactive_atom, height, spin)
combination, the molecule is oriented so the reactive atom (O1, O2, N) points
toward the surface, then placed at the target distance and relaxed.

Rationale: starting from a range of initial heights avoids committing to one
transition state pathway and lets FIRE converge to the nearest local minimum,
which may be chemisorbed or physisorbed depending on the landscape.

Outputs
-------
  phase2/results/chemisorption/Si100/       — ranked VASP + CSV + extxyz
  phase2/results/chemisorption/SiO2_Si_term/ — ranked VASP + CSV + extxyz
  phase2/results/chemisorption/chemi_summary.txt
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

from autoflow_srxn.surface.chemisorption_builder import (
    build_chemisorption_structures,
    analyze_surface_reactivity,
)
from autoflow_srxn.surface.surface_utils import (
    standardize_vasp_atoms,
    get_pair_bond_cutoff,
)
from autoflow_srxn.simulation.potentials import SimulationEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FROZEN_Z    = 5.5       # A — atoms below z_min + FROZEN_Z fixed during relax
FMAX        = 0.05      # eV/A
RELAX_STEPS = 250
PRESELECT   = 12        # relax top N candidates per substrate/center combo

# SiO2_Si_term site-map extended distances
DIST_MIN    = 2.5       # A — minimum reactive-atom-to-surface distance
DIST_MAX    = 4.0       # A — maximum
DIST_STEP   = 0.5       # A — step size  → heights: 2.5, 3.0, 3.5, 4.0
N_SPIN      = 4         # azimuthal spins: 0, 90, 180, 270 deg

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

# Config passed to build_chemisorption_structures and analyze_surface_reactivity.
# Uses "precursor" key (package convention for chemisorption routing).
CHEM_CONFIG = dict(ENGINE_CONFIG)
CHEM_CONFIG["reaction_search"] = {
    "symprec": 1.0,          # A — looser to handle the reconstructed Si dimer surface
    "candidate_filter": {
        "overlap_scale": 0.65,
        "max_pair_dist": 5.0,
    },
    "mechanisms": {
        "precursor": {
            "enabled": True,
            "chemisorption": {
                "enabled": True,
                "rot_steps": 8,
                "coordination_analysis": {
                    "bond_slack": 0.25,
                    "max_neighbor_dist": 4.0,
                },
            },
        },
    },
}
# surface_prep is accessed at CONFIG top level by analyze_surface_reactivity
CHEM_CONFIG["surface_prep"] = {
    "surface_analysis": {
        "ideal_coordination": {"Si": 4, "O": 2, "H": 1, "N": 3, "C": 4},
    },
}

OUT_DIR = ROOT / "phase2/results/chemisorption"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INHIBITOR_PATH = ROOT / "structures/inhibitor_relaxed.vasp"
SI100_SLAB     = ROOT / "structures/slabs/Si100_slab.vasp"
SIO2_SI_SLAB   = ROOT / "structures/slabs/SiO2_Si_term_slab.vasp"
SIO2_SI_SITES  = ROOT / "structures/slabs/site_maps/SiO2_Si_term_sites.csv"

CENTER_C_IDX   = 13     # 0-based index of center C in inhibitor

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def orient_atom_toward_surface(mol, atom_idx):
    """Rotate mol (COM at origin) so atom_idx is at the lowest z (pointing toward surface).

    Algorithm:
      1. Translate COM to origin.
      2. Compute unit vector v = (atom_pos - COM).
      3. Apply Rodrigues rotation to align v → -z.
    """
    mol = mol.copy()
    pos = mol.get_positions()
    com = pos.mean(axis=0)
    pos -= com

    v = pos[atom_idx]
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-6:
        mol.set_positions(pos)
        return mol

    v_hat = v / norm_v
    target = np.array([0.0, 0.0, -1.0])  # toward surface = -z

    axis = np.cross(v_hat, target)
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(v_hat, target)

    if sin_a < 1e-6:
        if cos_a > 0:
            # v_hat already points toward -z
            mol.set_positions(pos)
        else:
            # v_hat points toward +z — flip 180 deg around x
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
    """Rotate mol in-plane (around z through origin)."""
    mol = mol.copy()
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    mol.set_positions(mol.get_positions() @ Rz.T)
    return mol


def place_atom_on_site(slab, mol, site_xyz, atom_idx, height):
    """Translate mol so atom_idx lands at (site_x, site_y, site_z + height).

    Returns combined slab+mol Atoms with molecule atoms tagged=2.
    """
    mol = mol.copy()
    pos = mol.get_positions()
    target = np.array([site_xyz[0], site_xyz[1], site_xyz[2] + height])
    mol.set_positions(pos + (target - pos[atom_idx]))

    combined = slab.copy()
    existing_tags = list(combined.get_tags())
    for a in mol:
        combined.append(a)
        existing_tags.append(2)
    combined.set_tags(existing_tags)
    return combined


def interface_analysis(atoms):
    """Return (min_dist, min_pair_str, n_covalent_bonds) at mol-slab interface."""
    tags = atoms.get_tags()
    mol_idx = [i for i, t in enumerate(tags) if t >= 2]
    sub_idx = [i for i, t in enumerate(tags) if t < 2]
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
                cut = get_pair_bond_cutoff(si, sj, bond_slack=0.25, max_cutoff=2.6)
            except Exception:
                cut = 2.2
            if dd < cut:
                nb += 1
    return mind, minpair, nb


def relax_and_score(atoms, engine, e_slab, e_gas):
    """Standardize → relax → return (atoms_r, e_ads, mind, mpair, nb)."""
    atoms_r = standardize_vasp_atoms(atoms.copy(), z_min_offset=0.5)
    engine.relax(atoms_r, frozen_z_ang=FROZEN_Z, steps=RELAX_STEPS, fmax=FMAX, verbose=False)
    e_r = float(atoms_r.get_potential_energy())
    e_ads = e_r - e_slab - e_gas
    mind, mpair, nb = interface_analysis(atoms_r)
    return atoms_r, e_ads, mind, mpair, nb


E_ADS_MAX = 5.0   # eV — reject clearly unphysical high-energy structures


def _clean_info(atoms):
    """Remove non-serializable entries from atoms.info before writing extxyz."""
    bad_keys = []
    for k, v in list(atoms.info.items()):
        if isinstance(v, (Atoms, dict, list)):
            bad_keys.append(k)
        else:
            try:
                import json
                json.dumps(v)
            except (TypeError, ValueError):
                bad_keys.append(k)
    for k in bad_keys:
        del atoms.info[k]
    return atoms


def write_results(sub_dir, label, relaxed):
    """Sort by E_ads, write VASP/extxyz/CSV.

    relaxed: list of (e_ads, atoms, mind, mpair, nb, extra_info_dict)

    Structures with E_ads > E_ADS_MAX are flagged as unphysical and excluded
    from extxyz but still saved as individual VASP files for inspection.
    Mixed atom-count structures (chemisorption fragments) are grouped by
    atom count for extxyz output.
    Returns list of csv_row dicts (all ranks, including unphysical).
    """
    relaxed.sort(key=lambda x: x[0])
    csv_rows = []
    atoms_by_size = {}   # natoms -> [Atoms, ...]

    for rank, (e_ads, atoms_r, mind, mpair, nb, info) in enumerate(relaxed, 1):
        out_name = f"{label}_chemi_rank{rank:02d}.vasp"
        out_path = sub_dir / out_name
        write(str(out_path), atoms_r, vasp5=True)

        physical = e_ads <= E_ADS_MAX
        row = {
            "rank": rank,
            "e_ads_eV": f"{e_ads:.6f}",
            "min_dist_A": f"{mind:.3f}",
            "min_pair": mpair,
            "interface_bonds": nb,
            "physical": "Y" if physical else "N(high-E)",
            "output": str(out_path.relative_to(ROOT)),
        }
        row.update(info)
        csv_rows.append(row)

        if physical:
            n = len(atoms_r)
            atoms_by_size.setdefault(n, []).append(_clean_info(atoms_r.copy()))

    # Write grouped extxyz (one file per unique atom count)
    for n, group in atoms_by_size.items():
        xyz_path = sub_dir / f"{label}_chemi_n{n}.extxyz"
        write(str(xyz_path), group)

    # Write combined extxyz if all structures have the same size
    all_sizes = list(atoms_by_size.keys())
    if len(all_sizes) == 1:
        write(str(sub_dir / f"{label}_chemi_ranked.extxyz"),
              atoms_by_size[all_sizes[0]])
    else:
        print(f"  [Note] Mixed atom counts {all_sizes}: wrote {len(all_sizes)} "
              f"separate extxyz files (one per size group).")

    fieldnames = ["rank", "e_ads_eV", "min_dist_A", "min_pair",
                  "interface_bonds", "physical"] \
                 + [k for k in csv_rows[0] if k not in (
                    "rank", "e_ads_eV", "min_dist_A", "min_pair",
                    "interface_bonds", "physical", "output")] \
                 + ["output"]
    with open(sub_dir / f"{label}_chemi_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    return csv_rows


# ---------------------------------------------------------------------------
# Si100 — package chemisorption builder
# ---------------------------------------------------------------------------

def run_si100_chemisorption(inhibitor, engine, calc, e_gas, summary_lines):
    name = "Si100"
    print(f"\n{'='*72}")
    print(f"  Si100 Chemisorption (autoflow_srxn builder)")
    print(f"{'='*72}")

    sub_dir = OUT_DIR / name
    sub_dir.mkdir(parents=True, exist_ok=True)

    slab = read(str(SI100_SLAB))
    slab.calc = calc
    e_slab = float(slab.get_potential_energy())
    print(f"  Slab energy: {e_slab:.6f} eV")

    # Show surface reactivity for diagnostics
    print("\n  [Surface Reactivity Analysis]")
    sites = analyze_surface_reactivity(slab, CHEM_CONFIG, verbose=True)
    print(f"  unique_single={len(sites['unique_single'])}, "
          f"pairs={len(sites['pairs'])}, "
          f"exchange={len(sites['exchange'])}")

    centers = [
        ("C", CENTER_C_IDX),  # center C
        ("N", None),          # N atom (auto-detected from element)
    ]

    all_relaxed = []
    for center_sym, center_hint in centers:
        print(f"\n  -- center_target='{center_sym}' --")
        raw_candidates = build_chemisorption_structures(
            inhibitor.copy(),
            center_target=center_sym,
            surface=slab.copy(),
            rot_steps=8,
            config=CHEM_CONFIG,
            verbose=True,
            tag=2,
        )
        print(f"  Generated {len(raw_candidates)} candidates for center='{center_sym}'")
        if not raw_candidates:
            print(f"  [WARNING] No candidates found for center='{center_sym}'. "
                  "Check if the inhibitor has detachable ligands around this atom.")
            summary_lines.append(
                f"  Si100 center={center_sym}: 0 candidates generated "
                "(no detachable ligands detected by discover_ligands)"
            )
            continue

        # Tag all molecule atoms as 2 (some builders may not set tags)
        for cand in raw_candidates:
            tags = list(cand.get_tags())
            # atoms beyond slab length are molecule atoms
            n_slab = len(slab)
            for i in range(n_slab, len(cand)):
                tags[i] = 2
            cand.set_tags(tags)

        # Single-point pre-screen
        screened = []
        for cand in raw_candidates:
            cand.calc = calc
            e0 = float(cand.get_potential_energy())
            screened.append((e0 - e_slab - e_gas, cand))
        screened.sort(key=lambda x: x[0])
        print(f"  Pre-screen top 5:")
        for e0, _ in screened[:5]:
            print(f"    E_ads_init={e0:+.4f} eV")

        selected = screened[:min(PRESELECT, len(screened))]
        for rank_i, (e_ads0, cand) in enumerate(selected):
            print(f"\n  Relaxing {rank_i+1}/{len(selected)}: "
                  f"E_ads_init={e_ads0:+.4f} eV")
            atoms_r, e_ads, mind, mpair, nb = relax_and_score(cand, engine, e_slab, e_gas)
            atoms_r.info.update({
                "substrate": name, "center": center_sym,
                "mechanism": atoms_r.info.get("mechanism", "chemisorption"),
                "e_ads": e_ads, "min_dist": mind, "interface_bonds": nb,
            })
            all_relaxed.append((e_ads, atoms_r, mind, mpair, nb,
                                 {"center": center_sym,
                                  "mechanism": atoms_r.info.get("reaction_type", "chem")}))
            print(f"    -> E_ads={e_ads:+.4f} eV, min_dist={mind:.3f} A ({mpair}), cov_bonds={nb}")

    if not all_relaxed:
        print("  [WARNING] No relaxed candidates for Si100 chemisorption.")
        summary_lines.append(f"{name}: 0 relaxed candidates.")
        return

    csv_rows = write_results(sub_dir, name, all_relaxed)

    print(f"\n  [{name}] Final ranking:")
    print(f"  {'Rank':<5} {'E_ads(eV)':<12} {'MinDist':<9} {'Pair':<9} {'CovBonds':<10} Center/Mechanism")
    for row in csv_rows:
        flag = " <CHEM" if int(row["interface_bonds"]) > 0 else " <PHYSI"
        print(f"  {row['rank']:<5} {row['e_ads_eV']:<12} {row['min_dist_A']:<9} "
              f"{row['min_pair']:<9} {row['interface_bonds']:<10} "
              f"{row.get('center','?')}/{row.get('mechanism','?')}{flag}")

    top = csv_rows[:3]
    summary_lines.append(
        f"{name}: slab_E={e_slab:.4f} eV, relaxed={len(all_relaxed)}"
    )
    for row in top:
        summary_lines.append(
            f"  rank {row['rank']}: E_ads={row['e_ads_eV']} eV, "
            f"min_dist={row['min_dist_A']} A ({row['min_pair']}), "
            f"cov={row['interface_bonds']}, center={row.get('center','?')}"
        )
    summary_lines.append("")


# ---------------------------------------------------------------------------
# SiO2_Si_term — site-map guided extended-distance approach
# ---------------------------------------------------------------------------

def run_sio2_siterm_chemisorption(inhibitor, engine, calc, e_gas, summary_lines):
    name = "SiO2_Si_term"
    print(f"\n{'='*72}")
    print(f"  SiO2_Si_term Chemisorption (site-map, extended distances)")
    print(f"{'='*72}")

    sub_dir = OUT_DIR / name
    sub_dir.mkdir(parents=True, exist_ok=True)

    slab = read(str(SIO2_SI_SLAB))
    slab.calc = calc
    e_slab = float(slab.get_potential_energy())
    print(f"  Slab energy: {e_slab:.6f} eV")

    # Load site map
    sites = list(csv.DictReader(open(SIO2_SI_SITES)))
    print(f"  Sites: {len(sites)}")
    for s in sites:
        print(f"    {s['site_id']:5s}  type={s['type']:8s}  "
              f"z={float(s['z_A']):.3f} A")

    # Identify reactive atoms: O (all) and N
    syms = inhibitor.get_chemical_symbols()
    o_indices = [i for i, s in enumerate(syms) if s == "O"]
    n_indices  = [i for i, s in enumerate(syms) if s == "N"]
    print(f"\n  Inhibitor reactive atoms:")
    print(f"    O: indices={o_indices}, N: indices={n_indices}")

    reactive_atoms = [(i, f"O{k+1}") for k, i in enumerate(o_indices)] \
                   + [(i, f"N")      for    i in n_indices]

    heights = np.arange(DIST_MIN, DIST_MAX + 1e-6, DIST_STEP)
    spins   = range(0, 360, 360 // N_SPIN)

    total_cands = len(sites) * len(reactive_atoms) * len(heights) * N_SPIN
    print(f"\n  Candidate grid: {len(sites)} sites x {len(reactive_atoms)} reactive atoms "
          f"x {len(heights)} heights x {N_SPIN} spins = {total_cands} total")

    # Pre-orient inhibitor with each reactive atom pointing toward -z
    oriented = {}  # atom_idx -> mol_with_atom_at_bottom
    for atom_idx, atom_label in reactive_atoms:
        oriented[atom_idx] = orient_atom_toward_surface(inhibitor.copy(), atom_idx)

    # Generate candidates
    candidates = []
    for site in sites:
        site_xyz = np.array([float(site["x_A"]), float(site["y_A"]), float(site["z_A"])])
        for atom_idx, atom_label in reactive_atoms:
            mol_base = oriented[atom_idx]
            for height in heights:
                for spin_deg in spins:
                    mol = spin_z(mol_base, spin_deg)
                    combined = place_atom_on_site(slab, mol, site_xyz, atom_idx, height)
                    cid = len(candidates)
                    candidates.append({
                        "cid": cid,
                        "site_id": site["site_id"],
                        "atom_label": atom_label,
                        "atom_idx": atom_idx,
                        "height": float(height),
                        "spin_deg": int(spin_deg),
                        "atoms": combined,
                    })

    print(f"  Generated {len(candidates)} candidates")

    # Single-point pre-screen
    print("\n  Running single-point pre-screen...")
    for cand in candidates:
        cand["atoms"].calc = calc
        e0 = float(cand["atoms"].get_potential_energy())
        cand["e_ads_init"] = e0 - e_slab - e_gas

    candidates.sort(key=lambda c: c["e_ads_init"])
    print(f"  Pre-screen top 10 (by initial E_ads):")
    for c in candidates[:10]:
        print(f"    site={c['site_id']:5s} atom={c['atom_label']:3s} "
              f"h={c['height']:.1f}A spin={c['spin_deg']:3d}deg "
              f"E_ads_init={c['e_ads_init']:+.4f} eV")

    selected = candidates[:min(PRESELECT, len(candidates))]

    # Relax selected
    relaxed = []
    for rank_i, cand in enumerate(selected):
        print(f"\n  Relaxing {rank_i+1}/{len(selected)}: "
              f"site={cand['site_id']}, atom={cand['atom_label']}, "
              f"h={cand['height']:.1f}A, spin={cand['spin_deg']}deg, "
              f"E_ads_init={cand['e_ads_init']:+.4f} eV")
        atoms_r, e_ads, mind, mpair, nb = relax_and_score(
            cand["atoms"], engine, e_slab, e_gas
        )
        atoms_r.info.update({
            "substrate": name,
            "site_id": cand["site_id"],
            "reactive_atom": cand["atom_label"],
            "init_height_A": cand["height"],
            "spin_deg": cand["spin_deg"],
            "e_ads": e_ads, "min_dist": mind, "interface_bonds": nb,
        })
        relaxed.append((e_ads, atoms_r, mind, mpair, nb, {
            "site_id":       cand["site_id"],
            "reactive_atom": cand["atom_label"],
            "init_height_A": f"{cand['height']:.1f}",
            "spin_deg":      cand["spin_deg"],
        }))
        print(f"    -> E_ads={e_ads:+.4f} eV, min_dist={mind:.3f} A ({mpair}), cov_bonds={nb}")

    if not relaxed:
        print("  [WARNING] No relaxed candidates for SiO2_Si_term chemisorption.")
        summary_lines.append(f"{name}: 0 relaxed candidates.")
        return

    csv_rows = write_results(sub_dir, name, relaxed)

    print(f"\n  [{name}] Final ranking:")
    print(f"  {'Rk':<4} {'E_ads(eV)':<12} {'Dist':<8} {'Pair':<8} "
          f"{'Bonds':<6} {'Site':<6} {'Atom':<4} {'H_init':<8} {'Spin'}")
    for row in csv_rows:
        flag = " <CHEM" if int(row["interface_bonds"]) > 0 else " <PHYSI"
        print(f"  {row['rank']:<4} {row['e_ads_eV']:<12} {row['min_dist_A']:<8} "
              f"{row['min_pair']:<8} {row['interface_bonds']:<6} "
              f"{row.get('site_id','?'):<6} {row.get('reactive_atom','?'):<4} "
              f"{row.get('init_height_A','?'):<8} {row.get('spin_deg','?')}{flag}")

    top = csv_rows[:3]
    summary_lines.append(
        f"{name}: slab_E={e_slab:.4f} eV, "
        f"candidates={len(candidates)}, relaxed={len(relaxed)}"
    )
    for row in top:
        summary_lines.append(
            f"  rank {row['rank']}: E_ads={row['e_ads_eV']} eV, "
            f"min_dist={row['min_dist_A']} A ({row['min_pair']}), "
            f"cov={row['interface_bonds']}, site={row.get('site_id','?')}, "
            f"atom={row.get('reactive_atom','?')}, h_init={row.get('init_height_A','?')} A"
        )
    summary_lines.append("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    targets = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    engine = SimulationEngine(ENGINE_CONFIG)
    calc   = engine.get_calculator()

    inhibitor = read(str(INHIBITOR_PATH))
    syms = inhibitor.get_chemical_symbols()
    print(f"Inhibitor: {inhibitor.get_chemical_formula()}  ({len(inhibitor)} atoms)")
    print(f"  Center C idx={CENTER_C_IDX} ({syms[CENTER_C_IDX]})")
    o_idx = [i for i, s in enumerate(syms) if s == "O"]
    n_idx = [i for i, s in enumerate(syms) if s == "N"]
    print(f"  O atoms: {o_idx}  ({[syms[i] for i in o_idx]})")
    print(f"  N atoms: {n_idx}  ({[syms[i] for i in n_idx]})")

    # Gas-phase energy
    gas = inhibitor.copy()
    gas.center(vacuum=10.0)
    gas.calc = calc
    e_gas = float(gas.get_potential_energy())
    print(f"  Gas energy: {e_gas:.6f} eV")

    summary_lines = [
        "Phase 2 Chemisorption Search",
        "=" * 72,
        f"Si100    : autoflow_srxn build_chemisorption_structures, center=C/N",
        f"SiO2_Si  : site-map guided, reactive atoms=O1,O2,N, "
        f"heights={DIST_MIN:.1f}-{DIST_MAX:.1f} A step={DIST_STEP:.1f} A",
        f"e_gas    = {e_gas:.6f} eV",
        f"FROZEN_Z = {FROZEN_Z} A, FMAX = {FMAX} eV/A, PRESELECT = {PRESELECT}",
        "",
    ]

    if not targets or "Si100" in targets:
        run_si100_chemisorption(inhibitor, engine, calc, e_gas, summary_lines)

    if not targets or "SiO2_Si_term" in targets:
        run_sio2_siterm_chemisorption(inhibitor, engine, calc, e_gas, summary_lines)

    summary_path = OUT_DIR / "chemi_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "=" * 72)
    print("\n".join(summary_lines))
    print(f"\nSummary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
