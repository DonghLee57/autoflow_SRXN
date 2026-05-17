"""Phase 3: AllylCpNi physisorption + chemisorption on Si100, SiO2_Si_term, SiO2_O_term.

Molecule
--------
AllylCpNi  =  (eta³-allyl)(eta⁵-Cp)Ni   ·  10H + 8C + 1Ni = 19 atoms
Center     :  Ni  (atom index 18, last atom in VASP order H→C→Ni)

Cell choice
-----------
AllylCpNi footprint ≈ 4.5 Å  ≪  cell (10.1-10.9 Å)  →  original cells used.

Physisorption (flat placement)
------------------------------
  HEIGHT_PHYSI = 3.5 Å  (Ni center above site z)
  N_SPIN       = 4      (0/90/180/270 deg)
  Sites from site_maps CSVs
  Pre-screen → relax top PRESELECT_PHYSI per substrate

Chemisorption
-------------
  Si100        : autoflow_srxn build_chemisorption_structures  (center_target="Ni")
  SiO2_Si_term : site-map guided, Ni as reactive atom, heights 2.5-4.5 Å
  SiO2_O_term  : site-map guided, Ni as reactive atom, heights 2.0-4.0 Å
  Pre-screen → relax top PRESELECT_CHEMI per substrate

Outputs
-------
  phase3/results/AllylCpNi/{substrate}/physi/  -- ranked VASP + CSV
  phase3/results/AllylCpNi/{substrate}/chemi/  -- ranked VASP + CSV
  phase3/results/AllylCpNi/allylcpni_summary.txt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))   # phase3/

import numpy as np
from ase.io import read

from autoflow_srxn.surface.chemisorption_builder import (
    build_chemisorption_structures,
    analyze_surface_reactivity,
)
from autoflow_srxn.simulation.potentials import SimulationEngine

from utils import (
    center_molecule, orient_atom_toward_surface, spin_z,
    place_center_on_site, place_atom_on_site,
    interface_analysis, relax_and_score, write_results, load_sites,
    E_ADS_MAX,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NI_IDX          = 18       # 0-based Ni atom index in AllylCpNi (last atom)
HEIGHT_PHYSI    = 3.5      # Å  Ni center above site z  (physisorption start)
N_SPIN          = 4
PRESELECT_PHYSI = 10       # relax top N from physi pre-screen
PRESELECT_CHEMI = 12       # relax top N from chemi pre-screen

FROZEN_Z    = 5.5
FMAX        = 0.05
RELAX_STEPS = 250

# Chemisorption heights for SiO2 site-map approach
CHEMI_HEIGHTS_SI   = np.arange(2.5, 4.6, 0.5)   # 2.5 3.0 3.5 4.0 4.5
CHEMI_HEIGHTS_O    = np.arange(2.0, 4.1, 0.5)   # 2.0 2.5 3.0 3.5 4.0

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

CHEM_CONFIG = dict(ENGINE_CONFIG)
CHEM_CONFIG["reaction_search"] = {
    "symprec": 1.0,
    "candidate_filter": {"overlap_scale": 0.65, "max_pair_dist": 5.5},
    "mechanisms": {
        "precursor": {
            "enabled": True,
            "chemisorption": {
                "enabled": True,
                "rot_steps": 8,
                "coordination_analysis": {"bond_slack": 0.25, "max_neighbor_dist": 4.5},
            },
        },
    },
}
CHEM_CONFIG["surface_prep"] = {
    "surface_analysis": {
        "ideal_coordination": {"Si": 4, "O": 2, "H": 1, "N": 3, "C": 4, "Ni": 4},
    },
}

# Paths
MOL_PATH       = ROOT / "structures/AllylCpNi_relaxed.vasp"
SI100_SLAB     = ROOT / "structures/slabs/Si100_slab.vasp"
SIO2_SI_SLAB   = ROOT / "structures/slabs/SiO2_Si_term_slab.vasp"
SIO2_O_SLAB    = ROOT / "structures/slabs/SiO2_O_term_slab.vasp"
SI100_SITES    = ROOT / "structures/slabs/site_maps/Si100_sites.csv"
SIO2_SI_SITES  = ROOT / "structures/slabs/site_maps/SiO2_Si_term_sites.csv"
SIO2_O_SITES   = ROOT / "structures/slabs/site_maps/SiO2_O_term_sites.csv"

OUT_DIR = ROOT / "phase3/results/AllylCpNi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Physisorption -- flat placement
# ---------------------------------------------------------------------------

def run_physi(name, slab, sites, mol, calc, engine, e_slab, e_gas, summary_lines):
    sub_dir = OUT_DIR / name / "physi"
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  AllylCpNi PHYSI -- {name}  (HEIGHT={HEIGHT_PHYSI} A, {N_SPIN} spins)")
    print(f"{'='*68}")

    # Center molecule at origin, keep original orientation
    mol_c = center_molecule(mol)
    spins = range(0, 360, 360 // N_SPIN)

    candidates = []
    for site in sites:
        site_xyz = np.array([site["x_A"], site["y_A"], site["z_A"]])
        for deg in spins:
            mol_s = spin_z(mol_c, deg)
            combined = place_center_on_site(slab, mol_s, site_xyz, NI_IDX, HEIGHT_PHYSI)
            candidates.append({
                "site_id": site["site_id"], "spin_deg": int(deg), "atoms": combined,
            })

    print(f"  {len(sites)} sites × {N_SPIN} spins = {len(candidates)} candidates")

    # Pre-screen
    for c in candidates:
        c["atoms"].calc = calc
        e0 = float(c["atoms"].get_potential_energy())
        c["e_ads_init"] = e0 - e_slab - e_gas

    candidates.sort(key=lambda c: c["e_ads_init"])
    print(f"  Pre-screen top 5:")
    for c in candidates[:5]:
        print(f"    site={c['site_id']:5s} spin={c['spin_deg']:3d}  E_init={c['e_ads_init']:+.3f} eV")

    selected = candidates[:min(PRESELECT_PHYSI, len(candidates))]
    relaxed  = []
    for i, c in enumerate(selected):
        print(f"\n  Relax {i+1}/{len(selected)}: site={c['site_id']}, spin={c['spin_deg']}  "
              f"E_init={c['e_ads_init']:+.3f} eV")
        ar, e_ads, mind, mpair, nb = relax_and_score(
            c["atoms"], engine, e_slab, e_gas, FROZEN_Z, RELAX_STEPS, FMAX)
        relaxed.append((e_ads, ar, mind, mpair, nb,
                        {"site_id": c["site_id"], "spin_deg": c["spin_deg"],
                         "mode": "physi"}))
        print(f"    -> E_ads={e_ads:+.4f} eV, dist={mind:.3f} A ({mpair}), cov={nb}")

    if not relaxed:
        summary_lines.append(f"  {name} PHYSI: no candidates")
        return

    rows = write_results(sub_dir, f"AllylCpNi_{name}_physi", relaxed,
                         extra_fields=["site_id", "spin_deg", "mode"])
    top = rows[:3]
    summary_lines.append(f"  {name} physi: relaxed={len(relaxed)}")
    for r in top:
        flag = "CHEM" if int(r["interface_bonds"]) > 0 else "PHYSI"
        summary_lines.append(
            f"    rank{r['rank']}: E_ads={r['e_ads_eV']} eV  "
            f"dist={r['min_dist_A']} A ({r['min_pair']})  cov={r['interface_bonds']}  [{flag}]"
        )


# ---------------------------------------------------------------------------
# Chemisorption -- Si100 via builder
# ---------------------------------------------------------------------------

def run_chemi_si100(slab, mol, calc, engine, e_slab, e_gas, summary_lines):
    name    = "Si100"
    sub_dir = OUT_DIR / name / "chemi"
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  AllylCpNi CHEMI -- Si100  (build_chemisorption_structures, center=Ni)")
    print(f"{'='*68}")

    sites_info = analyze_surface_reactivity(slab, CHEM_CONFIG, verbose=True)
    print(f"  unique_single={len(sites_info['unique_single'])}, "
          f"pairs={len(sites_info['pairs'])}")

    raw = build_chemisorption_structures(
        mol.copy(), center_target="Ni", surface=slab.copy(),
        rot_steps=8, config=CHEM_CONFIG, verbose=True, tag=2,
    )
    print(f"  Builder generated {len(raw)} candidates")

    if not raw:
        summary_lines.append(f"  {name} CHEMI: 0 candidates from builder")
        return

    n_slab = len(slab)
    screened = []
    for cand in raw:
        tags = list(cand.get_tags())
        for i in range(n_slab, len(cand)):
            tags[i] = 2
        cand.set_tags(tags)
        cand.calc = calc
        e0 = float(cand.get_potential_energy())
        screened.append((e0 - e_slab - e_gas, cand))

    screened.sort(key=lambda x: x[0])
    print(f"  Pre-screen top 5:")
    for e0, _ in screened[:5]:
        print(f"    E_init={e0:+.4f} eV")

    selected = screened[:min(PRESELECT_CHEMI, len(screened))]
    relaxed  = []
    for i, (e0, cand) in enumerate(selected):
        print(f"\n  Relax {i+1}/{len(selected)}: E_init={e0:+.4f} eV")
        ar, e_ads, mind, mpair, nb = relax_and_score(
            cand, engine, e_slab, e_gas, FROZEN_Z, RELAX_STEPS, FMAX)
        mech = ar.info.get("reaction_type", "chem")
        relaxed.append((e_ads, ar, mind, mpair, nb,
                        {"mode": "chemi", "mechanism": mech}))
        print(f"    -> E_ads={e_ads:+.4f} eV  dist={mind:.3f} A ({mpair})  cov={nb}")

    if not relaxed:
        summary_lines.append(f"  {name} CHEMI: no relaxed candidates")
        return

    rows = write_results(sub_dir, f"AllylCpNi_{name}_chemi", relaxed)
    top  = rows[:3]
    summary_lines.append(f"  {name} chemi: relaxed={len(relaxed)}")
    for r in top:
        flag = "CHEM" if int(r["interface_bonds"]) > 0 else "PHYSI"
        summary_lines.append(
            f"    rank{r['rank']}: E_ads={r['e_ads_eV']} eV  "
            f"dist={r['min_dist_A']} A ({r['min_pair']})  cov={r['interface_bonds']}  [{flag}]"
        )


# ---------------------------------------------------------------------------
# Chemisorption -- SiO2 via site-map (Ni as reactive atom)
# ---------------------------------------------------------------------------

def run_chemi_sio2(name, slab, sites, mol, calc, engine, e_slab, e_gas,
                   summary_lines, heights):
    sub_dir = OUT_DIR / name / "chemi"
    sub_dir.mkdir(parents=True, exist_ok=True)

    spins = range(0, 360, 360 // N_SPIN)
    n_h   = len(heights)

    print(f"\n{'='*68}")
    print(f"  AllylCpNi CHEMI -- {name}  (site-map, Ni-down, h={heights[0]:.1f}-{heights[-1]:.1f} A)")
    print(f"{'='*68}")

    # Orient Ni toward surface
    mol_oriented = orient_atom_toward_surface(mol.copy(), NI_IDX)

    total = len(sites) * n_h * N_SPIN
    print(f"  Grid: {len(sites)} sites × {n_h} heights × {N_SPIN} spins = {total} candidates")

    candidates = []
    for site in sites:
        site_xyz = np.array([site["x_A"], site["y_A"], site["z_A"]])
        for h in heights:
            for deg in spins:
                mol_s    = spin_z(mol_oriented, deg)
                combined = place_atom_on_site(slab, mol_s, site_xyz, NI_IDX, float(h))
                candidates.append({
                    "site_id": site["site_id"], "height": float(h),
                    "spin_deg": int(deg), "atoms": combined,
                })

    # Pre-screen
    for c in candidates:
        c["atoms"].calc = calc
        e0 = float(c["atoms"].get_potential_energy())
        c["e_ads_init"] = e0 - e_slab - e_gas

    candidates.sort(key=lambda c: c["e_ads_init"])
    print(f"  Pre-screen top 5:")
    for c in candidates[:5]:
        print(f"    site={c['site_id']:5s} h={c['height']:.1f} spin={c['spin_deg']:3d}  "
              f"E_init={c['e_ads_init']:+.3f} eV")

    selected = candidates[:min(PRESELECT_CHEMI, len(candidates))]
    relaxed  = []
    for i, c in enumerate(selected):
        print(f"\n  Relax {i+1}/{len(selected)}: site={c['site_id']} h={c['height']:.1f} "
              f"spin={c['spin_deg']}  E_init={c['e_ads_init']:+.3f} eV")
        ar, e_ads, mind, mpair, nb = relax_and_score(
            c["atoms"], engine, e_slab, e_gas, FROZEN_Z, RELAX_STEPS, FMAX)
        relaxed.append((e_ads, ar, mind, mpair, nb, {
            "site_id": c["site_id"],
            "init_height_A": f"{c['height']:.1f}",
            "spin_deg": c["spin_deg"],
            "mode": "chemi",
        }))
        print(f"    -> E_ads={e_ads:+.4f} eV  dist={mind:.3f} A ({mpair})  cov={nb}")

    if not relaxed:
        summary_lines.append(f"  {name} CHEMI: no relaxed candidates")
        return

    rows = write_results(sub_dir, f"AllylCpNi_{name}_chemi", relaxed)
    top  = rows[:3]
    summary_lines.append(f"  {name} chemi: relaxed={len(relaxed)}")
    for r in top:
        flag = "CHEM" if int(r["interface_bonds"]) > 0 else "PHYSI"
        summary_lines.append(
            f"    rank{r['rank']}: E_ads={r['e_ads_eV']} eV  "
            f"dist={r['min_dist_A']} A ({r['min_pair']})  cov={r['interface_bonds']}  [{flag}]"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    engine = SimulationEngine(ENGINE_CONFIG)
    calc   = engine.get_calculator()

    mol = read(str(MOL_PATH))
    print(f"Molecule: {mol.get_chemical_formula()}  ({len(mol)} atoms)")
    print(f"  Ni at idx={NI_IDX}  symbol={mol.get_chemical_symbols()[NI_IDX]}")

    gas = mol.copy()
    gas.center(vacuum=10.0)
    gas.calc = calc
    e_gas = float(gas.get_potential_energy())
    print(f"  Gas energy: {e_gas:.6f} eV")

    # Load slabs
    si100_slab   = read(str(SI100_SLAB))
    sio2si_slab  = read(str(SIO2_SI_SLAB))
    sio2o_slab   = read(str(SIO2_O_SLAB))

    for slab in [si100_slab, sio2si_slab, sio2o_slab]:
        slab.calc = calc

    e_si100  = float(si100_slab.get_potential_energy())
    e_sio2si = float(sio2si_slab.get_potential_energy())
    e_sio2o  = float(sio2o_slab.get_potential_energy())
    print(f"\nSlab energies:")
    print(f"  Si100          = {e_si100:.4f} eV")
    print(f"  SiO2_Si_term   = {e_sio2si:.4f} eV")
    print(f"  SiO2_O_term    = {e_sio2o:.4f} eV")

    # Load site maps
    si100_sites  = load_sites(SI100_SITES)
    sio2si_sites = load_sites(SIO2_SI_SITES)
    sio2o_sites  = load_sites(SIO2_O_SITES)

    summary_lines = [
        "Phase 3 AllylCpNi Adsorption Search",
        "=" * 68,
        f"Molecule  : AllylCpNi  {mol.get_chemical_formula()}  Ni-idx={NI_IDX}",
        f"e_gas     = {e_gas:.6f} eV",
        f"PHYSI     : flat placement, HEIGHT={HEIGHT_PHYSI} A, {N_SPIN} spins, top-{PRESELECT_PHYSI}",
        f"CHEMI Si  : build_chemisorption_structures center=Ni, top-{PRESELECT_CHEMI}",
        f"CHEMI SiO2: site-map, Ni-down, {N_SPIN} spins, top-{PRESELECT_CHEMI}",
        f"FROZEN_Z  = {FROZEN_Z} A, FMAX={FMAX} eV/A",
        "",
        "Si100:",
    ]

    # ---------- Si100 ----------
    si100_slab.calc = calc
    run_physi("Si100", si100_slab, si100_sites, mol, calc, engine,
              e_si100, e_gas, summary_lines)
    run_chemi_si100(si100_slab, mol, calc, engine, e_si100, e_gas, summary_lines)

    # ---------- SiO2_Si_term ----------
    summary_lines.append("")
    summary_lines.append("SiO2_Si_term:")
    sio2si_slab.calc = calc
    run_physi("SiO2_Si_term", sio2si_slab, sio2si_sites, mol, calc, engine,
              e_sio2si, e_gas, summary_lines)
    run_chemi_sio2("SiO2_Si_term", sio2si_slab, sio2si_sites, mol, calc, engine,
                   e_sio2si, e_gas, summary_lines, CHEMI_HEIGHTS_SI)

    # ---------- SiO2_O_term ----------
    summary_lines.append("")
    summary_lines.append("SiO2_O_term:")
    sio2o_slab.calc = calc
    run_physi("SiO2_O_term", sio2o_slab, sio2o_sites, mol, calc, engine,
              e_sio2o, e_gas, summary_lines)
    run_chemi_sio2("SiO2_O_term", sio2o_slab, sio2o_sites, mol, calc, engine,
                   e_sio2o, e_gas, summary_lines, CHEMI_HEIGHTS_O)

    # Summary
    summary_path = OUT_DIR / "allylcpni_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "=" * 68)
    print("\n".join(summary_lines))
    print(f"\nSummary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
