"""Inhibitor full physisorption + chemisorption on [1,1]/[-1,1] supercell slabs.

This provides proper comparison with Ni(PF3)4 supercell results.
Phase2 results used original cells; here we use 2x supercell for all substrates.

Inhibitor: C5H13NO2, 21 atoms
  center C idx=13,  N idx=18,  O1 idx=19,  O2 idx=20

Physi  : flat placement, center C (idx=13), HEIGHT=2.5 A, 4 spins, all sites, top-10
Chemi  :
  Si100 2x     : site-map, reactive=N(18)/O1(19), heights 2.0-3.5 A  (N-Si ~1.74 A)
  SiO2_Si 2x  : site-map, reactive=O1(19)/O2(20)/N(18), heights 2.5-4.0 A
  SiO2_O  2x  : site-map, reactive=N(18)/O1(19), heights 2.0-3.5 A

Note: chemisorption_builder not used here (Unicode issues in library print calls).
      Site-map approach captures the same chemisorption basins (confirmed in phase2).

Outputs: phase3/results/inhibitor_supercell/{substrate}/physi/
                                                         /chemi/
         phase3/results/inhibitor_supercell/inhibitor_sc_summary.txt
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from ase.io import read

from autoflow_srxn.simulation.potentials import SimulationEngine

from utils import (
    center_molecule, orient_atom_toward_surface, spin_z,
    place_center_on_site, place_atom_on_site,
    relax_and_score, write_results, load_sites,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CENTER_C = 13      # center C
N_IDX    = 18      # N
O1_IDX   = 19      # O1
O2_IDX   = 20      # O2

HEIGHT_PHYSI  = 2.5   # A  (slightly higher than phase2's 2.3 for supercell safety)
N_SPIN        = 4
PRESELECT_PHYSI = 10
PRESELECT_CHEMI = 12

FROZEN_Z    = 5.5
FMAX        = 0.05
RELAX_STEPS = 250

# Chemi heights
HEIGHTS_SI100  = np.arange(2.0, 3.6, 0.5)   # 2.0 2.5 3.0 3.5  (N-Si ~1.74 A)
HEIGHTS_SIO2SI = np.arange(2.5, 4.1, 0.5)   # 2.5 3.0 3.5 4.0
HEIGHTS_SIO2O  = np.arange(1.8, 3.4, 0.5)   # 1.8 2.3 2.8 3.3

ENGINE_CONFIG = {
    "engine": {"potential": {"backend": "sevennet", "model": "7net-0",
                             "device": "cpu", "dtype": "float32"}},
    "relaxation": {"fmax": FMAX, "steps": RELAX_STEPS,
                   "optimizer": "FIRE", "frozen_z_ang": FROZEN_Z},
}

# Supercell slabs
SC_DIR         = ROOT / "structures/slabs/supercell"
SI100_SC_SLAB  = SC_DIR / "Si100_2x_slab.vasp"
SIO2SI_SC_SLAB = SC_DIR / "SiO2_Si_term_2x_slab.vasp"
SIO2O_SC_SLAB  = SC_DIR / "SiO2_O_term_2x_slab.vasp"

# Site maps (original Cartesian positions, valid in supercell)
SI100_SITES    = ROOT / "structures/slabs/site_maps/Si100_sites.csv"
SIO2SI_SITES   = ROOT / "structures/slabs/site_maps/SiO2_Si_term_sites.csv"
SIO2O_SITES    = ROOT / "structures/slabs/site_maps/SiO2_O_term_sites.csv"

MOL_PATH = ROOT / "structures/inhibitor_relaxed.vasp"
OUT_DIR  = ROOT / "phase3/results/inhibitor_supercell"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Physisorption -- flat placement, center C at HEIGHT
# ---------------------------------------------------------------------------
def run_physi(name, slab, sites, mol, calc, engine, e_slab, e_gas, summary_lines):
    sub_dir = OUT_DIR / name / "physi"
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  Inhibitor PHYSI [supercell] -- {name}  "
          f"(CENTER_C={CENTER_C}, HEIGHT={HEIGHT_PHYSI} A, {N_SPIN} spins)")
    print(f"{'='*68}")

    mol_c = center_molecule(mol)
    spins = range(0, 360, 360 // N_SPIN)

    candidates = []
    for site in sites:
        site_xyz = np.array([site["x_A"], site["y_A"], site["z_A"]])
        for deg in spins:
            mol_s    = spin_z(mol_c, deg)
            combined = place_center_on_site(slab, mol_s, site_xyz, CENTER_C, HEIGHT_PHYSI)
            candidates.append({"site_id": site["site_id"], "spin_deg": int(deg),
                                "atoms": combined})

    print(f"  {len(sites)} sites x {N_SPIN} spins = {len(candidates)} candidates")

    for c in candidates:
        c["atoms"].calc = calc
        c["e_ads_init"] = float(c["atoms"].get_potential_energy()) - e_slab - e_gas

    candidates.sort(key=lambda c: c["e_ads_init"])
    print("  Pre-screen top 5:")
    for c in candidates[:5]:
        print(f"    site={c['site_id']:5s} spin={c['spin_deg']:3d}  "
              f"E_init={c['e_ads_init']:+.3f} eV")

    selected = candidates[:min(PRESELECT_PHYSI, len(candidates))]
    relaxed  = []
    for i, c in enumerate(selected):
        print(f"\n  Relax {i+1}/{len(selected)}: site={c['site_id']} spin={c['spin_deg']}  "
              f"E_init={c['e_ads_init']:+.3f} eV")
        ar, e_ads, mind, mpair, nb = relax_and_score(
            c["atoms"], engine, e_slab, e_gas, FROZEN_Z, RELAX_STEPS, FMAX)
        relaxed.append((e_ads, ar, mind, mpair, nb,
                        {"site_id": c["site_id"], "spin_deg": c["spin_deg"], "mode": "physi"}))
        print(f"    -> E_ads={e_ads:+.4f} eV  dist={mind:.3f} A ({mpair})  cov={nb}")

    if not relaxed:
        summary_lines.append(f"  {name} PHYSI: no candidates"); return

    rows = write_results(sub_dir, f"inhibitor_{name}_sc_physi", relaxed)
    top  = rows[:3]
    summary_lines.append(f"  {name} physi [2x]: relaxed={len(relaxed)}")
    for r in top:
        flag = "CHEM" if int(r["interface_bonds"]) > 0 else "PHYSI"
        summary_lines.append(
            f"    rank{r['rank']}: E_ads={r['e_ads_eV']} eV  "
            f"dist={r['min_dist_A']} A ({r['min_pair']})  cov={r['interface_bonds']}  [{flag}]"
        )


# ---------------------------------------------------------------------------
# Chemisorption -- site-map, multiple reactive atoms
# ---------------------------------------------------------------------------
def run_chemi(name, slab, sites, mol, calc, engine, e_slab, e_gas,
              reactive_atoms, heights, summary_lines):
    """
    reactive_atoms: list of (atom_idx, label) tuples
    """
    sub_dir = OUT_DIR / name / "chemi"
    sub_dir.mkdir(parents=True, exist_ok=True)

    spins = range(0, 360, 360 // N_SPIN)
    atom_labels = [lbl for _, lbl in reactive_atoms]

    print(f"\n{'='*68}")
    print(f"  Inhibitor CHEMI [supercell] -- {name}")
    print(f"  Reactive atoms: {atom_labels}  heights: {heights[0]:.1f}-{heights[-1]:.1f} A")
    print(f"{'='*68}")

    # Pre-orient each reactive atom toward surface
    oriented = {}
    for idx, lbl in reactive_atoms:
        oriented[idx] = orient_atom_toward_surface(mol.copy(), idx)

    total = len(sites) * len(reactive_atoms) * len(heights) * N_SPIN
    print(f"  Grid: {len(sites)} sites x {len(reactive_atoms)} atoms x "
          f"{len(heights)} heights x {N_SPIN} spins = {total}")

    candidates = []
    for site in sites:
        site_xyz = np.array([site["x_A"], site["y_A"], site["z_A"]])
        for idx, lbl in reactive_atoms:
            mol_base = oriented[idx]
            for h in heights:
                for deg in spins:
                    mol_s    = spin_z(mol_base, deg)
                    combined = place_atom_on_site(slab, mol_s, site_xyz, idx, float(h))
                    candidates.append({
                        "site_id": site["site_id"], "atom_lbl": lbl,
                        "height": float(h), "spin_deg": int(deg), "atoms": combined,
                    })

    for c in candidates:
        c["atoms"].calc = calc
        c["e_ads_init"] = float(c["atoms"].get_potential_energy()) - e_slab - e_gas

    candidates.sort(key=lambda c: c["e_ads_init"])
    print("  Pre-screen top 5:")
    for c in candidates[:5]:
        print(f"    site={c['site_id']:5s} atom={c['atom_lbl']:3s} "
              f"h={c['height']:.1f} spin={c['spin_deg']:3d}  "
              f"E_init={c['e_ads_init']:+.3f} eV")

    selected = candidates[:min(PRESELECT_CHEMI, len(candidates))]
    relaxed  = []
    for i, c in enumerate(selected):
        print(f"\n  Relax {i+1}/{len(selected)}: site={c['site_id']} "
              f"atom={c['atom_lbl']} h={c['height']:.1f} spin={c['spin_deg']}  "
              f"E_init={c['e_ads_init']:+.3f} eV")
        ar, e_ads, mind, mpair, nb = relax_and_score(
            c["atoms"], engine, e_slab, e_gas, FROZEN_Z, RELAX_STEPS, FMAX)
        relaxed.append((e_ads, ar, mind, mpair, nb, {
            "site_id": c["site_id"], "reactive_atom": c["atom_lbl"],
            "init_height_A": f"{c['height']:.1f}", "spin_deg": c["spin_deg"],
            "mode": "chemi",
        }))
        print(f"    -> E_ads={e_ads:+.4f} eV  dist={mind:.3f} A ({mpair})  cov={nb}")

    if not relaxed:
        summary_lines.append(f"  {name} CHEMI: no candidates"); return

    rows = write_results(sub_dir, f"inhibitor_{name}_sc_chemi", relaxed)
    top  = rows[:3]
    summary_lines.append(f"  {name} chemi [2x]: relaxed={len(relaxed)}")
    for r in top:
        flag = "CHEM" if int(r["interface_bonds"]) > 0 else "PHYSI"
        summary_lines.append(
            f"    rank{r['rank']}: E_ads={r['e_ads_eV']} eV  "
            f"dist={r['min_dist_A']} A ({r['min_pair']})  cov={r['interface_bonds']}  "
            f"atom={r.get('reactive_atom','?')}  [{flag}]"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    engine = SimulationEngine(ENGINE_CONFIG)
    calc   = engine.get_calculator()

    mol  = read(str(MOL_PATH))
    syms = mol.get_chemical_symbols()
    print(f"Inhibitor: {mol.get_chemical_formula()}  ({len(mol)} atoms)")
    print(f"  center C={CENTER_C}({syms[CENTER_C]}), N={N_IDX}({syms[N_IDX]}), "
          f"O1={O1_IDX}({syms[O1_IDX]}), O2={O2_IDX}({syms[O2_IDX]})")

    gas = mol.copy(); gas.center(vacuum=10.0); gas.calc = calc
    e_gas = float(gas.get_potential_energy())
    print(f"  Gas energy: {e_gas:.6f} eV")

    # Load supercell slabs
    sc_si100  = read(str(SI100_SC_SLAB))
    sc_sio2si = read(str(SIO2SI_SC_SLAB))
    sc_sio2o  = read(str(SIO2O_SC_SLAB))
    for s in [sc_si100, sc_sio2si, sc_sio2o]:
        s.calc = calc
    e_si100  = float(sc_si100.get_potential_energy())
    e_sio2si = float(sc_sio2si.get_potential_energy())
    e_sio2o  = float(sc_sio2o.get_potential_energy())
    print(f"\nSupercell slab energies:")
    print(f"  Si100 2x     = {e_si100:.4f} eV  ({len(sc_si100)} atoms)")
    print(f"  SiO2_Si 2x   = {e_sio2si:.4f} eV  ({len(sc_sio2si)} atoms)")
    print(f"  SiO2_O  2x   = {e_sio2o:.4f} eV  ({len(sc_sio2o)} atoms)")

    si100_sites  = load_sites(SI100_SITES)
    sio2si_sites = load_sites(SIO2SI_SITES)
    sio2o_sites  = load_sites(SIO2O_SITES)

    summary_lines = [
        "Inhibitor Supercell Adsorption Search (2x cells)",
        "=" * 68,
        f"Inhibitor : {mol.get_chemical_formula()}  center_C={CENTER_C}",
        f"e_gas     = {e_gas:.6f} eV",
        f"PHYSI     : flat placement (CENTER_C), HEIGHT={HEIGHT_PHYSI} A, top-{PRESELECT_PHYSI}",
        f"CHEMI     : site-map, N/O reactive atoms, top-{PRESELECT_CHEMI}",
        f"FROZEN_Z  = {FROZEN_Z} A, FMAX={FMAX} eV/A",
        "",
        "Si100 [2x supercell]:",
    ]

    # Si100
    sc_si100.calc = calc
    run_physi("Si100", sc_si100, si100_sites, mol, calc, engine,
              e_si100, e_gas, summary_lines)
    run_chemi("Si100", sc_si100, si100_sites, mol, calc, engine, e_si100, e_gas,
              [(N_IDX, "N"), (O1_IDX, "O1")], HEIGHTS_SI100, summary_lines)

    # SiO2_Si_term
    summary_lines.append(""); summary_lines.append("SiO2_Si_term [2x supercell]:")
    sc_sio2si.calc = calc
    run_physi("SiO2_Si_term", sc_sio2si, sio2si_sites, mol, calc, engine,
              e_sio2si, e_gas, summary_lines)
    run_chemi("SiO2_Si_term", sc_sio2si, sio2si_sites, mol, calc, engine, e_sio2si, e_gas,
              [(O1_IDX, "O1"), (O2_IDX, "O2"), (N_IDX, "N")], HEIGHTS_SIO2SI, summary_lines)

    # SiO2_O_term
    summary_lines.append(""); summary_lines.append("SiO2_O_term [2x supercell]:")
    sc_sio2o.calc = calc
    run_physi("SiO2_O_term", sc_sio2o, sio2o_sites, mol, calc, engine,
              e_sio2o, e_gas, summary_lines)
    run_chemi("SiO2_O_term", sc_sio2o, sio2o_sites, mol, calc, engine, e_sio2o, e_gas,
              [(N_IDX, "N"), (O1_IDX, "O1")], HEIGHTS_SIO2O, summary_lines)

    summary_path = OUT_DIR / "inhibitor_sc_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "=" * 68)
    print("\n".join(summary_lines))
    print(f"\nSummary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
