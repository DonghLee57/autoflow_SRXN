"""AllylCpNi SiO2 chemisorption FIX.

Root cause of original failure:
  run_allylcpni.py used orient_atom_toward_surface(mol, NI_IDX=18) for SiO2 chemi.
  AllylCpNi geometry:
    allyl C (idx 15-17)  z ~10.0-10.2 A  <- BELOW Ni, closest to surface
    Ni      (idx 18)     z ~11.7 A        <- middle
    Cp C    (idx 10-14)  z ~13.6-13.9 A  <- ABOVE Ni

  Orienting Ni toward -z puts Ni at bottom but LIFTS allyl C upward,
  so allyl C (the actual reactive fragment seen forming C-Si bonds on Si100)
  never contacts the SiO2 surface.

Fix:
  Use allyl C idx=15 as the reactive atom for orient_atom_toward_surface.
  allyl C -> surface heights: 1.8-3.5 A for SiO2_Si_term (C-Si ~1.87 A)
                               1.5-3.0 A for SiO2_O_term  (C-O  ~1.43 A)

  Also run SiO2 physi with allyl C at flat-placement heights 2.5-4.0 A to
  check if any C-O/C-Si physisorption basin exists.

Outputs: phase3/results/AllylCpNi/{substrate}/chemi_fix/
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
ALLYL_C_IDX  = 15    # allyl C atom used as reactive (others: 16, 17)
NI_IDX       = 18
N_SPIN       = 4
PRESELECT    = 12

FROZEN_Z     = 5.5
FMAX         = 0.05
RELAX_STEPS  = 250

# Heights for allyl C -> surface chemi (Cartesian z of atom above site z)
HEIGHTS_SI   = np.arange(1.8, 3.6, 0.5)   # 1.8 2.3 2.8 3.3  (C-Si ~1.87 A)
HEIGHTS_O    = np.arange(1.5, 3.1, 0.5)   # 1.5 2.0 2.5 3.0  (C-O  ~1.43 A)

# Heights for flat physi (Ni center above site)
HEIGHTS_PHYSI_SI = np.arange(3.0, 5.1, 0.5)   # 3.0 3.5 4.0 4.5 5.0
HEIGHTS_PHYSI_O  = np.arange(2.5, 4.6, 0.5)   # 2.5 3.0 3.5 4.0 4.5

ENGINE_CONFIG = {
    "engine": {"potential": {"backend": "sevennet", "model": "7net-0",
                             "device": "cpu", "dtype": "float32"}},
    "relaxation": {"fmax": FMAX, "steps": RELAX_STEPS,
                   "optimizer": "FIRE", "frozen_z_ang": FROZEN_Z},
}

MOL_PATH      = ROOT / "structures/AllylCpNi_relaxed.vasp"
SIO2_SI_SLAB  = ROOT / "structures/slabs/SiO2_Si_term_slab.vasp"
SIO2_O_SLAB   = ROOT / "structures/slabs/SiO2_O_term_slab.vasp"
SIO2_SI_SITES = ROOT / "structures/slabs/site_maps/SiO2_Si_term_sites.csv"
SIO2_O_SITES  = ROOT / "structures/slabs/site_maps/SiO2_O_term_sites.csv"
OUT_DIR       = ROOT / "phase3/results/AllylCpNi"


# ---------------------------------------------------------------------------
# Site-map chemi with allyl C as reactive atom
# ---------------------------------------------------------------------------
def run_sio2_chemi_fix(name, slab, sites, mol, calc, engine, e_slab, e_gas,
                       heights, summary_lines):
    sub_dir = OUT_DIR / name / "chemi_fix"
    sub_dir.mkdir(parents=True, exist_ok=True)

    spins = range(0, 360, 360 // N_SPIN)
    print(f"\n{'='*68}")
    print(f"  AllylCpNi CHEMI FIX -- {name}  (allyl-C idx={ALLYL_C_IDX} down)")
    print(f"  Heights: {heights[0]:.1f}-{heights[-1]:.1f} A,  {N_SPIN} spins")
    print(f"{'='*68}")

    mol_oriented = orient_atom_toward_surface(mol.copy(), ALLYL_C_IDX)

    total = len(sites) * len(heights) * N_SPIN
    print(f"  Grid: {len(sites)} sites x {len(heights)} heights x {N_SPIN} spins = {total}")

    candidates = []
    for site in sites:
        site_xyz = np.array([site["x_A"], site["y_A"], site["z_A"]])
        for h in heights:
            for deg in spins:
                mol_s    = spin_z(mol_oriented, deg)
                combined = place_atom_on_site(slab, mol_s, site_xyz,
                                              ALLYL_C_IDX, float(h))
                candidates.append({
                    "site_id": site["site_id"], "height": float(h),
                    "spin_deg": int(deg), "atoms": combined,
                })

    for c in candidates:
        c["atoms"].calc = calc
        e0 = float(c["atoms"].get_potential_energy())
        c["e_ads_init"] = e0 - e_slab - e_gas

    candidates.sort(key=lambda c: c["e_ads_init"])
    print("  Pre-screen top 5:")
    for c in candidates[:5]:
        print(f"    site={c['site_id']:5s} h={c['height']:.1f} "
              f"spin={c['spin_deg']:3d}  E_init={c['e_ads_init']:+.3f} eV")

    selected = candidates[:min(PRESELECT, len(candidates))]
    relaxed  = []
    for i, c in enumerate(selected):
        print(f"\n  Relax {i+1}/{len(selected)}: site={c['site_id']} "
              f"h={c['height']:.1f} spin={c['spin_deg']}  "
              f"E_init={c['e_ads_init']:+.3f} eV")
        ar, e_ads, mind, mpair, nb = relax_and_score(
            c["atoms"], engine, e_slab, e_gas, FROZEN_Z, RELAX_STEPS, FMAX)
        relaxed.append((e_ads, ar, mind, mpair, nb, {
            "site_id": c["site_id"], "init_height_A": f"{c['height']:.1f}",
            "spin_deg": c["spin_deg"], "reactive_atom": f"allylC(idx={ALLYL_C_IDX})",
            "mode": "chemi_fix",
        }))
        print(f"    -> E_ads={e_ads:+.4f} eV  dist={mind:.3f} A ({mpair})  cov={nb}")

    if not relaxed:
        summary_lines.append(f"  {name} CHEMI FIX: no relaxed candidates")
        return

    rows = write_results(sub_dir, f"AllylCpNi_{name}_chemi_fix", relaxed)
    top  = rows[:3]
    summary_lines.append(f"  {name} chemi_fix (allyl C down): relaxed={len(relaxed)}")
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
    syms = mol.get_chemical_symbols()
    print(f"Molecule: {mol.get_chemical_formula()}  ({len(mol)} atoms)")
    print(f"  Allyl C idx={ALLYL_C_IDX} ({syms[ALLYL_C_IDX]}),  Ni idx={NI_IDX} ({syms[NI_IDX]})")

    gas = mol.copy(); gas.center(vacuum=10.0); gas.calc = calc
    e_gas = float(gas.get_potential_energy())
    print(f"  Gas energy: {e_gas:.6f} eV")

    sio2si_slab = read(str(SIO2_SI_SLAB))
    sio2o_slab  = read(str(SIO2_O_SLAB))
    for s in [sio2si_slab, sio2o_slab]:
        s.calc = calc
    e_si = float(sio2si_slab.get_potential_energy())
    e_o  = float(sio2o_slab.get_potential_energy())
    print(f"  SiO2_Si_term slab: {e_si:.4f} eV")
    print(f"  SiO2_O_term  slab: {e_o:.4f} eV")

    sio2si_sites = load_sites(SIO2_SI_SITES)
    sio2o_sites  = load_sites(SIO2_O_SITES)

    summary_lines = [
        "AllylCpNi SiO2 Chemisorption Fix (allyl C as reactive atom)",
        "=" * 68,
        f"Reactive atom: allyl C idx={ALLYL_C_IDX}",
        f"e_gas = {e_gas:.6f} eV",
        "",
        "SiO2_Si_term:",
    ]

    run_sio2_chemi_fix("SiO2_Si_term", sio2si_slab, sio2si_sites,
                       mol, calc, engine, e_si, e_gas, HEIGHTS_SI, summary_lines)

    summary_lines.append("")
    summary_lines.append("SiO2_O_term:")
    run_sio2_chemi_fix("SiO2_O_term", sio2o_slab, sio2o_sites,
                       mol, calc, engine, e_o, e_gas, HEIGHTS_O, summary_lines)

    summary_path = OUT_DIR / "allylcpni_sio2_fix_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "=" * 68)
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
