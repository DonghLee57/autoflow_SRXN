"""Phase 3: Ni(PF3)4 physisorption + chemisorption on Si100, SiO2_Si_term, SiO2_O_term.

Molecule
--------
Ni(PF3)4  =  12F + 4P + 1Ni = 17 atoms
Ni index  :  16 (last atom, VASP order F→P→Ni)
P indices :  12, 13, 14, 15

Cell choice
-----------
Ni(PF3)4 footprint ≈ 5.7-6.0 Å; F vdW radius 1.47 Å.
  SiO2 (10.1 Å): image gap ~1.4 Å after vdW  → too tight  → supercell needed
  Si100 (10.9 Å): image gap ~2.1 Å after vdW → borderline  → supercell for safety

Supercell : [1,1]/[-1,1] in-plane, area×2
  Si100     15.45 Å × 15.45 Å
  SiO2      14.27 Å × 14.27 Å

Inhibitor reference on supercell
---------------------------------
  Best physi structure from phase2 is replicated on supercell slab and relaxed
  to yield consistent E_ads reference for comparison.

Physisorption  :  flat placement, Ni center at HEIGHT_PHYSI above site, 4 spins
Chemisorption  :
  Si100        :  build_chemisorption_structures (center_target="Ni"),
                  uses ORIGINAL Si100 cell (same atom count, slab+mol supercell avoidable)
                  AND supercell attempt for consistency
  SiO2_Si_term :  site-map guided, P as reactive atom, heights 2.5-4.0 Å
  SiO2_O_term  :  site-map guided, P as reactive atom, heights 2.0-3.5 Å

Outputs
-------
  phase3/results/NiPF3_4/{substrate}/physi/
  phase3/results/NiPF3_4/{substrate}/chemi/
  phase3/results/NiPF3_4/inhibitor_ref/   -- inhibitor on supercell (reference)
  phase3/results/NiPF3_4/nipf3_4_summary.txt
"""

import sys
import io
from pathlib import Path

# Force UTF-8 stdout so library print() calls with non-ASCII chars don't crash
# on Windows cp949 console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from ase.io import read, write
from ase.build import make_supercell

from autoflow_srxn.simulation.potentials import SimulationEngine

# Note: build_chemisorption_structures is NOT used for Ni(PF3)4 -- the builder
# generates 0 candidates due to steric clashes of PF3 ligands on Si surface.
# Site-map approach is used for all substrates instead.

from utils import (
    center_molecule, orient_atom_toward_surface, spin_z,
    place_center_on_site, place_atom_on_site,
    interface_analysis, relax_and_score, write_results, load_sites,
    E_ADS_MAX,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NI_IDX         = 16            # Ni in Ni(PF3)4
P_IDX          = [12, 13, 14, 15]   # P atoms
HEIGHT_PHYSI   = 4.5           # Å -- Ni center above site z (larger molecule)
N_SPIN         = 4
PRESELECT_PHYSI = 10
PRESELECT_CHEMI = 10

FROZEN_Z    = 5.5
FMAX        = 0.05
RELAX_STEPS = 250

CHEMI_HEIGHTS_SI  = np.arange(2.5, 4.1, 0.5)   # 2.5 3.0 3.5 4.0  (P-Si bond ~2.3 Å)
CHEMI_HEIGHTS_O   = np.arange(2.0, 3.6, 0.5)   # 2.0 2.5 3.0 3.5  (P-O  bond ~1.8 Å)

# Supercell transform
SC_P = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 1]], dtype=int)

ENGINE_CONFIG = {
    "engine": {
        "potential": {
            "backend": "sevennet", "model": "7net-0",
            "device": "cpu", "dtype": "float32",
        }
    },
    "relaxation": {
        "fmax": FMAX, "steps": RELAX_STEPS,
        "optimizer": "FIRE", "frozen_z_ang": FROZEN_Z,
    },
}

# Paths -- original slabs (kept for reference; supercell used for all runs)
SI100_SLAB_ORIG   = ROOT / "structures/slabs/Si100_slab.vasp"
SIO2_SI_SLAB_ORIG = ROOT / "structures/slabs/SiO2_Si_term_slab.vasp"
SIO2_O_SLAB_ORIG  = ROOT / "structures/slabs/SiO2_O_term_slab.vasp"

# Paths -- supercell slabs (created by setup_supercells.py)
SC_DIR            = ROOT / "structures/slabs/supercell"
SI100_SLAB_SC     = SC_DIR / "Si100_2x_slab.vasp"
SIO2_SI_SLAB_SC   = SC_DIR / "SiO2_Si_term_2x_slab.vasp"
SIO2_O_SLAB_SC    = SC_DIR / "SiO2_O_term_2x_slab.vasp"

# Site maps (same Cartesian positions, valid in supercell)
SI100_SITES       = ROOT / "structures/slabs/site_maps/Si100_sites.csv"
SIO2_SI_SITES     = ROOT / "structures/slabs/site_maps/SiO2_Si_term_sites.csv"
SIO2_O_SITES      = ROOT / "structures/slabs/site_maps/SiO2_O_term_sites.csv"

# Inhibitor (for reference calculation on supercell)
INHIBITOR_PATH    = ROOT / "structures/inhibitor_relaxed.vasp"
INHIBITOR_CENTER  = 13   # center C idx in inhibitor

# Molecule
MOL_PATH = ROOT / "structures/NiPF3_4_relaxed.vasp"

OUT_DIR = ROOT / "phase3/results/NiPF3_4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Supercell slab helper
# ---------------------------------------------------------------------------

def make_or_load_sc(orig_path, sc_path, calc):
    """Load supercell slab if exists, else create on-the-fly."""
    if sc_path.exists():
        slab = read(str(sc_path))
    else:
        print(f"  [warn] {sc_path.name} not found -- creating on-the-fly")
        orig = read(str(orig_path))
        slab = make_supercell(orig, SC_P)
        sc_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(sc_path), slab, vasp5=True)
    slab.calc = calc
    return slab


# ---------------------------------------------------------------------------
# Inhibitor reference on supercell
# ---------------------------------------------------------------------------

def run_inhibitor_ref(sc_slab, sc_slab_name, calc, engine, e_sc_slab,
                      e_inh_gas, summary_lines):
    """Place inhibitor on supercell slab using flat-placement (4 sites × 4 spins).
    This gives a consistent reference E_ads for comparison with Ni(PF3)4.
    """
    sub_dir = OUT_DIR / "inhibitor_ref" / sc_slab_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    inhibitor = read(str(INHIBITOR_PATH))
    sites     = load_sites(SI100_SITES if "Si100" in sc_slab_name else
                           (SIO2_SI_SITES if "Si_term" in sc_slab_name else SIO2_O_SITES))
    mol_c     = center_molecule(inhibitor)
    spins     = range(0, 360, 360 // N_SPIN)

    print(f"\n  [Inhibitor ref on supercell: {sc_slab_name}]")
    candidates = []
    for site in sites[:4]:   # top-4 sites only (quick reference)
        site_xyz = np.array([site["x_A"], site["y_A"], site["z_A"]])
        for deg in spins:
            mol_s    = spin_z(mol_c, deg)
            combined = place_center_on_site(sc_slab, mol_s, site_xyz,
                                            INHIBITOR_CENTER, 2.5)
            candidates.append({"site_id": site["site_id"], "spin_deg": int(deg),
                                "atoms": combined})

    for c in candidates:
        c["atoms"].calc = calc
        e0 = float(c["atoms"].get_potential_energy())
        c["e_ads_init"] = e0 - e_sc_slab - e_inh_gas

    candidates.sort(key=lambda c: c["e_ads_init"])

    # Relax top 4
    relaxed = []
    for i, c in enumerate(candidates[:4]):
        print(f"    inh ref relax {i+1}/4: site={c['site_id']} spin={c['spin_deg']}  "
              f"E_init={c['e_ads_init']:+.3f} eV")
        ar, e_ads, mind, mpair, nb = relax_and_score(
            c["atoms"], engine, e_sc_slab, e_inh_gas, FROZEN_Z, RELAX_STEPS, FMAX)
        relaxed.append((e_ads, ar, mind, mpair, nb,
                        {"site_id": c["site_id"], "spin_deg": c["spin_deg"]}))
        print(f"      -> E_ads={e_ads:+.4f} eV  dist={mind:.3f} A ({mpair})  cov={nb}")

    if relaxed:
        rows = write_results(sub_dir, f"inhibitor_{sc_slab_name}_sc_physi", relaxed)
        r1   = rows[0]
        summary_lines.append(
            f"  Inhibitor ref ({sc_slab_name} 2x): "
            f"rank1 E_ads={r1['e_ads_eV']} eV  dist={r1['min_dist_A']} A ({r1['min_pair']})"
        )


# ---------------------------------------------------------------------------
# Physisorption -- flat placement on supercell
# ---------------------------------------------------------------------------

def run_physi(name, sc_slab, sites, mol, calc, engine, e_slab, e_gas, summary_lines):
    sub_dir = OUT_DIR / name / "physi"
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  Ni(PF3)4 PHYSI -- {name} [supercell]  (HEIGHT={HEIGHT_PHYSI} A, {N_SPIN} spins)")
    print(f"{'='*68}")

    mol_c = center_molecule(mol)
    spins = range(0, 360, 360 // N_SPIN)

    candidates = []
    for site in sites:
        site_xyz = np.array([site["x_A"], site["y_A"], site["z_A"]])
        for deg in spins:
            mol_s    = spin_z(mol_c, deg)
            combined = place_center_on_site(sc_slab, mol_s, site_xyz, NI_IDX, HEIGHT_PHYSI)
            candidates.append({"site_id": site["site_id"], "spin_deg": int(deg),
                                "atoms": combined})

    print(f"  {len(sites)} sites × {N_SPIN} spins = {len(candidates)} candidates")

    for c in candidates:
        c["atoms"].calc = calc
        e0 = float(c["atoms"].get_potential_energy())
        c["e_ads_init"] = e0 - e_slab - e_gas

    candidates.sort(key=lambda c: c["e_ads_init"])
    print("  Pre-screen top 5:")
    for c in candidates[:5]:
        print(f"    site={c['site_id']:5s} spin={c['spin_deg']:3d}  E_init={c['e_ads_init']:+.3f} eV")

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
        summary_lines.append(f"  {name} PHYSI: no candidates")
        return

    rows = write_results(sub_dir, f"NiPF3_4_{name}_physi", relaxed)
    top  = rows[:3]
    summary_lines.append(f"  {name} physi: relaxed={len(relaxed)}")
    for r in top:
        flag = "CHEM" if int(r["interface_bonds"]) > 0 else "PHYSI"
        summary_lines.append(
            f"    rank{r['rank']}: E_ads={r['e_ads_eV']} eV  "
            f"dist={r['min_dist_A']} A ({r['min_pair']})  cov={r['interface_bonds']}  [{flag}]"
        )


# ---------------------------------------------------------------------------
# Chemisorption -- Si100 via site-map on supercell (P as reactive atom)
# Note: build_chemisorption_structures generates 0 candidates for Ni(PF3)4
# because the large PF3 ligands cause severe steric clashes on the Si dimer
# surface. The site-map approach with P-down orientation is more flexible.
# ---------------------------------------------------------------------------

def run_chemi_si100(sc_slab, sites, mol, calc, engine, e_slab, e_gas, summary_lines):
    sub_dir = OUT_DIR / "Si100" / "chemi"
    sub_dir.mkdir(parents=True, exist_ok=True)

    name   = "Si100"
    spins  = range(0, 360, 360 // N_SPIN)
    heights = CHEMI_HEIGHTS_SI   # 2.5 3.0 3.5 4.0 A

    print(f"\n{'='*68}")
    print(f"  Ni(PF3)4 CHEMI -- Si100 [supercell] (site-map, P-down, "
          f"h={heights[0]:.1f}-{heights[-1]:.1f} A)")
    print(f"{'='*68}")
    print(f"  Note: builder skipped (generates 0 candidates for Ni(PF3)4 "
          f"due to steric constraints on Si dimer surface)")

    P_REPR       = P_IDX[0]
    mol_oriented = orient_atom_toward_surface(mol.copy(), P_REPR)
    total        = len(sites) * len(heights) * N_SPIN
    print(f"  Grid: {len(sites)} sites x {len(heights)} heights x {N_SPIN} spins = {total}")

    candidates = []
    for site in sites:
        site_xyz = np.array([site["x_A"], site["y_A"], site["z_A"]])
        for h in heights:
            for deg in spins:
                mol_s    = spin_z(mol_oriented, deg)
                combined = place_atom_on_site(sc_slab, mol_s, site_xyz, P_REPR, float(h))
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
            "site_id":       c["site_id"],
            "init_height_A": f"{c['height']:.1f}",
            "spin_deg":      c["spin_deg"],
            "reactive_atom": "P",
            "mode":          "chemi",
        }))
        print(f"    -> E_ads={e_ads:+.4f} eV  dist={mind:.3f} A ({mpair})  cov={nb}")

    if not relaxed:
        summary_lines.append("  Si100 CHEMI: no relaxed candidates")
        return

    rows = write_results(sub_dir, "NiPF3_4_Si100_chemi", relaxed)
    top  = rows[:3]
    summary_lines.append(f"  Si100 chemi: relaxed={len(relaxed)}")
    for r in top:
        flag = "CHEM" if int(r["interface_bonds"]) > 0 else "PHYSI"
        summary_lines.append(
            f"    rank{r['rank']}: E_ads={r['e_ads_eV']} eV  "
            f"dist={r['min_dist_A']} A ({r['min_pair']})  cov={r['interface_bonds']}  [{flag}]"
        )


# ---------------------------------------------------------------------------
# Chemisorption -- SiO2 via site-map, P as reactive atom, supercell
# ---------------------------------------------------------------------------

def run_chemi_sio2(name, sc_slab, sites, mol, calc, engine, e_slab, e_gas,
                   summary_lines, heights):
    sub_dir = OUT_DIR / name / "chemi"
    sub_dir.mkdir(parents=True, exist_ok=True)

    spins = range(0, 360, 360 // N_SPIN)

    print(f"\n{'='*68}")
    print(f"  Ni(PF3)4 CHEMI -- {name} [supercell] (P-down, h={heights[0]:.1f}-{heights[-1]:.1f} A)")
    print(f"{'='*68}")

    # Use first P atom (12) as representative reactive atom
    # (all P are equivalent by Td symmetry of Ni(PF3)4)
    P_REPR = P_IDX[0]

    # Orient P_REPR toward surface
    mol_oriented = orient_atom_toward_surface(mol.copy(), P_REPR)

    total = len(sites) * len(heights) * N_SPIN
    print(f"  Grid: {len(sites)} sites × {len(heights)} heights × {N_SPIN} spins = {total}")

    candidates = []
    for site in sites:
        site_xyz = np.array([site["x_A"], site["y_A"], site["z_A"]])
        for h in heights:
            for deg in spins:
                mol_s    = spin_z(mol_oriented, deg)
                combined = place_atom_on_site(sc_slab, mol_s, site_xyz, P_REPR, float(h))
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
            "spin_deg": c["spin_deg"], "reactive_atom": "P",
            "mode": "chemi",
        }))
        print(f"    -> E_ads={e_ads:+.4f} eV  dist={mind:.3f} A ({mpair})  cov={nb}")

    if not relaxed:
        summary_lines.append(f"  {name} CHEMI: no relaxed candidates")
        return

    rows = write_results(sub_dir, f"NiPF3_4_{name}_chemi", relaxed)
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
    syms = mol.get_chemical_symbols()
    print(f"  Ni idx={NI_IDX} ({syms[NI_IDX]}),  P idx={P_IDX} ({[syms[i] for i in P_IDX]})")

    gas = mol.copy()
    gas.center(vacuum=10.0)
    gas.calc = calc
    e_gas = float(gas.get_potential_energy())
    print(f"  Gas energy: {e_gas:.6f} eV")

    # Inhibitor gas energy (for reference calc)
    inhibitor = read(str(INHIBITOR_PATH))
    inh_gas   = inhibitor.copy()
    inh_gas.center(vacuum=10.0)
    inh_gas.calc = calc
    e_inh_gas = float(inh_gas.get_potential_energy())
    print(f"  Inhibitor gas: {e_inh_gas:.6f} eV")

    # Load supercell slabs
    sc_si100  = make_or_load_sc(SI100_SLAB_ORIG,   SI100_SLAB_SC,   calc)
    sc_sio2si = make_or_load_sc(SIO2_SI_SLAB_ORIG, SIO2_SI_SLAB_SC, calc)
    sc_sio2o  = make_or_load_sc(SIO2_O_SLAB_ORIG,  SIO2_O_SLAB_SC,  calc)

    e_sc_si100  = float(sc_si100.get_potential_energy())
    e_sc_sio2si = float(sc_sio2si.get_potential_energy())
    e_sc_sio2o  = float(sc_sio2o.get_potential_energy())
    print(f"\nSupercell slab energies:")
    print(f"  Si100     2x = {e_sc_si100:.4f} eV  ({len(sc_si100)} atoms)")
    print(f"  SiO2_Si   2x = {e_sc_sio2si:.4f} eV  ({len(sc_sio2si)} atoms)")
    print(f"  SiO2_O    2x = {e_sc_sio2o:.4f} eV  ({len(sc_sio2o)} atoms)")

    # Site maps
    si100_sites  = load_sites(SI100_SITES)
    sio2si_sites = load_sites(SIO2_SI_SITES)
    sio2o_sites  = load_sites(SIO2_O_SITES)

    summary_lines = [
        "Phase 3 Ni(PF3)4 Adsorption Search",
        "=" * 68,
        f"Molecule  : Ni(PF3)4  {mol.get_chemical_formula()}  Ni-idx={NI_IDX}",
        f"e_gas     = {e_gas:.6f} eV",
        f"PHYSI     : flat placement (supercell), HEIGHT={HEIGHT_PHYSI} A, {N_SPIN} spins, top-{PRESELECT_PHYSI}",
        f"CHEMI     : site-map (supercell), P-down, {N_SPIN} spins, top-{PRESELECT_CHEMI}",
        f"FROZEN_Z  = {FROZEN_Z} A, FMAX={FMAX} eV/A",
        "",
        "Inhibitor reference (supercell):",
    ]

    # Inhibitor reference on each supercell slab
    run_inhibitor_ref(sc_si100,  "Si100",        calc, engine, e_sc_si100,  e_inh_gas, summary_lines)
    run_inhibitor_ref(sc_sio2si, "SiO2_Si_term", calc, engine, e_sc_sio2si, e_inh_gas, summary_lines)
    run_inhibitor_ref(sc_sio2o,  "SiO2_O_term",  calc, engine, e_sc_sio2o,  e_inh_gas, summary_lines)

    summary_lines.append("")
    summary_lines.append("Si100:")
    # Physisorption on supercell Si100
    run_physi("Si100", sc_si100, si100_sites, mol, calc, engine,
              e_sc_si100, e_gas, summary_lines)
    # Chemisorption via site-map on supercell Si100 (builder gives 0 candidates)
    run_chemi_si100(sc_si100, si100_sites, mol, calc, engine,
                    e_sc_si100, e_gas, summary_lines)

    summary_lines.append("")
    summary_lines.append("SiO2_Si_term:")
    run_physi("SiO2_Si_term", sc_sio2si, sio2si_sites, mol, calc, engine,
              e_sc_sio2si, e_gas, summary_lines)
    run_chemi_sio2("SiO2_Si_term", sc_sio2si, sio2si_sites, mol, calc, engine,
                   e_sc_sio2si, e_gas, summary_lines, CHEMI_HEIGHTS_SI)

    summary_lines.append("")
    summary_lines.append("SiO2_O_term:")
    run_physi("SiO2_O_term", sc_sio2o, sio2o_sites, mol, calc, engine,
              e_sc_sio2o, e_gas, summary_lines)
    run_chemi_sio2("SiO2_O_term", sc_sio2o, sio2o_sites, mol, calc, engine,
                   e_sc_sio2o, e_gas, summary_lines, CHEMI_HEIGHTS_O)

    summary_path = OUT_DIR / "nipf3_4_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "=" * 68)
    print("\n".join(summary_lines))
    print(f"\nSummary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
