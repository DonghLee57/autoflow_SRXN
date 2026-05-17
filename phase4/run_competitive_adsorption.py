"""Phase 4: Competitive adsorption of AllylCpNi / Ni(PF3)4 on inhibitor-covered surfaces.

Concept
-------
E_ads_prec = E(slab+inh+prec) - E(slab+inh) - E(prec_gas)

Compare with Phase 3 clean-surface results:
  AllylCpNi  on Si100          : -2.359 eV  (chemi, C-Si)
  AllylCpNi  on SiO2_Si_term   : -0.957 eV  (physi only)
  Ni(PF3)4   on Si100          : -0.038 eV  (physi only)
  Ni(PF3)4   on SiO2_Si_term   : -1.540 eV  (physi F-O)

If E_ads_prec is significantly less negative than E_ads_clean, the inhibitor blocks
adsorption.  If it is comparably negative, the inhibitor does not block.

Substrates (inhibitor-covered)
-------------------------------
  Si100 [2x sc] + inhibitor physi rank01          -> 245-atom reference
  SiO2_Si_term [2x sc] + inhibitor physi rank01   -> 237-atom reference

SiO2_O_term is excluded: both inhibitor and precursors show O-H type bonding that
makes the raw E_ads unphysical (slab surface is over-reactive with the ML potential).

Precursors
----------
  AllylCpNi  : 19 atoms, Ni center idx=18, placement HEIGHT=3.5 A
  Ni(PF3)4   : 17 atoms, Ni center idx=16, placement HEIGHT=4.5 A

Grid sampling
-------------
  6x6 fractional-coordinate grid (f1,f2 in linspace(0.1,0.9,6))
  = 36 xy positions, each with N_SPIN=4 spin orientations
  = 144 candidates per (substrate, precursor)
  Pre-screen -> top PRESELECT=10 -> relax -> ranked output

Output
------
  phase4/results/{substrate}/{precursor}/   VASP + extxyz + CSV
  phase4/results/competitive_summary.txt
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / "phase3"))

import numpy as np
from ase.io import read

from autoflow_srxn.simulation.potentials import SimulationEngine

# Reuse the helpers from phase3/utils.py
from utils import (
    center_molecule, spin_z,
    place_center_on_site,
    relax_and_score, write_results,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SPIN     = 4
N_GRID     = 6          # 6x6 fractional grid
PRESELECT  = 10

FROZEN_Z    = 5.5       # A  -- same as phase3
FMAX        = 0.05      # eV/A
RELAX_STEPS = 250

# Ni center heights above topmost atom of the slab+inh reference structure
HEIGHT_ALLYL  = 3.5    # A  (AllylCpNi; physi-like starting height)
HEIGHT_NIPF3  = 4.5    # A  (Ni(PF3)4;  larger molecule, needs more clearance)

ENGINE_CONFIG = {
    "engine": {"potential": {"backend": "sevennet", "model": "7net-0",
                             "device": "cpu", "dtype": "float32"}},
    "relaxation": {"fmax": FMAX, "steps": RELAX_STEPS,
                   "optimizer": "FIRE", "frozen_z_ang": FROZEN_Z},
}

# Inhibitor-covered slab references (rank01 physi from phase3)
INH_SI100    = ROOT / "phase3/results/inhibitor_supercell/Si100/physi/inhibitor_Si100_sc_physi_rank01.vasp"
INH_SIO2SI   = ROOT / "phase3/results/inhibitor_supercell/SiO2_Si_term/physi/inhibitor_SiO2_Si_term_sc_physi_rank01.vasp"

# Precursor gas-phase structures
ALLYLCPNI_PATH = ROOT / "structures/AllylCpNi_relaxed.vasp"
NIPF3_PATH     = ROOT / "structures/NiPF3_4_relaxed.vasp"

OUT_DIR = ROOT / "phase4/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_inhibitor_centroid_xy(atoms):
    """Return xy centroid of inhibitor atoms.

    Inhibitor = C5H13NO2.  In VASP-sorted output, all C atoms and the N atom
    are unambiguously from the inhibitor (Si100 has no C/N; SiO2 has no C/N).
    """
    syms = atoms.get_chemical_symbols()
    inh_idx = [i for i, s in enumerate(syms) if s in ("C", "N")]
    if not inh_idx:
        raise ValueError("Cannot identify inhibitor atoms (no C or N found)")
    pos = atoms.positions[inh_idx]
    return pos[:, :2].mean(axis=0)


def build_grid_sites(atoms, n_grid=N_GRID):
    """Return list of dicts {x_A, y_A, z_A} spanning the 2D supercell.

    Fractional coords f1, f2 in linspace(0.1, 0.9, n_grid); the z coordinate
    is set to the topmost atom z of the combined slab+inh reference.
    """
    cell  = atoms.cell
    z_top = atoms.positions[:, 2].max()
    f_vals = np.linspace(0.1, 0.9, n_grid)
    sites = []
    for f1 in f_vals:
        for f2 in f_vals:
            cart = f1 * cell[0] + f2 * cell[1]
            sites.append({"x_A": float(cart[0]),
                          "y_A": float(cart[1]),
                          "z_A": float(z_top)})
    return sites


def run_precursor(substrate_name, inh_slab, prec_mol, prec_name,
                  center_idx, height, calc, engine,
                  e_inh_slab, e_prec_gas, summary_lines):
    """Run competitive adsorption search for one (substrate, precursor) pair."""

    sub_dir = OUT_DIR / substrate_name / prec_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  Competitive adsorption: {prec_name} on {substrate_name}")
    print(f"  Ni/center idx={center_idx}, HEIGHT={height} A, {N_GRID}x{N_GRID} grid, {N_SPIN} spins")
    print(f"{'='*68}")

    inh_center_xy = get_inhibitor_centroid_xy(inh_slab)
    print(f"  Inhibitor centroid xy = ({inh_center_xy[0]:.2f}, {inh_center_xy[1]:.2f}) A")

    sites = build_grid_sites(inh_slab, N_GRID)
    z_top = inh_slab.positions[:, 2].max()
    print(f"  z_top of slab+inh = {z_top:.3f} A -> precursor center z = {z_top+height:.3f} A")
    print(f"  Grid: {len(sites)} sites x {N_SPIN} spins = {len(sites)*N_SPIN} candidates")

    mol_c = center_molecule(prec_mol)
    spins = range(0, 360, 360 // N_SPIN)

    candidates = []
    for s in sites:
        site_xyz = np.array([s["x_A"], s["y_A"], s["z_A"]])
        # xy distance from inhibitor center (for labeling only, not filtering)
        inh_dist = float(np.linalg.norm(site_xyz[:2] - inh_center_xy))
        for deg in spins:
            mol_s    = spin_z(mol_c, deg)
            combined = place_center_on_site(
                inh_slab, mol_s, site_xyz, center_idx, height)
            candidates.append({
                "site_xy": (round(s["x_A"], 2), round(s["y_A"], 2)),
                "inh_dist_A": round(inh_dist, 2),
                "spin_deg": int(deg),
                "atoms": combined,
            })

    # --- Pre-screen ---
    for c in candidates:
        c["atoms"].calc = calc
        try:
            e0 = float(c["atoms"].get_potential_energy())
            c["e_ads_init"] = e0 - e_inh_slab - e_prec_gas
        except Exception:
            c["e_ads_init"] = 1e6   # penalise failures

    candidates.sort(key=lambda c: c["e_ads_init"])
    print("  Pre-screen top 5:")
    for c in candidates[:5]:
        print(f"    xy={c['site_xy']}  dist_inh={c['inh_dist_A']:.1f} A "
              f"spin={c['spin_deg']:3d}  E_init={c['e_ads_init']:+.3f} eV")

    selected = candidates[:min(PRESELECT, len(candidates))]
    relaxed  = []
    for i, c in enumerate(selected):
        print(f"\n  Relax {i+1}/{len(selected)}: xy={c['site_xy']} "
              f"dist_inh={c['inh_dist_A']:.1f} A  spin={c['spin_deg']}  "
              f"E_init={c['e_ads_init']:+.3f} eV")
        ar, e_ads, mind, mpair, nb = relax_and_score(
            c["atoms"], engine, e_inh_slab, e_prec_gas, FROZEN_Z, RELAX_STEPS, FMAX)
        relaxed.append((e_ads, ar, mind, mpair, nb, {
            "site_xy": str(c["site_xy"]),
            "inh_dist_A": str(c["inh_dist_A"]),
            "spin_deg": c["spin_deg"],
            "mode": "competitive",
        }))
        print(f"    -> E_ads={e_ads:+.4f} eV  dist={mind:.3f} A ({mpair})  cov={nb}")

    if not relaxed:
        summary_lines.append(f"  {substrate_name} / {prec_name}: no candidates")
        return

    rows = write_results(sub_dir, f"{prec_name}_{substrate_name}_comp", relaxed)
    top  = rows[:3]
    summary_lines.append(f"  {substrate_name} / {prec_name}: relaxed={len(relaxed)}")
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

    # --- Load precursor molecules ---
    allylcpni = read(str(ALLYLCPNI_PATH))
    nipf3     = read(str(NIPF3_PATH))

    syms_a = allylcpni.get_chemical_symbols()
    syms_n = nipf3.get_chemical_symbols()
    NI_IDX_ALLYL = 18
    NI_IDX_NIPF3 = 16
    print(f"AllylCpNi: {allylcpni.get_chemical_formula()}  ({len(allylcpni)} atoms)  "
          f"Ni={syms_a[NI_IDX_ALLYL]} @ idx={NI_IDX_ALLYL}")
    print(f"Ni(PF3)4 : {nipf3.get_chemical_formula()}  ({len(nipf3)} atoms)  "
          f"Ni={syms_n[NI_IDX_NIPF3]} @ idx={NI_IDX_NIPF3}")

    # --- Gas-phase energies ---
    def gas_energy(mol):
        g = mol.copy(); g.center(vacuum=10.0); g.calc = calc
        return float(g.get_potential_energy())

    e_allyl = gas_energy(allylcpni)
    e_nipf3 = gas_energy(nipf3)
    print(f"\nGas energies:")
    print(f"  AllylCpNi : {e_allyl:.6f} eV")
    print(f"  Ni(PF3)4  : {e_nipf3:.6f} eV")

    # --- Load inhibitor-covered slab references ---
    inh_si100  = read(str(INH_SI100))
    inh_sio2si = read(str(INH_SIO2SI))

    inh_si100.calc  = calc
    inh_sio2si.calc = calc
    e_inh_si100  = float(inh_si100.get_potential_energy())
    e_inh_sio2si = float(inh_sio2si.get_potential_energy())
    print(f"\nInhibitor-covered slab energies:")
    print(f"  Si100+inh     : {e_inh_si100:.4f} eV  ({len(inh_si100)} atoms)")
    print(f"  SiO2_Si+inh   : {e_inh_sio2si:.4f} eV  ({len(inh_sio2si)} atoms)")

    summary_lines = [
        "Phase 4: Competitive Adsorption (precursor on inhibitor-covered surfaces)",
        "=" * 68,
        "E_ads_prec = E(slab+inh+prec) - E(slab+inh) - E(prec_gas)",
        "",
        "Phase 3 clean-surface reference (best E_ads):",
        "  AllylCpNi  / Si100        : -2.359 eV  (chemi C-Si, cov=2)",
        "  AllylCpNi  / SiO2_Si_term : -0.957 eV  (physi only, cov=0)",
        "  Ni(PF3)4   / Si100        : -0.038 eV  (physi only, cov=0)",
        "  Ni(PF3)4   / SiO2_Si_term : -1.540 eV  (physi F-O, cov=0)",
        "",
        f"AllylCpNi  gas energy : {e_allyl:.6f} eV",
        f"Ni(PF3)4   gas energy : {e_nipf3:.6f} eV",
        f"Si100+inh  slab energy: {e_inh_si100:.4f} eV  ({len(inh_si100)} atoms)",
        f"SiO2_Si+inh slab energy: {e_inh_sio2si:.4f} eV  ({len(inh_sio2si)} atoms)",
        f"Grid: {N_GRID}x{N_GRID} fractional x {N_SPIN} spins -> top-{PRESELECT} relaxed",
        f"FROZEN_Z={FROZEN_Z} A, FMAX={FMAX} eV/A",
        "",
        "--- Results ---",
    ]

    # -----------------------------------------------------------------------
    # Si100 + inhibitor
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  Si100 + inhibitor substrate")
    print("=" * 68)

    run_precursor("Si100", inh_si100, allylcpni, "AllylCpNi",
                  NI_IDX_ALLYL, HEIGHT_ALLYL, calc, engine,
                  e_inh_si100, e_allyl, summary_lines)

    run_precursor("Si100", inh_si100, nipf3, "NiPF3_4",
                  NI_IDX_NIPF3, HEIGHT_NIPF3, calc, engine,
                  e_inh_si100, e_nipf3, summary_lines)

    # -----------------------------------------------------------------------
    # SiO2_Si_term + inhibitor
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  SiO2_Si_term + inhibitor substrate")
    print("=" * 68)

    run_precursor("SiO2_Si_term", inh_sio2si, allylcpni, "AllylCpNi",
                  NI_IDX_ALLYL, HEIGHT_ALLYL, calc, engine,
                  e_inh_sio2si, e_allyl, summary_lines)

    run_precursor("SiO2_Si_term", inh_sio2si, nipf3, "NiPF3_4",
                  NI_IDX_NIPF3, HEIGHT_NIPF3, calc, engine,
                  e_inh_sio2si, e_nipf3, summary_lines)

    # -----------------------------------------------------------------------
    # Write summary
    # -----------------------------------------------------------------------
    summary_path = OUT_DIR / "competitive_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "=" * 68)
    print("\n".join(summary_lines))
    print(f"\nSummary -> {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
