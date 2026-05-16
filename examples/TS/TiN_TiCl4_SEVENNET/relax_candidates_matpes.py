"""
Relax a minimal representative set of TiN(111)+TiCl4 candidates with
matpes_pbe and compare adsorption energies against 7net-0.

Strategy:
  - Load pre-relaxed matpes_pbe slab  (already done, saves ~4 min)
  - Load pre-relaxed gas TiCl4 energy (already done)
  - Generate physisorption + chemisorption candidates on the relaxed slab
  - Pick 1 physisorption + all chemisorption candidates (usually 2)
  - Relax each with matpes_pbe, compute E_ads
  - Print comparison table vs 7net-0 reference

Reference (7net-0, from results/):
  E_slab  = -1523.6171 eV
  E_TiCl4 = -23.1930  eV
  Best physisorption E_ads  = -0.0698 eV
  Only chemisorption E_ads  = -0.0299 eV
"""

import os
import sys
import time

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, ROOT)

SLAB_FILE     = os.path.join(SCRIPT_DIR, "slab_matpes_pbe_relaxed.vasp")
PRECURSOR_FILE = os.path.join(SCRIPT_DIR, "..", "..", "..", "structures", "TiCl4.vasp")
CONFIG_FILE   = os.path.join(SCRIPT_DIR, "config_matpes_pbe.yaml")
RESULTS_DIR   = os.path.join(SCRIPT_DIR, "results_matpes_pbe")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Known reference energies (already computed)
E_SLAB_MATPES   = -1512.6611   # eV
E_TiCl4_MATPES  = -22.9819    # eV

# 7net-0 reference
REF_7NET0 = {
    "E_slab":  -1523.6171,
    "E_TiCl4": -23.1930,
    "physi_best_eads": -0.0698,
    "chemi_eads":      -0.0299,
}

# ---------------------------------------------------------------------------
# Config + Engine
# ---------------------------------------------------------------------------
import yaml
from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.surface.ads_workflow_mgr import AdsorptionWorkflowManager
from autoflow_srxn.surface.chemisorption_builder import build_chemisorption_structures

with open(CONFIG_FILE, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Resolve paths relative to config file
config_dir = os.path.dirname(os.path.abspath(CONFIG_FILE))
for key in ["precursor", "substrate_bulk"]:
    val = config.get("paths", {}).get(key)
    if val and not os.path.isabs(val):
        abs_val = os.path.join(config_dir, val)
        if os.path.exists(abs_val):
            config["paths"][key] = abs_val

engine = SimulationEngine(config)


def get_calc():
    return engine.get_calculator()


def relax(atoms, fmax=0.05, steps=150, frozen_z_ang=6.0, label=""):
    atoms.calc = get_calc()
    t0 = time.time()
    e0 = atoms.get_potential_energy()

    # Freeze bottom half
    if frozen_z_ang is not None:
        from ase.constraints import FixAtoms
        z_max = atoms.positions[:, 2].max()
        z_thresh = z_max - frozen_z_ang
        fixed = [i for i, pos in enumerate(atoms.positions) if pos[2] < z_thresh]
        atoms.set_constraint(FixAtoms(indices=fixed))

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    e1 = atoms.get_potential_energy()
    dt = time.time() - t0
    print(f"  {label}: {e0:.4f} → {e1:.4f} eV  (Δ={e1-e0:+.4f} eV)  [{dt:.0f}s]")
    return e1


# ---------------------------------------------------------------------------
# Load relaxed slab
# ---------------------------------------------------------------------------
print("=" * 70)
print("matpes_pbe candidate relaxation — TiN(111) + TiCl4")
print("=" * 70)
print(f"\nLoading relaxed slab: {SLAB_FILE}")
slab = read(SLAB_FILE)
print(f"  {len(slab)} atoms, z=[{slab.positions[:,2].min():.3f}, {slab.positions[:,2].max():.3f}]")

mol = read(config["paths"]["precursor"])
print(f"  Precursor: {mol.get_chemical_formula()}")

# ---------------------------------------------------------------------------
# Generate candidates
# ---------------------------------------------------------------------------
print("\n[1] Generating physisorption candidates...")
physi_cfg = config["reaction_search"]["mechanisms"]["precursor"]["physisorption"]
symprec = config["reaction_search"].get("symprec", 1.0)

mgr = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=False)
phy_cands = mgr.generate_physisorption_candidates(
    mol,
    height=physi_cfg.get("placement_height", 3.5),
    tag=3,
    n_rot=physi_cfg.get("n_rot", 4),
    rot_center="Ti",
    height_mode="clearance",
    gravity_pull=physi_cfg.get("gravity_pull", {"enabled": False}),
)
for c in phy_cands:
    c.info.setdefault("reaction_type", "physisorption")
    c.info.setdefault("mechanism", "Physisorption")
print(f"  {len(phy_cands)} physisorption candidates generated")

print("\n[2] Generating chemisorption candidates...")
chem_cands = build_chemisorption_structures(
    molecule=mol,
    center_target="Ti",
    surface=slab,
    rot_steps=4,
    config=config,
    tag=3,
    results_dir=RESULTS_DIR,
    verbose=True,
)
for c in chem_cands:
    if "reaction_type" not in c.info:
        c.info["reaction_type"] = "chemisorption"
    if "mechanism" not in c.info:
        c.info["mechanism"] = "Chemisorption"
print(f"  {len(chem_cands)} chemisorption candidates generated")

# Save all candidates for reference
all_cands = phy_cands + chem_cands
write(os.path.join(RESULTS_DIR, "stage2_precursor_candidates.extxyz"), all_cands)
print(f"\nTotal candidates: {len(all_cands)} (saved to results_matpes_pbe/)")

# ---------------------------------------------------------------------------
# Select representative candidates
# ---------------------------------------------------------------------------
# Pick 1 physisorption (first / lowest clearance) + all chemisorption
if not phy_cands:
    print("\n[WARN] No physisorption candidates generated!")
    selected = []
else:
    selected_physi = [phy_cands[0]]  # just the first one for now
    selected = selected_physi + chem_cands

print(f"\nSelected for relaxation: {len(selected)} candidates")
for i, c in enumerate(selected):
    rt = c.info.get("reaction_type", "?")
    mech = c.info.get("mechanism", "?")
    print(f"  [{i}] {rt:25s}  {len(c)} atoms  {mech}")

# ---------------------------------------------------------------------------
# Relax selected candidates
# ---------------------------------------------------------------------------
print("\n[3] Relaxing selected candidates with matpes_pbe...")
rp = config.get("relaxation", {})
fmax  = rp.get("fmax", 0.05)
steps = rp.get("steps", 150)
frozen_z = rp.get("frozen_z_ang", 6.0)

results = []
t_total = time.time()
for i, atoms in enumerate(selected):
    rt   = atoms.info.get("reaction_type", "unknown")
    mech = atoms.info.get("mechanism", "?")
    print(f"\n  Candidate {i}: [{rt}] {mech}  ({len(atoms)} atoms)")
    try:
        atoms_w = atoms.copy()
        e_final = relax(atoms_w, fmax=fmax, steps=steps, frozen_z_ang=frozen_z,
                        label=f"  BFGS")
        e_ads = e_final - E_SLAB_MATPES - E_TiCl4_MATPES
        atoms_w.info.update({
            "e_final": e_final,
            "e_ads": e_ads,
            "reaction_type": rt,
            "mechanism": mech,
        })
        # Clear constraints before saving: FixAtoms stores a per-atom mask;
        # writing a mixed-size list to extxyz fails if masks have different lengths.
        atoms_w.set_constraint()
        results.append(atoms_w)
        print(f"    E_ads = {e_ads:+.4f} eV")
    except Exception as exc:
        print(f"    FAILED: {exc}")

print(f"\nRelaxation complete ({time.time()-t_total:.0f}s total)")

# ---------------------------------------------------------------------------
# Save and report
# ---------------------------------------------------------------------------
if results:
    write(os.path.join(RESULTS_DIR, "stage2_precursor_relaxed.extxyz"), results)
    print(f"Saved {len(results)} relaxed structures")

print("\n" + "=" * 70)
print("ADSORPTION ENERGY COMPARISON: matpes_pbe vs 7net-0")
print("=" * 70)
print(f"{'Type':<30} {'matpes_pbe':>12} {'7net-0':>12}  {'Diff':>10}")
print("-" * 70)

ref_physi = REF_7NET0["physi_best_eads"]
ref_chemi = REF_7NET0["chemi_eads"]

for r in results:
    rt   = r.info.get("reaction_type", "?")
    mech = r.info.get("mechanism", "?")
    e    = r.info.get("e_ads", float("nan"))
    if rt == "physisorption":
        ref = ref_physi
    elif "chemi" in rt.lower():
        ref = ref_chemi
    else:
        ref = float("nan")
    diff = e - ref if not (np.isnan(e) or np.isnan(ref)) else float("nan")
    print(f"{mech:<30} {e:>+12.4f} {ref:>+12.4f}  {diff:>+10.4f} eV")

print("-" * 70)
print(f"\nReference energies:")
print(f"  matpes_pbe : E_slab={E_SLAB_MATPES:.4f} eV, E_TiCl4={E_TiCl4_MATPES:.4f} eV")
print(f"  7net-0     : E_slab={REF_7NET0['E_slab']:.4f} eV, E_TiCl4={REF_7NET0['E_TiCl4']:.4f} eV")
print("=" * 70)
