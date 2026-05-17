"""Phase 0 — AllylCpNi haptic code validation.

Tests:
  1. discover_ligands correctly identifies eta3-allyl (hapticity=3) and
     eta5-Cp (hapticity=5) from AllylCpNi with center_target="Ni".

  2. _place_at_dangling_bond routes to the multi-atom haptic path for the
     departing haptic fragment (Route 2/dissociation).

  3. _form_byproduct does NOT crash when called with binding_idx_b[0] from
     a haptic ligand (it uses only the first C; this is acceptable for the
     informational byproduct label).

  4. Single-site chemisorption pipeline runs on a minimal mock surface and
     produces at least one candidate with AllylCpNi as precursor.

Prints PASS / FAIL for each check.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.io import read
from ase.build import bulk, surface
from ase import Atoms

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" | {detail}" if detail else ""))
    results.append((label, condition))


# ---------------------------------------------------------------------------
# Load molecule
# ---------------------------------------------------------------------------
mol_path = ROOT / "structures" / "AllylCpNi.vasp"
mol = read(str(mol_path))
print(f"\nLoaded AllylCpNi: {len(mol)} atoms  {mol.get_chemical_formula()}")

# ---------------------------------------------------------------------------
# Test 1: discover_ligands
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("TEST 1: discover_ligands(AllylCpNi, center_target='Ni')")
print("-" * 60)

from autoflow_srxn.surface.ads_workflow_mgr import AdsorptionWorkflowManager

# Build a tiny dummy slab for the manager (it won't be used here)
dummy_slab = Atoms("Si4", positions=[[0,0,0],[2,0,0],[0,2,0],[2,2,0]],
                   cell=[4,4,10], pbc=True)

mgr = AdsorptionWorkflowManager(dummy_slab, config={}, verbose=True)
c_idx, ligands = mgr.discover_ligands(mol, center_target="Ni")

check("Ni center found", c_idx is not None, f"c_idx={c_idx}")
check("Exactly 2 ligands found", len(ligands) == 2,
      f"found {len(ligands)}: {[l['formula'] for l in ligands]}")

hapticities = sorted([l["hapticity"] for l in ligands])
check("Hapticity values = [3, 5]", hapticities == [3, 5],
      f"got {hapticities}")

for l in ligands:
    n = l["hapticity"]
    check(
        f"Ligand formula (hapticity={n}) matches expected",
        (n == 3 and "C3" in l["formula"]) or (n == 5 and "C5" in l["formula"]),
        f"formula={l['formula']}, binding_atoms={l['binding_atoms']}"
    )
    check(
        f"Normal vector magnitude ~1 (hapticity={n})",
        abs(np.linalg.norm(l["normal_vector"]) - 1.0) < 1e-5,
        f"|n|={np.linalg.norm(l['normal_vector']):.6f}"
    )
    check(
        f"VBS position defined (hapticity={n})",
        l["vbs_pos"] is not None and len(l["vbs_pos"]) == 3,
        f"vbs={l['vbs_pos']}"
    )

# ---------------------------------------------------------------------------
# Test 2: _place_at_dangling_bond — haptic (multi-atom) path
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("TEST 2: _place_at_dangling_bond with haptic binding_idx list")
print("-" * 60)

haptic5_ligand = next(l for l in ligands if l["hapticity"] == 5)
indices_b = haptic5_ligand["indices"]
frag_b = mol[indices_b]
binding_idx_b = [indices_b.index(idx) for idx in haptic5_ligand["binding_atoms"]]

target_pos = np.array([2.0, 2.0, 5.0])
db_vector  = np.array([0.0, 0.0, 1.0])
bond_len   = 2.0
try:
    placed = mgr._place_at_dangling_bond(
        frag_b,
        binding_idx_b,          # list of 5 C indices — haptic path
        -haptic5_ligand["bond_vec"],
        target_pos,
        db_vector,
        bond_len,
        rot_angle=0,
        haptic_normal=haptic5_ligand["normal_vector"],
    )
    centroid_dist = np.linalg.norm(
        np.mean(placed.positions[binding_idx_b], axis=0) - (target_pos + bond_len * db_vector)
    )
    check("_place_at_dangling_bond (haptic) ran without error", True)
    check("Centroid of placed haptic fragment near target+bond_len",
          centroid_dist < 0.3, f"dist={centroid_dist:.4f} A")
except Exception as e:
    check("_place_at_dangling_bond (haptic) ran without error", False, str(e))
    check("Centroid of placed haptic fragment near target+bond_len", False, "skipped")

# ---------------------------------------------------------------------------
# Test 3: _form_byproduct with haptic ligand's first binding atom
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("TEST 3: _form_byproduct with binding_idx_b[0] (haptic first atom)")
print("-" * 60)

haptic3_ligand = next(l for l in ligands if l["hapticity"] == 3)
indices_allyl = haptic3_ligand["indices"]
frag_allyl = mol[indices_allyl]
binding_idx_allyl = [indices_allyl.index(idx) for idx in haptic3_ligand["binding_atoms"]]

try:
    byproduct = mgr._form_byproduct(frag_allyl, binding_idx_allyl[0], -haptic3_ligand["bond_vec"])
    check("_form_byproduct (haptic, first atom) ran without error", True)
    check("Byproduct has one extra H vs fragment",
          len(byproduct) == len(frag_allyl) + 1,
          f"frag={len(frag_allyl)}, byproduct={len(byproduct)}")
    added_sym = byproduct.symbols[-1]
    check("Added atom is H", added_sym == "H", f"got '{added_sym}'")
except Exception as e:
    check("_form_byproduct (haptic, first atom) ran without error", False, str(e))

# ---------------------------------------------------------------------------
# Test 4: Full single-site chemisorption pipeline on a minimal Si(100)-like slab
# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
print("TEST 4: build_chemisorption_structures on minimal Si slab")
print("-" * 60)

from autoflow_srxn.surface.chemisorption_builder import build_chemisorption_structures
from autoflow_srxn.surface.surface_utils import create_slab_from_bulk, passivate_surface_coverage_general

si_bulk = read(str(ROOT / "structures" / "Si_mp149.vasp"))
si_slab = create_slab_from_bulk(
    si_bulk, miller_indices=(1, 0, 0), thickness=8.0, vacuum=12.0, target_area=50.0
)
# Passivate bottom only
si_slab = passivate_surface_coverage_general(
    si_slab,
    coverage=1.0,
    valence_map={"Si": 4, "H": 1},
    element="H",
    side="bottom",
)
print(f"  Slab: {len(si_slab)} atoms, {si_slab.get_chemical_formula()}")

config_test = {
    "reaction_search": {
        "symprec": 0.2,
        "mechanisms": {
            "precursor": {
                "chemisorption": {
                    "coordination_analysis": {
                        "expected_coord": {"Si": 4, "H": 1},
                        "bond_slack": 0.2,
                        "max_neighbor_dist": 4.0,
                    }
                }
            }
        },
    }
}

try:
    candidates = build_chemisorption_structures(
        mol,
        center_target="Ni",
        surface=si_slab,
        rot_steps=4,   # fast test
        config=config_test,
        verbose=True,
    )
    check("build_chemisorption_structures ran without error", True)
    check("At least one candidate generated", len(candidates) > 0,
          f"got {len(candidates)} candidates")
    if candidates:
        mechs = set(c.info.get("reaction_type","?") for c in candidates)
        check("Candidates have reaction_type info", len(mechs) > 0, f"{mechs}")
except Exception as e:
    import traceback
    traceback.print_exc()
    check("build_chemisorption_structures ran without error", False, str(e))
    check("At least one candidate generated", False, "skipped")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
total = len(results)
passed = sum(1 for _, ok in results if ok)
for label, ok in results:
    sym = "OK" if ok else "NG"
    print(f"  {sym} {label}")
print(f"\n  {passed}/{total} checks passed.")
if passed < total:
    sys.exit(1)
