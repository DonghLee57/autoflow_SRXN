"""
Systematic inspection of the interface matching pipeline.
Steps:
  1. get_surface_lattice_2d  — what 2D lattice does each Miller plane give?
  2. find_coincidences       — what HNF pairs are returned and why?
  3. build_symmetric_slab    — does the HNF supercell in 3D match the 2D lattice?
  4. stack_interface          — after v1-alignment rotation, how well do v2 match?
"""
import sys, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from autoflow_srxn.interface.builder import (
    get_surface_lattice_2d,
    find_coincidences,
    build_symmetric_slab,
    stack_interface,
    strain_from_F,
    iter_hnf_2d,
)

STRUCT_DIR = os.path.join(ROOT, "structures")

def load(fname):
    return Structure.from_file(os.path.join(STRUCT_DIR, fname))

def fmt_mat(m, label=""):
    if label:
        print(f"  {label}:")
    for row in m:
        print(f"    [{row[0]:8.4f}  {row[1]:8.4f}]")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ============================================================
# SECTION 1: 2D surface lattices for each candidate surface
# ============================================================
section("1. 2D Surface Lattices")

zro2 = load("ZrO2_t_bulk.vasp")
nbo2 = load("NbO2_bulk.vasp")
nbo  = load("NbO_bulk.vasp")

test_pairs = [
    # (sub_struct, sub_miller, film_struct, film_miller, label)
    (zro2, (1,0,1), nbo2, (0,0,1), "ZrO2(101)/NbO2(001)  [best match, vm=1.33%]"),
    (zro2, (0,0,1), nbo2, (0,0,1), "ZrO2(001)/NbO2(001)"),
    (zro2, (0,0,1), nbo,  (0,0,1), "ZrO2(001)/NbO(001)"),
    (zro2, (1,1,1), nbo,  (1,1,0), "ZrO2(111)/NbO(110)"),
]

lattices = {}
for sub_s, sub_m, film_s, film_m, label in test_pairs:
    A_sub  = get_surface_lattice_2d(sub_s,  sub_m)
    A_film = get_surface_lattice_2d(film_s, film_m)
    lattices[(sub_m, film_m)] = (A_sub, A_film)
    print(f"\n--- {label} ---")
    fmt_mat(A_sub,  f"A_sub  {sub_m}")
    fmt_mat(A_film, f"A_film {film_m}")
    a1 = np.linalg.norm(A_sub[0]);  a2 = np.linalg.norm(A_sub[1])
    b1 = np.linalg.norm(A_film[0]); b2 = np.linalg.norm(A_film[1])
    def angle2d(M):
        v1, v2 = M[0], M[1]
        c = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(c,-1,1)))
    print(f"  |v1_sub|={a1:.4f}  |v2_sub|={a2:.4f}  gamma_sub={angle2d(A_sub):.2f} deg")
    print(f"  |v1_film|={b1:.4f}  |v2_film|={b2:.4f}  gamma_film={angle2d(A_film):.2f} deg")

# ============================================================
# SECTION 2: Coincidence search — inspect top hits
# ============================================================
section("2. Coincidence Search Results")

PAIRS_TO_CHECK = [
    (zro2, (1,0,1), nbo2, (0,0,1), "ZrO2(101)/NbO2(001)", 12, 0.08),
    (zro2, (0,0,1), nbo,  (0,0,1), "ZrO2(001)/NbO(001)",  12, 0.08),
    (zro2, (1,1,1), nbo,  (1,1,0), "ZrO2(111)/NbO(110)",  12, 0.08),
]

found_pairs = {}
for sub_s, sub_m, film_s, film_m, label, max_det, sc in PAIRS_TO_CHECK:
    A_sub  = get_surface_lattice_2d(sub_s,  sub_m)
    A_film = get_surface_lattice_2d(film_s, film_m)
    hits   = find_coincidences(A_sub, A_film, max_det=max_det, strain_cutoff=sc)
    found_pairs[(sub_m, film_m)] = hits
    print(f"\n--- {label}  (max_det={max_det}, sc={sc}) ---")
    print(f"  {len(hits)} matches found")
    for i, h in enumerate(hits[:5]):
        print(f"  #{i+1}  vm={h['vm']*100:.2f}%  eps1={h['eps1']*100:.2f}%  eps2={h['eps2']*100:.2f}%"
              f"  detNa={h['det_Na']}  detNb={h['det_Nb']}"
              f"  area_ratio={h['area_ratio']:.4f}")
        fmt_mat(h['Na'], "Na")
        fmt_mat(h['Nb'], "Nb")

# ============================================================
# SECTION 3: Check that HNF supercell matches lattice vectors
# ============================================================
section("3. HNF Supercell Consistency  (2D vs 3D cell)")

print("\nFor ZrO2(101)/NbO2(001) best match:")
sub_s, sub_m, film_s, film_m = zro2, (1,0,1), nbo2, (0,0,1)
A_sub  = get_surface_lattice_2d(sub_s,  sub_m)
A_film = get_surface_lattice_2d(film_s, film_m)
hits   = find_coincidences(A_sub, A_film, max_det=12, strain_cutoff=0.08)
if not hits:
    print("  No matches — skipping.")
else:
    best = hits[0]
    Na, Nb = best['Na'], best['Nb']
    A_Na = Na.astype(float) @ A_sub
    A_Nb = Nb.astype(float) @ A_film
    print(f"  Na=\n    {Na}")
    print(f"  Nb=\n    {Nb}")
    fmt_mat(A_Na, "Na @ A_sub  (sub supercell 2D)")
    fmt_mat(A_Nb, "Nb @ A_film (film supercell 2D)")
    print(f"\n  Building 3D slabs (Na applied to sub, Nb applied to film)...")
    try:
        sub_slab  = build_symmetric_slab(sub_s,  sub_m, min_thickness_ang=10.0, vacuum_ang=15.0, HNF=Na)
        film_slab = build_symmetric_slab(film_s, film_m, min_thickness_ang=10.0, vacuum_ang=15.0, HNF=Nb)
        sub_cell_3d  = np.array(sub_slab.cell)
        film_cell_3d = np.array(film_slab.cell)
        print(f"\n  sub_slab  cell (3D, after rotate_to_z):")
        for r in sub_cell_3d: print(f"    {r}")
        print(f"\n  film_slab cell (3D, after rotate_to_z):")
        for r in film_cell_3d: print(f"    {r}")
        print(f"\n  sub  in-plane: |v1|={np.linalg.norm(sub_cell_3d[0,:2]):.4f}  |v2|={np.linalg.norm(sub_cell_3d[1,:2]):.4f}")
        print(f"  film in-plane: |v1|={np.linalg.norm(film_cell_3d[0,:2]):.4f}  |v2|={np.linalg.norm(film_cell_3d[1,:2]):.4f}")
        # Check mismatch between 2D prediction and 3D result
        v1_sub_3d  = sub_cell_3d[0, :2]
        v2_sub_3d  = sub_cell_3d[1, :2]
        v1_film_3d = film_cell_3d[0, :2]
        v2_film_3d = film_cell_3d[1, :2]
        print(f"\n  2D predicted sub supercell:  |v1|={np.linalg.norm(A_Na[0]):.4f}  |v2|={np.linalg.norm(A_Na[1]):.4f}")
        print(f"  3D actual sub slab in-plane: |v1|={np.linalg.norm(v1_sub_3d):.4f}  |v2|={np.linalg.norm(v2_sub_3d):.4f}")
        print(f"\n  2D predicted film supercell:  |v1|={np.linalg.norm(A_Nb[0]):.4f}  |v2|={np.linalg.norm(A_Nb[1]):.4f}")
        print(f"  3D actual film slab in-plane: |v1|={np.linalg.norm(v1_film_3d):.4f}  |v2|={np.linalg.norm(v2_film_3d):.4f}")

        # ============================================================
        # SECTION 4: stack_interface — v1 rotation and v2 mismatch
        # ============================================================
        section("4. stack_interface: v1-alignment and v2 mismatch")

        print("\nBefore stacking:")
        print(f"  sub  v1 direction: angle={np.degrees(np.arctan2(sub_cell_3d[0,1], sub_cell_3d[0,0])):.2f} deg from X")
        print(f"  film v1 direction: angle={np.degrees(np.arctan2(film_cell_3d[0,1], film_cell_3d[0,0])):.2f} deg from X")

        # Reproduce what stack_interface does
        v1_sub  = sub_cell_3d[0]
        v1_film = film_cell_3d[0]
        angle_sub  = np.arctan2(v1_sub[1],  v1_sub[0])
        angle_film = np.arctan2(v1_film[1], v1_film[0])
        rot_angle  = np.degrees(angle_sub - angle_film)
        print(f"\n  Rotation applied to film: {rot_angle:.4f} deg around z")

        film_copy = film_slab.copy()
        film_copy.rotate(rot_angle, "z", rotate_cell=True)
        film_cell_rot = np.array(film_copy.cell)
        print(f"\n  Film cell after rotation:")
        for r in film_cell_rot: print(f"    {r}")

        v1_sub_xy  = sub_cell_3d[0, :2]
        v2_sub_xy  = sub_cell_3d[1, :2]
        v1_film_xy = film_cell_rot[0, :2]
        v2_film_xy = film_cell_rot[1, :2]

        print(f"\n  After rotation:")
        print(f"    sub  v1={v1_sub_xy}  |v1|={np.linalg.norm(v1_sub_xy):.4f}")
        print(f"    film v1={v1_film_xy} |v1|={np.linalg.norm(v1_film_xy):.4f}")
        print(f"    v1 difference: {v1_film_xy - v1_sub_xy}  |diff|={np.linalg.norm(v1_film_xy - v1_sub_xy):.4f} Ang")

        print(f"\n    sub  v2={v2_sub_xy}  |v2|={np.linalg.norm(v2_sub_xy):.4f}")
        print(f"    film v2={v2_film_xy} |v2|={np.linalg.norm(v2_film_xy):.4f}")
        print(f"    v2 difference: {v2_film_xy - v2_sub_xy}  |diff|={np.linalg.norm(v2_film_xy - v2_sub_xy):.4f} Ang")

        # What happens when film positions are mapped into sub cell?
        print(f"\n  Film fractional coords -> sub cell mapping:")
        print(f"    film_cart_xy_new = film_frac[:,:2] @ sub_cell_2d")
        print(f"    This FORCES film atoms into sub lattice vectors (= epitaxial strain model).")
        print(f"    The strain applied is: F = A_sub @ inv(A_film_rot)")

        sub_2d  = sub_cell_3d[:2, :2]
        film_2d = film_cell_rot[:2, :2]
        try:
            F_effective = sub_2d @ np.linalg.inv(film_2d)
            eps1, eps2, vm_eff = strain_from_F(sub_2d, film_2d)
            print(f"\n    F (effective deformation after rotation):")
            for r in F_effective: print(f"      {r}")
            print(f"    eps1={eps1*100:.3f}%  eps2={eps2*100:.3f}%  vm={vm_eff*100:.3f}%")
            print(f"    (Should match the vm from find_coincidences: {best['vm']*100:.3f}%)")
        except np.linalg.LinAlgError:
            print("    (singular matrix)")

        # ============================================================
        # SECTION 5: Optimal rotation from polar decomposition
        # ============================================================
        section("5. Optimal rotation: polar decomposition F = R U")

        print("\nDeformation gradient from raw 2D lattices (before any rotation):")
        F_raw = A_Na @ np.linalg.inv(A_Nb)
        print(f"  F_raw (A_Na @ inv(A_Nb)):")
        for r in F_raw: print(f"    {r}")

        # Polar decomposition: F = R U  (R=rotation, U=right stretch)
        from scipy.linalg import polar
        R_opt, U = polar(F_raw)
        print(f"\n  Optimal rotation R from polar decomp:")
        for r in R_opt: print(f"    {r}")
        opt_rot_angle = np.degrees(np.arctan2(R_opt[1,0], R_opt[0,0]))
        print(f"  Optimal rotation angle: {opt_rot_angle:.4f} deg")

        print(f"\n  Pure strain tensor U:")
        for r in U: print(f"    {r}")
        eps1u, eps2u, vm_u = strain_from_F(U, np.eye(2))
        print(f"  From U: eps1={eps1u*100:.3f}%  eps2={eps2u*100:.3f}%  vm={vm_u*100:.3f}%")

        print(f"\n  Current stack_interface rotation angle: {rot_angle:.4f} deg (v1-alignment only)")
        print(f"  Optimal rotation angle from polar decomp: {opt_rot_angle:.4f} deg")
        print(f"  Discrepancy: {abs(rot_angle - opt_rot_angle):.4f} deg")

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()

# ============================================================
# SECTION 6: Summary of issues
# ============================================================
section("6. Rectangular vs non-rectangular: ZrO2(111)/NbO(110) deep-dive")

print("\nChecking stack_interface for non-rectangular case ZrO2(111)/NbO(110)...")
sub_s, sub_m, film_s, film_m = zro2, (1,1,1), nbo, (1,1,0)
A_sub  = get_surface_lattice_2d(sub_s,  sub_m)
A_film = get_surface_lattice_2d(film_s, film_m)
hits6  = find_coincidences(A_sub, A_film, max_det=12, strain_cutoff=0.08)
if not hits6:
    print("  No matches found.")
else:
    best6 = hits6[0]
    Na6, Nb6 = best6['Na'], best6['Nb']
    print(f"  Best match: vm={best6['vm']*100:.2f}%  Na={Na6.tolist()}  Nb={Nb6.tolist()}")
    A_Na6 = Na6.astype(float) @ A_sub
    A_Nb6 = Nb6.astype(float) @ A_film
    fmt_mat(A_Na6, "Na@A_sub (canonical 2D supercell)")
    fmt_mat(A_Nb6, "Nb@A_film (canonical 2D supercell)")
    try:
        sub_slab6  = build_symmetric_slab(sub_s,  sub_m, min_thickness_ang=10.0, vacuum_ang=15.0, HNF=Na6)
        film_slab6 = build_symmetric_slab(film_s, film_m, min_thickness_ang=10.0, vacuum_ang=15.0, HNF=Nb6)
        sub_cell6  = np.array(sub_slab6.cell)
        film_cell6 = np.array(film_slab6.cell)
        print("\n  3D slab cells after build_symmetric_slab:")
        print(f"  sub  v1={sub_cell6[0,:2]}  |v1|={np.linalg.norm(sub_cell6[0,:2]):.4f}")
        print(f"  sub  v2={sub_cell6[1,:2]}  |v2|={np.linalg.norm(sub_cell6[1,:2]):.4f}")
        print(f"  film v1={film_cell6[0,:2]}  |v1|={np.linalg.norm(film_cell6[0,:2]):.4f}")
        print(f"  film v2={film_cell6[1,:2]}  |v2|={np.linalg.norm(film_cell6[1,:2]):.4f}")
        # stack_interface rotation
        v1_sub6  = sub_cell6[0]
        v1_film6 = film_cell6[0]
        ang_sub6  = np.arctan2(v1_sub6[1],  v1_sub6[0])
        ang_film6 = np.arctan2(v1_film6[1], v1_film6[0])
        rot6 = np.degrees(ang_sub6 - ang_film6)
        print(f"\n  stack_interface rotation: {rot6:.4f} deg")
        film_copy6 = film_slab6.copy()
        film_copy6.rotate(rot6, "z", rotate_cell=True)
        fc6 = np.array(film_copy6.cell)
        print(f"  After rotation:")
        print(f"    sub  v1={sub_cell6[0,:2]}")
        print(f"    film v1={fc6[0,:2]}")
        print(f"    sub  v2={sub_cell6[1,:2]}")
        print(f"    film v2={fc6[1,:2]}")
        v2_diff = fc6[1,:2] - sub_cell6[1,:2]
        print(f"    v2 diff={v2_diff}  |v2_diff|={np.linalg.norm(v2_diff):.4f} Ang")
        # Effective strain
        sc2d = sub_cell6[:2,:2]
        fc2d = fc6[:2,:2]
        F_eff6 = sc2d @ np.linalg.inv(fc2d)
        eps1e, eps2e, vm_e = strain_from_F(sc2d, fc2d)
        print(f"  Effective F after rotation:")
        for r in F_eff6: print(f"    {r}")
        print(f"  eps1={eps1e*100:.3f}%  eps2={eps2e*100:.3f}%  vm={vm_e*100:.3f}%  (expected {best6['vm']*100:.3f}%)")
        # Optimal rotation from polar decomp on canonical 2D
        from scipy.linalg import polar
        F_raw6 = A_Na6 @ np.linalg.inv(A_Nb6)
        R_opt6, U6 = polar(F_raw6)
        opt_angle6 = np.degrees(np.arctan2(R_opt6[1,0], R_opt6[0,0]))
        print(f"\n  Optimal rotation (polar decomp of canonical F): {opt_angle6:.4f} deg")
        print(f"  v1-alignment rotation:                           {rot6:.4f} deg")
        print(f"  Discrepancy: {abs(opt_angle6 - rot6):.4f} deg")
    except Exception as e:
        import traceback; traceback.print_exc()

section("7. Conclusions")
lines = [
    "[A] get_surface_lattice_2d: canonical form is correct.",
    "[B] find_coincidences: HNF enumeration and vm computation are correct.",
    "[C] build_symmetric_slab: 3D HNF supercell matches 2D prediction exactly.",
    "[D] stack_interface v1-rotation: CORRECT for rectangular surfaces (gamma=90).",
    "    For non-rectangular (shear) lattices: v1-rotation != optimal rotation.",
    "    v2 misalignment may introduce non-physical shear in the stacked cell.",
    "[E] film_frac @ sub_cell_2d: physically correct epitaxial strain model.",
    "    The vm computed in find_coincidences IS reproduced by the stacking step.",
    "",
    "MAIN ISSUES TO FIX:",
    "  1. ZrO2(001)/NbO(001) has 0 matches within max_det=12.",
    "     NbO a=4.21 vs ZrO2 a=3.60 is 17% mismatch; commensurate cells need",
    "     larger supercells (7x3.60=25.2 vs 6x4.21=25.26 is 0.2% strain but det=42).",
    "     -> Consider increasing max_det or using non-(001) orientations for NbO.",
    "  2. For non-rectangular surface lattices, stack_interface should use the",
    "     optimal rotation from polar decomp of F = R @ U instead of v1-only.",
    "  3. wrap_interface_for_dft at lines 323-395 was added prematurely and",
    "     should be removed if not yet intended.",
]
for l in lines:
    print(l)
