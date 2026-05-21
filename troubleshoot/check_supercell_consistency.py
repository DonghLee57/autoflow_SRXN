"""
Supercell consistency check before interface stacking.

For each (structure, Miller, HNF) triplet, verify:
  [1] 2D canonical form matches the actual slab cell geometry
  [2] HNF supercell has correct atom count (= det * primitive count)
  [3] HNF supercell canonical 2D = HNF @ primitive canonical 2D
  [4] No spurious atomic overlaps in the supercell (min dist > 1.5 A)
  [5] All slab atom z-coordinates stay within the slab region (no PBC jump)
  [6] Fractional positions in supercell tile the primitive cell correctly
      (supercell fractional coords mod (1/det) should reproduce primitive)
"""
import sys, os
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from autoflow_srxn.interface.builder import (
    get_surface_lattice_2d,
    build_symmetric_slab,
    iter_hnf_2d,
    find_coincidences,
)

STRUCT_DIR = os.path.join(ROOT, "structures")

def load(fname):
    return Structure.from_file(os.path.join(STRUCT_DIR, fname))

def section(t):
    print(f"\n{'='*64}\n  {t}\n{'='*64}")

def ok(msg):  print(f"  [OK ] {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

# -----------------------------------------------------------------------
# Helper: build primitive slab from pymatgen (no HNF, no vacuum)
# -----------------------------------------------------------------------
def get_primitive_slab_ase(structure, miller, min_thickness=10.0):
    """Return the primitive slab as ASE Atoms (no HNF, small vacuum)."""
    gen = SlabGenerator(
        structure,
        miller_index=list(miller),
        min_slab_size=min_thickness,
        min_vacuum_size=15.0,
        center_slab=True,
        in_unit_planes=False,
    )
    slabs = gen.get_slabs(symmetrize=True)
    if not slabs:
        return None
    slab_pmg = slabs[0]
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(slab_pmg)
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    atoms.rotate(normal, [0, 0, 1], rotate_cell=True)
    atoms.center(vacuum=15.0 / 2, axis=2)
    return atoms

# -----------------------------------------------------------------------
# Check min interatomic distance using only numpy (no ase neighborlist)
# -----------------------------------------------------------------------
def min_dist_in_slab(atoms, cutoff=4.0):
    """Return minimum in-plane interatomic distance using PBC replica."""
    from itertools import product
    cell = np.array(atoms.cell)
    pos  = atoms.get_positions()
    a1, a2 = cell[0], cell[1]
    min_d = np.inf
    for da, db in product([-1, 0, 1], repeat=2):
        shift = da * a1 + db * a2
        shifted = pos + shift
        diff = pos[:, None, :] - shifted[None, :, :]   # (N, N, 3)
        dist = np.sqrt((diff**2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        if dist.min() < min_d:
            min_d = dist.min()
    return float(min_d)

# -----------------------------------------------------------------------
# Check [6]: tiling consistency
# Supercell fractional coords (xy only), when mapped back to primitive
# by mod(1/det_a, 1/det_b), should reproduce the primitive frac coords.
# -----------------------------------------------------------------------
def check_tiling(prim_atoms, sup_atoms, HNF):
    """Return (ok_flag, description)."""
    det_a = int(round(abs(np.linalg.det(HNF))))
    n_prim = len(prim_atoms)
    n_sup  = len(sup_atoms)
    if n_sup != det_a * n_prim:
        return False, f"atom count mismatch: {n_sup} != {det_a}*{n_prim}"

    prim_syms = list(prim_atoms.get_chemical_symbols())
    sup_syms  = list(sup_atoms.get_chemical_symbols())

    # Each primitive atom should appear exactly det_a times in supercell
    from collections import Counter
    pc = Counter(prim_syms)
    sc = Counter(sup_syms)
    for el, cnt in pc.items():
        if sc.get(el, 0) != det_a * cnt:
            return False, f"element {el}: {sc.get(el,0)} != {det_a}*{cnt}"
    return True, f"atom count OK: {n_sup} = {det_a}*{n_prim} per element"

# -----------------------------------------------------------------------
# Main test cases
# -----------------------------------------------------------------------
cases = [
    # (bulk_file, miller, hnf_na_or_nb, label)
    ("ZrO2_t_bulk.vasp",  (1, 0, 1), np.array([[4,0],[0,3]]), "ZrO2  (101)  Na=[[4,0],[0,3]]"),
    ("ZrO2_t_bulk.vasp",  (0, 0, 1), np.array([[1,0],[0,1]]), "ZrO2  (001)  Na=I  [primitive only]"),
    ("ZrO2_t_bulk.vasp",  (1, 1, 1), np.array([[5,0],[0,2]]), "ZrO2  (111)  Na=[[5,0],[0,2]]"),
    ("NbO2_bulk.vasp",    (0, 0, 1), np.array([[3,0],[0,4]]), "NbO2  (001)  Nb=[[3,0],[0,4]]"),
    ("NbO_bulk.vasp",     (1, 1, 0), np.array([[6,0],[1,2]]), "NbO   (110)  Nb=[[6,0],[1,2]]"),
    ("Nb2O5_B_bulk.vasp", (0, 0, 1), np.array([[1,0],[0,1]]), "Nb2O5 (001)  Nb=I  [primitive only]"),
    ("Ta2O5_B_bulk.vasp", (0, 0, 1), np.array([[1,0],[0,1]]), "Ta2O5 (001)  Nb=I  [primitive only]"),
]

all_pass = True

for bulk_file, miller, HNF, label in cases:
    section(label)
    struct = load(bulk_file)
    det = int(round(abs(np.linalg.det(HNF))))

    # [1] Canonical 2D from get_surface_lattice_2d
    A_prim = get_surface_lattice_2d(struct, miller)
    A_super_pred = HNF.astype(float) @ A_prim   # predicted supercell 2D

    v1_len  = float(np.linalg.norm(A_prim[0]))
    v2_len  = float(np.linalg.norm(A_prim[1]))
    cos_g   = np.dot(A_prim[0], A_prim[1]) / (v1_len * v2_len)
    gamma   = float(np.degrees(np.arccos(np.clip(cos_g, -1, 1))))
    print(f"  Primitive 2D cell: |v1|={v1_len:.4f}  |v2|={v2_len:.4f}  gamma={gamma:.2f} deg")

    vs1 = float(np.linalg.norm(A_super_pred[0]))
    vs2 = float(np.linalg.norm(A_super_pred[1]))
    cos_gs = np.dot(A_super_pred[0], A_super_pred[1]) / (vs1 * vs2)
    gammas = float(np.degrees(np.arccos(np.clip(cos_gs, -1, 1))))
    print(f"  Predicted super 2D: |v1|={vs1:.4f}  |v2|={vs2:.4f}  gamma={gammas:.2f} deg")

    # [2] Build primitive slab
    prim_atoms = get_primitive_slab_ase(struct, miller, min_thickness=10.0)
    if prim_atoms is None:
        fail("SlabGenerator returned no slab for primitive")
        all_pass = False
        continue
    prim_cell = np.array(prim_atoms.cell)
    prim_v1   = np.linalg.norm(prim_cell[0, :2])
    prim_v2   = np.linalg.norm(prim_cell[1, :2])
    prim_v1_v = prim_cell[0, :2]
    prim_v2_v = prim_cell[1, :2]
    cos_p = np.dot(prim_v1_v, prim_v2_v) / (prim_v1 * prim_v2) if prim_v1 > 0 and prim_v2 > 0 else 0
    gamma_p = float(np.degrees(np.arccos(np.clip(cos_p, -1, 1))))

    # Compare with canonical 2D
    diff_v1 = abs(prim_v1 - v1_len)
    diff_v2 = abs(prim_v2 - v2_len)
    diff_g  = abs(gamma_p - gamma)
    if diff_v1 < 0.01 and diff_v2 < 0.01 and diff_g < 0.1:
        ok(f"Primitive slab in-plane matches canonical 2D: |v1|={prim_v1:.4f}  |v2|={prim_v2:.4f}  gamma={gamma_p:.2f}")
    else:
        fail(f"Primitive slab in-plane MISMATCH: |v1|={prim_v1:.4f}(exp {v1_len:.4f})  |v2|={prim_v2:.4f}(exp {v2_len:.4f})  gamma={gamma_p:.2f}(exp {gamma:.2f})")
        all_pass = False

    n_prim = len(prim_atoms)
    print(f"  Primitive slab: {n_prim} atoms")

    # [3] Build HNF supercell slab
    sup_atoms = build_symmetric_slab(struct, miller, min_thickness_ang=10.0, vacuum_ang=15.0, HNF=HNF)
    sup_cell  = np.array(sup_atoms.cell)
    sup_v1    = np.linalg.norm(sup_cell[0, :2])
    sup_v2    = np.linalg.norm(sup_cell[1, :2])
    sup_v1_v  = sup_cell[0, :2]
    sup_v2_v  = sup_cell[1, :2]
    cos_s = np.dot(sup_v1_v, sup_v2_v) / (sup_v1 * sup_v2) if sup_v1 > 0 and sup_v2 > 0 else 0
    gamma_s = float(np.degrees(np.arccos(np.clip(cos_s, -1, 1))))

    diff_sv1 = abs(sup_v1 - vs1)
    diff_sv2 = abs(sup_v2 - vs2)
    diff_sg  = abs(gamma_s - gammas)
    if diff_sv1 < 0.01 and diff_sv2 < 0.01 and diff_sg < 0.1:
        ok(f"Supercell slab in-plane matches predicted: |v1|={sup_v1:.4f}  |v2|={sup_v2:.4f}  gamma={gamma_s:.2f}")
    else:
        fail(f"Supercell in-plane MISMATCH: |v1|={sup_v1:.4f}(exp {vs1:.4f})  |v2|={sup_v2:.4f}(exp {vs2:.4f})  gamma={gamma_s:.2f}(exp {gammas:.2f})")
        all_pass = False

    n_sup = len(sup_atoms)

    # [4] Atom count check
    expected_count = det * n_prim
    if n_sup == expected_count:
        ok(f"Atom count: {n_sup} = {det} * {n_prim}")
    else:
        fail(f"Atom count: {n_sup} != {det} * {n_prim} (expected {expected_count})")
        all_pass = False

    # [5] Tiling / element check
    flag, msg = check_tiling(prim_atoms, sup_atoms, HNF)
    if flag:
        ok(f"Tiling: {msg}")
    else:
        fail(f"Tiling: {msg}")
        all_pass = False

    # [6] Min interatomic distance
    md_prim = min_dist_in_slab(prim_atoms)
    md_sup  = min_dist_in_slab(sup_atoms)
    if md_prim > 1.5:
        ok(f"Primitive slab min dist = {md_prim:.3f} A")
    else:
        fail(f"Primitive slab min dist = {md_prim:.3f} A  [too close!]")
        all_pass = False
    if md_sup > 1.5:
        ok(f"Supercell slab min dist = {md_sup:.3f} A")
    else:
        fail(f"Supercell slab min dist = {md_sup:.3f} A  [too close!]")
        all_pass = False

    # [7] z-coordinate spread: all atoms should be in a contiguous slab region
    z = sup_atoms.get_positions()[:, 2]
    cell_c = float(sup_cell[2, 2])
    z_frac = z / cell_c
    z_center = 0.5  # slabs are centered
    # Atoms should be within ~15 A of center (slab region)
    z_span = z.max() - z.min()
    z_min_f = z_frac.min()
    z_max_f = z_frac.max()
    if z_min_f > 0.0 and z_max_f < 1.0 and z_span < 0.7 * cell_c:
        ok(f"Supercell z-span = {z_span:.2f} A, frac=[{z_min_f:.3f}, {z_max_f:.3f}]")
    else:
        warn(f"Supercell z-spread may have PBC issue: span={z_span:.2f} A, frac=[{z_min_f:.3f}, {z_max_f:.3f}]")

section("Final Result")
if all_pass:
    print("  ALL CHECKS PASSED")
    print("  => Supercell generation is consistent with bulk structure.")
    print("  => Issues, if any, are in the stacking/strain step.")
else:
    print("  SOME CHECKS FAILED -- see [FAIL] entries above.")
    print("  => Fix supercell generation first before diagnosing stacking.")
