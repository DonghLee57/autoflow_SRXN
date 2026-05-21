"""Print the raw 3x3 cell matrices for Nb2O5(001) slabs to see what's actually stored."""
import sys, os, warnings
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

STRUCT_DIR = os.path.join(ROOT, "structures")
struct = Structure.from_file(os.path.join(STRUCT_DIR, "Nb2O5_B_bulk.vasp"))

print("=== BULK CELL ===")
print(struct.lattice.matrix)

for (min_t, sym, label) in [
    (1, False, "sym=False min_t=1  (current get_surface_lattice_2d)"),
    (10, True,  "sym=True  min_t=10 (current build_symmetric_slab)"),
    (15, True,  "sym=True  min_t=15"),
]:
    print(f"\n=== (001) slab: {label} ===")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen = SlabGenerator(struct, miller_index=[0,0,1],
                            min_slab_size=min_t, min_vacuum_size=0,
                            center_slab=False, in_unit_planes=False)
        slabs = gen.get_slabs(symmetrize=sym)
    if not slabs:
        print("  [NO SLAB]")
        continue
    slab = slabs[0]
    lm = slab.lattice.matrix
    print(f"  lattice.matrix:")
    for row in lm:
        print(f"    {row}")
    print(f"  |v1|={np.linalg.norm(lm[0]):.4f}  |v2|={np.linalg.norm(lm[1]):.4f}  |v3|={np.linalg.norm(lm[2]):.4f}")
    cos_g = np.dot(lm[0], lm[1])/(np.linalg.norm(lm[0])*np.linalg.norm(lm[1]))
    print(f"  gamma(v1,v2)={np.degrees(np.arccos(np.clip(cos_g,-1,1))):.2f} deg")
    print(f"  n_atoms={len(slab)}")

    # Check: are v1, v2 actually in-plane?
    # Normal direction of surface = direction perpendicular to slab (should be ~ v3 direction)
    # For pymatgen Slab, the surface normal is slab.normal
    print(f"  slab.normal={slab.normal}")

    # Check if v1, v2 are perpendicular to the normal
    n = np.array(slab.normal)
    print(f"  v1 · normal = {np.dot(lm[0], n):.6f}  (should be ~0 if in-plane)")
    print(f"  v2 · normal = {np.dot(lm[1], n):.6f}  (should be ~0 if in-plane)")
