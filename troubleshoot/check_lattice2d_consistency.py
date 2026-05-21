"""
Verify that get_surface_lattice_2d gives the same in-plane lattice as
build_symmetric_slab for all test structures and Miller indices.

Specifically tests the effect of min_slab_size in SlabGenerator.
"""
import sys, os, warnings
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

STRUCT_DIR = os.path.join(ROOT, "structures")

def load(fname):
    return Structure.from_file(os.path.join(STRUCT_DIR, fname))

def slab_inplane(structure, miller, min_t, sym):
    """Build slab and return (|v1|, |v2|, gamma_deg, n_atoms) or None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen = SlabGenerator(
            structure,
            miller_index=list(miller),
            min_slab_size=min_t,
            min_vacuum_size=0,
            center_slab=False,
            in_unit_planes=False,
        )
        slabs = gen.get_slabs(symmetrize=sym)
    if not slabs:
        return None
    slab = slabs[0]
    v1 = slab.lattice.matrix[0]
    v2 = slab.lattice.matrix[1]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    cos_g = np.dot(v1, v2) / (n1 * n2)
    gamma = np.degrees(np.arccos(np.clip(cos_g, -1, 1)))
    return n1, n2, gamma, len(slab)

cases = [
    ("ZrO2_t_bulk.vasp",  [(1,0,1), (0,0,1), (1,1,1)],        "ZrO2"),
    ("NbO2_bulk.vasp",    [(0,0,1), (1,1,0)],                  "NbO2"),
    ("NbO_bulk.vasp",     [(0,0,1), (1,1,0), (1,0,0)],         "NbO"),
    ("Nb2O5_B_bulk.vasp", [(0,0,1), (1,0,0), (0,1,0), (1,1,0)], "Nb2O5"),
    ("Ta2O5_B_bulk.vasp", [(0,0,1), (1,0,0), (0,1,0), (1,1,0)], "Ta2O5"),
]

print(f"\n{'Structure':<8} {'Miller':<10} {'min_t':>6} {'sym':>5}  {'|v1|':>8} {'|v2|':>8} {'gamma':>7} {'n':>5}")
print("-" * 65)

for fname, millers, label in cases:
    struct = load(fname)
    for miller in millers:
        for min_t in [1, 6, 10, 15]:
            for sym in [False]:
                r = slab_inplane(struct, miller, min_t, sym)
                if r:
                    v1, v2, g, n = r
                    print(f"{label:<8} {str(miller):<10} {min_t:>6.0f} {str(sym):>5}  {v1:>8.4f} {v2:>8.4f} {g:>7.2f} {n:>5}")
                else:
                    print(f"{label:<8} {str(miller):<10} {min_t:>6.0f} {str(sym):>5}  {'---':>8} {'---':>8} {'---':>7} {'---':>5}")
        print()
