"""
Compare:
  (a) gen.get_slabs(sym=False)[0].lattice  <- current get_surface_lattice_2d
  (b) gen.get_slabs(sym=True)[0].lattice   <- build_symmetric_slab
  (c) gen.oriented_unit_cell.lattice        <- possible fix (no slab needed)

For all key cases.
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

def inplane_from_lattice(latt_matrix):
    """Given 3x3 pymatgen lattice matrix, return |v1|, |v2|, gamma."""
    v1, v2 = latt_matrix[0], latt_matrix[1]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    cos_g = np.dot(v1, v2) / (n1 * n2)
    gamma = np.degrees(np.arccos(np.clip(cos_g, -1, 1)))
    return n1, n2, gamma, v1, v2

def inplane_after_rotate_z(latt_matrix):
    """Get in-plane info AFTER rotating cell so that v1×v2 points to z."""
    from ase import Atoms
    v1 = latt_matrix[0]
    v2 = latt_matrix[1]
    v3 = latt_matrix[2]
    cell_3d = np.array([v1, v2, v3])
    n_atoms_fake = 1
    atoms = Atoms('H', positions=[[0,0,0]], cell=cell_3d, pbc=True)
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    atoms.rotate(normal, [0, 0, 1], rotate_cell=True)
    new_v1 = atoms.cell[0]
    new_v2 = atoms.cell[1]
    n1 = np.linalg.norm(new_v1[:2])
    n2 = np.linalg.norm(new_v2[:2])
    if n1 < 1e-8 or n2 < 1e-8:
        return 0, 0, 0, new_v1, new_v2
    cos_g = np.dot(new_v1[:2], new_v2[:2]) / (n1 * n2)
    gamma = np.degrees(np.arccos(np.clip(cos_g, -1, 1)))
    return n1, n2, gamma, new_v1, new_v2

cases = [
    ("ZrO2_t_bulk.vasp",  (1, 0, 1), "ZrO2(101)"),
    ("ZrO2_t_bulk.vasp",  (1, 1, 1), "ZrO2(111)"),
    ("NbO2_bulk.vasp",    (0, 0, 1), "NbO2(001)"),
    ("NbO_bulk.vasp",     (1, 1, 0), "NbO(110)"),
    ("Nb2O5_B_bulk.vasp", (0, 0, 1), "Nb2O5(001)"),
    ("Nb2O5_B_bulk.vasp", (1, 0, 0), "Nb2O5(100)"),
    ("Ta2O5_B_bulk.vasp", (0, 0, 1), "Ta2O5(001)"),
]

print(f"{'Case':<15}  {'Method':<30}  {'|v1|':>8} {'|v2|':>8} {'gamma':>7}")
print("-" * 75)

for fname, miller, label in cases:
    struct = load(fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen = SlabGenerator(struct, miller_index=list(miller),
                            min_slab_size=1, min_vacuum_size=0,
                            center_slab=False, in_unit_planes=False)

    # (a) current method: raw slab lattice matrix (sym=False, min_t=1)
    slabs = gen.get_slabs(symmetrize=False)
    if slabs:
        n1, n2, g, v1, v2 = inplane_from_lattice(slabs[0].lattice.matrix)
        print(f"{label:<15}  {'(a) sym=F raw matrix':<30}  {n1:>8.4f} {n2:>8.4f} {g:>7.2f}")
    else:
        print(f"{label:<15}  {'(a) sym=F raw matrix':<30}  {'---':>8} {'---':>8} {'---':>7}")

    # (b) current method + rotate to z
    if slabs:
        n1, n2, g, v1, v2 = inplane_after_rotate_z(slabs[0].lattice.matrix)
        print(f"{label:<15}  {'(b) sym=F + rotate_z':<30}  {n1:>8.4f} {n2:>8.4f} {g:>7.2f}")

    # (c) oriented_unit_cell raw
    ouc = gen.oriented_unit_cell
    n1, n2, g, v1, v2 = inplane_from_lattice(ouc.lattice.matrix)
    print(f"{label:<15}  {'(c) oriented_unit_cell':<30}  {n1:>8.4f} {n2:>8.4f} {g:>7.2f}")

    # (d) oriented_unit_cell + rotate to z
    n1, n2, g, v1, v2 = inplane_after_rotate_z(ouc.lattice.matrix)
    print(f"{label:<15}  {'(d) ouc + rotate_z':<30}  {n1:>8.4f} {n2:>8.4f} {g:>7.2f}")

    # (e) sym=True slab (reference: what build_symmetric_slab uses)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen2 = SlabGenerator(struct, miller_index=list(miller),
                             min_slab_size=15, min_vacuum_size=0,
                             center_slab=False, in_unit_planes=False)
        slabs2 = gen2.get_slabs(symmetrize=True)
    if slabs2:
        n1, n2, g, v1, v2 = inplane_after_rotate_z(slabs2[0].lattice.matrix)
        print(f"{label:<15}  {'(e) sym=T min_t=15 +rot_z':<30}  {n1:>8.4f} {n2:>8.4f} {g:>7.2f} [TARGET]")
    else:
        print(f"{label:<15}  {'(e) sym=T min_t=15 +rot_z':<30}  {'---':>8} {'---':>8} {'---':>7} [FAIL]")

    print()
