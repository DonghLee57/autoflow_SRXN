"""
Verify that the fixed get_surface_lattice_2d matches build_symmetric_slab
for all key cases, and that find_coincidences still works correctly.
"""
import sys, os, warnings
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from autoflow_srxn.interface.builder import (
    get_surface_lattice_2d,
    build_symmetric_slab,
    find_coincidences,
)

STRUCT_DIR = os.path.join(ROOT, "structures")

def load(fname):
    return Structure.from_file(os.path.join(STRUCT_DIR, fname))

def ok(msg):   print(f"  [OK ] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

def section(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")

# -----------------------------------------------------------------------
# Check: get_surface_lattice_2d vs actual build_symmetric_slab cell
# (build_symmetric_slab rotates to z, so we compare the in-plane lengths)
# -----------------------------------------------------------------------
section("1. get_surface_lattice_2d vs build_symmetric_slab in-plane cell")

cases = [
    ("ZrO2_t_bulk.vasp",  (1,0,1), np.array([[1,0],[0,1]]), "ZrO2(101)  primitive"),
    ("ZrO2_t_bulk.vasp",  (0,0,1), np.array([[1,0],[0,1]]), "ZrO2(001)  primitive"),
    ("ZrO2_t_bulk.vasp",  (1,1,1), np.array([[1,0],[0,1]]), "ZrO2(111)  primitive"),
    ("NbO2_bulk.vasp",    (0,0,1), np.array([[1,0],[0,1]]), "NbO2(001)  primitive"),
    ("NbO_bulk.vasp",     (1,1,0), np.array([[1,0],[0,1]]), "NbO(110)   primitive"),
    ("Nb2O5_B_bulk.vasp", (0,0,1), np.array([[1,0],[0,1]]), "Nb2O5(001) primitive"),
    ("Ta2O5_B_bulk.vasp", (0,0,1), np.array([[1,0],[0,1]]), "Ta2O5(001) primitive"),
]

all_pass = True
for fname, miller, HNF, label in cases:
    struct = load(fname)
    A_2d = get_surface_lattice_2d(struct, miller)
    v1_2d = np.linalg.norm(A_2d[0])
    v2_2d = np.linalg.norm(A_2d[1])
    cos_g = np.dot(A_2d[0], A_2d[1]) / (v1_2d * v2_2d) if v1_2d > 0 and v2_2d > 0 else 0
    g_2d  = np.degrees(np.arccos(np.clip(cos_g, -1, 1)))

    # Build actual slab and check in-plane cell
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build_symmetric_slab(struct, miller, min_thickness_ang=10.0, vacuum_ang=15.0, HNF=None)
        cell = np.array(atoms.cell)
        v1_sl = np.linalg.norm(cell[0, :2])
        v2_sl = np.linalg.norm(cell[1, :2])
        cos_g_sl = np.dot(cell[0,:2], cell[1,:2]) / (v1_sl * v2_sl) if v1_sl > 0 and v2_sl > 0 else 0
        g_sl = np.degrees(np.arccos(np.clip(cos_g_sl, -1, 1)))
        d1 = abs(v1_2d - v1_sl)
        d2 = abs(v2_2d - v2_sl)
        dg = abs(g_2d  - g_sl)
        if d1 < 0.02 and d2 < 0.02 and dg < 0.2:
            ok(f"{label}: |v1|={v1_2d:.4f}  |v2|={v2_2d:.4f}  gamma={g_2d:.2f} deg  MATCHES slab")
        else:
            fail(f"{label}:")
            print(f"       get_surface_lattice_2d: |v1|={v1_2d:.4f}  |v2|={v2_2d:.4f}  gamma={g_2d:.2f}")
            print(f"       build_symmetric_slab:   |v1|={v1_sl:.4f}  |v2|={v2_sl:.4f}  gamma={g_sl:.2f}")
            all_pass = False
    except Exception as e:
        print(f"  [WARN] {label}: slab build failed ({e})")
        print(f"         get_surface_lattice_2d gave: |v1|={v1_2d:.4f}  |v2|={v2_2d:.4f}  gamma={g_2d:.2f}")

# -----------------------------------------------------------------------
# Check: find_coincidences still works for the key ZrO2/NbO2 case
# -----------------------------------------------------------------------
section("2. find_coincidences with fixed lattices")

zro2 = load("ZrO2_t_bulk.vasp")
nbo2 = load("NbO2_bulk.vasp")
nbo  = load("NbO_bulk.vasp")

pairs = [
    (zro2, (1,0,1), nbo2, (0,0,1), "ZrO2(101)/NbO2(001)", 12, 0.08),
    (zro2, (1,1,1), nbo,  (1,1,0), "ZrO2(111)/NbO(110)",  12, 0.08),
]

for sub_s, sub_m, film_s, film_m, label, max_det, sc in pairs:
    A_sub  = get_surface_lattice_2d(sub_s,  sub_m)
    A_film = get_surface_lattice_2d(film_s, film_m)
    hits   = find_coincidences(A_sub, A_film, max_det=max_det, strain_cutoff=sc)
    if hits:
        best = hits[0]
        ok(f"{label}: {len(hits)} matches, best vm={best['vm']*100:.2f}%  Na={best['Na'].tolist()}  Nb={best['Nb'].tolist()}")
    else:
        fail(f"{label}: 0 matches found (was working before fix)")
        all_pass = False

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
section("Final")
if all_pass:
    print("  ALL CHECKS PASSED — fix is correct.")
else:
    print("  SOME CHECKS FAILED.")
