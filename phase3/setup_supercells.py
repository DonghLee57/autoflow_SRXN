"""Phase 3 setup: create [1,1]/[-1,1] x2 supercell slabs for all three substrates.

Transformation:  a' =  a + b,  b' = -a + b  (45-deg rotated, 2x area)
For square cell with a=b=a0:  |a'| = |b'| = a0*sqrt(2)
  Si100     10.926 A -> 15.45 A
  SiO2      10.085 A -> 14.27 A

Outputs:
  structures/slabs/supercell/{name}_2x_slab.vasp
  structures/slabs/supercell/site_maps/{name}_2x_sites.csv  (same Cartesian XY, valid in supercell)
"""
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.io import read, write
from ase.build import make_supercell

# [1,1]/[-1,1] in-plane, keep z unchanged
P = np.array([[1, 1, 0],
              [-1, 1, 0],
              [0, 0, 1]], dtype=int)

SLABS = {
    "Si100":        ROOT / "structures/slabs/Si100_slab.vasp",
    "SiO2_Si_term": ROOT / "structures/slabs/SiO2_Si_term_slab.vasp",
    "SiO2_O_term":  ROOT / "structures/slabs/SiO2_O_term_slab.vasp",
}

OUT_DIR  = ROOT / "structures/slabs/supercell"
SITE_DIR = ROOT / "structures/slabs/site_maps"
SC_SITE  = OUT_DIR / "site_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SC_SITE.mkdir(parents=True, exist_ok=True)

print("Creating [1,1]/[-1,1] supercell slabs")
print("=" * 60)

for name, src in SLABS.items():
    slab = read(str(src))
    sc   = make_supercell(slab, P)

    out = OUT_DIR / f"{name}_2x_slab.vasp"
    write(str(out), sc, vasp5=True)

    a0 = slab.cell.lengths()[0]
    a_new = sc.cell.lengths()[0]
    b_new = sc.cell.lengths()[1]
    print(f"\n{name}:")
    print(f"  Original  : {len(slab):4d} atoms, a = {a0:.4f} A")
    print(f"  Supercell : {len(sc):4d} atoms, a = {a_new:.4f} A, b = {b_new:.4f} A")
    print(f"  Written   : {out.relative_to(ROOT)}")

    # Copy site map unchanged — Cartesian site positions are valid within the
    # new supercell (they lie in the lower-left quadrant of the 2x cell)
    orig_site = SITE_DIR / f"{name}_sites.csv"
    if orig_site.exists():
        dst = SC_SITE / f"{name}_2x_sites.csv"
        shutil.copy(str(orig_site), str(dst))
        print(f"  Site map  : copied -> {dst.relative_to(ROOT)}")

print("\nDone.")
