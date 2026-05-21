"""
Diagnose why SlabGenerator fails for Nb2O5/Ta2O5 (001) and find working faces.

B-phase Nb2O5/Ta2O5: monoclinic C2/m (#12), beta~105 deg.
The (001) failure is likely because:
  a) beta != 90 deg means c is not perpendicular to the ab-plane
  b) symmetrize=True removes too many atoms for a thin slab
  c) The surface might be polar/non-stoichiometric

Strategy: try several Miller indices and both symmetrize=True and False.
"""
import sys, os
import numpy as np
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.io.ase import AseAtomsAdaptor
from autoflow_srxn.interface.builder import get_surface_lattice_2d

STRUCT_DIR = os.path.join(ROOT, "structures")

def load(fname):
    return Structure.from_file(os.path.join(STRUCT_DIR, fname))

def section(t):
    print(f"\n{'='*64}\n  {t}\n{'='*64}")

def try_slab(struct, miller, sym, min_t):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen = SlabGenerator(
            struct, miller_index=list(miller),
            min_slab_size=min_t, min_vacuum_size=15.0,
            center_slab=True, in_unit_planes=False,
        )
        slabs = gen.get_slabs(symmetrize=sym)
    return slabs

for fname, label in [("Nb2O5_B_bulk.vasp", "B-Nb2O5"), ("Ta2O5_B_bulk.vasp", "B-Ta2O5")]:
    struct = load(fname)
    section(f"{label}: lattice info")
    latt = struct.lattice
    print(f"  a={latt.a:.4f}  b={latt.b:.4f}  c={latt.c:.4f} Ang")
    print(f"  alpha={latt.alpha:.2f}  beta={latt.beta:.2f}  gamma={latt.gamma:.2f} deg")
    print(f"  n_atoms={len(struct)}")
    print(f"  SG = {struct.get_space_group_info()}")

    # Get all distinct Miller indices up to max=2
    distinct = get_symmetrically_distinct_miller_indices(struct, 2)
    print(f"\n  Distinct Miller indices (max=2): {[tuple(m) for m in distinct]}")

    section(f"{label}: slab generation test")
    results = []
    for miller in distinct:
        miller = tuple(miller)
        for sym in [True, False]:
            for min_t in [10.0, 15.0, 20.0]:
                slabs = try_slab(struct, miller, sym, min_t)
                if slabs:
                    slab = slabs[0]
                    n = len(slab)
                    lm = slab.lattice.matrix
                    v1_len = np.linalg.norm(lm[0])
                    v2_len = np.linalg.norm(lm[1])
                    cos_g = np.dot(lm[0], lm[1]) / (v1_len * v2_len)
                    gamma = np.degrees(np.arccos(np.clip(cos_g, -1, 1)))
                    # check symmetry of top/bottom termination
                    sym_ok = slab.is_symmetric()
                    results.append((miller, sym, min_t, n, v1_len, v2_len, gamma, sym_ok))
                    break  # found smallest thickness that works
            else:
                continue
            break

    if not results:
        print(f"  [FAIL] No slab could be generated for any Miller index!")
        continue

    print(f"\n  {'Miller':<10} {'sym':<6} {'min_t':<8} {'n_atoms':<10} {'|v1|':>8} {'|v2|':>8} {'gamma':>8} {'is_sym'}")
    print(f"  {'-'*75}")
    for r in results:
        miller, sym, min_t, n, v1, v2, g, is_sym = r
        print(f"  {str(miller):<10} {str(sym):<6} {min_t:<8.1f} {n:<10d} {v1:>8.3f} {v2:>8.3f} {g:>8.2f} {is_sym}")

    # Recommend the best face for interface matching
    # Priority: (1) sym_ok, (2) small n_atoms, (3) reasonable gamma
    sym_results = [r for r in results if r[7]]  # is_symmetric
    if sym_results:
        best = min(sym_results, key=lambda r: r[3])
        print(f"\n  Best face for interface: {best[0]}  ({best[3]} atoms, gamma={best[6]:.1f}, symmetric)")
    else:
        best = min(results, key=lambda r: r[3])
        print(f"\n  Best face (non-sym): {best[0]}  ({best[3]} atoms, gamma={best[6]:.1f})")
        print(f"  WARNING: No symmetric slab found - DFT surface may be polar/unphysical")

    # Print 2D lattice for the best face
    A_2d = get_surface_lattice_2d(struct, best[0])
    print(f"\n  2D canonical lattice for {best[0]}:")
    print(f"    v1 = {A_2d[0]}  |v1|={np.linalg.norm(A_2d[0]):.4f}")
    print(f"    v2 = {A_2d[1]}  |v2|={np.linalg.norm(A_2d[1]):.4f}")
