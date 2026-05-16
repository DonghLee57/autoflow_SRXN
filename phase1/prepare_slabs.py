"""Phase 1 — Substrate slab preparation and relaxation.

Slabs prepared:
  1. Si(100)        : 2x1 buckled-dimer seed -> 7net-0 relax
  2. SiO2(001) O-terminated  : O-top/O-bottom -> ionic rumpling seed -> relax
  3. SiO2(001) Si-terminated : Si-top/Si-bottom -> ionic rumpling seed -> relax

Common settings:
  - Bottom 1.0 coverage H passivation
  - frozen_z_ang = 5.5 A (atoms below z_min + 5.5 A fixed during relax)
  - fmax = 0.05 eV/A, optimizer = FIRE

Outputs (structures/slabs/):
  Si100_slab.vasp
  SiO2_O_term_slab.vasp
  SiO2_Si_term_slab.vasp
  slabs_summary.txt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.io import read, write

from autoflow_srxn.surface.surface_utils import (
    create_slab_from_bulk,
    apply_surface_reconstruction,
    passivate_surface_coverage_general,
    find_surface_indices,
    standardize_vasp_atoms,
    get_all_dangling_bonds_general,
)
from autoflow_srxn.surface.chemisorption_builder import analyze_surface_reactivity
from autoflow_srxn.simulation.potentials import SimulationEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FROZEN_Z = 5.5      # A — fix atoms below z_min + FROZEN_Z during relax
FMAX     = 0.05     # eV/A
STEPS    = 300

ENGINE_CONFIG = {
    "engine": {
        "potential": {
            "backend": "sevennet",
            "model": "7net-0",
            "device": "cpu",
            "dtype": "float32",
        }
    },
    "relaxation": {
        "fmax": FMAX,
        "steps": STEPS,
        "optimizer": "FIRE",
        "frozen_z_ang": FROZEN_Z,
    },
}

VALENCE_MAP_SI   = {"Si": 4, "H": 1}
VALENCE_MAP_SIO2 = {"Si": 4, "O": 2, "H": 1}

OUT_DIR = ROOT / "structures" / "slabs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLAB_CONFIGS = [
    {
        "name":       "Si100",
        "bulk":       ROOT / "structures/Si_relaxed.vasp",
        "miller":     (1, 0, 0),
        "thickness":  12.0,
        "vacuum":     15.0,
        "target_area": 120.0,   # 2x2 supercell (~10.93 x 10.93 A); doubled x for adsorbate studies
        "top_term":   None,
        "bot_term":   None,
        "valence":    VALENCE_MAP_SI,
        "recon_strategy": "auto",
        "output":     OUT_DIR / "Si100_slab.vasp",
        # Shift bulk by a/4 along [100] to expose the x=3a/4 plane as the
        # surface. The a=5.4631 A lattice constant is read at runtime below;
        # we store a sentinel here and resolve it in build_and_relax().
        "bulk_shift": "a/4",
    },
    {
        "name":       "SiO2_O_term",
        "bulk":       ROOT / "structures/SiO2_relaxed.vasp",
        "miller":     (0, 0, 1),
        "thickness":  15.0,
        "vacuum":     15.0,
        "target_area": 100.0,   # 2x2 supercell (~10.09 x 10.09 A); doubled x for adsorbate studies
        "top_term":   "O",
        "bot_term":   "O",
        "valence":    VALENCE_MAP_SIO2,
        "recon_strategy": "auto",
        "output":     OUT_DIR / "SiO2_O_term_slab.vasp",
    },
    {
        "name":       "SiO2_Si_term",
        "bulk":       ROOT / "structures/SiO2_relaxed.vasp",
        "miller":     (0, 0, 1),
        "thickness":  15.0,
        "vacuum":     15.0,
        "target_area": 100.0,   # 2x2 supercell (~10.09 x 10.09 A); doubled x for adsorbate studies
        "top_term":   "Si",
        "bot_term":   "Si",
        "valence":    VALENCE_MAP_SIO2,
        "recon_strategy": "auto",
        "output":     OUT_DIR / "SiO2_Si_term_slab.vasp",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_dangling_bonds(slab, valence_map):
    """Return number of upward-pointing dangling bonds on the top surface."""
    dbs = get_all_dangling_bonds_general(slab, valence_map, side="top")
    return len(dbs)


def analyze_slab(slab, name, valence_map):
    """Print structural statistics for the slab."""
    z = slab.positions[:, 2]
    top_idx = find_surface_indices(slab, "top", threshold=1.0)
    bot_idx = find_surface_indices(slab, "bottom", threshold=1.0)
    n_db = count_dangling_bonds(slab, valence_map)

    top_syms = [slab.symbols[i] for i in top_idx]
    bot_syms = [slab.symbols[i] for i in bot_idx]

    print(f"  Atoms        : {len(slab)}  ({slab.get_chemical_formula()})")
    print(f"  Cell (a,b,c) : {slab.cell.lengths()}")
    print(f"  Z range      : {z.min():.2f} - {z.max():.2f} A  (thickness ~ {z.max()-z.min():.1f} A)")
    print(f"  Top surface  : {set(top_syms)}  ({len(top_idx)} atoms)")
    print(f"  Bot surface  : {set(bot_syms)}  ({len(bot_idx)} atoms)")
    print(f"  Dangling bonds (top): {n_db}")
    return n_db


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_and_relax(cfg, engine):
    name = cfg["name"]
    print("\n" + "=" * 70)
    print(f"  SLAB: {name}")
    print("=" * 70)

    # 1. Load bulk
    bulk = read(str(cfg["bulk"]))
    print(f"  Bulk: {cfg['bulk'].name}  ({len(bulk)} atoms, {bulk.get_chemical_formula()})")

    # 2. Create slab and wrap atoms into PBC cell
    # Si(100) special case: shift bulk by a/4 along [100] before slicing.
    # The x=0 plane of the conventional diamond-cubic cell has a corner atom at
    # fractional (0,0) which ASE excludes to avoid PBC duplicates in supercell
    # operations, leaving only 4 atoms per 2x2 surface layer (half the expected 8).
    # Shifting by +a/4 exposes the x=3a/4 plane whose atoms lie at (a/4,3a/4) and
    # (3a/4,a/4) — never on a boundary — so all 8 atoms per 2x2 layer are included,
    # and nearest-neighbor distance = a/sqrt(2) = 3.86 A, correct for 2x1 dimers.
    bulk_for_slab = bulk.copy()
    shift_spec = cfg.get("bulk_shift")
    if shift_spec == "a/4":
        a = bulk.cell.lengths()[0]
        bulk_for_slab.translate([a / 4, 0, 0])
        bulk_for_slab.wrap()
    elif shift_spec is not None:
        bulk_for_slab.translate(shift_spec)
        bulk_for_slab.wrap()

    slab = create_slab_from_bulk(
        bulk_for_slab,
        miller_indices=cfg["miller"],
        thickness=cfg["thickness"],
        vacuum=cfg["vacuum"],
        target_area=cfg["target_area"],
        top_termination=cfg["top_term"],
        bottom_termination=cfg["bot_term"],
        verbose=True,
    )
    slab.wrap()   # bring boundary atoms (frac=1.0) into the cell (frac=0.0)
    print(f"  Raw slab: {len(slab)} atoms")

    # 3. Bottom H passivation (coverage=1.0)
    slab = passivate_surface_coverage_general(
        slab,
        coverage=1.0,
        valence_map=cfg["valence"],
        element="H",
        side="bottom",
        verbose=False,
    )
    print(f"  After bottom H passivation: {len(slab)} atoms")

    # 4. Surface reconstruction seed
    slab = apply_surface_reconstruction(
        slab,
        strategy=cfg["recon_strategy"],
        side="top",
        miller=cfg["miller"],
        verbose=True,
    )

    # 5. Pre-relax analysis
    print("\n  [PRE-RELAX]")
    analyze_slab(slab, name, cfg["valence"])

    # 6. Slab relaxation with frozen bottom
    print(f"\n  Relaxing slab (frozen_z_ang={FROZEN_Z} A, fmax={FMAX}, FIRE)...")
    engine.relax(slab, frozen_z_ang=FROZEN_Z, verbose=True)

    # 7. Post-relax analysis
    print("\n  [POST-RELAX]")
    n_db = analyze_slab(slab, name, cfg["valence"])

    # 8. Save
    slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
    write(str(cfg["output"]), slab, vasp5=True)
    print(f"\n  -> Saved: {cfg['output'].relative_to(ROOT)}")

    return slab, n_db


def main(targets=None):
    engine = SimulationEngine(config=ENGINE_CONFIG)

    summary_lines = []
    for cfg in SLAB_CONFIGS:
        if targets and cfg["name"] not in targets:
            continue
        slab, n_db = build_and_relax(cfg, engine)
        e = slab.get_potential_energy()
        summary_lines.append(
            f"{cfg['name']:20s}  atoms={len(slab):3d}  "
            f"E={e:.4f} eV  E/atom={e/len(slab):.4f} eV  dangling_bonds={n_db}"
        )

    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    for line in summary_lines:
        print(" ", line)

    summary_path = OUT_DIR / "slabs_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Phase 1 Slab Summary\n")
        f.write("=" * 70 + "\n")
        for line in summary_lines:
            f.write(line + "\n")
    print(f"\n  Summary written to: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    main(targets)
