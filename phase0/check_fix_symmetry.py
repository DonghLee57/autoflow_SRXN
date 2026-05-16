"""Phase 0 — Bulk symmetry check and symmetry-constrained re-relaxation.

Workflow per bulk structure:
  1. Check space group of original vs relaxed using spglib (symprec=0.01 A).
  2. If broken: re-relax from the original structure with a cell filter that
     preserves the Bravais lattice shape.
     - Cubic        : ExpCellFilter(hydrostatic_strain=True)  [1 DOF: isotropic]
     - Tetragonal/
       Hexagonal/
       Trigonal      : ExpCellFilter(mask=[1,1,1,0,0,0]) then enforce a=b (=mean)
     - Orthorhombic  : ExpCellFilter(mask=[1,1,1,0,0,0])       [3 DOF: diag only]
     - Others        : full ExpCellFilter
  3. Post-symmetrize the cell analytically.
  4. Final position-only relax with the symmetrized fixed cell.
  5. Re-check symmetry.

Output: structures/{Si,SiO2}_relaxed.vasp  (overwritten if re-relaxed)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import spglib
from ase.filters import ExpCellFilter
from ase.io import read, write
from ase.optimize import FIRE

from autoflow_srxn.simulation.potentials import SimulationEngine

SYMPREC = 0.01   # A — tight check for bulk
FMAX_CELL = 0.01  # eV/A
FMAX_POS = 0.005  # eV/A  (tighter for position-only final pass)
STEPS = 600

CONFIG = {
    "engine": {
        "potential": {
            "backend": "sevennet",
            "model": "7net-0",
            "device": "cpu",
            "dtype": "float32",
        }
    }
}

BULKS = [
    {
        "name": "Si",
        "original": ROOT / "structures/Si_mp149.vasp",
        "relaxed": ROOT / "structures/Si_relaxed.vasp",
    },
    {
        "name": "SiO2",
        "original": ROOT / "structures/SiO2_mp-546794.vasp",
        "relaxed": ROOT / "structures/SiO2_relaxed.vasp",
    },
]


# ---------------------------------------------------------------------------
# Symmetry helpers
# ---------------------------------------------------------------------------

def _spglib_input(atoms):
    return (atoms.get_cell()[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())


def get_sg_info(atoms, symprec=SYMPREC):
    inp = _spglib_input(atoms)
    sg_str = spglib.get_spacegroup(inp, symprec=symprec)
    dataset = spglib.get_symmetry_dataset(inp, symprec=symprec)
    if dataset is None:
        return {"number": -1, "international": "?", "crystal_system": "?", "sg_str": sg_str or "?"}
    num = dataset.number
    cs = _crystal_system(num)
    return {"number": num, "international": dataset.international,
            "crystal_system": cs, "sg_str": sg_str or f"{dataset.international} ({num})"}


def _crystal_system(sg_num):
    if sg_num <= 2:   return "triclinic"
    if sg_num <= 15:  return "monoclinic"
    if sg_num <= 74:  return "orthorhombic"
    if sg_num <= 142: return "tetragonal"
    if sg_num <= 194: return "hexagonal"  # includes trigonal
    return "cubic"


# ---------------------------------------------------------------------------
# Cell post-symmetrize
# ---------------------------------------------------------------------------

def symmetrize_cell(atoms, crystal_system):
    """Analytically enforce the ideal Bravais cell shape after numeric relaxation."""
    lengths = atoms.cell.lengths()
    angles  = atoms.cell.angles()

    if crystal_system == "cubic":
        a = lengths.mean()
        new_cell = np.diag([a, a, a])

    elif crystal_system in ("tetragonal",):
        a = (lengths[0] + lengths[1]) / 2.0   # enforce a=b
        c = lengths[2]
        new_cell = np.diag([a, a, c])

    elif crystal_system == "hexagonal":
        a = (lengths[0] + lengths[1]) / 2.0
        c = lengths[2]
        # Hexagonal cell: a along x, b at 120 degrees
        new_cell = np.array([
            [a,           0.0,  0.0],
            [-0.5 * a,    a * np.sqrt(3) / 2,  0.0],
            [0.0,          0.0,  c],
        ])

    elif crystal_system == "orthorhombic":
        # a, b, c all different but angles = 90
        new_cell = np.diag(lengths)

    else:
        # Cannot make simple assumptions; keep as-is
        print(f"    [SymCell] No analytical symmetrization for {crystal_system}, keeping numeric cell.")
        return atoms

    delta_max = np.max(np.abs(new_cell - atoms.cell[:]))
    print(f"    [SymCell] Max cell correction: {delta_max:.6f} A")
    atoms_out = atoms.copy()
    atoms_out.set_cell(new_cell, scale_atoms=True)
    atoms_out.wrap()
    return atoms_out


# ---------------------------------------------------------------------------
# Symmetry-constrained cell relaxation
# ---------------------------------------------------------------------------

def make_cell_filter(atoms, crystal_system):
    """Return appropriate ExpCellFilter for the crystal system."""
    if crystal_system == "cubic":
        # 1 DOF: isotropic volume, no shape change
        return ExpCellFilter(atoms, hydrostatic_strain=True)

    elif crystal_system in ("tetragonal", "hexagonal"):
        # 3 diagonal DOF; shear locked; a=b enforced post-hoc
        return ExpCellFilter(atoms, mask=[True, True, True, False, False, False])

    elif crystal_system == "orthorhombic":
        # 3 independent diagonal DOF
        return ExpCellFilter(atoms, mask=[True, True, True, False, False, False])

    else:
        # Monoclinic, triclinic: full cell
        return ExpCellFilter(atoms)


def relax_positions_only(atoms, calc, fmax=FMAX_POS, steps=300):
    """Run atomic-position relaxation with cell fixed."""
    atoms.calc = calc
    opt = FIRE(atoms, logfile="-")
    opt.run(fmax=fmax, steps=steps)
    return atoms


def relax_with_symmetry_constraint(original_path, calc, crystal_system,
                                   fmax_cell=FMAX_CELL, fmax_pos=FMAX_POS, steps=STEPS):
    atoms = read(str(original_path))
    print(f"  Loaded original: {original_path.name}  ({len(atoms)} atoms)")
    print(f"  Cell (initial): {atoms.cell.lengths()}")
    print(f"  Using constraint: {crystal_system}")

    # Stage 1 — cell + position relaxation with symmetry constraint
    atoms.calc = calc
    cell_filter = make_cell_filter(atoms, crystal_system)
    opt = FIRE(cell_filter, logfile="-")
    opt.run(fmax=fmax_cell, steps=steps)
    print(f"  Cell (after relax): {atoms.cell.lengths()}")

    # Stage 2 — post-symmetrize cell
    atoms = symmetrize_cell(atoms, crystal_system)
    print(f"  Cell (after symmetrize): {atoms.cell.lengths()}")

    # Stage 3 — final position-only relax with symmetrized fixed cell
    print("  Final position-only relax...")
    atoms = relax_positions_only(atoms, calc, fmax=fmax_pos)
    print(f"  E = {atoms.get_potential_energy():.6f} eV")

    return atoms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    engine = SimulationEngine(config=CONFIG)
    calc = engine.get_calculator()

    for bulk in BULKS:
        print("\n" + "=" * 70)
        print(f"  SYMMETRY CHECK: {bulk['name']}")
        print("=" * 70)

        orig = read(str(bulk["original"]))
        relaxed = read(str(bulk["relaxed"]))

        sg_orig    = get_sg_info(orig)
        sg_relaxed = get_sg_info(relaxed)

        print(f"  Original : {sg_orig['sg_str']}  ({sg_orig['crystal_system']})")
        print(f"  Relaxed  : {sg_relaxed['sg_str']}  ({sg_relaxed['crystal_system']})")

        same_sg = (sg_orig["number"] == sg_relaxed["number"])
        print(f"  Space group preserved (symprec={SYMPREC} A): {'YES' if same_sg else 'NO'}")

        if same_sg:
            print("  -> No re-relaxation needed.")
            # Still post-symmetrize the existing relaxed structure
            cs = sg_orig["crystal_system"]
            sym = symmetrize_cell(relaxed, cs)
            write(str(bulk["relaxed"]), sym, vasp5=True)
            print(f"  -> Post-symmetrized cell written to {bulk['relaxed'].name}")
        else:
            print("  -> Re-relaxing with symmetry constraint from original structure...")
            cs = sg_orig["crystal_system"]
            fixed = relax_with_symmetry_constraint(
                bulk["original"], calc, cs
            )

            # Final symmetry check
            sg_final = get_sg_info(fixed)
            print(f"  Final SG : {sg_final['sg_str']}  ({sg_final['crystal_system']})")
            ok = (sg_final["number"] == sg_orig["number"])
            print(f"  Space group restored: {'YES' if ok else 'STILL BROKEN'}")

            write(str(bulk["relaxed"]), fixed, vasp5=True)
            print(f"  -> Saved: {bulk['relaxed'].name}")

        # Summary cell parameters
        final = read(str(bulk["relaxed"]))
        sg_f = get_sg_info(final)
        print(f"  Final cell: a={final.cell.lengths()[0]:.5f} "
              f"b={final.cell.lengths()[1]:.5f} "
              f"c={final.cell.lengths()[2]:.5f} A")
        print(f"  Final SG check: {sg_f['sg_str']}")


if __name__ == "__main__":
    main()
