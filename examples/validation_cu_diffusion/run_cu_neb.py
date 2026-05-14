import numpy as np
import os
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.metadynamics.ts_engine import NEBSearcher, ARTSearcher
from ase.io import write

def run_validation():
    # 1. Setup Simulation Engine (EMT)
    config = {
        "engine": {
            "potential": {
                "backend": "emt"
            },
            "relaxation": {
                "fmax": 0.01,
                "steps": 200
            }
        }
    }
    engine = SimulationEngine(config)
    
    print("--- Step 1: Building Cu(100) Surface ---")
    # 3x3x2 Cu(100) slab
    slab = fcc100('Cu', size=(3, 3, 2), vacuum=10.0)
    # Fix the bottom layer
    slab.set_constraint(FixAtoms(indices=[atom.index for atom in slab if atom.tag == 2]))
    
    # 2. Define Initial State (Hollow site)
    initial = slab.copy()
    add_adsorbate(initial, 'Cu', height=1.5, position='hollow')
    initial.calc = engine.get_calculator()
    engine.relax(initial, fmax=0.01)
    e_initial = initial.get_potential_energy()
    print(f"Initial State Energy: {e_initial:.4f} eV")
    
    # 3. Define Final State (Adjacent Hollow site)
    # Cu(100) hollow sites are at (0.5, 0.5) scaled coordinates relative to the unit cell
    # Next hollow site is 1 unit cell over in x or y.
    final = slab.copy()
    # Manual placement to adjacent hollow
    cell = slab.get_cell()
    hollow_pos = initial[-1].position + cell[0]/3.0 # Move by 1 unit in 3x3 supercell
    add_adsorbate(final, 'Cu', height=1.5, position=(hollow_pos[0], hollow_pos[1]))
    final.calc = engine.get_calculator()
    engine.relax(final, fmax=0.01)
    e_final = final.get_potential_energy()
    print(f"Final State Energy: {e_final:.4f} eV")

    # 4. Run NEB Validation
    print("\n--- Step 2: Running NEB for Hopping Mechanism ---")
    neb_searcher = NEBSearcher(engine)
    # Use 7 images to get a good resolution of the barrier
    images = neb_searcher.run(initial, final, n_images=5, fmax=0.05, steps=100, interpolate='idpp', trajectory='cu_diffusion_neb.traj')
    
    energies = [img.get_potential_energy() for img in images]
    barrier_neb = max(energies) - e_initial
    print(f"NEB Calculated Barrier: {barrier_neb:.4f} eV")
    
    # 5. Run ARTn Validation
    print("\n--- Step 3: Running ARTn starting from Initial Minimum ---")
    art_searcher = ARTSearcher(engine)
    # Perturb toward the bridge site (midpoint between hollows)
    direction = np.zeros((len(initial), 3))
    direction[-1] = [1.0, 0.0, 0.0] # Move adatom along x
    
    ts_structure = art_engine = art_searcher.find_saddle(initial, direction=direction, fmax=0.05, steps=100, displacement_ang=0.3)
    e_ts_art = ts_structure.get_potential_energy()
    barrier_art = e_ts_art - e_initial
    print(f"ARTn Calculated Barrier: {barrier_art:.4f} eV")
    
    # Write results
    write('initial.vasp', initial)
    write('final.vasp', final)
    write('ts_art.vasp', ts_structure)
    
    print("\n--- Summary ---")
    print(f"NEB Barrier: {barrier_neb:.4f} eV")
    print(f"ARTn Barrier: {barrier_art:.4f} eV")
    print("Literature (EMT/Cu100 Hopping): ~0.5 eV range (varies with slab size/layers)")

if __name__ == "__main__":
    run_validation()
