import os
import sys
import yaml
import numpy as np
from ase.build import bulk, add_adsorbate
from ase.constraints import FixAtoms
from ase.io import write, read
from ase.eos import EquationOfState

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.transition.engine import NEBSearcher
from autoflow_srxn.surface.surface_utils import create_slab_from_bulk
from autoflow_srxn.utils.logger_utils import setup_logger

def run_cu_diffusion_example():
    """
    Cu Diffusion Example (EMT) following core code workflow:
    1. Bulk relaxation to find equilibrium lattice constant.
    2. Slab generation using core surface utilities.
    3. Initial/Final state preparation with adatoms.
    4. NEB calculation using the core transition engine.
    """
    # 1. Workspace setup
    example_dir = os.path.dirname(os.path.abspath(__file__))
    inputs_dir = os.path.join(example_dir, "inputs")
    results_dir = os.path.join(example_dir, "results")
    for d in [inputs_dir, results_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Setup logger to both file and console
    logger = setup_logger(os.path.join(results_dir, "cu_example.log"), verbose=True)
    logger.info("Starting Cu Diffusion Example (Core Logic Compliance)")

    # 2. Engine configuration (EMT backend)
    # This structure mirrors the expected core configuration
    config = {
        "engine": {
            "potential": {
                "backend": "emt",
                "device": "cpu"
            }
        },
        "relaxation": {
            "fmax": 0.05,
            "steps": 200
        }
    }
    engine = SimulationEngine(config)
    calc = engine.get_calculator()

    # 3. Bulk relaxation to find equilibrium lattice constant for EMT
    logger.info("--- Stage 1: Bulk Lattice Optimization ---")
    volumes = []
    energies = []
    # Scan around typical EMT Cu value (~3.60 A)
    a_guesses = np.linspace(3.5, 3.7, 7)
    for a in a_guesses:
        cu_bulk = bulk('Cu', 'fcc', a=a, cubic=True)
        cu_bulk.calc = calc
        volumes.append(cu_bulk.get_volume())
        energies.append(cu_bulk.get_potential_energy())
    
    eos = EquationOfState(volumes, energies)
    v0, e0, B = eos.fit()
    # For cubic FCC cell (4 atoms), V = a^3
    a_relaxed = v0**(1/3) 
    logger.info(f"Relaxed Cu lattice constant (EMT): {a_relaxed:.4f} A")

    # 4. Surface Slab Generation
    logger.info("--- Stage 2: Slab Generation ---")
    # We use ase.build.fcc111 to ensure the Atoms object has 'adsorbate_info'
    # so that add_adsorbate(..., position='fcc') works correctly.
    from ase.build import fcc111
    slab = fcc111('Cu', size=(3, 3, 4), a=a_relaxed, vacuum=10.0)
    
    # Apply constraints: Fix bottom atoms (approx 2 layers)
    z_min = slab.positions[:, 2].min()
    fixed_indices = [atom.index for atom in slab if atom.position[2] < z_min + 3.5]
    slab.set_constraint(FixAtoms(indices=fixed_indices))
    slab.calc = calc
    
    logger.info(f"Slab atoms: {len(slab)}")
    engine.relax(slab, fmax=0.05)
    write(os.path.join(inputs_dir, "slab_relaxed.vasp"), slab)

    # 5. Build Initial and Final States for Diffusion
    logger.info("--- Stage 3: Building Initial and Final States ---")
    
    # Initial State: Cu adatom at FCC hollow site
    initial = slab.copy()
    add_adsorbate(initial, 'Cu', height=1.5, position='fcc')
    initial.calc = calc
    logger.info("Relaxing Initial State (FCC hollow)...")
    engine.relax(initial, fmax=0.05)
    write(os.path.join(inputs_dir, "initial.vasp"), initial)

    # Final State: Cu adatom at HCP hollow site (adjacent to FCC)
    final = slab.copy()
    add_adsorbate(final, 'Cu', height=1.5, position='hcp')
    final.calc = calc
    logger.info("Relaxing Final State (HCP hollow)...")
    engine.relax(final, fmax=0.05)
    write(os.path.join(inputs_dir, "final.vasp"), final)

    # 6. NEB Calculation using core searcher
    logger.info("--- Stage 4: NEB Reaction Search ---")
    neb_searcher = NEBSearcher(engine)
    
    # Parameters following config_full.yaml TS search defaults
    n_images = 5
    fmax_neb = 0.05
    steps_neb = 100
    
    images = neb_searcher.run(
        initial, final,
        n_images=n_images,
        fmax=fmax_neb,
        steps=steps_neb,
        trajectory=os.path.join(results_dir, "neb_path.extxyz")
    )

    logger.info(f"Example run complete. NEB trajectory saved to {os.path.relpath(os.path.join(results_dir, 'neb_path.extxyz'))}")

if __name__ == "__main__":
    run_cu_diffusion_example()
