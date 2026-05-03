import os
import numpy as np
from ase.io import write
from ase.mep import NEB
from ase.optimize import BFGS
from ase.mep.neb import IDPP

class TSSearcher:
    """
    Automated Transition State search using NEB (Nudged Elastic Band).
    """
    def __init__(self, engine, log_dir: str = "ts_results"):
        self.engine = engine
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def find_barrier(self, initial_atoms, final_atoms, n_images: int = 5, fmax: float = 0.05):
        """
        Runs NEB to find the barrier between initial and final states.
        """
        print(f"[TS] Starting NEB calculation with {n_images} images...", flush=True)
        
        # 1. Create images
        images = [initial_atoms.copy() for _ in range(n_images + 2)]
        for img in images:
            img.calc = self.engine.get_calculator()
        
        # 2. Linear/IDPP Interpolation
        neb = NEB(images, climb=True)
        # Use IDPP for better initial path
        idpp = IDPP(images)
        idpp.minimize(fmax=0.1)
        
        # 3. Optimize path
        # images[0] and images[-1] are fixed
        optimizer = BFGS(neb, logfile=os.path.join(self.log_dir, "neb.log"))
        
        try:
            optimizer.run(fmax=fmax, steps=100)
        except Exception as e:
            print(f"  [TS] NEB optimization warning: {e}", flush=True)

        # 4. Extract energies
        energies = [img.get_potential_energy() for img in images]
        e_initial = energies[0]
        e_ts = max(energies)
        barrier = e_ts - e_initial
        
        # Save results
        ts_index = energies.index(e_ts)
        ts_atoms = images[ts_index]
        write(os.path.join(self.log_dir, "neb_path.extxyz"), images)
        write(os.path.join(self.log_dir, "ts_structure.extxyz"), ts_atoms)
        
        print(f"  [TS] Barrier found: {barrier:.4f} eV", flush=True)
        return barrier, ts_atoms
