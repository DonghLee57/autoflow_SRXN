import os
import numpy as np
from ase.io import write
from ase.mep import SingleCalculatorNEB
from ase.optimize import BFGS

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
        images[-1].positions = final_atoms.positions
        images[-1].cell = final_atoms.cell
        
        # 2. Interpolation
        neb = SingleCalculatorNEB(images, climb=True)
        neb.interpolate(method='idpp', mic=True)
        
        # 3. Attach calculator to the NEB object 
        # In SingleCalculatorNEB, we attach the calculator to the images or use it directly
        calc = self.engine.get_calculator()
        for img in images:
            img.calc = calc
        
        # 4. Optimize path
        optimizer = BFGS(neb, logfile=os.path.join(self.log_dir, "neb.log"))
        
        try:
            optimizer.run(fmax=fmax, steps=100)
        except Exception as e:
            print(f"  [TS] NEB optimization warning: {e}", flush=True)

        # 5. Extract energies
        # SingleCalculatorNEB might need manual energy extraction to be safe
        energies = []
        for img in images:
            energies.append(img.get_potential_energy())
            
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
