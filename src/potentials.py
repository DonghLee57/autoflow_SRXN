import os
import sys
import numpy as np
from ase.optimize import BFGS, FIRE
from ase.calculators.emt import EMT

class SimulationEngine:
    """
    High-level engine for structural relaxation and energy extraction.
    Supports Multiple-Interatomic Potentials (MLIPs) like MACE.
    """
    def __init__(self, model_type='mace', device='cpu'):
        self.model_type = model_type.lower()
        self.device = device
        self._calculator = None

    def get_calculator(self):
        """
        Lazy-loads the calculator to prevent DLL issues on initialization.
        """
        if self._calculator is not None:
            return self._calculator

        if self.model_type == 'emt':
            self._calculator = EMT()
        elif self.model_type in ['mace', 'mace0']:
            try:
                # Lazy import to avoid Windows DLL load issues (WinError 1114)
                from mace.calculators import mace_mp
                # "mace0" or "mace" defaults to the foundation MACE-MP models
                self._calculator = mace_mp(model="medium", device=self.device, default_dtype="float32")
                print(f"  [SimulationEngine] Loaded MACE-MP calculator on {self.device}.")
            except ImportError:
                print("  [Warning] MACE-torch not found. Falling back to EMT.")
                self._calculator = EMT()
            except Exception as e:
                print(f"  [Error] Failed to load MACE: {e}. Falling back to EMT.")
                self._calculator = EMT()
        else:
            self._calculator = EMT()
            
        return self._calculator

    def relax(self, atoms, fmax=0.05, steps=200, optimizer='BFGS', verbose=True):
        """
        Performs structural relaxation.
        Units: Energy (eV), Forces (eV/A)
        """
        calc = self.get_calculator()
        atoms.calc = calc
        
        opt_class = BFGS if optimizer.upper() == 'BFGS' else FIRE
        logfile = sys.stdout if verbose else None
        
        dyn = opt_class(atoms, logfile=logfile)
        dyn.run(fmax=fmax, steps=steps)
        
        e_final = atoms.get_potential_energy()
        if verbose:
            print(f"  [Relaxation] Completed. Final Energy: {e_final:.4f} eV")
        return e_final

    def get_energy(self, atoms):
        """
        Extracts the potential energy of the current configuration.
        """
        calc = self.get_calculator()
        atoms.calc = calc
        return atoms.get_potential_energy()

    def get_forces(self, atoms):
        """
        Extracts the atomic forces.
        """
        calc = self.get_calculator()
        atoms.calc = calc
        return atoms.get_forces()
