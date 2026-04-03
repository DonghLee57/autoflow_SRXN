from ase.calculators.emt import EMT
from ase.optimize import BFGS

class PotentialFactory:
    """
    Manages calculator selection. 
    Environment Fix: Removed top-level torch import to avoid WinError 1114.
    """
    def __init__(self, model_type='emt'):
        self.model_type = model_type.lower()
        
    def get_calculator(self):
        # Strictly using EMT in this Windows environment due to torch DLL issues
        return EMT()

    def relax(self, atoms, fmax=0.05, steps=100):
        """Perform local relaxation using BFGS."""
        calc = self.get_calculator()
        atoms.calc = calc
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=fmax, steps=steps)
        return atoms.get_potential_energy()
