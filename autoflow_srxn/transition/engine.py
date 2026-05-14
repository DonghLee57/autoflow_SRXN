import os
import numpy as np
from ase import Atoms
from ase.mep import NEB
from ase.optimize import FIRE
from ase.io import write
from ase.calculators.calculator import Calculator, all_changes

from ..utils.knowledge_engine import chem_kb
from ..utils.logger_utils import get_workflow_logger

# Forward declarations or imports to avoid circularity
# from ..vibrational.vibrational_analyzer import VibrationalAnalyzer (Moved to local imports)

class _OvershotError(Exception):
    """Internal exception to halt optimization if bond breaks."""
    pass

# ---------------------------------------------------------------------------
# Calculators
# ---------------------------------------------------------------------------

class GradientFlippingCalculator(Calculator):
    """Custom ASE Calculator for climbing-image gradient-flipping."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, v_ts: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.v_ts = v_ts.ravel() / np.linalg.norm(v_ts)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        self.base_calc.calculate(atoms, properties, system_changes)
        energy = self.base_calc.results["energy"]
        g = self.base_calc.results["forces"].ravel()
        overlap = np.dot(g, self.v_ts)
        f_mod = g - 2.0 * overlap * self.v_ts
        self.results["energy"] = energy
        self.results["forces"] = f_mod.reshape(atoms.positions.shape)

class AdaptiveGradientFlippingCalculator(Calculator):
    """Gradient-flipping with dynamic climbing direction tracking."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, central_idx: int, ligand_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.c_idx = central_idx
        self.l_idx = ligand_idx
        self._last_energy = float("nan")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        r_c = atoms.positions[self.c_idx]
        r_l = atoms.positions[self.l_idx]
        v_dir = r_l - r_c
        v_hat = v_dir / np.linalg.norm(v_dir)
        n = len(atoms)
        v_ts = np.zeros((n, 3))
        v_ts[self.l_idx] = v_hat
        v_ts[self.c_idx] = -v_hat
        v_ts = v_ts.ravel()
        v_ts /= np.linalg.norm(v_ts)
        
        self.base_calc.calculate(atoms, properties, system_changes)
        energy = self.base_calc.results["energy"]
        g = self.base_calc.results["forces"].ravel()
        self._last_energy = float(energy)
        overlap = np.dot(g, v_ts)
        f_mod = g - 2.0 * overlap * v_ts
        self.results["energy"] = energy
        self.results["forces"] = f_mod.reshape(atoms.positions.shape)

# ---------------------------------------------------------------------------
# Search Engines
# ---------------------------------------------------------------------------

class NEBSearcher:
    """Nudged Elastic Band (NEB) searcher."""
    def __init__(self, engine, config=None):
        self.engine = engine
        self.config = config or {}
        self.logger = get_workflow_logger()

    def run(self, initial: Atoms, final: Atoms, n_images: int = 7, 
            fmax: float = 0.05, steps: int = 200, 
            interpolate: str = 'idpp', trajectory: str = "neb.traj"):
        self.logger.info(f"  [NEB] Starting NEB with {n_images} images")
        images = [initial.copy()]
        for _ in range(n_images):
            images.append(initial.copy())
        images.append(final.copy())
        
        neb = NEB(images, allow_shared_calculator=True)
        mic = any(initial.pbc)
        if interpolate.lower() == 'idpp':
            try:
                neb.interpolate(method='idpp', mic=mic)
            except Exception:
                from ase.mep.neb import IDPP
                idpp = IDPP(images, mic=mic)
                FIRE(idpp, logfile=None).run(fmax=0.1, steps=100)
        else:
            neb.interpolate(mic=mic)
            
        calc = self.engine.get_calculator()
        for image in images: image.calc = calc
            
        dyn = FIRE(neb, trajectory=trajectory, logfile="-")
        dyn.run(fmax=fmax, steps=steps)

        # Ensure path is saved in extxyz if requested
        if trajectory and trajectory.endswith(".extxyz"):
            write(trajectory, images)

        return images

class ARTSearcher:
    """Activation Relaxation Technique (ARTn) searcher."""
    def __init__(self, engine, config=None):
        self.engine = engine
        self.config = config or {}
        self.logger = get_workflow_logger()

    def find_saddle(self, atoms: Atoms, direction: np.ndarray = None, 
                    fmax: float = 0.05, steps: int = 200,
                    displacement_ang: float = 0.2):
        self.logger.info("  [ARTn] Starting Activation-Relaxation search...")
        work_atoms = atoms.copy()
        work_atoms.calc = self.engine.get_calculator()
        
        if direction is None:
            from ..vibrational.vibrational_analyzer import VibrationalAnalyzer
            vib = VibrationalAnalyzer(work_atoms, self.engine)
            _, eigs = vib.run_analysis()
            v_ts = eigs[:, 0]
        else:
            v_ts = direction.ravel()
            
        v_ts /= np.linalg.norm(v_ts)
        work_atoms.set_positions(work_atoms.positions + displacement_ang * v_ts.reshape(-1, 3))
        
        gf_calc = GradientFlippingCalculator(self.engine.get_calculator(), v_ts)
        work_atoms.calc = gf_calc
        FIRE(work_atoms, logfile="-").run(fmax=fmax, steps=steps)
        return work_atoms

class TSSearcher:
    """Hessian-Based Gradient Flipping Searcher."""
    def __init__(self, engine, atoms, config: dict | None = None):
        self.engine = engine
        self.atoms = atoms.copy()
        self.config = config or {}
        self.logger = get_workflow_logger()

    def find_transition_state(self, bond_indices: list[int], fmax: float = 0.05, steps: int = 200):
        c_idx, l_idx = bond_indices
        self.atoms.calc = self.engine.get_calculator()
        
        # 1. Compute Hessian to get initial mode
        from ..vibrational.vibrational_analyzer import VibrationalAnalyzer
        vib = VibrationalAnalyzer(self.atoms, self.engine)
        _, eigs = vib.run_analysis()
        
        # 2. Select mode by overlap with Si-N vector
        v_dir_3d = self.atoms.positions[l_idx] - self.atoms.positions[c_idx]
        v_hat = v_dir_3d / np.linalg.norm(v_dir_3d)
        v_dir_3n = np.zeros_like(self.atoms.positions)
        v_dir_3n[l_idx] = v_hat
        v_dir_3n[c_idx] = -v_hat
        v_dir_3n = v_dir_3n.ravel()
        
        overlaps = np.abs(eigs.T @ v_dir_3n)
        k_star = np.argmax(overlaps)
        v_ts = eigs[:, k_star]
        
        # 3. Perturb and Optimize
        disp = self.config.get("displacement_ang", 0.2)
        self.atoms.positions += disp * v_ts.reshape(-1, 3) * np.sign(np.dot(v_ts, v_dir_3n))
        
        gf_calc = AdaptiveGradientFlippingCalculator(self.engine.get_calculator(), c_idx, l_idx)
        self.atoms.calc = gf_calc
        FIRE(self.atoms, logfile="-").run(fmax=fmax, steps=steps)
        return self.atoms
