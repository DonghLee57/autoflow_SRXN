import os
import numpy as np
from ase import Atoms
from ase.mep import NEB
from ase.optimize import FIRE
from ase.io import write
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

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
            fmax: float = 0.05, steps: int = 100,
            interpolate: str = 'idpp', climbing_image: bool = False,
            trajectory: str = "neb.extxyz"):
        self.logger.info(f"  [NEB] Starting NEB with {n_images} images")
        images = [initial.copy()]
        for _ in range(n_images):
            images.append(initial.copy())
        images.append(final.copy())
        
        neb = NEB(images, allow_shared_calculator=True, climb=climbing_image)
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
            
        # 1. Trajectory and Output Handling
        results_dir = os.path.dirname(trajectory) if trajectory else "."
        if trajectory and not trajectory.endswith(".extxyz"):
            trajectory += ".extxyz"
        # Remove stale trajectory file so append-writes start clean
        if trajectory and os.path.exists(trajectory):
            os.remove(trajectory)

        def _snapshot_from_cache(neb):
            """Build a list of Atoms with SinglePointCalculator from NEB's
            internal cache (neb.energies / neb.real_forces).
            Zero new calculator calls — avoids shared-calculator cache misses."""
            energies   = getattr(neb, 'energies',    None)   # shape (nimages,)
            real_forces = getattr(neb, 'real_forces', None)  # shape (nimages, natoms, 3)
            snapshots = []
            for i, img in enumerate(neb.images):
                a = img.copy()
                kwargs = {}
                if energies is not None:
                    e = energies[i]
                    if np.isfinite(e):
                        kwargs['energy'] = float(e)
                if real_forces is not None:
                    kwargs['forces'] = real_forces[i].copy()
                if kwargs:
                    a.calc = SinglePointCalculator(a, **kwargs)
                snapshots.append(a)
            return snapshots

        # Callback: append one NEB snapshot per step to the trajectory file.
        # Uses cached neb.energies / neb.real_forces — zero redundant evals.
        class NEBLogger:
            def __init__(self, neb, traj_path, log_interval):
                self.neb = neb
                self.traj_path = traj_path
                self.log_interval = log_interval
                self.count = 0
            def __call__(self):
                if self.traj_path and (self.count % self.log_interval == 0):
                    snapshots = _snapshot_from_cache(self.neb)
                    write(self.traj_path, snapshots, append=True)
                self.count += 1

        log_interval = max(1, steps // 20)   # ~20 snapshots over the full run
        neb_logger = NEBLogger(neb, trajectory, log_interval)

        dyn = FIRE(neb, trajectory=None, logfile="-")
        dyn.attach(neb_logger, interval=1)
        dyn.run(fmax=fmax, steps=steps)

        # 2. Append final converged path (always saved regardless of interval)
        if trajectory:
            self.logger.info(f"  [NEB] Saving converged path to {trajectory}")
            write(trajectory, _snapshot_from_cache(neb), append=True)

        # 3. Visualization — read energies from cache, not from calculator
        cached_energies = list(getattr(neb, 'energies', [None] * len(images)))
        self.plot_profile(images, results_dir, cached_energies=cached_energies)

        return images

    def plot_profile(self, images, output_dir, cached_energies=None):
        """Generates energy profile plots (neb_profile.png)."""
        try:
            import matplotlib.pyplot as plt
            if cached_energies is not None and all(
                    e is not None and np.isfinite(e) for e in cached_energies):
                energies = [float(e) for e in cached_energies]
            else:
                energies = [img.get_potential_energy() for img in images]
            rel_energies = [e - min(energies) for e in energies]
            barrier = max(energies) - energies[0]
            
            # ── Final Profile ──────────────────────────────────────────────────
            plt.figure(figsize=(8, 5))
            x = range(len(images))
            plt.plot(x, rel_energies, 'ro-', linewidth=2, markersize=8)
            plt.xlabel("Configurations", fontsize=12)
            plt.ylabel("Relative Energy (eV)", fontsize=12)
            plt.title("NEB Energy Profile", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Label the barrier
            plt.text(len(images)//2, max(rel_energies) * 1.05, f"Barrier: {barrier:.2f} eV", 
                     ha='center', fontsize=12, color='blue', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "neb_profile.png"), dpi=300)
            plt.savefig(os.path.join(output_dir, "neb_opt_profile.png"), dpi=300) # Duplicate for now or use history
            plt.close()
            self.logger.info(f"  [NEB] Energy profiles saved to {output_dir}")
        except Exception as e:
            self.logger.warning(f"  [NEB] Could not generate plots: {e}")

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
        
        artn_freqs, artn_eigs = None, None
        if direction is None:
            from ..vibrational.vibrational_analyzer import VibrationalAnalyzer
            vib = VibrationalAnalyzer(work_atoms, self.engine)
            artn_freqs, artn_eigs = vib.run_analysis()
            v_ts = artn_eigs[:, 0]
        else:
            v_ts = direction.ravel()

        v_ts /= np.linalg.norm(v_ts)
        work_atoms.set_positions(work_atoms.positions + displacement_ang * v_ts.reshape(-1, 3))

        gf_calc = GradientFlippingCalculator(self.engine.get_calculator(), v_ts)
        work_atoms.calc = gf_calc
        FIRE(work_atoms, logfile="-").run(fmax=fmax, steps=steps)
        return work_atoms, artn_freqs, artn_eigs

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
