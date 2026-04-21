import numpy as np
import os
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from logger_utils import get_workflow_logger

from ase.vibrations import Vibrations
import shutil

class VibrationalAnalyzer:
    """
    Handles vibrational frequency analysis using ASE Vibrations (supporting PHVA) 
    or Phonopy.
    """
    def __init__(self, atoms, engine, indices=None, displacement=0.01, name="vib_analysis"):
        """
        Args:
            atoms: ASE Atoms object.
            engine: SimulationEngine (ASE-compatible).
            indices: List of atomic indices to include in the Partial Hessian. 
                     If None, all atoms are treated as active (Full Hessian).
            displacement: Finite difference displacement (A).
            name: Name for the vibration log directory.
        """
        self.atoms = atoms
        self.engine = engine
        self.indices = indices
        self.displacement = displacement
        self.name = name
        self.logger = get_workflow_logger()
        
        # Attach calculator
        self.atoms.calc = self.engine.get_calculator()
        
    def run_analysis(self):
        """
        Performs (Partial) Hessian Vibrational Analysis using ASE Vibrations.
        Returns:
            frequencies_cm_inv: List of frequencies in cm^-1.
            eigenvectors: Placeholder/None for now (ASE Vibrations handles this differently).
        """
        self.logger.info(f"  [VibAnalyzer] Starting PHVA/FHVA (active atoms: {len(self.indices) if self.indices else len(self.atoms)}).")
        
        if os.path.exists(self.name):
            shutil.rmtree(self.name)
            
        vib = Vibrations(self.atoms, indices=self.indices, name=self.name, delta=self.displacement)
        vib.run()
        
        # Get frequencies in cm^-1
        frequencies = vib.get_frequencies()
        
        # Log basic summary
        n_imag = sum(1 for f in frequencies if (hasattr(f, "imag") and f.imag != 0) or f < 0)
        self.logger.info(f"  [VibAnalyzer] Analysis complete. Total modes: {len(frequencies)}, Imaginary: {n_imag}")
        
        return frequencies

    def generate_qpoints_file(self, filename='qpoints.yaml'):
        """Runs the analysis pipeline and outputs qpoints.yaml including eigenvectors."""
        # Ensure analysis has been run
        if self.phonopy.force_constants is None:
            self.run_analysis()
            
        self.logger.info(f"  [VibAnalyzer] Generating {filename} with eigenvectors...")
        # Run q-points at Gamma point
        self.phonopy.run_qpoints([[0, 0, 0]], with_eigenvectors=True)
        
        # Phonopy's write_yaml_qpoints_phonon() usually writes to 'qpoints.yaml'
        self.phonopy.write_yaml_qpoints_phonon()
        
        # Handle custom filename if necessary
        if filename != 'qpoints.yaml':
            import shutil
            if os.path.exists('qpoints.yaml'):
                if os.path.exists(filename):
                    os.remove(filename)
                shutil.move('qpoints.yaml', filename)
            else:
                self.logger.warning("  [VibAnalyzer] Expected 'qpoints.yaml' but it was not found after Phonopy call.")
                
        self.logger.info(f"  [VibAnalyzer] {filename} successfully generated.")

from qpoint_handler import QPointParser

class ModeFollowingOptimizer:
    """
    Orchestrates iterative stability enrichment by following imaginary modes.
    Used for internal integrated workflows.
    """
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.logger = get_workflow_logger()
        self.alpha = config.get('vibrational_analysis', {}).get('perturbation', {}).get('alpha', 0.1)
        self.fmax = config.get('vibrational_analysis', {}).get('fmax', 0.01)
        self.max_iter = config.get('vibrational_analysis', {}).get('max_iter', 5)
        
    def optimize(self, atoms):
        """Iteratively removes imaginary modes."""
        current_atoms = atoms.copy()
        
        for i in range(self.max_iter):
            self.logger.info(f"--- Stability Iteration {i+1} ---")
            
            # 1. Relax structure
            self.engine.relax(current_atoms, fmax=self.fmax)
            
            # 2. Run vibrational analysis
            analyzer = VibrationalAnalyzer(current_atoms, self.engine)
            freqs, eigs = analyzer.run_analysis()
            
            # 3. Check for imaginary modes (Sorted ascending)
            neg_indices = np.where(freqs < -0.1)[0] # Threshold of 0.1 THz for numerical noise
            
            if len(neg_indices) == 0:
                self.logger.info("  [Success] No significant imaginary frequencies found. Structure is stable.")
                return current_atoms, freqs
            
            self.logger.info(f"  [Stability] Found {len(neg_indices)} imaginary modes. Most negative: {freqs[0]:.2f} THz")
            
            # 4. Displace along the most negative mode
            mode_idx = neg_indices[0]
            vector = eigs[:, mode_idx].real
            
            # Normalize and apply alpha displacement
            displacement = self.alpha * (vector / np.linalg.norm(vector))
            current_atoms.positions += displacement.reshape(-1, 3)
            
            self.logger.info(f"  [Stability] Perturbed structure along mode {mode_idx} (alpha={self.alpha}).")
            
        self.logger.warning(f"  [Warning] Stability loop reached max iterations ({self.max_iter}).")
        return current_atoms, freqs

class MultiModeFollower:
    """
    Advanced stability refinement using external qpoints.yaml.
    Supports multi-mode filtering and safety constraints.
    """
    def __init__(self, engine, config):
        self.engine = engine
        self.all_config = config
        self.config = config.get('vibrational_analysis', {})
        self.logger = get_workflow_logger()
        
        # Hierarchical parameters
        self.selection = self.config.get('selection', {})
        self.perturbation = self.config.get('perturbation', {})
        self.constraints = self.config.get('constraints', {})

    def _save_mode_trajectory(self, atoms, displacement, mode_idx, freq):
        """Generates a multi-frame extxyz file showing the mode transformation."""
        viz_config = self.config.get('visualization', {})
        if not viz_config.get('save_trajectory', True):
            return
            
        n_frames = viz_config.get('n_frames', 10)
        output_dir = viz_config.get('output_dir', 'mode_anims')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        from ase.io import write
        frames = []
        for i in range(n_frames + 1):
            frame = atoms.copy()
            # Linear interpolation from initial(0) to final(1)
            step_disp = (i / n_frames) * displacement
            frame.positions += step_disp
            
            # Attach the direction vector (eigenvector proxy) as 'forces' for visualization
            # This allows software like OVITO to show arrows for the vibrational mode.
            frame.arrays['forces'] = np.array(displacement)
            
            frame.info['mode'] = mode_idx + 1
            frame.info['frequency_thz'] = freq
            frame.info['interpolation_step'] = i
            frames.append(frame)
            
        file_path = os.path.join(output_dir, f"mode_{mode_idx+1}_refinement.extxyz")
        write(file_path, frames, format='extxyz')
        self.logger.info(f"  [Visualization] Saved mode animation to {file_path}")

    def optimize(self, atoms, **kwargs):
        """Sequential multi-mode refinement."""
        qpath = self.config.get('qpoints_path', 'qpoints.yaml')
        if not os.path.exists(qpath):
            self.logger.error(f"  [MultiMode] qpoints.yaml not found at {qpath}")
            return atoms
            
        parser = QPointParser(qpath)
        
        # STEP 1: Selection (Filtering)
        modes = parser.get_filtered_modes(
            freq_threshold=self.selection.get('freq_threshold', -0.5),
            max_modes=self.selection.get('max_modes', 3)
        )
        
        if not modes:
            self.logger.info("  [MultiMode] No modes identified for refinement based on selection criteria.")
            return atoms
            
        self.logger.info(f"  [MultiMode] Starting refinement for {len(modes)} identifying modes.")
        current_atoms = atoms.copy()
        
        for i, mode in enumerate(modes):
            freq = mode['frequency']
            self.logger.info(f"--- Refinement Mode {i+1}/{len(modes)} (Freq: {freq:.2f} THz) ---")
            
            # STEP 2: Perturbation (Scaling)
            alpha = self.perturbation.get('alpha', 0.1)
            raw_displacement = alpha * mode['eigenvector']
            
            # STEP 3: Constraints (Safety)
            max_d = self.constraints.get('max_displacement', 0.3)
            
            # Norm-scaling to maintain direction while respecting hard limit
            atom_norms = np.linalg.norm(raw_displacement, axis=1)
            max_norm_found = np.max(atom_norms)
            
            if max_norm_found > max_d:
                scale_factor = max_d / max_norm_found
                raw_displacement *= scale_factor
                self.logger.info(f"  [Constraint] Scaled displacement by {scale_factor:.3f} (Max norm: {max_norm_found:.3f} -> {max_d:.3f})")

            # NEW: Save trajectory for visual understanding
            self._save_mode_trajectory(current_atoms, raw_displacement, i, freq)

            # Apply displacement and perform relaxation
            current_atoms.positions += raw_displacement
            
            # Use provided kwargs or fallback to config fmax
            relax_kwargs = {'fmax': self.config.get('fmax', 0.05)}
            relax_kwargs.update(kwargs)
            self.engine.relax(current_atoms, **relax_kwargs)
            
            self.logger.info(f"  [MultiMode] Iteration {i+1} relaxation complete.")
            
        return current_atoms
