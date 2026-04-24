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
        
    def run_analysis(self, overwrite=False):
        """
        Performs (Partial) Hessian Vibrational Analysis using ASE Vibrations.
        Args:
            overwrite: If True, delete any existing cache before running.
                       Set False (default) to resume from a partially-completed cache.
        Returns:
            freqs_thz: List of frequencies in THz (negative for imaginary).
            eigs: Eigenvectors at Gamma point.
        """
        self.logger.info(f"  [VibAnalyzer] Starting PHVA/FHVA (active atoms: {len(self.indices) if self.indices else len(self.atoms)}).")

        if overwrite and os.path.exists(self.name):
            shutil.rmtree(self.name)

        vib = Vibrations(self.atoms, indices=self.indices, name=self.name, delta=self.displacement)
        vib.run()

        # Get raw frequencies
        freqs_raw = vib.get_frequencies()
        freqs_thz = []
        for f in freqs_raw:
            cf = complex(f)
            if abs(cf.imag) > abs(cf.real):
                freqs_thz.append(-abs(cf.imag) / 33.3564)
            else:
                freqs_thz.append(cf.real / 33.3564)

        # Get raw modes
        vib_data = vib.get_vibrations()
        modes = vib_data.get_modes()  # Shape: (num_modes, num_active, 3)
        
        N_total = len(self.atoms)
        num_modes = modes.shape[0]
        eigs = np.zeros((3 * N_total, num_modes))
        
        indices = self.indices if self.indices is not None else list(range(N_total))
        
        for i in range(num_modes):
            mode_3d = np.zeros((N_total, 3))
            mode_3d[indices] = modes[i]
            eigs[:, i] = mode_3d.reshape(-1)

        # Log basic summary
        n_imag = sum(1 for f in freqs_thz if f < -0.01)
        self.logger.info(f"  [VibAnalyzer] Analysis complete. Total modes: {len(freqs_thz)}, Imaginary: {n_imag}")

        self._freqs_thz = np.array(freqs_thz)
        self._eigs = eigs
        return self._freqs_thz, self._eigs

    def generate_qpoints_file(self, filename='qpoints.yaml'):
        """Write a phonopy-compatible qpoints.yaml at *filename*.

        Eigenvector convention (matches phonopy):
            e_{k,α} = u_{k,α} · sqrt(m_k)   (mass-weighted, normalised so Σ|e|²=1)
        where u_{k,α} is the Cartesian displacement from ASE Vibrations.get_modes().

        An additional ``masses`` key (AutoFlow-SRXN extension) is written so that
        QPointParser can back-convert eigenvectors to displacements for mode-following
        without needing the structure file at read time.
        """
        if not hasattr(self, '_freqs_thz') or not hasattr(self, '_eigs'):
            self.run_analysis()

        import yaml
        parent = os.path.dirname(os.path.abspath(filename))
        if parent:
            os.makedirs(parent, exist_ok=True)

        n_total_atoms = len(self.atoms)
        masses    = self.atoms.get_masses()          # (N,)
        mass_sqrt = np.sqrt(masses)                  # (N,)
        num_modes = len(self._freqs_thz)

        bands = []
        for i in range(num_modes):
            freq = float(self._freqs_thz[i])

            if self._eigs is not None and self._eigs.shape[1] > i:
                vec_3d = self._eigs[:, i].reshape(n_total_atoms, 3)

                # u_{k,α}  →  e_{k,α} = u_{k,α} · sqrt(m_k)
                # then normalise: Σ_{k,α} |e_{k,α}|² = 1
                e_vec = vec_3d * mass_sqrt[:, np.newaxis]   # (N, 3)
                norm  = np.linalg.norm(e_vec)
                if norm > 1e-10:
                    e_vec = e_vec / norm

                # phonopy format: nested list.
                # [ [ [ux,ix], [uy,iy], [uz,iz] ], [ [ux,ix], ... ], ... ]
                eig_list = []
                for k in range(n_total_atoms):
                    atom_vec = []
                    for j in range(3):
                        atom_vec.append([float(e_vec[k, j]), 0.0])
                    eig_list.append(atom_vec)
            else:
                eig_list = []

            bands.append({'frequency': freq, 'eigenvector': eig_list})

        # Reciprocal lattice (Phonopy convention: 1/Angstrom, without 2pi)
        recip_cell = self.atoms.cell.reciprocal().tolist()

        data = {
            'nqpoint': 1,
            'natom':   n_total_atoms,
            'reciprocal_lattice': recip_cell,
            # AutoFlow-SRXN extension: atom masses (amu) for displacement back-conversion.
            # Not part of the standard phonopy spec; ignored by unaware readers.
            'masses':  [float(m) for m in masses],
            'phonon':  [{
                'q-position': [0.0, 0.0, 0.0],
                'weight':     1,
                'band':       bands,
            }],
        }

        with open(filename, 'w', encoding='utf-8') as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

        self.logger.info(f"  [VibAnalyzer] {filename} written ({num_modes} modes, {n_total_atoms} atoms).")

def calculate_thermo(freqs_thz, T):
    """Calculates vibrational free energy and ZPE given THz frequencies."""
    from thermo_engine import ThermoCalculator, eV_to_J_mol
    thermo = ThermoCalculator(freqs_thz)
    G_vib_J = thermo.calculate_vib_free_energy(T)
    ZPE_J   = thermo.calculate_zpe()
    return float(G_vib_J / eV_to_J_mol), float(ZPE_J / eV_to_J_mol)

def build_phva_active_indices(atoms, n_adsorbate, cutoff_angstrom):
    from ase.neighborlist import neighbor_list
    n_total = len(atoms)
    ads_set = set(range(n_total - n_adsorbate, n_total))
    i_arr, j_arr = neighbor_list('ij', atoms, cutoff_angstrom)
    slab_neighbors = {
        int(j_arr[k]) for k, i in enumerate(i_arr) if i in ads_set and j_arr[k] not in ads_set
    }
    return sorted(ads_set | slab_neighbors)


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
        _mr = config.get('analysis', {}).get('vibrational', {}).get('mode_refinement', {})
        self.alpha    = _mr.get('perturbation_alpha', 0.1)
        self.fmax     = _mr.get('fmax', 0.001)
        self.max_iter = _mr.get('max_iter', 10)
        
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
        # Navigate to analysis.vibrational in the full config tree
        self.vib_config = config.get('analysis', {}).get('vibrational', {})
        self.config     = self.vib_config.get('mode_refinement', {})
        self.logger     = get_workflow_logger()

    def _save_mode_trajectory(self, atoms, displacement, mode_idx, freq):
        """Generates a multi-frame extxyz file showing the mode transformation."""
        viz_config = self.vib_config.get('visualization', {})
        if not viz_config.get('enabled', True):
            return

        n_frames   = viz_config.get('n_frames', 10)
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
        # `or` handles both missing key and explicit `null` (Python None) in config
        qpath = self.vib_config.get('qpoints_file') or 'qpoints.yaml'
        if not os.path.exists(qpath):
            self.logger.error(f"  [MultiMode] qpoints file not found at '{qpath}'")
            return atoms

        parser = QPointParser(qpath)

        # STEP 1: Selection — filter imaginary modes
        modes = parser.get_filtered_modes(
            freq_threshold=self.config.get('freq_threshold_thz', -0.5),
            max_modes=self.config.get('max_modes', 3),
        )

        if not modes:
            self.logger.info("  [MultiMode] No modes meet the selection criteria.")
            return atoms

        self.logger.info(f"  [MultiMode] Starting refinement for {len(modes)} mode(s).")
        current_atoms = atoms.copy()

        for i, mode in enumerate(modes):
            freq = mode['frequency']
            self.logger.info(f"--- Refinement Mode {i+1}/{len(modes)} (Freq: {freq:.2f} THz) ---")

            # STEP 2: Perturbation
            alpha           = self.config.get('perturbation_alpha', 0.1)
            raw_displacement = alpha * mode['eigenvector']

            # STEP 3: Constraints
            max_d       = self.config.get('max_displacement', 0.3)
            atom_norms  = np.linalg.norm(raw_displacement, axis=1)
            max_norm    = np.max(atom_norms)
            if max_norm > max_d:
                scale           = max_d / max_norm
                raw_displacement *= scale
                self.logger.info(f"  [Constraint] Scaled displacement by {scale:.3f} "
                                 f"(max norm {max_norm:.3f} → {max_d:.3f} Å)")

            self._save_mode_trajectory(current_atoms, raw_displacement, i, freq)

            # Apply displacement and relax
            current_atoms.positions += raw_displacement
            relax_kwargs = {'fmax': self.config.get('fmax', 0.001)}
            relax_kwargs.update(kwargs)
            self.engine.relax(current_atoms, **relax_kwargs)

            self.logger.info(f"  [MultiMode] Mode {i+1} relaxation complete.")

        return current_atoms
