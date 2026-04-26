import numpy as np
import os
from ase.io import write
from logger_utils import get_workflow_logger

from ase.vibrations import Vibrations
import shutil
# NOTE: ase.optimize.dimer is intentionally excluded due to environment import issues.
# TSSearcher uses a self-contained Hessian-based gradient flipping strategy instead.
from ase.optimize import FIRE

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
                     If None, it will be automatically determined from config.
            displacement: Finite difference displacement (A).
            name: Name for the vibration log directory.
        """
        self.atoms = atoms
        self.engine = engine
        self._indices = indices
        self.displacement = displacement
        self.name = name
        self.logger = get_workflow_logger()
        
        # Attach calculator
        self.atoms.calc = self.engine.get_calculator()
        self._freqs_thz = None
        self._eigs = None

    @property
    def indices(self):
        """
        Returns the active indices for the Hessian.
        If not explicitly set, resolves them from the configuration.
        """
        if self._indices is not None:
            return self._indices
        
        # Automatic resolution logic requested by USER
        config = self.engine.all_config
        vib_cfg = config.get('analysis', {}).get('vibrational', {})
        radius = vib_cfg.get('phva_radius_ang')
        
        # Check for frozen_z_ang in vibrational or surface_prep.equilibration
        frozen_z = vib_cfg.get('frozen_z_ang')
        if frozen_z is None:
            frozen_z = config.get('surface_prep', {}).get('equilibration', {}).get('frozen_z_ang')
            
        if radius is None and frozen_z is None:
            # Default: Full Hessian
            return None
            
        indices_set = set(range(len(self.atoms)))
        
        # 1. Exclude frozen atoms by height if requested
        if frozen_z is not None:
            z_min = self.atoms.positions[:, 2].min()
            mask = self.atoms.positions[:, 2] >= z_min + frozen_z
            indices_set &= set(np.where(mask)[0])
            
        # 2. If radius is set, focus on adsorbate + neighbors
        if radius is not None:
            from surface_utils import identify_protectors
            _, ads_idx = identify_protectors(self.atoms, config)
            
            if len(ads_idx) > 0:
                from ase.neighborlist import neighbor_list
                # Build neighborhood around adsorbate
                i_list, j_list = neighbor_list('ij', self.atoms, radius)
                neighbor_set = set()
                for a_idx in ads_idx:
                    neighbor_set.update(j_list[i_list == a_idx])
                
                # Combine adsorbate + its neighbors within radius, 
                # then intersect with non-frozen indices
                phva_set = set(ads_idx) | neighbor_set
                indices_set &= phva_set
            else:
                self.logger.warning("  [VibAnalyzer] PHVA radius requested but no adsorbate found. Using height-based selection.")
        
        # Convert to sorted list
        return sorted(list(indices_set))

    @indices.setter
    def indices(self, value):
        self._indices = value

    @property
    def min_freq(self):
        """Returns the minimum frequency in THz."""
        if self._freqs_thz is None:
            return None
        return float(np.min(self._freqs_thz))

    @property
    def modes(self):
        """Returns the list of modes (freq and eigenvector) for refinement."""
        if self._freqs_thz is None:
            return []
        
        modes_list = []
        n_atoms = len(self.atoms)
        mass_sqrt = np.sqrt(self.atoms.get_masses())
        
        for i, freq in enumerate(self._freqs_thz):
            u_vec = self._eigs[:, i].reshape(n_atoms, 3)
            # Standardizing to mass-weighted eigenvector (e = u * sqrt(m))
            e_vec = u_vec * mass_sqrt[:, np.newaxis]
            norm = np.linalg.norm(e_vec)
            if norm > 1e-10:
                e_vec /= norm
                
            modes_list.append({
                'frequency': float(freq),
                'eigenvector': e_vec.tolist()
            })
        return modes_list
        
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
            self._robust_rmtree(self.name)

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

        # Clean up temporary displacement data directory
        if os.path.exists(self.name):
            self._robust_rmtree(self.name)

        # Default qpoints generation requested by USER
        # Save in the same parent directory as the vib cache
        parent_dir = os.path.dirname(self.name) if os.path.dirname(self.name) else "."
        self.generate_qpoints_file(os.path.join(parent_dir, 'qpoints.yaml'))

        return self._freqs_thz, self._eigs

    def _robust_rmtree(self, path):
        """Robustly remove a directory, retrying on failure (common on Windows)."""
        import time
        for i in range(3):
            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
                return
            except Exception:
                time.sleep(0.5)
        # Last resort: ignore errors
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    def generate_qpoints_file(self, filename='qpoints.yaml'):
        """Write a phonopy-compatible qpoints.yaml at *filename* using 
        manual formatting to match Phonopy's exact style.
        """
        if self._freqs_thz is None or self._eigs is None:
            self.run_analysis()

        n_total_atoms = len(self.atoms)
        masses    = self.atoms.get_masses()
        mass_sqrt = np.sqrt(masses)
        num_modes = len(self._freqs_thz)
        lattice   = self.atoms.cell

        with open(filename, 'w', encoding='utf-8') as w:
            # Header
            w.write("nqpoint: %-7d\n" % 1)
            w.write("natom:   %-7d\n" % n_total_atoms)

            # Reciprocal lattice
            if lattice.volume > 1e-6:
                rec_lattice = np.linalg.inv(lattice)  # column vectors
                w.write("reciprocal_lattice:\n")
                for vec, axis in zip(rec_lattice.T, ("a*", "b*", "c*"), strict=True):
                    w.write("- [ %12.8f, %12.8f, %12.8f ] # %2s\n" % (tuple(vec) + (axis,)))
            
            w.write("phonon:\n")
            # Q-point (Gamma only)
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % (0.0, 0.0, 0.0))
            w.write("  band:\n")

            for j in range(num_modes):
                freq = float(self._freqs_thz[j])
                w.write("  - # %d\n" % (j + 1))
                w.write("    frequency: %15.10f\n" % freq)
                
                if self._eigs is not None:
                    # Reconstruct mass-weighted eigenvector (Phonopy convention)
                    # e = u * sqrt(m)
                    u_vec = self._eigs[:, j].reshape(n_total_atoms, 3)
                    e_vec = u_vec * mass_sqrt[:, np.newaxis]
                    norm  = np.linalg.norm(e_vec)
                    if norm > 1e-10:
                        e_vec = e_vec / norm

                    w.write("    eigenvector:\n")
                    for k in range(n_total_atoms):
                        w.write("    - # atom %d\n" % (k + 1))
                        for ll in (0, 1, 2):
                            # [real, imag] pair
                            w.write("      - [ %17.14f, %17.14f ]\n" % (float(e_vec[k, ll]), 0.0))
            w.write("\n")

        self.logger.info(f"  [VibAnalyzer] {os.path.relpath(filename)} written in Phonopy-style ({num_modes} modes).")

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

class MultiModeFollower:
    """
    Advanced stability refinement using linear combination of imaginary modes.
    """
    def __init__(self, engine, config):
        self.engine = engine
        self.all_config = config
        # Navigate to analysis.vibrational in the full config tree
        self.vib_config = config.get('analysis', {}).get('vibrational', {})
        self.config     = self.vib_config.get('mode_refinement', {})
        self.viz_config = self.vib_config.get('visualization', {})
        self.logger     = get_workflow_logger()

    def optimize(self, atoms, modes=None, **kwargs):
        """
        Refines structure using linear combination of unstable modes.
        
        Args:
            atoms: ASE Atoms object.
            modes: Optional list of modes (dicts with 'frequency' and 'eigenvector').
            **kwargs: Passed to engine.relax (e.g. fmax, steps).
        """
        # 1. Selection — filter imaginary modes
        if modes is None:
            qpath = self.vib_config.get('qpoints_file') or 'qpoints.yaml'
            if not os.path.exists(qpath):
                self.logger.error(f"  [MultiMode] qpoints file not found at '{qpath}'")
                return atoms
            parser = QPointParser(qpath)
            modes = [b for phon in parser.data['phonon'] for b in phon['band']]

        threshold = self.config.get('freq_threshold_thz', -0.1)
        max_modes = self.config.get('max_modes', 3)
        target_modes = [m for m in modes if m['frequency'] < threshold][:max_modes]

        if not target_modes:
            self.logger.info("  [MultiMode] No imaginary modes found below threshold. Skipping.")
            return atoms

        # 2. Combine displacements
        n_atoms = len(atoms)
        masses  = atoms.get_masses()
        m_sqrt  = np.sqrt(masses)
        
        # Resultant displacement vector in Cartesian coordinates
        total_u = np.zeros((n_atoms, 3))
        
        for mode in target_modes:
            # eigenvector is usually [real, imag] pairs. 
            e_raw = np.array(mode['eigenvector'])
            if e_raw.size == 2 * n_atoms * 3:
                # Handle both [[r,i],...] and [r,i,r,i,...] formats
                e_vec = e_raw.reshape(-1, 2)[:, 0].reshape(n_atoms, 3)
            else:
                e_vec = e_raw.reshape(n_atoms, 3)
            # u = e / sqrt(m)
            u_vec = e_vec / m_sqrt[:, np.newaxis]
            total_u += u_vec
            
        # 3. Apply perturbation scale (alpha)
        alpha = self.config.get('perturbation_alpha', 0.1)
        total_u *= alpha
        
        # 4. Enforce max_displacement constraint globally
        max_d = np.linalg.norm(total_u, axis=1).max()
        limit = self.config.get('max_displacement', 0.5)
        if max_d > limit:
            scale = limit / max_d
            self.logger.warning(f"  [MultiMode] Combined max displacement {max_d:.3f} > limit {limit:.3f}. "
                               f"Scaling entire vector by {scale:.3f}")
            total_u *= scale
            
        # 5. Backup initial positions for interpolation
        initial_atoms = atoms.copy()
        
        # 6. Perturb and Relax
        current_atoms = atoms.copy()
        current_atoms.set_positions(current_atoms.get_positions() + total_u)
        
        self.logger.info(f"  [MultiMode] Combined {len(target_modes)} modes. Starting single relaxation...")
        
        # Ensure 'modes' is NOT in kwargs when calling relax
        relax_kwargs = kwargs.copy()
        if 'modes' in relax_kwargs:
            relax_kwargs.pop('modes')
        if 'trajectory' in relax_kwargs:
            relax_kwargs.pop('trajectory')
            
        self.engine.relax(current_atoms, **relax_kwargs)
        
        # 7. Visualization: Interpolated Animation
        if self.viz_config.get('enabled', False):
            n_frames = self.viz_config.get('n_frames', 10)
            traj_name = self.viz_config.get('output_traj', 'relaxation.extxyz')
            self.logger.info(f"  [MultiMode] Generating {n_frames} interpolation frames -> {os.path.relpath(traj_name)}")
            
            final_pos = current_atoms.get_positions()
            start_pos = initial_atoms.get_positions()
            
            # Prepare the 'forces' array to store displacement vectors for visualization
            # This allows tools like OVITO/VESTA to show arrows for the mode direction.
            viz_forces = total_u.copy()
            
            animation = []
            for i in range(n_frames):
                # Linear interpolation: t from 0 to 1
                t = i / (n_frames - 1) if n_frames > 1 else 1.0
                frame = initial_atoms.copy()
                frame.set_positions((1.0 - t) * start_pos + t * final_pos)
                
                # Store displacement vectors in the 'forces' array
                # In extxyz, this maps to FX, FY, FZ columns
                from ase.calculators.singlepoint import SinglePointCalculator
                frame.calc = SinglePointCalculator(frame, forces=viz_forces)
                
                animation.append(frame)
                
            write(traj_name, animation)
        
        return current_atoms

# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------

class _OvershotError(Exception):
    """Raised by the FIRE observer when the tracked bond exceeds max_bond_dist.

    args: (bond_dist_A: float, energy_eV: float)
    """


# ---------------------------------------------------------------------------
# Gradient Flipping Calculator
# ---------------------------------------------------------------------------

from ase.calculators.calculator import Calculator, all_changes


class GradientFlippingCalculator(Calculator):
    """
    Custom ASE Calculator that modifies the force vector returned by an
    underlying calculator according to the climbing-image gradient-flipping rule:

    Physics / Algorithm
    -------------------
    Given the true force vector  **g** = -∇E  and the unit eigenvector **v_TS**
    corresponding to the target transition-mode, the modified force is:

        **f_mod** = **g** - 2 (g · v_TS) v_TS          [units: eV/Å]

    This inverts the force component along **v_TS** so that the FIRE optimizer
    *climbs* the PES along the TS direction while relaxing in all perpendicular
    directions, driving the structure to a 1st-order saddle point.

    Reference: Henkelman & Jónsson, J. Chem. Phys. 111, 7010 (1999).
    DOI: 10.1063/1.480097
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, base_calc, v_ts: np.ndarray, **kwargs):
        """
        Args:
            base_calc: Any ASE-compatible Calculator (e.g. MACE).
            v_ts:      Normalised 3N eigenvector of the target TS mode  (units: dimensionless).
        """
        super().__init__(**kwargs)
        self.base_calc = base_calc
        # Flatten and normalise defensively
        self.v_ts = v_ts.ravel() / np.linalg.norm(v_ts)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        # Delegate to the underlying calculator directly — do NOT set atoms.calc,
        # because that would overwrite this wrapper on the atoms object and break
        # gradient flipping on every step after the first.
        self.base_calc.calculate(atoms, properties, system_changes)

        energy = self.base_calc.results['energy']        # units: eV
        g      = self.base_calc.results['forces'].ravel()  # units: eV/Å, shape (3N,)

        # Gradient-flipping: f_mod = g - 2(g · v_TS) v_TS
        overlap   = np.dot(g, self.v_ts)             # scalar projection  [eV/Å]
        f_mod     = g - 2.0 * overlap * self.v_ts    # modified force     [eV/Å]

        self.results['energy'] = energy
        self.results['forces'] = f_mod.reshape(atoms.positions.shape)


# ---------------------------------------------------------------------------
# Adaptive Gradient Flipping Calculator
# ---------------------------------------------------------------------------

class AdaptiveGradientFlippingCalculator(Calculator):
    """
    Gradient-flipping calculator whose climbing direction is recomputed at
    every force evaluation as the current unit vector along the bond being
    broken (central_idx → ligand_idx).

    Motivation
    ----------
    A fixed v_TS (from the Hessian at the initial geometry) stops tracking
    the actual bond-stretch direction once the structure deforms.  Keeping
    v_TS aligned with the live bond vector ensures:

    • The climbing force always acts along the correct coordinate.
    • After the structure crosses the saddle (Si-N force reverses sign),
      the gradient-flip automatically creates a restoring force that
      drives FIRE to converge *at* the TS rather than overshooting to
      the fully-dissociated limit.

    Only the two bond atoms have their forces modified; all other atoms
    experience unmodified MACE forces and relax normally.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, base_calc, central_idx: int, ligand_idx: int, **kwargs):
        """
        Args:
            base_calc:    Any ASE-compatible Calculator (e.g. MACE).
            central_idx:  Index of the central atom (Si).
            ligand_idx:   Index of the ligand atom (N).
        """
        super().__init__(**kwargs)
        self.base_calc    = base_calc
        self.c_idx        = central_idx
        self.l_idx        = ligand_idx
        self._last_energy = float('nan')  # updated every calculate(); read by FIRE observer

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        # --- Dynamic climbing direction: current Si→N unit vector in 3N space ---
        r_c = atoms.positions[self.c_idx]
        r_l = atoms.positions[self.l_idx]
        v_dir   = r_l - r_c                               # units: Å
        v_hat   = v_dir / np.linalg.norm(v_dir)           # dimensionless unit vector

        n = len(atoms)
        v_ts = np.zeros((n, 3))
        v_ts[self.l_idx] =  v_hat   # ligand atom moves away from central
        v_ts[self.c_idx] = -v_hat   # central atom moves away from ligand
        v_ts = v_ts.ravel()
        v_ts /= np.linalg.norm(v_ts)  # norm = 1/sqrt(2) * sqrt(2) = 1

        # --- Base forces (do NOT set atoms.calc — see GradientFlippingCalculator) ---
        self.base_calc.calculate(atoms, properties, system_changes)
        energy = self.base_calc.results['energy']          # units: eV
        g      = self.base_calc.results['forces'].ravel()  # units: eV/Å

        # Cache energy so the FIRE observer can read it reliably via closure.
        # (self.atoms.calc.results is not guaranteed to be populated when the
        # observer fires between FIRE steps.)
        self._last_energy = float(energy)

        # --- Gradient-flipping: f_mod = g - 2(g · v_TS) v_TS ---
        overlap = np.dot(g, v_ts)
        f_mod   = g - 2.0 * overlap * v_ts

        self.results['energy'] = energy
        self.results['forces'] = f_mod.reshape(atoms.positions.shape)


# ---------------------------------------------------------------------------
# TSSearcher  (Hessian-Based Gradient Flipping)
# ---------------------------------------------------------------------------

class TSSearcher:
    """
    Transition State Searcher using a Simplified Hessian-Based Gradient
    Flipping strategy.

    Algorithm Overview
    ------------------
    1. Compute the full molecular Hessian at the initial geometry via
       VibrationalAnalyzer (finite-difference, δ = 0.01 Å).
    2. Identify the ligand fragment attached to bond_indices[1] through
       graph partitioning of the covalent-bond adjacency matrix (bond to
       be broken is excluded from the graph).
    3. Compute the dissociation direction:
           V_dir = COM(Ligand) - r(central_atom)          [units: Å]
           V_hat = V_dir / |V_dir|                        [dimensionless]
    4. Select the target TS eigenvector **v_TS** by maximum dot-product
       overlap with V_hat:
           k* = argmax_k |v_k · V_hat_3N|
    5. Apply an initial perturbation along **v_TS** and relax with FIRE
       using GradientFlippingCalculator.
    6. Return the converged structure (1st-order saddle point candidate).
    """

    def __init__(self, engine, atoms, config: dict | None = None):
        """
        Args:
            engine: SimulationEngine with a MACE (or other ASE-compatible) backend.
            atoms:  ASE Atoms object (initial geometry, already relaxed).
            config: Optional dict from config.yaml ts_search block.
        """
        self.engine = engine
        self.atoms  = atoms.copy()
        self.config = config or {}
        self.logger = get_workflow_logger()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _identify_ligand_fragment(
        self, atoms, bond_indices: list[int]
    ) -> list[int]:
        """
        Graph-partitions the covalent bond network, excluding the target
        bond, and returns the indices belonging to the ligand fragment.

        Args:
            atoms:        ASE Atoms object.
            bond_indices: [central_idx, ligand_start_idx]

        Returns:
            Sorted list of atom indices in the ligand fragment.
        """
        from ase.data import covalent_radii
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        c_idx, l_idx = bond_indices
        n_atoms      = len(atoms)
        adj          = np.zeros((n_atoms, n_atoms), dtype=int)
        pos          = atoms.positions

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Exclude the bond being broken
                if (i == c_idx and j == l_idx) or (i == l_idx and j == c_idx):
                    continue
                dist   = np.linalg.norm(pos[i] - pos[j])
                cutoff = (covalent_radii[atoms.numbers[i]]
                          + covalent_radii[atoms.numbers[j]] + 0.3)  # units: Å
                if dist < cutoff:
                    adj[i, j] = adj[j, i] = 1

        _, labels    = connected_components(csr_matrix(adj), directed=False)
        ligand_label = labels[l_idx]
        return sorted(np.where(labels == ligand_label)[0].tolist())

    def _compute_hessian_eigensystem(
        self, atoms, displacement: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the full (3N × 3N) Hessian via finite differences and
        returns its eigensystem.

        Physics
        -------
        The Hessian element is approximated by central differences:

            H_{ij} = [ F_i(+δ_j) - F_i(-δ_j) ] / (2δ)     [units: eV/Å²]

        Diagonalisation:  H v_k = λ_k v_k

        Args:
            atoms:       ASE Atoms object with calculator attached.
            displacement: Finite-difference step δ  (units: Å, default 0.01 Å).

        Returns:
            eigenvalues:  shape (3N,)    [units: eV/Å²]
            eigenvectors: shape (3N, 3N) [dimensionless], columns are modes
        """
        self.logger.info(
            f"  [TSSearch] Computing full Hessian "
            f"(δ={displacement} Å, N={len(atoms)} atoms, "
            f"{2 * 3 * len(atoms)} MACE evaluations)..."
        )

        n_atoms = len(atoms)
        dof     = 3 * n_atoms
        H       = np.zeros((dof, dof))  # units: eV/Å²
        pos0    = atoms.get_positions().copy()

        for atom_i in range(n_atoms):
            for cart in range(3):  # x, y, z
                col = 3 * atom_i + cart

                # Forward displacement
                pos_fwd = pos0.copy()
                pos_fwd[atom_i, cart] += displacement           # units: Å
                atoms.set_positions(pos_fwd)
                f_fwd = atoms.get_forces().ravel()              # units: eV/Å

                # Backward displacement
                pos_bwd = pos0.copy()
                pos_bwd[atom_i, cart] -= displacement           # units: Å
                atoms.set_positions(pos_bwd)
                f_bwd = atoms.get_forces().ravel()              # units: eV/Å

                # Central-difference gradient of force = -Hessian column
                H[:, col] = -(f_fwd - f_bwd) / (2.0 * displacement)  # units: eV/Å²

        # Restore original positions
        atoms.set_positions(pos0)

        # Symmetrise to suppress numerical asymmetry
        H = 0.5 * (H + H.T)

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        self.logger.info("  [TSSearch] Hessian diagonalisation complete.")
        return eigenvalues, eigenvectors  # cols of eigenvectors are modes

    def _select_ts_mode(
        self,
        eigenvectors: np.ndarray,
        direction_3n: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """
        Selects the eigenvector v_TS with the highest overlap with the
        3N-dimensional dissociation direction vector V_hat:

            k* = argmax_k |v_k · V_hat|,   V_hat = V_dir / |V_dir|

        Args:
            eigenvectors: (3N × 3N) matrix, columns are mode eigenvectors.
            direction_3n: Raw 3N dissociation vector (need not be normalised).

        Returns:
            (k_star, v_ts)  —  selected mode index and unit eigenvector.
        """
        v_hat    = direction_3n / np.linalg.norm(direction_3n)  # dimensionless
        overlaps = np.abs(eigenvectors.T @ v_hat)               # shape (3N,)
        k_star   = int(np.argmax(overlaps))
        v_ts     = eigenvectors[:, k_star]                      # unit eigenvector
        self.logger.info(
            f"  [TSSearch] TS mode selected: index {k_star}, "
            f"overlap = {overlaps[k_star]:.4f}"
        )
        return k_star, v_ts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_transition_state(
        self,
        bond_indices: list[int],
        fmax: float = 0.05,
        steps: int  = 200,
        trajectory: str | None = None,
    ):
        """
        Finds the 1st-order saddle point (Transition State) for the bond
        specified by bond_indices using Hessian-based gradient flipping.

        Physics / Algorithm
        -------------------
        See class docstring for full derivation.

        Args:
            bond_indices: [central_atom_idx, ligand_start_idx]  (e.g. [Si, N]).
            fmax:         Force convergence threshold   (units: eV/Å).
            steps:        Maximum FIRE optimizer steps.
            trajectory:   Optional path to save the optimisation trajectory.

        Returns:
            ASE Atoms object at the TS geometry.
        """
        c_idx, l_idx = bond_indices
        n_atoms      = len(self.atoms)
        displacement  = self.config.get('hessian_displacement', 0.01)  # units: Å

        # --- Step 1: Attach calculator and compute Hessian ---
        self.atoms.calc = self.engine.get_calculator()
        eigenvalues, eigenvectors = self._compute_hessian_eigensystem(
            self.atoms, displacement=displacement
        )

        # --- Step 2: Identify ligand fragment ---
        self.logger.info(
            f"  [TSSearch] Identifying ligand fragment "
            f"(central={c_idx}, ligand_start={l_idx})..."
        )
        ligand_indices = self._identify_ligand_fragment(self.atoms, bond_indices)
        self.logger.info(
            f"  [TSSearch] Fragment: {len(ligand_indices)} atoms → indices {ligand_indices}"
        )

        # --- Step 3: Build 3N dissociation direction vector ---
        # Use only the two bond atoms (central and ligand-start) moving in opposite
        # directions. Distributing the direction across the whole ligand fragment gives
        # a large net-drift vector that inadvertently overlaps with translational
        # zero-modes instead of the actual bond-stretching mode.
        pos_central  = self.atoms.positions[c_idx]                       # units: Å
        pos_ligstart = self.atoms.positions[l_idx]                       # units: Å
        v_dir_3d     = pos_ligstart - pos_central                        # units: Å
        v_hat        = v_dir_3d / np.linalg.norm(v_dir_3d)              # unit vector

        v_dir_3n         = np.zeros((n_atoms, 3))                       # units: Å
        v_dir_3n[l_idx]  =  v_hat   # ligand-start atom moves away from central
        v_dir_3n[c_idx]  = -v_hat   # central atom moves away from ligand-start
        v_dir_3n         = v_dir_3n.ravel()

        self.logger.info(
            f"  [TSSearch] Dissociation direction |V_dir| = "
            f"{np.linalg.norm(v_dir_3d):.3f} Å (Si-N bond vector)"
        )

        # --- Step 4: Select TS mode by maximum overlap ---
        k_star, v_ts = self._select_ts_mode(eigenvectors, v_dir_3n)

        n_imag_init = int(np.sum(eigenvalues < 0.0))
        self.logger.info(
            f"  [TSSearch] Initial Hessian: {n_imag_init} negative eigenvalue(s). "
            f"Selected mode eigenvalue λ_{k_star} = {eigenvalues[k_star]:.4f} eV/Å²"
        )

        # --- Step 5: Apply initial perturbation along v_TS ---
        disp_ang = self.config.get('displacement_ang', 0.2)  # units: Å
        perturbation = (v_ts.reshape(n_atoms, 3)
                        * np.sign(np.dot(v_ts, v_dir_3n)))   # align sign with direction
        self.atoms.set_positions(
            self.atoms.positions + disp_ang * perturbation   # units: Å
        )
        self.logger.info(
            f"  [TSSearch] Applied perturbation: {disp_ang} Å along v_TS."
        )

        # --- Step 6: Adaptive Gradient-Flipping optimisation with FIRE ---
        # AdaptiveGradientFlippingCalculator recomputes the climbing direction
        # at every force evaluation using the current bond vector, so the
        # climbing always follows the actual Si-N axis regardless of molecular
        # rotation or other conformational changes during the trajectory.
        base_calc     = self.engine.get_calculator()
        gf_calc       = AdaptiveGradientFlippingCalculator(
            base_calc=base_calc, central_idx=c_idx, ligand_idx=l_idx
        )
        self.atoms.calc = gf_calc

        max_bond_dist = self.config.get('max_bond_dist', 3.5)   # Å
        log_interval  = self.config.get('log_interval',   10)   # steps

        r_init = float(np.linalg.norm(
            self.atoms.positions[l_idx] - self.atoms.positions[c_idx]
        ))

        self.logger.info(
            f"  [TSSearch] Starting FIRE with adaptive gradient flipping "
            f"(fmax={fmax} eV/A, max_steps={steps}, "
            f"max_bond_dist={max_bond_dist:.2f} A)..."
        )

        # --- Overshoot monitor ---
        # Called every `log_interval` FIRE steps via dyn.attach().
        # Raises _OvershotError when the bond length exceeds max_bond_dist,
        # which propagates through dyn.run() and is caught below.
        step_log = []   # list of (step, bond_dist_A, energy_eV)

        dyn = FIRE(self.atoms, trajectory=trajectory, logfile='-')

        def _check_overshoot():
            r = float(np.linalg.norm(
                self.atoms.positions[l_idx] - self.atoms.positions[c_idx]
            ))
            e = gf_calc._last_energy  # cached by AdaptiveGradientFlippingCalculator.calculate()
            step_log.append((dyn.nsteps, r, e))
            if r > max_bond_dist:
                raise _OvershotError(r, e)

        dyn.attach(_check_overshoot, interval=log_interval)

        overshoot_detected = False
        try:
            converged = dyn.run(fmax=fmax, steps=steps)
        except _OvershotError as exc:
            converged           = False
            overshoot_detected  = True
            r_over, e_over      = exc.args

        # --- Report ---
        if overshoot_detected:
            e_start = step_log[0][2] if step_log else float('nan')
            r_final = step_log[-1][1] if step_log else float('nan')
            e_final = step_log[-1][2] if step_log else float('nan')

            self.logger.warning(
                "  [TSSearch] *** BARRIERLESS DISSOCIATION DETECTED ***"
            )
            self.logger.warning(
                f"  [TSSearch]   Bond length exceeded max_bond_dist = "
                f"{max_bond_dist:.2f} A during gradient-flipping."
            )
            self.logger.warning(
                f"  [TSSearch]   The Si-N PES appears to have no classical "
                f"barrier with this potential/model."
            )
            self.logger.warning(
                f"  [TSSearch]   Initial : bond = {r_init:.3f} A, "
                f"E = {e_start:.4f} eV"
            )
            self.logger.warning(
                f"  [TSSearch]   At stop : bond = {r_final:.3f} A, "
                f"E = {e_final:.4f} eV  "
                f"(dE = {e_final - e_start:+.3f} eV)"
            )
            self.logger.warning(
                f"  [TSSearch]   Bond / energy profile "
                f"(every {log_interval} steps):"
            )
            for step, r, e in step_log:
                self.logger.warning(
                    f"  [TSSearch]     step {step:4d}:  "
                    f"bond = {r:.3f} A,  "
                    f"E = {e:.4f} eV  "
                    f"(dE = {e - e_start:+.3f} eV)"
                )
            self.logger.warning(
                "  [TSSearch]   Recommendation: use a surface slab model "
                "so that adsorption energy stabilises the TS."
            )
        elif converged:
            self.logger.info(
                "  [TSSearch] Gradient-flipping optimisation converged."
            )
        else:
            self.logger.warning(
                f"  [TSSearch] Did not converge within {steps} steps. "
                "Consider increasing 'steps' or checking the initial geometry."
            )

        return self.atoms
