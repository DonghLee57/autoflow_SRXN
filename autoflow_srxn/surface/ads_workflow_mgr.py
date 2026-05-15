from itertools import combinations

import numpy as np
import spglib
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from ..utils.knowledge_engine import chem_kb
from ..utils.logger_utils import get_workflow_logger
from .surface_utils import calculate_haptic_normal, calculate_haptic_vbs

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


class AdsorptionWorkflowManager:
    """Generalized Adsorption Manager with Mechanistic Logging and Visual Clarity."""

    def __init__(self, slab, config=None, symprec=0.2, verbose=False):
        from .surface_utils import standardize_vasp_atoms
        self.slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
        self.config = config if config is not None else {}
        self.verbose = verbose
        self.symprec = symprec
        self.logger = get_workflow_logger()

        tags = self.slab.get_tags()
        self.slab_tags = set(tags)
        # Identify substrate surface (tags < 2 or tag 4)
        sub_mask = (tags < 2) | (tags == 4)
        if np.any(sub_mask):
            z_max_sub = self.slab.positions[sub_mask, 2].max()
            sub_surface = np.where(sub_mask & (self.slab.positions[:, 2] > z_max_sub - 1.5))[0]
        else:
            sub_surface = np.array([], dtype=int)

        # Identify inhibitor surface (tags in [2, 3])
        inh_mask = (tags == 2) | (tags == 3)
        if np.any(inh_mask):
            z_max_inh = self.slab.positions[inh_mask, 2].max()
            inh_surface = np.where(inh_mask & (self.slab.positions[:, 2] > z_max_inh - 1.5))[0]
        else:
            inh_surface = np.array([], dtype=int)

        all_surface = np.unique(np.concatenate([sub_surface, inh_surface]))
        target_label = "Substrate" if len(self.slab) > 10 else "Molecule/Fragment"
        self.surface_indices = self.get_unique_surface_indices(self.slab, all_surface, symprec=self.symprec)
        
        # Clarified logging message
        site_label = "adsorption" if target_label == "Substrate" else "coordination"
        self.logger.info(
            f"[Symmetry] {target_label} analysis (symprec={self.symprec}): "
            f"Grouped {len(all_surface)} surface atoms into {len(self.surface_indices)} symmetrically distinct {site_label} sites."
        )

    def calculate_molecule_lateral_extent(self, molecule):
        """Calculates the maximum lateral (XY) span of the molecule to detect potential PBC overlaps.
        Returns the max distance between any two atoms projected on the XY plane.
        """
        pos_xy = molecule.positions[:, :2]
        if len(pos_xy) < 2:
            return 0.0

        # Max distance between any two atoms in XY
        from scipy.spatial.distance import pdist

        dists = pdist(pos_xy)
        return float(np.max(dists))

    def get_unique_surface_indices(self, slab, indices, symprec=0.2):
        lattice, positions, numbers = (
            slab.get_cell(),
            slab.get_scaled_positions(),
            slab.get_atomic_numbers(),
        )

        try_precisions = [symprec]
        if symprec < 0.5:
            try_precisions += [0.5]

        for prec in try_precisions:
            try:
                dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=prec)
                if dataset is None:
                    continue

                if hasattr(dataset, "equivalent_atoms"):
                    equiv = dataset.equivalent_atoms
                else:
                    equiv = dataset["equivalent_atoms"]

                unique_classes = np.unique(equiv[indices])

                if len(unique_classes) < len(indices) or prec == try_precisions[-1]:
                    centered_indices = []
                    for c in unique_classes:
                        class_members = [i for i in indices if equiv[i] == c]
                        dist_sq = np.sum((positions[class_members][:, :2] - 0.5) ** 2, axis=1)
                        best_idx = class_members[np.argmin(dist_sq)]
                        centered_indices.append(best_idx)

                    if len(centered_indices) == len(indices):
                        return self.get_unique_geometric_sites(slab, indices)
                    return centered_indices
            except Exception:
                pass
        return self.get_unique_geometric_sites(slab, indices)

    def get_unique_coordinates(self, slab, coords, symprec=0.2):
        """Reduce arbitrary Cartesian coordinates to symmetry-unique representatives.

        Surface sites (top, bridge, hollow) require **2D surface symmetry** only.
        3D spglib operations include vertical rotations that can falsely map top-surface
        sites to bottom-surface sites, preventing correct orbit merging.  This method
        therefore filters to *2D-compatible operations* — those that leave the fractional
        Z coordinate unchanged (R[2,:] ≈ [0,0,1], t_z ≈ 0) — and compares XY distances
        only for the equivalence test and orbit deduplication.

        For each equivalence class the representative is the orbit member whose
        fractional XY coordinate is closest to (0.5, 0.5) so that placed candidates
        sit near the centre of the unit cell rather than at an edge or corner.
        Z is taken from the original candidate (surface reference level).
        """
        if not coords:
            return []
        sub_indices = np.where(slab.get_tags() < 2)[0]
        sub_slab = slab[sub_indices]

        lattice = sub_slab.get_cell()
        positions = sub_slab.get_scaled_positions()
        numbers = sub_slab.get_atomic_numbers()

        sym = spglib.get_symmetry((lattice, positions, numbers), symprec=symprec)
        if not sym:
            return coords

        rotations  = sym['rotations']
        translations = sym['translations']
        inv_lattice = np.linalg.inv(lattice)

        # --- Filter to 2D (surface-parallel) operations only ---
        # Keep operations where R leaves Z invariant: R[2,0]=R[2,1]=0, R[2,2]=1, t_z≈0.
        # Fractional t_z tolerance 0.15 covers non-primitive-cell translations that are
        # pure lattice vectors in Z (e.g. 0, ±1, ±2, …) but since t is reduced mod 1
        # by spglib, any non-zero t_z flags a Z-mixing operation.
        ops_2d = [
            (r, t) for r, t in zip(rotations, translations)
            if (abs(r[2, 0]) < 0.1 and abs(r[2, 1]) < 0.1
                and abs(r[2, 2] - 1.0) < 0.1 and abs(t[2]) < 0.15)
        ]
        if not ops_2d:
            # Fallback: use all operations (e.g. symmetric slab where top≡bottom)
            ops_2d = list(zip(rotations, translations))

        # Use the user-defined symprec for coordinate equivalence
        EQUIV_TOL = self.symprec

        def get_order(c_frac):
            count = 0
            for r, t in ops_2d:
                mapped = np.dot(r, c_frac) + t
                diff = mapped[:2] - c_frac[:2]
                diff -= np.round(diff)
                d_cart = np.linalg.norm(diff[0] * lattice[0] + diff[1] * lattice[1])
                if d_cart < 0.2:
                    count += 1
            return count

        accepted_info = [] # List of {frac, cart, order}

        for c in coords:
            c_frac = np.dot(c, inv_lattice)
            c_order = get_order(c_frac)
            
            found_idx = -1
            for i, info in enumerate(accepted_info):
                is_equiv = False
                for r, t in ops_2d:
                    mapped = np.dot(r, c_frac) + t
                    diff = mapped[:2] - info['frac'][:2]
                    diff -= np.round(diff)
                    d_cart = np.linalg.norm(diff[0] * lattice[0] + diff[1] * lattice[1])
                    
                    if d_cart < EQUIV_TOL:
                        is_equiv = True
                        break
                if is_equiv:
                    found_idx = i
                    break
            
            if found_idx == -1:
                accepted_info.append({'frac': c_frac, 'cart': c, 'order': c_order})
            else:
                if c_order > accepted_info[found_idx]['order']:
                    accepted_info[found_idx] = {'frac': c_frac, 'cart': c, 'order': c_order}

        return [info['cart'] for info in accepted_info]

    def get_unique_geometric_sites(self, slab, indices, cutoff=1.5):
        if not len(indices):
            return []
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        pos = slab.positions[indices]
        if len(pos) == 1:
            return indices

        dist_matrix = pdist(pos)
        Z = linkage(dist_matrix, method="complete")
        labels = fcluster(Z, t=cutoff, criterion="distance")

        centered_representatives = []
        scaled_pos = slab.get_scaled_positions()[indices]
        for c in np.unique(labels):
            members_idx = np.where(labels == c)[0]
            dists = np.linalg.norm(scaled_pos[members_idx][:, :2] - 0.5, axis=1)
            centered_representatives.append(indices[members_idx[np.argmin(dists)]])

        return centered_representatives

    def get_all_adjacent_sites(self, slab, core_idx, k, max_dist=4.5):
        from ase.geometry import get_distances

        _, d_list = get_distances(slab.positions[core_idx], slab.positions, cell=slab.cell, pbc=slab.pbc)
        dists = d_list[0]
        z_max = slab.positions[:, 2].max()
        surface_mask = slab.positions[:, 2] > z_max - 1.5
        adj_indices = np.where((dists > 0.1) & (dists < max_dist) & surface_mask)[0]
        for cluster_indices in combinations(adj_indices, k):
            yield (core_idx,) + cluster_indices

    def generate_rdkit_conformer(self, smiles, sanitize_fallback=True):
        import re

        mol = Chem.MolFromSmiles(smiles)
        if mol is None and sanitize_fallback:
            temp_smiles = re.sub(r"SiH(\d+)", r"[SiH\1]", smiles)
            temp_smiles = re.sub(r"Si(?!H|\[)", r"[Si]", temp_smiles)
            mol = Chem.MolFromSmiles(temp_smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass

        if mol.GetNumConformers() == 0:
            return None
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        return Atoms(symbols=symbols, positions=positions)

    def check_overlap(self, atoms, skip_indices=None, skip_pairs=None,
                      overlap_scale=None, cutoff=None, verbose=False, check_internal=True):
        """Element-aware overlap check using Alvarez (2013) vdW radii by default.

        Two mutually exclusive threshold modes:
        - ``overlap_scale`` (default): pair threshold = overlap_scale * (r_vdw_i + r_vdw_j)
          using ALVAREZ_VDW_RADII.  ``overlap_scale`` defaults to
          ``config.reaction_search.candidate_filter.overlap_scale`` (fallback 0.65).
        - ``cutoff`` (A): explicit flat threshold applied to every pair, regardless of
          element identity.  Useful for chemisorption geometry checks where the newly
          formed bond length is already known (e.g. cutoff=1.4 A).  If both are supplied,
          ``cutoff`` takes precedence.

        Z-periodicity is disabled to avoid spurious collisions with the slab bottom image.
        """
        from ase.geometry import get_distances

        tags = atoms.get_tags()
        # New atoms are those whose tags were not in the original slab
        new_idx = np.array([i for i, t in enumerate(tags) if t not in self.slab_tags], dtype=int)
        env_idx = np.array([i for i, t in enumerate(tags) if t in self.slab_tags], dtype=int)

        if len(new_idx) == 0:
            # Fallback for when tags are not maintained: assume last atoms are new
            # (But this shouldn't happen with our workflow)
            return False

        pos = atoms.positions
        symbols = atoms.get_chemical_symbols()
        skip_pairs = set(tuple(sorted(p)) for p in (skip_pairs or []))

        # Resolve overlap_scale from config if not explicitly provided
        if overlap_scale is None:
            overlap_scale = self.config.get("reaction_search", {}).get(
                "candidate_filter", {}).get("overlap_scale", 0.65)

        def _thresh(i, j):
            """Return the overlap threshold (A) for atom pair (i, j)."""
            if cutoff is not None:
                return cutoff
            ri = chem_kb.get_radius(symbols[i], "vdw")
            rj = chem_kb.get_radius(symbols[j], "vdw")
            return overlap_scale * (ri + rj)

        # Disable Z-periodicity to avoid wrap-around hits with slab bottom
        effective_pbc = [True, True, False]

        # 1. Internal check (new atoms vs. each other)
        if check_internal and len(new_idx) > 1:
            _, int_dists = get_distances(pos[new_idx], pos[new_idx], cell=atoms.cell, pbc=effective_pbc)
            for i in range(len(new_idx)):
                for j in range(i + 1, len(new_idx)):
                    idx_i, idx_j = new_idx[i], new_idx[j]
                    if skip_pairs and tuple(sorted((idx_i, idx_j))) in skip_pairs:
                        continue
                    thresh = _thresh(idx_i, idx_j)
                    if int_dists[i, j] < thresh:
                        if verbose:
                            self.logger.info(f"  [Overlap] Internal clash: {symbols[idx_i]}({idx_i}) - {symbols[idx_j]}({idx_j}) | Dist: {int_dists[i, j]:.2f} < {thresh:.2f} A")
                        return True

        # 2. External check (new atoms vs. environment)
        if len(env_idx) > 0:
            _, ext_dists = get_distances(pos[new_idx], pos[env_idx], cell=atoms.cell, pbc=effective_pbc)
            for i, idx_i in enumerate(new_idx):
                if skip_indices and idx_i in skip_indices:
                    continue
                for j, idx_j in enumerate(env_idx):
                    if skip_indices and idx_j in skip_indices:
                        continue
                    if skip_pairs and tuple(sorted((idx_i, idx_j))) in skip_pairs:
                        continue
                    thresh = _thresh(idx_i, idx_j)
                    if ext_dists[i, j] < thresh:
                        if verbose:
                            self.logger.info(f"  [Overlap] External clash: {symbols[idx_i]}({idx_i}) - {symbols[idx_j]}({idx_j}) | Dist: {ext_dists[i, j]:.2f} < {thresh:.2f} A")
                        return True
        return False

    def _get_steric_fitness(self, atoms, overlap_scale=None, cutoff=None, check_internal=True):
        if self.check_overlap(atoms, overlap_scale=overlap_scale, cutoff=cutoff,
                              verbose=True, check_internal=check_internal):
            return -1e9

        from ase.geometry import get_distances
        tags = atoms.get_tags()
        max_tag = np.max(tags)
        new_indices = np.where(tags == max_tag)[0]
        env_indices = np.where(tags < max_tag)[0]

        if len(env_indices) == 0:
            return 0.0

        _, dists = get_distances(
            atoms.positions[new_indices],
            atoms.positions[env_indices],
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        score = -np.sum(1.0 / (dists**6 + 1e-6))
        return score

    def _get_diverse_top_poses(self, poses, n_out=5):
        if not poses:
            return []
        poses.sort(key=lambda x: x[0], reverse=True)
        return [p[1] for p in poses[:n_out]]

    def _get_rotation_center(self, atoms, mode="com"):
        if mode == "com":
            return atoms.get_center_of_mass()
        elif mode == "closest":
            com = atoms.get_center_of_mass()
            idx = np.argmin(np.linalg.norm(atoms.positions - com, axis=1))
            return atoms.positions[idx]
        elif isinstance(mode, int):
            return atoms.positions[mode]
        elif isinstance(mode, str):
            indices = [a.index for a in atoms if a.symbol == mode]
            if indices:
                return np.mean(atoms.positions[indices], axis=0)
        return atoms.get_center_of_mass()

    def _get_physi_alignment(self, molecule, mode="com"):
        """Analyzes molecule and returns a copy aligned in a favorable orientation for physisorption.
        Ensures the rotation center (mode) is at [0, 0, 0] at the end.
        """
        m = molecule.copy()
        
        # 1. Orientation
        if mode == "com" or mode == "closest":
            m.translate(-m.get_center_of_mass())
            pos = m.positions
            cov = np.cov(pos.T)
            evals, evecs = np.linalg.eigh(cov)
            
            # Align shortest axis (evecs[:, 0]) with Z-axis
            m.rotate(evecs[:, 0], [0, 0, 1], center=[0, 0, 0])
            
            # --- H-up Flip Logic ---
            # Calculate the average direction of Hydrogen atoms relative to COM.
            h_indices = [a.index for a in m if a.symbol == "H"]
            if h_indices:
                h_pos = m.positions[h_indices]
                avg_h_vec = np.mean(h_pos, axis=0) # Relative to COM [0,0,0]
                
                # If hydrogens are pointing down (-z), flip the molecule 180 deg
                if avg_h_vec[2] < 0:
                    m.rotate(180, 'x', center=[0, 0, 0])
        elif isinstance(mode, str) and len(mode) <= 2:
            # Anchor element alignment
            indices = [a.index for a in m if a.symbol == mode]
            if indices:
                anchor_pos = np.mean(m.positions[indices], axis=0)
                # Align vector from anchor to COM with +Z (points UP)
                vec = m.get_center_of_mass() - anchor_pos
                if np.linalg.norm(vec) > 1e-3:
                    m.rotate(vec, [0, 0, 1], center=anchor_pos)
        
        # 2. Final Centering: ensure the requested rot_center is at [0,0,0]
        # This is CRITICAL for placement height to be accurate.
        c_pos = self._get_rotation_center(m, mode=mode)
        m.translate(-c_pos)
        return m

    def _generate_surface_sites(self, z_surface_ref):
        """Generate top, bridge, and 3-fold hollow sites via Delaunay triangulation.

        Returns a list of [x, y, z_surface_ref] arrays.  Z carries only the substrate
        surface level; the height offset is applied per-orientation inside
        generate_physisorption_candidates (clearance vs center modes differ).
        """
        from .surface_utils import find_surface_indices
        from scipy.spatial import Delaunay

        all_surface = find_surface_indices(self.slab, side="top")
        if not len(all_surface):
            return []

        pos = self.slab.positions[all_surface]
        sites = [np.array([p[0], p[1], z_surface_ref]) for p in pos]  # top sites

        if len(all_surface) < 3:
            return sites

        try:
            tri = Delaunay(pos[:, :2])
            seen_edges = set()
            for s in tri.simplices:
                for a, b in [(0, 1), (1, 2), (0, 2)]:
                    key = tuple(sorted((int(s[a]), int(s[b]))))
                    if key not in seen_edges:
                        seen_edges.add(key)
                        mid = (pos[s[a]] + pos[s[b]]) / 2
                        sites.append(np.array([mid[0], mid[1], z_surface_ref]))
                centroid = pos[s].mean(axis=0)
                sites.append(np.array([centroid[0], centroid[1], z_surface_ref]))
        except Exception:
            # Fallback for collinear atoms: pairwise bridges only
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    if 1.5 < np.linalg.norm(pos[i] - pos[j]) < 5.5:
                        mid = (pos[i] + pos[j]) / 2
                        sites.append(np.array([mid[0], mid[1], z_surface_ref]))
        return sites

    def _sample_molecule_orientations(self, m_aligned, n_rot):
        """Sample n_rot orientations on SO(3) via Fibonacci sphere.

        Decomposes n_rot into n_polar × n_spin where n_polar directions are
        distributed uniformly in solid angle (Fibonacci lattice, uniform in
        cos θ) and n_spin azimuthal spins are applied around Z for each.
        The rotation center stays at [0, 0, 0] throughout.
        """
        golden = (1.0 + 5.0 ** 0.5) / 2.0
        n_polar = max(1, int(n_rot ** 0.5))
        n_spin = max(1, n_rot // n_polar)
        poses = []
        for i in range(n_polar):
            cos_theta = 1.0 - 2.0 * (i + 0.5) / n_polar
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            phi = 2.0 * np.pi * i / golden
            direction = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            for j in range(n_spin):
                m = m_aligned.copy()
                if np.linalg.norm(direction - np.array([0.0, 0.0, 1.0])) > 1e-6:
                    m.rotate([0, 0, 1], direction, center=[0, 0, 0])
                if j > 0:
                    m.rotate(360.0 * j / n_spin, [0, 0, 1], center=[0, 0, 0])
                poses.append(m)
        return poses

    def _find_contact_z(self, m_template, target_xy, z_lo, z_hi, overlap_scale, tag):
        """Binary search for the lowest non-overlapping Z of the rotation center.

        Performs 12 bisection steps (~4096× resolution over the window), giving
        sub-0.01 Å precision for a typical 5 Å search window.
        Returns the lowest Z where m_template does not overlap the slab.
        """
        for _ in range(12):
            z_mid = (z_lo + z_hi) * 0.5
            probe = m_template.copy()
            probe.translate([target_xy[0], target_xy[1], z_mid])
            for a in probe:
                a.tag = tag
            combined = self.slab.copy()
            combined += probe
            if self.check_overlap(combined, overlap_scale=overlap_scale, check_internal=False):
                z_lo = z_mid  # overlap → must raise
            else:
                z_hi = z_mid  # clear   → try lower
        return z_hi

    def generate_physisorption_candidates(self, molecule, height=3.5, n_rot=32, rot_center="com",
                                           height_mode="clearance", gravity_pull=None,
                                           config=None, tag=2):
        """Generate physisorption candidate structures.

        Site generation uses Delaunay triangulation to produce top, bridge, and
        3-fold hollow sites.  Orientations are sampled uniformly on SO(3) via the
        Fibonacci sphere (n_rot total, respecting the n_rot parameter).
        Deduplication is per-site so candidates at different adsorption sites are
        always preserved regardless of molecular orientation.

        Parameters
        ----------
        height : float
            Placement height in Å interpreted by ``height_mode``.
        height_mode : str
            ``"clearance"`` — lowest molecule atom sits ``height`` Å above surface.
            ``"center"``    — rotation center placed at ``height`` Å above surface.
        n_rot : int
            Number of orientations sampled per site.
        gravity_pull : dict or None
            ``{"enabled": True}`` — descend to vdW contact via binary search.
            When disabled, molecule is fixed at the height defined by height_mode.
        """
        from .surface_utils import CavityDetector, identify_protectors, standardize_vasp_atoms
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist

        overlap_scale = self.config.get("reaction_search", {}).get(
            "candidate_filter", {}).get("overlap_scale", 0.65)

        _grav = gravity_pull if gravity_pull is not None else {}
        grav_enabled = _grav.get("enabled", False)

        sub_tags = self.slab.get_tags()
        sub_mask_z = sub_tags < 2
        z_surface_ref = (float(np.max(self.slab.positions[sub_mask_z, 2]))
                         if np.any(sub_mask_z)
                         else float(np.max(self.slab.positions[:, 2])))

        # --- Site generation ---
        _protex = self.config.get("reaction_search", {}).get("mechanisms", {}).get("protector", {})
        _inh_cfg = self.config.get("reaction_search", {}).get("mechanisms", {}).get("inhibitor", {})
        slab_has_inhibitors = np.any(self.slab.get_tags() >= 2)

        raw_centers = []
        if self.config and (_protex.get("enabled", False) or
                            (_inh_cfg.get("enabled", False) and slab_has_inhibitors)):
            sub_idx, prot_idx = identify_protectors(self.slab, self.config, verbose=self.verbose)
            grid_res = _protex.get("cavity_grid_ang", _protex.get("grid_resolution", 0.2))
            detector = CavityDetector(self.slab, sub_idx, prot_idx, grid_res=grid_res, verbose=self.verbose)
            raw_centers = detector.find_void_centers(top_clearance=height)
            if raw_centers:
                raw_centers = [c + np.array([0, 0, 0.5]) for c in raw_centers]

        if not raw_centers:
            raw_centers = self._generate_surface_sites(z_surface_ref)

        target_centers = self.get_unique_coordinates(self.slab, raw_centers, symprec=self.symprec)

        self.logger.info(
            f"  [Physisorption] {len(target_centers)} unique sites "
            f"(height={height:.1f} Å [{height_mode}], n_rot={n_rot}, "
            f"overlap_scale={overlap_scale:.2f}, gravity={'on' if grav_enabled else 'off'})"
        )

        # --- Pre-compute 2D operations for site-symmetry dedup ---
        lattice = self.slab.get_cell()
        positions = self.slab.get_scaled_positions()
        numbers = self.slab.get_atomic_numbers()
        sym = spglib.get_symmetry((lattice, positions, numbers), symprec=self.symprec)
        rotations = sym['rotations']
        translations = sym['translations']
        ops_2d = [
            (r, t) for r, t in zip(rotations, translations)
            if (abs(r[2, 0]) < 0.1 and abs(r[2, 1]) < 0.1
                and abs(r[2, 2] - 1.0) < 0.1 and abs(t[2]) < 0.15)
        ]

        # --- Pre-compute all orientations once; tags set here, reused across sites ---
        m_aligned = self._get_physi_alignment(molecule, mode=rot_center)
        sampled_poses = self._sample_molecule_orientations(m_aligned, n_rot)
        for m_pose in sampled_poses:
            for a in m_pose:
                a.tag = tag

        candidates = []
        stats = {"total": 0, "overlap": 0, "dedup": 0}

        site_iter = (
            _tqdm(target_centers, desc="[Physi] sites", unit="site", leave=True, dynamic_ncols=True)
            if _tqdm else target_centers
        )

        for site_idx, target_pos in enumerate(site_iter):
            target_xy = target_pos[:2]
            target_frac = np.dot(target_pos, np.linalg.inv(lattice))
            site_candidates = []
            site_rel_poses = []  # within-site orientation dedup registry

            # --- Identify Site Point Group ---
            # Subset of ops_2d that leave target_pos invariant (modulo lattice)
            site_ops = []
            for r, t in ops_2d:
                mapped = np.dot(r, target_frac) + t
                diff = mapped[:2] - target_frac[:2]
                diff -= np.round(diff)
                if np.linalg.norm(np.dot(diff, lattice[:2, :2])) < 0.5:
                    site_ops.append(r)

            orient_iter = (
                _tqdm(
                    sampled_poses,
                    desc=f"  site {site_idx + 1}/{len(target_centers)} orientations",
                    unit="rot",
                    leave=False,
                    dynamic_ncols=True,
                )
                if _tqdm else sampled_poses
            )

            for m_pose in orient_iter:
                stats["total"] += 1

                # 1. Deduplicate orientations based on site symmetry BEFORE expensive gravity pull
                # (rel_pos is invariant to Z, so we can check it now)
                ads_pos = m_pose.positions
                ads_sym = np.array(m_pose.get_chemical_symbols())
                rel_pos = ads_pos - np.mean(ads_pos, axis=0)

                is_dup = False
                for ref_rel, ref_sym in site_rel_poses:
                    if len(ads_sym) != len(ref_sym):
                        continue
                    if not np.array_equal(np.sort(ads_sym), np.sort(ref_sym)):
                        continue

                    # Check if any site operation maps ref_rel to current rel_pos
                    for r_site in site_ops:
                        mapped_ref = np.dot(ref_rel, r_site.T)
                        match = True
                        for sym_type in np.unique(ads_sym):
                            ci = np.where(ads_sym == sym_type)[0]
                            ri = np.where(ref_sym == sym_type)[0]
                            D = cdist(rel_pos[ci], mapped_ref[ri])
                            rw, cw = linear_sum_assignment(D)
                            if np.any(D[rw, cw] > 0.4):
                                match = False
                                break
                        if match:
                            is_dup = True
                            break
                    if is_dup:
                        break

                if is_dup:
                    stats["dedup"] += 1
                    continue

                # 2. Only perform placement logic (fixed or gravity) for unique orientations
                min_z_template = m_pose.positions[:, 2].min()
                if height_mode == "clearance":
                    z_nominal = z_surface_ref + height - min_z_template
                else:  # "center"
                    z_nominal = z_surface_ref + height

                if grav_enabled:
                    z_final = self._find_contact_z(
                        m_pose, target_xy,
                        z_lo=z_surface_ref,
                        z_hi=z_nominal + 5.0,
                        overlap_scale=overlap_scale,
                        tag=tag,
                    )
                else:
                    # Fixed placement
                    probe = m_pose.copy()
                    probe.translate([target_xy[0], target_xy[1], z_nominal])
                    combined_check = self.slab.copy()
                    combined_check += probe
                    if self.check_overlap(combined_check, overlap_scale=overlap_scale, check_internal=False):
                        stats["overlap"] += 1
                        continue
                    z_final = z_nominal

                # 3. Build final combined structure
                final_pose = m_pose.copy()
                final_pose.translate([target_xy[0], target_xy[1], z_final])
                combined = self.slab.copy()
                combined += final_pose
                
                # Clean labeling
                combined.info["mechanism"] = f"Physisorption (Site {site_idx + 1})"
                combined.info["site_id"] = site_idx + 1
                combined.info["reaction_type"] = "physisorption"

                site_candidates.append(combined)
                site_rel_poses.append((rel_pos, ads_sym))

            candidates.extend(site_candidates)

        candidates = [standardize_vasp_atoms(c, z_min_offset=0.5) for c in candidates]

        self.logger.info(
            f"  [Physisorption] {len(candidates)} unique candidates "
            f"({stats['overlap']} overlap-rejected, {stats['dedup']} deduplicated within sites)"
        )

        if not candidates:
            _total   = stats["total"]
            _overlap = stats["overlap"]
            _dedup   = stats["dedup"]
            _placed  = _total - _overlap - _dedup
            if not target_centers:
                self.logger.warning(
                    "[Physisorption] WARNING: 0 candidates — no surface sites were generated. "
                    "Check that the slab has a valid surface layer and that site-generation "
                    "parameters (symprec, grid_resolution) are appropriate."
                )
            else:
                self.logger.warning(
                    f"[Physisorption] WARNING: 0 candidates generated.\n"
                    f"  Sites: {len(target_centers)}  |  Orientations sampled: {len(sampled_poses)}\n"
                    f"  Rejection breakdown (total {_total} orientation trials):\n"
                    f"    Deduplicated (symmetry)         : {_dedup}\n"
                    f"    External overlap (mol vs slab)  : {_overlap}\n"
                    f"    Placed but not added (bug?)     : {_placed}\n"
                    f"  Common causes:\n"
                    f"    - placement_height too small → molecule placed inside surface.\n"
                    f"    - overlap_scale too large → valid poses rejected as overlapping.\n"
                    f"    - Internal molecular bond overlap (old bug): occurs when "
                    f"check_internal=True is used (should be False for physisorption)."
                )

        return candidates

    def discover_ligands(self, molecule, center_target="Si", skin=0.3, verbose=None):
        if verbose is None: verbose = self.verbose
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from ase.neighborlist import neighbor_list as ase_nl

        if isinstance(center_target, int):
            c_idx = center_target
        else:
            center_indices = [a.index for a in molecule if a.symbol == center_target]
            if not center_indices: return None, []
            c_idx = center_indices[0]

        n_atoms = len(molecule)
        # Use a broad cutoff then apply per-pair element-specific tightening.
        # 2.5 A covers all common covalent bonds; avoids O(n²) distance matrix.
        max_cutoff = 2.5 + skin
        ni_arr, nj_arr, nd_arr = ase_nl("ijd", molecule, max_cutoff)
        adj_matrix = np.zeros((n_atoms, n_atoms), dtype=int)
        for ni, nj, nd in zip(ni_arr, nj_arr, nd_arr):
            cutoff = chem_kb.get_radius(molecule.symbols[ni], "covalent") + chem_kb.get_radius(molecule.symbols[nj], "covalent") + skin
            if 0.1 < nd < cutoff:
                adj_matrix[ni, nj] = 1
                adj_matrix[nj, ni] = 1

        bonded_indices = np.where(adj_matrix[c_idx, :] == 1)[0]
        adj_matrix[c_idx, :] = 0
        adj_matrix[:, c_idx] = 0

        graph = csr_matrix(adj_matrix)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        ligands = []
        center_label = labels[c_idx]

        for comp_id in range(n_components):
            if comp_id == center_label: continue
            frag_indices = np.where(labels == comp_id)[0]
            binding_atoms = list(set(frag_indices).intersection(bonded_indices))
            if len(binding_atoms) > 0:
                frag_atoms = molecule[frag_indices]
                formula = frag_atoms.get_chemical_formula()
                binding_pos = np.mean(molecule.positions[binding_atoms], axis=0)
                bond_vec = binding_pos - molecule.positions[c_idx]
                vbs = calculate_haptic_vbs(molecule, binding_atoms)
                normal = calculate_haptic_normal(molecule, binding_atoms)
                vec_to_metal = molecule.positions[c_idx] - vbs
                if np.dot(normal, vec_to_metal) < 0: normal = -normal
                ligands.append({"formula": formula, "indices": list(frag_indices), "binding_atoms": binding_atoms, 
                                "hapticity": len(binding_atoms), "bond_vec": bond_vec, "vbs_pos": vbs, "normal_vector": normal})

        if verbose:
            print(f"Precursor Fragmentation Analysis ({center_target} centered): Found {len(ligands)} ligands attached to index {c_idx}.")
        return c_idx, ligands

    def _place_at_dangling_bond(self, fragment, binding_idx, internal_bond_vec, target_site_pos, db_vector, bond_length, rot_angle=0, haptic_normal=None):
        f = fragment.copy()
        if isinstance(binding_idx, (list, np.ndarray)) and len(binding_idx) > 1:
            anchor_pos = np.mean(f.positions[binding_idx], axis=0)
            align_vec = haptic_normal if haptic_normal is not None else internal_bond_vec
        else:
            b_idx = binding_idx[0] if isinstance(binding_idx, (list, np.ndarray)) else binding_idx
            anchor_pos = f.positions[b_idx]
            align_vec = internal_bond_vec
        f.rotate(align_vec, -db_vector, center=anchor_pos)
        f.rotate(rot_angle, db_vector, center=anchor_pos)
        placement_pos = target_site_pos + (db_vector / np.linalg.norm(db_vector)) * bond_length
        f.translate(placement_pos - anchor_pos)
        return f

    def _form_byproduct(self, fragment, binding_idx, internal_bond_vec):
        from ase import Atoms
        f = fragment.copy()
        sym = f.symbols[binding_idx]
        b_len = 1.0 if sym in ["N", "O"] else 1.1 if sym == "C" else 1.5
        h_pos = f.positions[binding_idx] + (internal_bond_vec / np.linalg.norm(internal_bond_vec)) * b_len
        f += Atoms("H", positions=[h_pos])
        return f

