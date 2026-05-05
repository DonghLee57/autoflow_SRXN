from itertools import combinations

import numpy as np
import spglib
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from .knowledge_engine import chem_kb
from .logger_utils import get_workflow_logger
from .surface_utils import calculate_haptic_normal, calculate_haptic_vbs


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
        self.surface_indices = self.get_unique_surface_indices(self.slab, all_surface, symprec=self.symprec)
        self.logger.info(
            f"Surface Symmetry Analysis (symprec={self.symprec}): {len(all_surface)} atoms reduced to {len(self.surface_indices)} sites."
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
        """Reduces a set of arbitrary Cartesian coordinates to symmetry-unique ones."""
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

        rotations = sym['rotations']
        translations = sym['translations']
        
        unique_coords = []
        inv_lattice = np.linalg.inv(lattice)

        for c in coords:
            c_frac = np.dot(c, inv_lattice)
            is_new = True
            
            for uc in unique_coords:
                uc_frac = np.dot(uc, inv_lattice)
                for r, t in zip(rotations, translations):
                    mapped = np.dot(r, c_frac) + t
                    diff = mapped - uc_frac
                    diff -= np.round(diff)
                    cart_dist = np.linalg.norm(np.dot(diff, lattice))
                    if cart_dist < 1.0: 
                        is_new = False
                        break
                if not is_new:
                    break
            if is_new:
                unique_coords.append(c)
        return unique_coords

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
        except:
            pass

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
        
        # print(f"DEBUG check_overlap: check_internal={check_internal}")

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

    def _get_diverse_top_poses(self, poses, n_out=5, angle_threshold=45.0):
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
            m.rotate(evecs[:, 0], [0, 0, 1], center=[0, 0, 0])
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

    def generate_physisorption_candidates(self, molecule, height=3.5, n_rot=32, rot_center="com",
                                           height_mode="clearance", gravity_pull=None,
                                           config=None, tag=2):
        """Generate physisorption candidate structures using Fibonacci-sphere orientational sampling.

        Parameters
        ----------
        height : float
            Placement height in A, interpreted according to ``height_mode``.
        height_mode : str
            ``"clearance"`` (default) — lowest atom of the molecule sits exactly
            ``height`` A above the substrate surface.  Physically correct for large
            molecules where the COM can be far above the binding atom.
            ``"center"`` — rotation center (COM or specified element) is placed at
            ``height`` A above the surface.  Useful when comparing multiple molecules
            at a consistent center-to-surface distance.
        gravity_pull : dict or None
            Optional downward-descent after initial placement.  Example::

                {"enabled": True, "step_size": 0.2}

            The molecule descends by ``step_size`` A per step until either the first
            vdW contact (Alvarez radii, ``overlap_scale``) or the substrate surface
            (hard floor at z_surface_ref) is encountered.  Default: disabled.
        """
        from .surface_utils import CavityDetector, identify_protectors

        # --- Overlap scale: Alvarez (2013) vdW fraction from config ---
        overlap_scale = self.config.get("reaction_search", {}).get(
            "candidate_filter", {}).get("overlap_scale", 0.65)

        # --- Gravity pull settings ---
        _grav = gravity_pull if gravity_pull is not None else {}
        grav_enabled = _grav.get("enabled", False)
        grav_step = float(_grav.get("step_size", 0.2))

        # --- Identify the substrate surface Z reference (top of substrate atoms, tag < 2) ---
        sub_tags = self.slab.get_tags()
        sub_mask_z = sub_tags < 2
        z_surface_ref = (float(np.max(self.slab.positions[sub_mask_z, 2]))
                         if np.any(sub_mask_z)
                         else float(np.max(self.slab.positions[:, 2])))

        # --- CavityDetector path (only when inhibitor atoms are ACTUALLY in the slab) ---
        _protex = self.config.get("reaction_search", {}).get("mechanisms", {}).get("protector", {})
        _inh_cfg = self.config.get("reaction_search", {}).get("mechanisms", {}).get("inhibitor", {})
        is_inh_active = _inh_cfg.get("enabled", False)
        slab_has_inhibitors = np.any(self.slab.get_tags() >= 2)

        target_centers = []
        if self.config and (_protex.get("enabled", False) or (is_inh_active and slab_has_inhibitors)):
            sub_idx, prot_idx = identify_protectors(self.slab, self.config, verbose=self.verbose)
            grid_res = _protex.get("cavity_grid_ang", _protex.get("grid_resolution", 0.2))
            detector = CavityDetector(self.slab, sub_idx, prot_idx, grid_res=grid_res, verbose=self.verbose)
            raw_centers = detector.find_void_centers(top_clearance=height)
            target_centers = self.get_unique_coordinates(self.slab, raw_centers, symprec=self.symprec)
            if target_centers:
                target_centers = [c + np.array([0, 0, 0.5]) for c in target_centers]

        # --- Regular surface-atom sites ---
        if not target_centers:
            from .surface_utils import find_surface_indices
            all_surface = find_surface_indices(self.slab, side="top")
            raw_target_centers = []

            for idx in all_surface:
                site_xy = self.slab.positions[idx, :2].copy()
                raw_target_centers.append(np.array([site_xy[0], site_xy[1], z_surface_ref + height]))

            # Add bridge/hollow mid-points for all adjacent surface atoms
            if len(all_surface) >= 2:
                for i in range(len(all_surface)):
                    for j in range(i + 1, len(all_surface)):
                        p1 = self.slab.positions[all_surface[i]]
                        p2 = self.slab.positions[all_surface[j]]
                        if 2.0 < np.linalg.norm(p1 - p2) < 5.0:
                            hollow = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, z_surface_ref + height])
                            raw_target_centers.append(hollow)

            # Apply symmetry reduction AFTER topological generation
            target_centers = self.get_unique_coordinates(self.slab, raw_target_centers, symprec=self.symprec)

        self.logger.info(
            f"  [Physisorption] {len(target_centers)} placement sites "
            f"(inh_in_slab={slab_has_inhibitors}, height={height:.1f} A [{height_mode}], "
            f"overlap_scale={overlap_scale:.2f}, gravity={'on @{:.2f}A'.format(grav_step) if grav_enabled else 'off'})"
        )

        # --- Roll + Tilt sampling (32 orientations total) ---
        n_rolls = 8
        n_tilts = 4
        
        candidates = []
        stats = {"total": 0, "overlap": 0, "grav_steps": 0}

        # Align molecule once: center at [0,0,0] with anchor element/COM strategy
        m_aligned = self._get_physi_alignment(molecule, mode=rot_center)

        for target_pos in target_centers:
            current_site_poses = []
            
            # Safe sampling height (far enough to rotate freely)
            # We use 10.0A as a safe ceiling for initial rotation
            safe_height = 10.0
            
            for i_roll in range(n_rolls):
                roll_angle = (360.0 / n_rolls) * i_roll
                for i_tilt in range(n_tilts):
                    stats["total"] += 1
                    tilt_angle = 15.0 * i_tilt
                    
                    m_pose = m_aligned.copy()
                    m_pose.rotate(roll_angle, [0, 0, 1], center=[0, 0, 0])
                    if tilt_angle > 0:
                        m_pose.rotate(tilt_angle, [1, 0, 0], center=[0, 0, 0])
                    
                    # Start at safe height above the target site
                    # Initial placement for descent
                    m_pose.translate(target_pos + np.array([0, 0, safe_height - height]))
                    
                    # --- Gravity Pulling: descend until first vdW contact ---
                    step_size = grav_step if grav_enabled else 0.2
                    last_safe_atoms = None
                    final_step = 0
                    
                    # Descend from safe_height (10A) down to surface (approx 0A relative)
                    max_steps = int(safe_height / step_size) + 5
                    for step in range(max_steps):
                        combined = self.slab.copy()
                        for a in m_pose: a.tag = tag
                        combined += m_pose
                        
                        if self.check_overlap(combined, overlap_scale=overlap_scale, check_internal=False):
                            break
                        else:
                            last_safe_atoms = combined.copy()
                            final_step = step
                            m_pose.translate([0, 0, -step_size])
                    
                    if last_safe_atoms is not None:
                        f_h = safe_height - (final_step * step_size)
                        if self.verbose and stats["total"] % 8 == 0:
                            self.logger.info(f"    [Gravity] Orientation {stats['total']}: Pulled down to h={f_h:.2f} A")
                        
                        last_safe_atoms.info["mechanism"] = f"Physisorption (Gravity), roll={roll_angle:.0f}, tilt={tilt_angle:.0f}, h={f_h:.2f}"
                        # Save the pose
                        candidates.append(last_safe_atoms)
                    else:
                        stats["overlap"] += 1

        from .surface_utils import standardize_vasp_atoms
        standardized_candidates = [standardize_vasp_atoms(c, z_min_offset=0.5) for c in candidates]

        # --- Candidate Deduplication (Hungarian Algorithm + Site-Relative) ---
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
        
        unique_candidates = []
        unique_orientations = [] # List of (n_atoms, 3) arrays representing relative poses
        
        for cand in standardized_candidates:
            # Identify adsorbate atoms and their relative positions to their anchor/center
            ads_idx = np.where(cand.get_tags() >= 2)[0]
            if len(ads_idx) == 0:
                unique_candidates.append(cand)
                continue
            
            # Use the center of the adsorbate as a reference for "orientation"
            # Or better, the rotation center if we had it. 
            # Let's use the average position of adsorbate atoms as a local origin.
            ads_pos = cand.positions[ads_idx]
            ads_symbols = cand.symbols[ads_idx]
            local_origin = np.mean(ads_pos, axis=0)
            rel_pos = ads_pos - local_origin
            
            is_duplicate = False
            for target_rel_pos, target_symbols in unique_orientations:
                if len(ads_symbols) != len(target_symbols):
                    continue
                if not np.array_equal(np.sort(ads_symbols), np.sort(target_symbols)):
                    continue
                
                # Check element-wise matching
                total_match = True
                for sym in np.unique(ads_symbols):
                    c_idx = [i for i, s in enumerate(ads_symbols) if s == sym]
                    t_idx = [i for i, s in enumerate(target_symbols) if s == sym]
                    
                    d_matrix = cdist(rel_pos[c_idx], target_rel_pos[t_idx])
                    row_ind, col_ind = linear_sum_assignment(d_matrix)
                    if np.any(d_matrix[row_ind, col_ind] > 0.2):
                        total_match = False
                        break
                
                if total_match:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates.append(cand)
                unique_orientations.append((rel_pos, ads_symbols))

        grav_note = f", using gravity descent" if grav_enabled else ""
        self.logger.info(
            f"  [Physisorption] {len(unique_candidates)} unique candidates from {len(target_centers)} sites "
            f"(Reduced from {len(standardized_candidates)} initial poses, {stats['overlap']} rejected by "
            f"overlap_scale={overlap_scale:.2f}{grav_note})."
        )
        return unique_candidates

    def discover_ligands(self, molecule, center_target="Si", skin=0.2, verbose=None):
        if verbose is None: verbose = self.verbose
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from ase.geometry import get_distances

        if isinstance(center_target, int):
            c_idx = center_target
        else:
            center_indices = [a.index for a in molecule if a.symbol == center_target]
            if not center_indices: return None, []
            c_idx = center_indices[0]

        n_atoms = len(molecule)
        adj_matrix = np.zeros((n_atoms, n_atoms), dtype=int)
        D, d = get_distances(molecule.positions, molecule.positions, cell=molecule.cell, pbc=molecule.pbc)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist_cutoff = chem_kb.get_radius(molecule.symbols[i], "covalent") + chem_kb.get_radius(molecule.symbols[j], "covalent") + skin
                if d[i, j] < dist_cutoff and d[i, j] > 0.1:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

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

