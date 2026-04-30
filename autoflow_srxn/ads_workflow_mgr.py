from itertools import combinations

import numpy as np
import spglib
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from .logger_utils import get_workflow_logger
from .surface_utils import calculate_haptic_normal, calculate_haptic_vbs


class AdsorptionWorkflowManager:
    """Generalized Adsorption Manager with Mechanistic Logging and Visual Clarity."""

    def __init__(self, slab, config=None, symprec=0.2, verbose=False):
        self.slab = slab
        self.config = config if config is not None else {}
        self.verbose = verbose
        self.symprec = symprec
        self.logger = get_workflow_logger()

        z_max = slab.positions[:, 2].max()
        all_surface = np.where(slab.positions[:, 2] > z_max - 1.5)[0]
        self.surface_indices = self.get_unique_surface_indices(slab, all_surface, symprec=self.symprec)
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

        # We first try the user-provided symprec. If it fails to reduce anything AND it's low, we try to increment it up to 0.5 to force reduction.
        # But generally we respect the user's symprec if it works.
        try_precisions = [symprec]
        if symprec < 0.5:
            try_precisions += [0.5]

        for prec in try_precisions:
            try:
                dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=prec)
                if dataset is None:
                    continue

                # Handling SPGlib >= 2.0 where dataset is an object
                if hasattr(dataset, "equivalent_atoms"):
                    equiv = dataset.equivalent_atoms
                else:
                    # Fallback for older dict interface
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

    def get_unique_geometric_sites(self, slab, indices, cutoff=1.5):
        # Distance-based agglomeration clustering fallback
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
            # Pick the one closest to fractional center (0.5, 0.5) to avoid edge artifacts
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
            # Try to fix common Silicon-based groups if they are not bracketed
            # Handles SiH3, SiH2, SiH, and generic Si
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
            # Fallback if optimization fail
            pass

        # Convert RDKit Mol to ASE Atoms
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        return Atoms(symbols=symbols, positions=positions)

    def check_overlap(self, atoms, cutoff=None, verbose=False):
        """Rigid-body overlap check using a configurable threshold.
        Supports multi-stage tagging:
          - Substrate: Tags 0, 1
          - Adsorbates: Tags >= 2
        """
        import numpy as np
        from ase.geometry import get_distances

        cand_filter = self.config.get("reaction_search", {}).get("candidate_filter", {})
        env_cutoff = cand_filter.get("overlap_cutoff", 2.5)
        # Tighter threshold for substrate-adsorbate contacts (allows bonding proximity)
        sub_cutoff = 1.5

        if cutoff is not None:
            # If a local cutoff is provided (e.g. from chemisorption builder), use it for everything
            env_cutoff = cutoff
            sub_cutoff = cutoff

        pos = atoms.positions
        cell = atoms.cell
        pbc = atoms.pbc

        tags = atoms.get_tags()
        max_tag = np.max(tags)

        if max_tag < 2:
            return False

        # We only check distance between the atoms added in the CURRENT stage (highest tag)
        # and all other atoms.
        new_indices = np.where(tags == max_tag)[0]
        if len(new_indices) == 0:
            return False

        for idx in new_indices:
            mask = np.ones(len(atoms), dtype=bool)
            mask[tags == max_tag] = False  # Skip internal checks
            mask_indices = np.where(mask)[0]

            ref_pos = pos[idx]
            other_pos = pos[mask]
            other_tags = tags[mask]

            _, d_list = get_distances(ref_pos, other_pos, cell=cell, pbc=pbc)
            d_list = d_list.flatten()  # Ensure 1D array for zipping

            for d, o_idx, o_tag in zip(d_list, mask_indices, other_tags):
                # Threshold depends on whether we are hitting the substrate or another adsorbate
                threshold = sub_cutoff if o_tag < 2 else env_cutoff
                if d < threshold:
                    if verbose:
                        print(
                            f"  [Overlap] Collision: Atom {idx}(tag {tags[idx]}) and {o_idx}(tag {o_tag}) at {d:.2f} A (Threshold: {threshold} A)"
                        )
                    return True
        return False

    def _get_steric_fitness(self, atoms, cutoff=None):
        """Calculates a 'fitness' score.
        Checks for hard collisions and then soft-repulsion.
        """
        # 1. Hard Collision Check (using context-aware logic)
        if self.check_overlap(atoms, cutoff=cutoff, verbose=False):
            return -1e9  # Overlap

        # 2. Soft-repulsion score
        # Calculate distances between NEW atoms and ALL environment atoms
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

        # Soft-repulsion score: favor larger distances
        score = -np.sum(1.0 / (dists**6 + 1e-6))
        return score

    def _get_diverse_top_poses(self, poses, n_out=5, angle_threshold=45.0):
        """Filters a list of (score, atoms, rotation_vec) to return top N diverse poses."""
        if not poses:
            return []
        # Sort by score descending
        poses.sort(key=lambda x: x[0], reverse=True)

        selected = [poses[0]]
        for p in poses[1:]:
            if len(selected) >= n_out:
                break

            # Check rotation diversity
            is_diverse = True
            for s in selected:
                # Dot product of rotation vectors
                v1, v2 = p[2], s[2]
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                if angle < angle_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(p)

        return [s[1] for s in selected]

    def _get_rotation_center(self, atoms, mode="com"):
        """Helper to get rotation/placement center. Supports 'com', 'closest', index, or element symbol."""
        if mode == "com":
            return atoms.get_center_of_mass()
        elif mode == "closest":
            com = atoms.get_center_of_mass()
            idx = np.argmin(np.linalg.norm(atoms.positions - com, axis=1))
            return atoms.positions[idx]
        elif isinstance(mode, int):
            return atoms.positions[mode]
        elif isinstance(mode, str):
            # Check if it is an element symbol
            indices = [a.index for a in atoms if a.symbol == mode]
            if indices:
                return atoms.positions[indices[0]]
        return np.array([0.0, 0.0, 0.0])

    def generate_physisorption_candidates(self, molecule, height=3.5, n_rot=32, rot_center="com", config=None, tag=2):
        from .surface_utils import CavityDetector, identify_protectors

        phi = np.pi * (3.0 - np.sqrt(5.0))
        # Unique rotations
        rot_vectors, sampled_coords = [], []

        # Determine center for rotation/sampling
        initial_center = self._get_rotation_center(molecule, mode=rot_center)

        # Heuristic: If rot_center is a specific atom/element, try no-rotation first.
        is_fixed_center = rot_center not in ["com", "closest"]

        for i in range(n_rot):
            if n_rot > 1:
                y = 1 - (i / float(n_rot - 1)) * 2
            else:
                y = 1.0
            r = np.sqrt(1 - y * y)
            theta = phi * i
            vec = np.array([np.cos(theta) * r, y, np.sin(theta) * r])

            # Simple check to avoid duplicated vectors
            if not any(np.allclose(vec, rv, atol=0.01) for rv in rot_vectors):
                rot_vectors.append(vec)

        candidates = []
        stats = {"total": 0, "overlap": 0}

        target_centers = []
        conf = config if config is not None else self.config
        _protex = (
            conf.get("reaction_search", {}).get("mechanisms", {}).get("protector_exchange", conf.get("protector", {}))
        )
        if conf and _protex.get("enabled", False):
            sub_idx, prot_idx = identify_protectors(self.slab, conf, verbose=self.verbose)
            grid_res = _protex.get("cavity_grid_ang", _protex.get("grid_resolution", 0.2))
            detector = CavityDetector(self.slab, sub_idx, prot_idx, grid_res=grid_res, verbose=self.verbose)
            target_centers = detector.find_void_centers(top_clearance=height)
        else:
            z_max = self.slab.positions[:, 2].max()
            for idx in self.surface_indices:
                site = self.slab.positions[idx]
                target_centers.append(np.array([site[0], site[1], z_max + height]))

        # Get global overlap cutoff from config
        global_overlap = self.config.get("reaction_search", {}).get("candidate_filter", {}).get("overlap_cutoff", 2.0)

        for target_pos in target_centers:
            current_site_poses = []

            # Stage A: If fixed center is specified, try identity rotation first.
            if is_fixed_center:
                m_identity = molecule.copy()
                c_pos = self._get_rotation_center(m_identity, mode=rot_center)
                m_identity.translate(target_pos - c_pos)

                combined_id = self.slab.copy()
                for a in m_identity:
                    a.tag = tag
                combined_id += m_identity

                score_id = self._get_steric_fitness(combined_id, cutoff=global_overlap)
                if score_id > -1e8:
                    # Found a valid no-rotation pose!
                    combined_id.info["mechanism"] = f"Physisorption, center={rot_center}, identity_rot, tag={tag}"
                    candidates.append(combined_id)
                    stats["total"] += 1
                    continue  # Move to next site, skipping rotation search for this site.

            # Stage B: Fallback to rotation search (or normal rotation search if not fixed center)
            for rv in rot_vectors:
                stats["total"] += 1
                m_copy = molecule.copy()
                c_pos_init = self._get_rotation_center(m_copy, mode=rot_center)
                m_copy.rotate([0, 0, 1], rv, center=c_pos_init)

                c_pos_rotated = self._get_rotation_center(m_copy, mode=rot_center)
                m_copy.translate(target_pos - c_pos_rotated)

                combined = self.slab.copy()
                for a in m_copy:
                    a.tag = tag
                combined += m_copy

                # Use Steric Fitness to evaluate pose
                score = self._get_steric_fitness(combined, cutoff=global_overlap)
                if score > -1e8:  # Valid pose
                    combined.info["mechanism"] = f"Physisorption, center={rot_center}, tag={tag}"
                    current_site_poses.append((score, combined, rv))
                else:
                    stats["overlap"] += 1

            # Select Top 5 diverse poses for this site
            best_poses = self._get_diverse_top_poses(current_site_poses, n_out=5)
            candidates.extend(best_poses)

        if self.verbose:
            print(
                f"Physisorption Search (tag={tag}): Generated {len(candidates)} candidates from {len(target_centers)} sites ({stats['total']} total orientation attempts, {stats['overlap']} skipped)."
            )
        return candidates

    def discover_ligands(self, molecule, center_target="Si", skin=0.2, verbose=None):
        if verbose is None:
            verbose = self.verbose
        """
        Discover ligands and their hapticity using graph partitioning.
        Includes the bond vector (from center to ligand) for alignment.
        """
        from ase.data import covalent_radii
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        if isinstance(center_target, int):
            c_idx = center_target
            if c_idx < 0 or c_idx >= len(molecule):
                return None, []
        else:
            center_indices = [a.index for a in molecule if a.symbol == center_target]
            if not center_indices:
                return None, []
            c_idx = center_indices[0]

        n_atoms = len(molecule)
        adj_matrix = np.zeros((n_atoms, n_atoms), dtype=int)

        from ase.geometry import get_distances

        D, d = get_distances(molecule.positions, molecule.positions, cell=molecule.cell, pbc=molecule.pbc)

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist_cutoff = covalent_radii[molecule.numbers[i]] + covalent_radii[molecule.numbers[j]] + skin
                if d[i, j] < dist_cutoff and d[i, j] > 0.1:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        center_bonded_mask = adj_matrix[c_idx, :] == 1
        bonded_indices = np.where(center_bonded_mask)[0]

        adj_matrix[c_idx, :] = 0
        adj_matrix[:, c_idx] = 0

        graph = csr_matrix(adj_matrix)
        n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

        ligands = []
        center_label = labels[c_idx]

        for comp_id in range(n_components):
            if comp_id == center_label:
                continue

            frag_indices = np.where(labels == comp_id)[0]
            binding_atoms = list(set(frag_indices).intersection(bonded_indices))
            hapticity = len(binding_atoms)

            if hapticity > 0:
                # Calculate reference bond vector (from center to binding geometric center)
                frag_atoms = molecule[frag_indices]
                formula = frag_atoms.get_chemical_formula()

                binding_pos = np.mean(molecule.positions[binding_atoms], axis=0)
                bond_vec = binding_pos - molecule.positions[c_idx]

                # Haptic Geometry: VBS and Normal
                vbs = calculate_haptic_vbs(molecule, binding_atoms)
                normal = calculate_haptic_normal(molecule, binding_atoms)

                # Orient normal toward the metal center for consistent alignment logic
                vec_to_metal = molecule.positions[c_idx] - vbs
                if np.dot(normal, vec_to_metal) < 0:
                    normal = -normal

                ligands.append(
                    {
                        "formula": formula,
                        "indices": list(frag_indices),
                        "binding_atoms": binding_atoms,
                        "hapticity": hapticity,
                        "bond_vec": bond_vec,  # Vector from center to ligand
                        "vbs_pos": vbs,
                        "normal_vector": normal,
                    }
                )

        if verbose:
            print(f"Precursor Fragmentation Analysis ({center_target} centered):")
            print(f"  Found {len(ligands)} ligands attached to index {c_idx}.")
            for i, l in enumerate(ligands):
                print(f"  - Ligand {i}: {l['formula']} (hapticity={l['hapticity']}), atoms: {l['indices']}")
        return c_idx, ligands

    def _place_at_dangling_bond(
        self,
        fragment,
        binding_idx,
        internal_bond_vec,
        target_site_pos,
        db_vector,
        bond_length,
        rot_angle=0,
        haptic_normal=None,
    ):
        """Precise placement and rotation of a fragment on a surface site."""
        f = fragment.copy()

        # Determine anchor and alignment vector
        if isinstance(binding_idx, (list, np.ndarray)) and len(binding_idx) > 1:
            anchor_pos = np.mean(f.positions[binding_idx], axis=0)
            align_vec = haptic_normal if haptic_normal is not None else internal_bond_vec
        else:
            b_idx = binding_idx[0] if isinstance(binding_idx, (list, np.ndarray)) else binding_idx
            anchor_pos = f.positions[b_idx]
            align_vec = internal_bond_vec

        # Alignment: align_vec must point TOWARD the surface (-db_vector)
        # Note: haptic_normal was oriented toward metal in fragmentation,
        # so it acts like the bond vector from ligand to metal.
        f.rotate(align_vec, -db_vector, center=anchor_pos)
        f.rotate(rot_angle, db_vector, center=anchor_pos)

        # Position anchor at target_site_pos + normalized(db_vector) * bond_length
        placement_pos = target_site_pos + (db_vector / np.linalg.norm(db_vector)) * bond_length
        f.translate(placement_pos - anchor_pos)
        return f

    def _form_byproduct(self, fragment, binding_idx, internal_bond_vec):
        """Helper to create a byproduct molecule (Ligand + H)."""
        from ase import Atoms

        f = fragment.copy()
        sym = f.symbols[binding_idx]
        b_len = 1.0 if sym in ["N", "O"] else 1.1 if sym == "C" else 1.5

        h_pos = f.positions[binding_idx] + (internal_bond_vec / np.linalg.norm(internal_bond_vec)) * b_len
        f += Atoms("H", positions=[h_pos])
        return f
