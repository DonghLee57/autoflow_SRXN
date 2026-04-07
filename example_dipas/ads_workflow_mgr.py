import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import add_adsorbate
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import spglib
from itertools import combinations

class AdsorptionWorkflowManager:
    """
    Generalized Adsorption Manager with Mechanistic Logging and Visual Clarity.
    """
    def __init__(self, slab, verbose=False):
        self.slab = slab
        self.verbose = verbose
        z_max = slab.positions[:, 2].max()
        all_surface = np.where(slab.positions[:, 2] > z_max - 1.5)[0]
        self.surface_indices = self.get_unique_surface_indices(slab, all_surface)
        if self.verbose:
            print(f"Surface Symmetry Analysis: {len(all_surface)} atoms reduced to {len(self.surface_indices)} sites.")
    
    def _get_rotation_center(self, atoms, mode='com'):
        """Helper to get rotation/placement center."""
        if mode == 'com':
            return atoms.get_center_of_mass()
        elif mode == 'closest':
            com = atoms.get_center_of_mass()
            idx = np.argmin(np.linalg.norm(atoms.positions - com, axis=1))
            return atoms.positions[idx]
        elif isinstance(mode, int):
            return atoms.positions[mode]
        else:
            return np.array([0.0, 0.0, 0.0])


    def get_unique_surface_indices(self, slab, indices):
        lattice, positions, numbers = slab.get_cell(), slab.get_scaled_positions(), slab.get_atomic_numbers()
        try:
            dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=0.05)
            equiv = dataset['equivalent_atoms']
            # Find the representative atom for each symmetry class that is closest to (0.5, 0.5) in fractional coords
            unique_classes = np.unique(equiv[indices])
            centered_indices = []
            for c in unique_classes:
                class_members = [i for i in indices if equiv[i] == c]
                dist_sq = np.sum((positions[class_members][:, :2] - 0.5)**2, axis=1)
                best_idx = class_members[np.argmin(dist_sq)]
                centered_indices.append(best_idx)
            return centered_indices
        except Exception:
            return self.get_unique_geometric_sites(slab, indices)


    def get_unique_geometric_sites(self, slab, indices, precision=1):
        # Use fractional coordinates for hashing to be lattice-agnostic
        scaled_pos = slab.get_scaled_positions()
        groups = {}
        for idx in indices:
            # Hash based on rounded Z height and fractional XY coordinates
            s_pos = scaled_pos[idx]
            z_pos = slab.positions[idx, 2]
            # Use precision for rounding (precision=1 means 0.1 A or 0.01 fractional)
            h = (round(z_pos, precision), round(s_pos[0] % 1.0, precision + 1), round(s_pos[1] % 1.0, precision + 1))
            if h not in groups: groups[h] = []
            groups[h].append(idx)
        
        # Pick the one closest to the center for each hash group (using fractional coordinates)
        centered_representatives = []
        for members in groups.values():
            dists = np.linalg.norm(scaled_pos[members][:, :2] - 0.5, axis=1)
            centered_representatives.append(members[np.argmin(dists)])
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
            temp_smiles = re.sub(r'SiH(\d+)', r'[SiH\1]', smiles)
            temp_smiles = re.sub(r'Si(?!H|\[)', r'[Si]', temp_smiles)
            mol = Chem.MolFromSmiles(temp_smiles)
            
        if mol is None: return None
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            # Fallback if optimization fail (molecule might be too small or complex)
            pass
            
        conf = mol.GetConformer()
        return Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=conf.GetPositions())



    def align_and_place(self, slab, molecule, reactive_indices, target_positions):
        m_copy = molecule.copy()
        m_center = m_copy.positions[reactive_indices[0]]
        m_copy.translate(target_positions[0] - m_center)
        if len(reactive_indices) > 1:
            m_vec = m_copy.positions[reactive_indices[1]] - m_copy.positions[reactive_indices[0]]
            s_vec = target_positions[1] - target_positions[0]
            m_copy.rotate(m_vec, s_vec, center=target_positions[0])
        slab_with_ads = slab.copy()
        for a in m_copy: a.tag = 2
        slab_with_ads += m_copy
        return slab_with_ads

    def check_overlap(self, atoms, cutoff=1.2, verbose=None):
        if verbose is None: verbose = self.verbose
        from ase.geometry import get_distances
        tags = atoms.get_tags()
        substrate, adsorbate = atoms[tags <= 1], atoms[tags >= 2]
        if not len(adsorbate): return False
        
        # 1. Adsorbate vs Substrate
        indices_ads = np.where(tags >= 2)[0]
        indices_sub = np.where(tags <= 1)[0]
        D_sub, d_sub = get_distances(adsorbate.positions, substrate.positions, cell=atoms.cell, pbc=atoms.pbc)
        # d_sub is (n_ads, n_sub)
        clashes = np.where(d_sub < cutoff)
        if len(clashes[0]) > 0:
            if verbose:
                for i_a, i_s in zip(clashes[0], clashes[1]):
                    a_idx, s_idx = indices_ads[i_a], indices_sub[i_s]
                    print(f"    [Overlap] {atoms.symbols[a_idx]}({a_idx}) - {atoms.symbols[s_idx]}({s_idx}) clash (dist: {d_sub[i_a, i_s]:.2f} A)")
            return True
            
        # 2. Adsorbate internal (Core vs Ligand)
        if len(np.unique(tags[tags >= 2])) > 1:
            unique_tags = np.sort(np.unique(tags[tags >= 2]))
            for i_t, t1 in enumerate(unique_tags):
                for t2 in unique_tags[i_t+1:]:
                    idx1 = np.where(tags == t1)[0]
                    idx2 = np.where(tags == t2)[0]
                    D_int, d_int = get_distances(atoms.positions[idx1], atoms.positions[idx2], cell=atoms.cell, pbc=atoms.pbc)
                    clashes_int = np.where(d_int < cutoff)
                    if len(clashes_int[0]) > 0:
                        if verbose:
                            for i1, i2 in zip(clashes_int[0], clashes_int[1]):
                                a1_idx, a2_idx = idx1[i1], idx2[i2]
                                print(f"    [Overlap] {atoms.symbols[a1_idx]}({a1_idx}) - {atoms.symbols[a2_idx]}({a2_idx}) clash (dist: {d_int[i1, i2]:.2f} A)")
                        return True
        return False


    def generate_physisorption_candidates(self, molecule, height=3.5, n_rot=16, rot_center='com'):
        from itertools import product
        phi = np.pi * (3.0 - np.sqrt(5.0))
        # Unique rotations
        rot_vectors, sampled_coords = [], []
        
        # Determine center for rotation/sampling
        initial_center = self._get_rotation_center(molecule, mode=rot_center)
        
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
        z_max = self.slab.positions[:, 2].max()
        stats = {'total': 0, 'overlap': 0}
        for idx in self.surface_indices:
            site = self.slab.positions[idx]
            for rv in rot_vectors:
                stats['total'] += 1
                m_copy = molecule.copy()
                # Rotate around chosen center
                c_pos_init = self._get_rotation_center(m_copy, mode=rot_center)
                m_copy.rotate([0,0,1], rv, center=c_pos_init)
                
                # Tag adsorbate for overlap detection
                for a in m_copy: a.tag = 2
                
                # Manual placement: place chosen center at site + height
                c_pos_rotated = self._get_rotation_center(m_copy, mode=rot_center)
                target_pos = np.array([site[0], site[1], z_max + height])
                m_copy.translate(target_pos - c_pos_rotated)
                
                slab_copy = self.slab.copy()
                slab_copy += m_copy # Manual addition instead of add_adsorbate
                
                if not self.check_overlap(slab_copy, cutoff=1.2):
                    slab_copy.info['mechanism'] = f"Physisorption on Site {idx}, center={rot_center}"
                    candidates.append(slab_copy)
                else:
                    stats['overlap'] += 1
        
        if self.verbose:
            print(f"Physisorption Search: Generated {len(candidates)} candidates from {stats['total']} attempts ({stats['overlap']} skipped due to overlaps).")
        return candidates


    def discover_ligands(self, molecule, center_target='Si', skin=0.2, verbose=None):
        if verbose is None: verbose = self.verbose
        """
        Discover ligands and their hapticity using graph partitioning.
        Includes the bond vector (from center to ligand) for alignment.
        """
        from ase.data import covalent_radii
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        
        if isinstance(center_target, int):
            c_idx = center_target
            if c_idx < 0 or c_idx >= len(molecule): return None, []
        else:
            center_indices = [a.index for a in molecule if a.symbol == center_target]
            if not center_indices: return None, []
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
            if comp_id == center_label: continue
                
            frag_indices = np.where(labels == comp_id)[0]
            binding_atoms = list(set(frag_indices).intersection(bonded_indices))
            hapticity = len(binding_atoms)
            
            if hapticity > 0:
                # Calculate reference bond vector (from center to binding geometric center)
                frag_atoms = molecule[frag_indices]
                formula = frag_atoms.get_chemical_formula()
                
                binding_pos = np.mean(molecule.positions[binding_atoms], axis=0)
                bond_vec = binding_pos - molecule.positions[c_idx]
                
                ligands.append({
                    'formula': formula,
                    'indices': list(frag_indices),
                    'binding_atoms': binding_atoms,
                    'hapticity': hapticity,
                    'bond_vec': bond_vec # Vector from center to ligand
                })

                
        if verbose:
            print(f"Precursor Fragmentation Analysis ({center_target} centered):")
            print(f"  Found {len(ligands)} ligands attached to index {c_idx}.")
            for i, l in enumerate(ligands):
                print(f"  - Ligand {i}: {l['formula']} (hapticity={l['hapticity']}), atoms: {l['indices']}")
                
        return c_idx, ligands

    def _place_at_dangling_bond(self, fragment, binding_idx, internal_bond_vec, target_site_pos, db_vector, bond_length, rot_angle=0):
        """Precise placement and rotation of a fragment on a surface site."""
        f = fragment.copy()
        # db_vector points AWAY from surface. 
        # internal_bond_vec points AWAY from fragment core. 
        # To bond, fragment's internal_bond_vec must point TOWARD the surface (-db_vector).
        f.rotate(internal_bond_vec, -db_vector, center=f.positions[binding_idx])
        f.rotate(rot_angle, db_vector, center=f.positions[binding_idx])
        
        # Position binding_idx at target_site_pos + normalized(db_vector) * bond_length
        placement_pos = target_site_pos + (db_vector / np.linalg.norm(db_vector)) * bond_length
        f.translate(placement_pos - f.positions[binding_idx])
        return f



    def _form_byproduct(self, fragment, binding_idx, internal_bond_vec):
        """Helper to create a byproduct molecule (Ligand + H)."""
        from ase import Atoms
        f = fragment.copy()
        sym = f.symbols[binding_idx]
        b_len = 1.0 if sym in ['N', 'O'] else 1.1 if sym == 'C' else 1.5
        
        h_pos = f.positions[binding_idx] + (internal_bond_vec / np.linalg.norm(internal_bond_vec)) * b_len
        f += Atoms('H', positions=[h_pos])
        return f




