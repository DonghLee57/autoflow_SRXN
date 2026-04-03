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
    def __init__(self, slab):
        self.slab = slab
        z_max = slab.positions[:, 2].max()
        all_surface = np.where(slab.positions[:, 2] > z_max - 1.5)[0]
        self.surface_indices = self.get_unique_surface_indices(slab, all_surface)
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

    def check_overlap(self, atoms, cutoff=1.2, verbose=False):
        from ase.geometry import get_distances
        tags = atoms.get_tags()
        substrate, adsorbate = atoms[tags <= 1], atoms[tags >= 2]
        if not len(adsorbate): return False
        
        # 1. Adsorbate vs Substrate
        _, d_sub = get_distances(adsorbate.positions, substrate.positions, cell=atoms.cell, pbc=atoms.pbc)
        if np.any(d_sub < cutoff):
            if verbose: print(f"    [Overlap] Adsorbate-Substrate clash detected ({np.sum(d_sub < cutoff)} pairs)")
            return True
            
        # 2. Adsorbate internal (Core vs Ligand)
        if len(np.unique(tags[tags >= 2])) > 1:
            # Check overlap between fragments with different tags
            for t1 in np.unique(tags[tags >= 2]):
                for t2 in np.unique(tags[tags >= 2]):
                    if t1 >= t2: continue
                    p1 = atoms.positions[tags == t1]
                    p2 = atoms.positions[tags == t2]
                    _, d_int = get_distances(p1, p2, cell=atoms.cell, pbc=atoms.pbc)
                    if np.any(d_int < cutoff):
                        if verbose: print(f"    [Overlap] Fragment-Fragment clash detected ({np.sum(d_int < cutoff)} pairs)")
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
        for idx in self.surface_indices:
            site = self.slab.positions[idx]
            for rv in rot_vectors:
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
        return candidates


    def discover_ligands(self, molecule, center_symbol='Si', skin=0.2):
        """
        Discover ligands and their hapticity using graph partitioning.
        Includes the bond vector (from center to ligand) for alignment.
        """
        from ase.data import covalent_radii
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        
        center_indices = [a.index for a in molecule if a.symbol == center_symbol]
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

    def generate_chemisorption_candidates(self, molecule, center_symbol='Si', rot_steps=12):
        """
        High-fidelity chemisorption generation with rotational steric screening.
        Focuses on Core+1Ligand pair on a Dimer.
        """
        from si_surface_utils import get_dangling_bond_info, reconstruct_2x1_buckled, find_existing_dimers
        
        c_idx, ligands = self.discover_ligands(molecule, center_symbol=center_symbol)
        if not ligands: return []
        
        # 1. Identify Dimer pairs for site mapping
        # Try reconstruction first, then fallback to existing dimer discovery
        dimers = reconstruct_2x1_buckled(self.slab)
        if not dimers:
            dimers = find_existing_dimers(self.slab)
            
        if not dimers:
            print("Warning: No dimers found on surface. Chemisorption requires site pairs.")
            return []
        print(f"DEBUG: Found {len(dimers)} dimers for chemisorption.")
        candidates = []



        
        # 2. Iterate each dissociation pathway (Cohesive Dissociation)
        seen_formulas = set()
        for l_info in ligands:
            formula = l_info.get('formula', 'Unknown')
            if formula in seen_formulas: continue
            seen_formulas.add(formula)
            
            # Fragment B: The leaving ligand

            indices_b = l_info['indices']
            frag_b = molecule[indices_b]
            binding_idx_b = indices_b.index(l_info['binding_atoms'][0])
            
            # Fragment A: The Heavy Piece (Everything else including the Center)
            indices_a = list(set(range(len(molecule))) - set(indices_b))
            frag_a = molecule[indices_a]
            # Mapping c_idx to frag_a
            binding_idx_a = indices_a.index(c_idx)
            
            # For each Dimer, try both pairing orientations
            for (idx1, idx2) in dimers:
                db1 = get_dangling_bond_info(self.slab, idx1)
                db2 = get_dangling_bond_info(self.slab, idx2)
                if not db1 or not db2: continue
                
                for s1, s2 in [(db1, db2), (db2, db1)]:
                    best_pose = None
                    
                    # Rotational Screening (Simplified 1D scan for now)
                    for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                        # Place Heavy Piece (A) 
                        # Vector it lost was l_info['bond_vec']
                        p_a = self._place_at_dangling_bond(frag_a, binding_idx_a, l_info['bond_vec'], 
                                                           s1['pos'], s1['db_vector'], 2.35, rot_angle=angle)
                        
                        # Place Leaving Piece (B)
                        # Vector it lost was -l_info['bond_vec']
                        bond_len_b = 1.48 if frag_b.symbols[binding_idx_b] == 'H' else 2.1
                        p_b = self._place_at_dangling_bond(frag_b, binding_idx_b, -l_info['bond_vec'], 
                                                           s2['pos'], s2['db_vector'], bond_len_b, rot_angle=0)
                        
                        combined = self.slab.copy()
                        for a in p_a: a.tag = 2
                        combined += p_a
                        for a in p_b: a.tag = 3
                        combined += p_b
                        
                        # Check overlap
                        if not self.check_overlap(combined, cutoff=1.2, verbose=False):
                            comp_a = "".join(frag_a.symbols)
                            combined.info['mechanism'] = f"Cohesive Chemisorption: {comp_a} on {s1['index']}, {frag_b.symbols[binding_idx_b]} on {s2['index']}, rot={angle:.1f}"
                            best_pose = combined
                            break
                    
                    if best_pose:
                        candidates.append(best_pose)
                        break 
        return candidates

    def generate_h_exchange_candidates(self, molecule, center_symbol='Si', rot_steps=12):
        """
        Specialized candidate generator for passivated surfaces.
        Mechanism: Precursor-L + Surface-H -> Surface-Precursor (Fragment) + L-H (By-product)
        """
        from si_surface_utils import get_surface_h_mapping
        c_idx, ligands = self.discover_ligands(molecule, center_symbol=center_symbol)
        if not ligands: return []
        
        # 1. Identify Surface-H sites
        h_mapping = get_surface_h_mapping(self.slab)
        if not h_mapping:
            print("Warning: No surface-H found. Perhaps the surface is not passivated?")
            return []
        
        candidates = []
        
        # 2. Iterate each dissociation pathway
        seen_formulas = set()
        for l_info in ligands:
            formula = l_info.get('formula', 'Unknown')
            if formula in seen_formulas: continue
            seen_formulas.add(formula)
            
            # Piece B: The leaving ligand

            indices_b = l_info['indices']
            frag_b = molecule[indices_b]
            binding_idx_b = indices_b.index(l_info['binding_atoms'][0])
            
            # Piece A: The Heavy Piece (remaining precursor)
            indices_a = list(set(range(len(molecule))) - set(indices_b))
            frag_a = molecule[indices_a]
            binding_idx_a = indices_a.index(c_idx)
            
            # For each available surface-H site
            for si_idx, h_idx in h_mapping.items():
                # Define bond vector: From Si to H
                si_pos = self.slab.positions[si_idx]
                h_pos = self.slab.positions[h_idx]
                h_vec = h_pos - si_pos
                h_vec_norm = h_vec / np.linalg.norm(h_vec)
                
                # Active vacancy site is si_idx + h_vec_norm * bond_length
                
                # Rotational scan for Fragment A on the surface vacancy
                for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                    # Place Fragment A (Si-Si bond)
                    # Vector it lost was l_info['bond_vec']
                    p_a = self._place_at_dangling_bond(frag_a, binding_idx_a, l_info['bond_vec'], 
                                                       si_pos, h_vec_norm, 2.35, rot_angle=angle)
                    
                    # Form Fragment B + H (Byproduct)
                    p_b = self._form_byproduct(frag_b, binding_idx_b, -l_info['bond_vec'])
                    
                    # Offset byproduct ~4A above the highest surface atom
                    z_clearance = np.max(self.slab.positions[:, 2]) + 4.0
                    p_b.translate([si_pos[0], si_pos[1], z_clearance] - p_b.positions[0])
                    
                    # Combine: Surface (without this H) + Fragment A + Byproduct
                    final = self.slab.copy()
                    # Assign tags for visualization/analysis
                    for i_at in range(len(final)): final[i_at].tag = 0 # Surface Si
                    # Mark the specific H we'll delete (actually just delete it)
                    
                    # Instead of deleting from an atoms object while iterating, we reconstruct:
                    # new_slab = [ slab minus h_idx ]
                    # This is tricky because indices shift. Safer to zero out H but that's messy.
                    
                    # Better: create a list of indices to keep
                    keep_indices = [i for i in range(len(self.slab)) if i != h_idx]
                    reduced_slab = self.slab[keep_indices]
                    
                    combined = reduced_slab.copy()
                    for a in p_a: a.tag = 2 # Adsorbed Core
                    combined += p_a
                    for a in p_b: a.tag = 3 # Byproduct
                    combined += p_b
                    
                    if not self.check_overlap(combined, cutoff=1.2, verbose=False):
                        comp_a = "".join(frag_a.symbols)
                        combined.info['mechanism'] = f"H-Exchange: {comp_a} on Si_{si_idx}, byproduct {frag_b.symbols[binding_idx_b]}-H"
                        candidates.append(combined)
                        break
        return candidates

        return candidates

    def _form_byproduct(self, fragment, binding_idx, internal_bond_vec):
        """Helper to create a byproduct molecule (Ligand + H)."""
        from ase import Atoms
        f = fragment.copy()
        sym = f.symbols[binding_idx]
        b_len = 1.0 if sym in ['N', 'O'] else 1.1 if sym == 'C' else 1.5
        
        h_pos = f.positions[binding_idx] + (internal_bond_vec / np.linalg.norm(internal_bond_vec)) * b_len
        f += Atoms('H', positions=[h_pos])
        return f




