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

    def get_unique_surface_indices(self, slab, indices):
        lattice, positions, numbers = slab.get_cell(), slab.get_scaled_positions(), slab.get_atomic_numbers()
        try:
            dataset = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=0.05)
            equiv = dataset['equivalent_atoms']
            unique_indices, seen_classes = [], set()
            for idx in indices:
                if equiv[idx] not in seen_classes:
                    unique_indices.append(idx); seen_classes.add(equiv[idx])
            return unique_indices
        except Exception:
            return self.get_unique_geometric_sites(slab, indices)

    def get_unique_geometric_sites(self, slab, indices):
        unique, hashes = [], []
        for idx in indices:
            pos = slab.positions[idx]
            h = (round(pos[2], 1), round(pos[0] % 5.43, 1), round(pos[1] % 5.43, 1))
            if h not in hashes: unique.append(idx); hashes.append(h)
        return unique

    def get_all_adjacent_sites(self, slab, core_idx, k, max_dist=4.5):
        from ase.geometry import get_distances
        _, d_list = get_distances(slab.positions[core_idx], slab.positions, cell=slab.cell, pbc=slab.pbc)
        dists = d_list[0]
        z_max = slab.positions[:, 2].max()
        surface_mask = slab.positions[:, 2] > z_max - 1.5
        adj_indices = np.where((dists > 0.1) & (dists < max_dist) & surface_mask)[0]
        for cluster_indices in combinations(adj_indices, k):
            yield (core_idx,) + cluster_indices

    def generate_rdkit_conformer(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles.replace("SiH3", "[SiH3]").replace("SiH2", "[SiH2]"))
        if mol is None: return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
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

    def check_overlap(self, atoms, cutoff=1.2):
        from ase.geometry import get_distances
        tags = atoms.get_tags()
        substrate, adsorbate = atoms[tags <= 1], atoms[tags >= 2]
        if not len(adsorbate): return False
        _, d = get_distances(adsorbate.positions, substrate.positions, cell=atoms.cell, pbc=atoms.pbc)
        return np.any(d < cutoff)

    def generate_physisorption_candidates(self, molecule, height=3.5, n_rot=16):
        from itertools import product
        phi = np.pi * (3.0 - np.sqrt(5.0))
        # Unique rotations
        rot_vectors, sampled_coords = [], []
        for i in range(n_rot):
            y = 1 - (i / float(n_rot - 1)) * 2
            r = np.sqrt(1 - y * y)
            theta = phi * i
            vec = np.array([np.cos(theta) * r, y, np.sin(theta) * r])
            m_test = molecule.copy()
            m_test.rotate([0, 0, 1], vec)
            m_test.center()
            d = np.sort(np.linalg.norm(m_test.positions, axis=1))
            if not any(np.allclose(d, pd, atol=0.1) for pd in sampled_coords):
                rot_vectors.append(vec); sampled_coords.append(d)
        
        candidates = []
        for idx in self.surface_indices:
            site = self.slab.positions[idx]
            for rv in rot_vectors:
                m_copy = molecule.copy()
                m_copy.rotate([0,0,1], rv)
                slab_copy = self.slab.copy()
                add_adsorbate(slab_copy, m_copy, height=height, position=(site[0], site[1]))
                if not self.check_overlap(slab_copy, cutoff=1.2):
                    slab_copy.info['mechanism'] = f"Physisorption on Site {idx}"
                    candidates.append(slab_copy)
        return candidates

    def get_fragment_indices(self, molecule, start_atom, broken_bond_atom):
        """Find all indices belonging to the fragment after breaking a bond."""
        from ase.neighborlist import neighbor_list
        # Higher cutoff to ensure all atoms (including H) are captured
        i_list, j_list = neighbor_list('ij', molecule, 2.2)
        
        fragment = {broken_bond_atom}
        queue = [broken_bond_atom]
        while queue:
            curr = queue.pop(0)
            for neighbor in j_list[i_list == curr]:
                if neighbor != start_atom and neighbor not in fragment:
                    fragment.add(neighbor)
                    queue.append(neighbor)
        return list(fragment)

    def generate_multidentate_candidates(self, molecule, center_symbol='Si', k_max=2):
        center_indices = [a.index for a in molecule if a.symbol == center_symbol]
        if not center_indices: return []
        c_idx = center_indices[0]
        from ase.neighborlist import neighbor_list
        i_list, j_list = neighbor_list('ij', molecule, 2.2)
        all_ligands = j_list[i_list == c_idx]
        
        candidates = []
        for k in range(1, k_max + 1):
            for ligand_subset in combinations(all_ligands, k):
                for s_core in self.surface_indices:
                    for s_cluster in self.get_all_adjacent_sites(self.slab, s_core, k):
                        m_copy = molecule.copy()
                        s_positions = [self.slab.positions[s] for s in s_cluster]
                        
                        # TRUE DISSOCIATION: Displace fragments independently
                        tag_map = np.zeros(len(m_copy), dtype=int)
                        # Move each dissociated ligand fragment to its target site
                        for i, l_idx in enumerate(ligand_subset):
                            frag_indices = self.get_fragment_indices(molecule, c_idx, l_idx)
                            # Target is 2.2 A above the dimer partner
                            target_p = s_positions[i+1] + np.array([0,0,2.2])
                            current_p = m_copy.positions[l_idx]
                            m_copy.positions[frag_indices] += (target_p - current_p)
                            tag_map[frag_indices] = 1 
                        
                        # Move the core to the core site
                        core_indices = np.where(tag_map == 0)[0]
                        target_p_core = s_positions[0] + np.array([0,0,2.2])
                        current_p_core = m_copy.positions[c_idx]
                        m_copy.positions[core_indices] += (target_p_core - current_p_core)
                        
                        slab_copy = self.slab.copy()
                        for a in m_copy: a.tag = 2
                        slab_copy += m_copy
                        
                        if not self.check_overlap(slab_copy, cutoff=1.2):
                            ligands_str = ", ".join([molecule.symbols[l] for l in ligand_subset])
                            slab_copy.info['mechanism'] = f"True Dissociation (k={k}): {ligands_str} cleaved, Fragments on {s_cluster}"
                            candidates.append(slab_copy)
                            break 
        return candidates
