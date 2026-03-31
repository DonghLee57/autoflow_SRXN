import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances

def find_surface_indices(atoms, side='top', threshold=0.2):
    """Find indices of atoms at the top or bottom surface based on Z-coordinates."""
    z_coords = atoms.positions[:, 2]
    if side == 'top':
        z_target = np.max(z_coords)
    else:
        z_target = np.min(z_coords)
    return np.where(np.abs(z_coords - z_target) < threshold)[0]

def get_missing_tetrahedral_vectors(atoms, idx, cutoff=2.6, bond_length=1.48):
    """
    Given a Si atom index, identify its missing tetrahedral bond vectors.
    PBC-Aware: Uses the distance vectors 'D' returned by neighbor_list.
    """
    # 1. Get neighbors with MIC/PBC
    i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)
    neighbors_D = D_list[i_list == idx]
    
    # Normalize neighbors to unit vectors
    unit_vectors = []
    for d in neighbors_D:
        mag = np.linalg.norm(d)
        if mag > 0.1:
            unit_vectors.append(d / mag)
            
    num_neighbors = len(unit_vectors)
    if num_neighbors >= 4:
        return [] # Fully coordinated
        
    if num_neighbors == 2:
        v1, v2 = unit_vectors[0], unit_vectors[1]
        w = v1 + v2 
        u = v1 - v2 
        if np.linalg.norm(w) < 1e-5: return [] 

        w_unit = w / np.linalg.norm(w)
        u_unit = u / np.linalg.norm(u)
        p_unit = np.cross(w_unit, u_unit)
        p_unit /= np.linalg.norm(p_unit)
        
        a_coeff = -1.0 / np.sqrt(3)
        b_coeff = np.sqrt(2.0 / 3.0)
        
        v3 = a_coeff * w_unit + b_coeff * p_unit
        v4 = a_coeff * w_unit - b_coeff * p_unit
        
        return [v3 * bond_length, v4 * bond_length]
        
    return []

def get_natural_pairing_vector(atoms, idx):
    """Determine the lateral direction where dangling bonds point toward a neighbor."""
    vecs = get_missing_tetrahedral_vectors(atoms, idx)
    if len(vecs) == 2:
        diff = vecs[0] - vecs[1]
        diff[2] = 0 # Projection onto XY plane
        if np.linalg.norm(diff) > 1e-3:
            return diff / np.linalg.norm(diff)
    return None

def passivate_slab(atoms, species='H', side='bottom', bond_length=1.48):
    """Robust passivation by identifying dangling bonds."""
    indices = find_surface_indices(atoms, side)
    new_h_pos = []
    for idx in indices:
        missing_vecs = get_missing_tetrahedral_vectors(atoms, idx, bond_length=bond_length)
        pos = atoms.positions[idx]
        for vec in missing_vecs:
            if (side == 'bottom' and vec[2] < 0) or (side == 'top' and vec[2] > 0):
                new_h_pos.append(pos + vec)
                
    if new_h_pos:
        atoms += Atoms(species * len(new_h_pos), positions=new_h_pos)
    return atoms

def reconstruct_2x1_buckled(atoms, bond_length=2.30, buckle=0.7):
    """
    PBC-Aware 2x1 reconstruction (Minimum Image Convention).
    Correctly pairs surface atoms across cell boundaries.
    """
    indices = find_surface_indices(atoms, 'top')
    # Use neighbor list to find coordination
    i_list, j_list = neighbor_list('ij', atoms, 2.6)
    
    # Filter for undercoordinated Si (Coord < 4)
    surface_si = []
    for idx in indices:
        if np.sum(i_list == idx) < 4:
            surface_si.append(idx)
            
    paired = set()
    dimer_pairs = []
    
    for idx1 in surface_si:
        if idx1 in paired: continue
        
        pref_vec = get_natural_pairing_vector(atoms, idx1)
        if pref_vec is None: continue
        
        pos1 = atoms.positions[idx1]
        best_idx2 = -1
        best_dist = 10.0
        
        # PBC-Aware distance search
        # Using get_distances with mic=True
        other_indices = [i for i in surface_si if i != idx1 and i not in paired]
        if not other_indices: continue
        
        # D_all has vectors; d_all has distances
        D_all, d_all = get_distances(pos1, atoms.positions[other_indices], cell=atoms.cell, pbc=atoms.pbc)
        # Flatten (since we only passed one pos1)
        D_all = D_all[0]
        d_all = d_all[0]
        
        for idx_in_sub, idx2 in enumerate(other_indices):
            dist = d_all[idx_in_sub]
            dist_vec = D_all[idx_in_sub]
            dist_unit = dist_vec / dist
            alignment = abs(np.dot(dist_unit, pref_vec))
            
            if dist < 4.5 and alignment > 0.8: # Must be close and aligned
                if dist < best_dist:
                    best_dist = dist
                    best_idx2 = idx2
                    best_dist_vec = dist_vec
        
        if best_idx2 != -1:
            dimer_pairs.append((idx1, best_idx2, best_dist_vec))
            paired.add(idx1)
            paired.add(best_idx2)
            
    # Apply Buckling and Shift (PBC-Aware)
    d_xy = np.sqrt(bond_length**2 - buckle**2)
    for idx1, idx2, dist_vec in dimer_pairs:
        p1 = atoms.positions[idx1]
        p2_eff = p1 + dist_vec # Use MIC-corrected position image
        center = (p1 + p2_eff) / 2
        vec = (p1 - p2_eff) / np.linalg.norm(p1 - p2_eff)
        
        # Shift positions
        atoms.positions[idx1] = center + vec * (d_xy / 2) + np.array([0, 0, buckle / 2])
        # For idx2, it might be a periodic image. We apply the shift and let it wrap?
        # Actually, if we modify idx2's absolute position, it works.
        atoms.positions[idx2] = center - vec * (d_xy / 2) - np.array([0, 0, buckle / 2])
        
    print(f"Applied PBC-aware 2x1 reconstruction to {len(dimer_pairs)} pairs.")
    return dimer_pairs
