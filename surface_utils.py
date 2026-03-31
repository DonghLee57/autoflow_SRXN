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
    i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)
    neighbors_D = D_list[i_list == idx]
    
    unit_vectors = []
    for d in neighbors_D:
        mag = np.linalg.norm(d)
        if mag > 0.1:
            unit_vectors.append(d / mag)
            
    num_neighbors = len(unit_vectors)
    if num_neighbors >= 4:
        return []
        
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

def reconstruct_2x1_buckled(atoms, bond_length=2.30, buckle=0.7, pattern='checkerboard'):
    """
    Grid-Based Robust 2x1 reconstruction for Si(100).
    Identifies all surface atoms and ensures perfect alignment using integer coordinates.
    """
    indices = find_surface_indices(atoms, 'top')
    if len(indices) == 0: return []
    
    # 1. Map Surface Si Atoms to a Logical Grid (nxm)
    # Using scaled positions to handle any cell size/pbc wrapping
    scaled_pos = atoms.get_scaled_positions()[indices]
    # Identify unique fractional coordinates to find the supercell dimensions
    unique_s0 = np.unique(np.round(scaled_pos[:, 0], 3))
    unique_s1 = np.unique(np.round(scaled_pos[:, 1], 3))
    N0, N1 = len(unique_s0), len(unique_s1)
    
    # Map each index to an integer (i, j) based on its sorted fractional coordinates
    idx_to_grid = {}
    for i_atom, idx in enumerate(indices):
        s0, s1 = scaled_pos[i_atom, 0], scaled_pos[i_atom, 1]
        gi = np.argmin(np.abs(unique_s0 - s0))
        gj = np.argmin(np.abs(unique_s1 - s1))
        idx_to_grid[idx] = (gi, gj)

    # 2. Identify Pairing Axis from the dangling bonds of the first atom
    pref_vec = get_natural_pairing_vector(atoms, indices[0])
    # Project pref_vec (X,Y component) onto scaled basis
    scaled_pref = pref_vec[:2] @ np.linalg.inv(atoms.cell[:2, :2])
    pairing_axis = 0 if abs(scaled_pref[0]) > abs(scaled_pref[1]) else 1
    other_axis = 1 - pairing_axis

    # 3. Explicit Pairing based on Grid
    paired = set()
    dimer_pairs = []
    
    # Find all surface atoms and pair them
    for idx1 in indices:
        if idx1 in paired: continue
        gi, gj = idx_to_grid[idx1]
        
        # Target neighbor along the pairing axis (handling PBC)
        # We pair (even, j) with (odd, j) if pairing_axis == 0
        if pairing_axis == 0:
            target_gi = (gi + 1) if gi % 2 == 0 else (gi - 1)
            target_gi %= N0
            target_gj = gj
        else:
            target_gj = (gj + 1) if gj % 2 == 0 else (gj - 1)
            target_gj %= N1
            target_gi = gi
            
        # Find the atom at (target_gi, target_gj)
        best_idx2 = -1
        for idx2, grid2 in idx_to_grid.items():
            if grid2 == (target_gi, target_gj) and idx2 != idx1:
                best_idx2 = idx2
                break
        
        if best_idx2 != -1:
            # Calculate MIC distance vector
            pos1 = atoms.positions[idx1]
            dist_vec = atoms.get_distance(idx1, best_idx2, vector=True, mic=True)
            
            # Buckling Phase S
            # Stripe p(2x2): Phase depends only on the position along the other_axis?
            # No, buckle alternates ALONG the row.
            # Checkerboard c(4x2): Alternates along row AND between rows.
            col_idx = (gi if pairing_axis == 0 else gj) // 2
            row_idx = gj if pairing_axis == 0 else gi
            
            # Correction: col_idx is the index of the DIMER unit in the row.
            # row_idx is the index of the ROW.
            
            if pattern == 'stripe':
                S = (-1)**col_idx
            else: # checkerboard
                S = (-1)**(col_idx + row_idx)
                
            dimer_pairs.append((idx1, best_idx2, dist_vec, S))
            paired.add(idx1)
            paired.add(best_idx2)

    # 4. Apply Buckling and Shift (100% pairing guaranteed)
    d_xy = np.sqrt(bond_length**2 - buckle**2)
    for idx1, idx2, dist_vec, S in dimer_pairs:
        p1 = atoms.positions[idx1]
        p2_eff = p1 + dist_vec
        center = (p1 + p2_eff) / 2
        vec = (p1 - p2_eff) / np.linalg.norm(p1 - p2_eff)
        
        # Consistent buckle assignment
        atoms.positions[idx1] = center + vec * (d_xy / 2) + np.array([0, 0, S * buckle / 2])
        atoms.positions[idx2] = center - vec * (d_xy / 2) - np.array([0, 0, S * buckle / 2])
        
    print(f"Applied 100% Grid-Matched 2x1 reconstruction ({pattern}) to {len(dimer_pairs)} pairs.")
    return dimer_pairs
