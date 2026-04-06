import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from surface_utils import find_surface_indices

def get_missing_tetrahedral_vectors(atoms, idx, cutoff=2.8, bond_length=2.35, neighbor_data=None):

    """Detect missing tetrahedral bonds for bulk-like Silicon using optional pre-computed neighbors."""
    
    if neighbor_data:
        i_list, j_list, D_list = neighbor_data
    else:
        from ase.neighborlist import neighbor_list
        i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)

    # Displacement vectors FROM idx TO neighbors
    idx_mask = (i_list == idx)
    neighbors_D = D_list[idx_mask]
    
    unit_vectors = []
    for d in neighbors_D:
        mag = np.linalg.norm(d)
        if 0.1 < mag < cutoff:
            unit_vectors.append(d / mag)
            
    num_neighbors = len(unit_vectors)
    if num_neighbors == 0 or num_neighbors >= 4:
        return []



        
    # Robust General Approach: The missing vectors should point "away" 
    # from the average direction of existing neighbors.
    v_sum = np.sum(unit_vectors, axis=0)
    v_target = -v_sum / (np.linalg.norm(v_sum) if np.linalg.norm(v_sum) > 1e-3 else 1.0)
    
    if num_neighbors == 3:
        # One missing bond - point exactly opposite to the sum of 3
        return [v_target]

    if num_neighbors == 2:
        # Two missing bonds on a Si(100) surface atom (e.g. pristine)
        w_unit = v_target # Points "up" generally
        u = unit_vectors[0] - unit_vectors[1]
        u_unit = u / np.linalg.norm(u)
        p_unit = np.cross(w_unit, u_unit)
        p_unit /= np.linalg.norm(p_unit)
        
        # Ideal tetrahedral angle: mix of bisector and perpendicular
        v3 = w_unit * 0.577 + p_unit * 0.816
        v4 = w_unit * 0.577 - p_unit * 0.816
        return [v3, v4]

    if num_neighbors == 1:
        # Three missing bonds
        return [v_target]

    return []

def get_natural_pairing_vector(atoms, idx, neighbor_data=None):
    """Determine the lateral axis where dangling bonds point toward a neighbor for Si(100)."""
    vecs = get_missing_tetrahedral_vectors(atoms, idx, neighbor_data=neighbor_data)
    if len(vecs) == 2:
        diff = vecs[0] - vecs[1]
        diff[2] = 0
        mag = np.linalg.norm(diff)
        if mag > 1e-3:
            return diff / mag
    return None

def reconstruct_2x1_buckled(atoms, buckle=0.7, bond_length=2.30, verbose=False):
    """Refined Si(100) 2x1 reconstruction with aligned buckling."""
    if verbose: print("  [Reconstruction] Starting 2x1 buckling alignment...")
    bulk_indices = np.where(atoms.symbols == 'Si')[0]
    z_coords = atoms.positions[bulk_indices, 2]
    z_max = np.max(z_coords)
    surface_indices = bulk_indices[np.abs(z_coords - z_max) < 1.0]
    i_list, j_list, D_list = neighbor_list('ijD', atoms, 4.2)
    neighbor_data = (i_list, j_list, D_list)
    
    # Coordination count should use a smaller cutoff to distinguish surface vs bulk
    i_coord, _, _ = neighbor_list('ijD', atoms, 3.1)
    counts = np.bincount(i_coord, minlength=len(atoms))

    paired = set()
    for idx1 in surface_indices:
        if idx1 in paired or counts[idx1] >= 4: continue
        
        pref_vec = get_natural_pairing_vector(atoms, idx1, neighbor_data=neighbor_data)
        if pref_vec is None: continue
        
        mask = (i_list == idx1)
        potential_ids = j_list[mask]
        d_all = np.linalg.norm(D_list[mask], axis=1)
        D_all = D_list[mask]

        best_idx2 = -1
        for sub_id, idx2 in enumerate(potential_ids):
            if idx2 not in surface_indices or idx2 in paired or counts[idx2] >= 4: continue
            dist = d_all[sub_id]
            if 2.0 < dist < 4.2:
                dot_prod = abs(np.dot(D_all[sub_id]/dist, pref_vec))
                if dot_prod > 0.7:
                    best_idx2 = idx2
                    break
        
        if best_idx2 != -1:
            D12 = D_all[sub_id] # pos2 - pos1 with MIC
            mid = atoms.positions[idx1] + D12 / 2.0
            p12 = -D12 # pos1 - pos2
            p12[2] = 0
            p12_norm = np.linalg.norm(p12)
            if p12_norm < 1e-3: continue
            
            p12_unit = p12 / p12_norm
            
            atoms.positions[idx1] = mid + p12_unit * (bond_length/2.0)
            atoms.positions[best_idx2] = mid - p12_unit * (bond_length/2.0)

            
            # Buckling: align based on the pairing axis direction (PBC robust)
            # Use the displacement vector D12 (pos2 - pos1) with MIC
            # Decide which axis to use (dominant component of pref_vec)
            if abs(pref_vec[1]) > abs(pref_vec[0]): # Y-oriented pairing
                idx2_is_up = (D12[1] > 0)
            else: # X-oriented pairing
                idx2_is_up = (D12[0] > 0)
                
            if idx2_is_up:
                atoms.positions[best_idx2][2] = z_max + buckle/2.0
                atoms.positions[idx1][2] = z_max - buckle/2.0
            else:
                atoms.positions[best_idx2][2] = z_max - buckle/2.0
                atoms.positions[idx1][2] = z_max + buckle/2.0
                
            paired.add(best_idx2)
            
    # Wrap atoms to ensure they are within the unit cell after shifts
    atoms.wrap()
    
    if verbose: print(f"  [Reconstruction] Successfully formed and aligned {len(paired)} dimers.")
    # Return list of reconstructed dimer pairs for downstream manager
    dimers, _ = identify_surface_bonds(atoms)
    return dimers




def identify_surface_bonds(atoms, cutoff=2.6):
    """Categorize Si-Si bonds into surface dimers and subsurface backbonds."""
    l1_indices = find_surface_indices(atoms, 'top', threshold=0.8, species='Si')
    z_coords = atoms.positions[:, 2]
    z_top = np.max(z_coords[l1_indices])
    l2_candidates = np.where((atoms.symbols == 'Si') & (z_coords < z_top - 0.5) & (z_coords > z_top - 2.5))[0]
    
    i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)
    
    dimer_bonds = []
    backbonds = []
    seen_bonds = set()
    
    for idx1 in l1_indices:
        mask = (i_list == idx1)
        neighbors = j_list[mask]
        for n_idx in neighbors:
            if n_idx == idx1: continue
            if atoms.symbols[n_idx] != 'Si': continue
            bond = tuple(sorted((idx1, n_idx)))

            if bond in seen_bonds: continue
            
            if n_idx in l1_indices:
                dimer_bonds.append(bond)
            elif n_idx in l2_candidates:
                backbonds.append(bond)
            seen_bonds.add(bond)
            
    # print(f"DEBUG: identify_surface_bonds Dimer bonds count: {len(dimer_bonds)}")
    return dimer_bonds, backbonds


def oxidize_si_surface(slab, dimer_coverage=0.0, backbond_coverage=0.0, verbose=False):
    """
    Produce an oxidized Si(100) surface using a Greedy Max-Min Distance algorithm.
    """
    from ase.geometry import get_distances
    
    # 1. Identify all target bonds
    dimers, backbonds = identify_surface_bonds(slab)
    n_dim_target = int(round(len(dimers) * dimer_coverage))
    n_bb_target = int(round(len(backbonds) * backbond_coverage))
    
    oxidized = slab.copy()
    oxidation_count = {i: 0 for i in range(len(slab))}
    MAX_OXIDATION_PER_SI = 2

    def get_greedy_best_bond(atoms, candidates, count):
        current_atoms = atoms.copy()
        success = 0
        available_bonds = list(candidates)
        
        while success < count and available_bonds:
            o_indices = [i for i, sym in enumerate(current_atoms.symbols) if sym == 'O']
            o_positions = current_atoms.positions[o_indices] if o_indices else []
            
            best_bond = None
            best_score = -1.0 
            best_bond_idx = -1
            
            for i_b, (b_idx1, b_idx2) in enumerate(available_bonds):
                if oxidation_count[b_idx1] >= MAX_OXIDATION_PER_SI or \
                   oxidation_count[b_idx2] >= MAX_OXIDATION_PER_SI:
                    continue
                
                midpoint = (current_atoms.positions[b_idx1] + current_atoms.positions[b_idx2]) / 2.0
                
                if len(o_positions) == 0:
                    slab_center = np.sum(current_atoms.cell, axis=0) / 2.0
                    score = 100.0 - np.linalg.norm(midpoint[:2] - slab_center[:2]) 
                else:
                    dists = get_distances(midpoint, o_positions, cell=current_atoms.cell, pbc=current_atoms.pbc)[1]
                    score = np.min(dists)
                
                if score > best_score:
                    dists_to_all = np.linalg.norm(current_atoms.positions - midpoint, axis=1)
                    mask = np.ones(len(dists_to_all), dtype=bool)
                    mask[b_idx1] = False
                    mask[b_idx2] = False
                    if np.any(dists_to_all[mask] < 1.5): continue
                    
                    best_score = score
                    best_bond = (b_idx1, b_idx2)
                    best_bond_idx = i_b
            
            if best_bond:
                current_atoms = insert_o_bridge_pure_geo(current_atoms, best_bond[0], best_bond[1])
                oxidation_count[best_bond[0]] += 1
                oxidation_count[best_bond[1]] += 1
                success += 1
                available_bonds.pop(best_bond_idx)
            else:
                break 
        return current_atoms, success

    if verbose: print(f"  [Oxidation] Targeting {n_dim_target} dimers and {n_bb_target} backbonds for uniform coverage...")
    oxidized, d_count = get_greedy_best_bond(oxidized, dimers, n_dim_target)
    oxidized, b_count = get_greedy_best_bond(oxidized, backbonds, n_bb_target)
    
    if verbose: print(f"  [Oxidation] Successfully oxidized {d_count} dimers and {b_count} backbonds uniformly.")
    return oxidized


def insert_o_bridge_pure_geo(atoms, idx1, idx2, target_si_o=1.63, target_angle=144.0):
    """Geometrically insert Oxygen between Si atoms (Si-O-Si bridge)."""
    pos1 = atoms.positions[idx1].copy()
    pos2 = atoms.positions[idx2].copy()
    
    if pos2[2] > pos1[2]:
        idx1, idx2 = idx2, idx1
        pos1, pos2 = pos2, pos1
        
    bond_vec = pos1 - pos2
    bond_len = np.linalg.norm(bond_vec)
    bond_unit = bond_vec / bond_len
    
    midpoint = (pos1 + pos2) / 2.0
    half_angle_rad = np.deg2rad(target_angle / 2.0)
    dist_perp = target_si_o * np.sin(half_angle_rad)
    dist_along = target_si_o * np.cos(half_angle_rad)
    
    # Use normalized cross with [0,0,1] for perp direction
    up = np.array([0.0, 0.0, 1.0])
    perp_vec = np.cross(bond_unit, up)
    if np.linalg.norm(perp_vec) < 1e-3:
        perp_vec = np.cross(bond_unit, np.array([0.0, 1.0, 0.0]))
    perp_unit = perp_vec / np.linalg.norm(perp_vec)
    
    o_pos = midpoint + perp_unit * dist_perp
    
    # Stress relieving shifts (80/20)
    shift1 = bond_unit * (dist_along - bond_len/2.0) * 0.8
    shift2 = -bond_unit * (dist_along - bond_len/2.0) * 0.2
    
    atoms.positions[idx1] += shift1
    atoms.positions[idx2] += shift2
    atoms += Atoms('O', positions=[o_pos])
    return atoms

def passivate_si_surface(atoms, h_coverage=1.0, side='top', verbose=False):
    """Hybrid H/OH passivation optimized for Si(100) surfaces."""
    from surface_utils import get_all_dangling_bonds_general
    
    # We use a wrapper for dangling bonds that knows about Si/O valence
    def si_val_map(symbol):
        return {'Si': 4, 'O': 2, 'H': 1}.get(symbol, 0)

    def si_vector_gen(atoms, idx, neighbor_data):
        sym = atoms.symbols[idx]
        if sym == 'Si':
            return get_missing_tetrahedral_vectors(atoms, idx, neighbor_data=neighbor_data)
        elif sym == 'O':
            # Identify OH vector: point away from the parent Si
            i_list, j_list, D_list = neighbor_data
            mask = (i_list == idx)
            dists = np.linalg.norm(D_list[mask], axis=1)
            # Filter neighbors within covalent range (Si-O is ~1.63, steric neighbors are ~2.48)
            valid_mask = (dists > 0.1) & (dists < 2.0) 
            if np.sum(valid_mask) == 1:
                v = -D_list[mask][valid_mask][0] # points away from parent atom
                return [v / np.linalg.norm(v)]

        return []

    from surface_utils import passivate_surface_coverage_general
    return passivate_surface_coverage_general(atoms, h_coverage, si_val_map, si_vector_gen, side=side, verbose=verbose)

def build_si100_slab(bulk_atoms, size=(4,4), layers=8, vacuum=10.0):
    """
    Standardized (100) slab generation from bulk Si.
    Applies expansion and identifies surface atoms by tag.
    """
    from ase.build import surface
    # Generate the (100) surface. Conventional cell (100) is 5.43x5.43.
    # To get a 4x4 area, we multiply by (4, 4, 1).
    slab = surface(bulk_atoms, (1, 0, 0), layers=layers, vacuum=vacuum)
    slab = slab * (size[0], size[1], 1)
    
    # Tag surface atoms for easier identification
    # Top surface = 1, Bottom = 4 (to be passivated/fixed)
    z_max = slab.positions[:, 2].max()
    z_min = slab.positions[:, 2].min()
    for a in slab:
        if a.position[2] > z_max - 0.5:
            a.tag = 1 # Top surface
        elif a.position[2] < z_min + 0.5:
            a.tag = 4 # Bottom surface
            
    return slab

def generate_standard_surfaces(bulk_si, verbose=False):
    """
    Generate 4 standard Si(100) surfaces for adsorption study:
    S1: Clean 2x1 Reconstructed
    S2: H-passivated 2x1 Reconstructed
    S3: Oxidized 2x1 Reconstructed (50% Dimer/BB coverage)
    S4: Oxidized + H-passivated
    """
    if verbose: print("Generating Standard Silicon Surfaces...")
    # Base slab
    base = build_si100_slab(bulk_si, size=(4,4), layers=8)
    
    # S1: Reconstructed
    s1 = base.copy()
    reconstruct_2x1_buckled(s1, verbose=verbose)
    s1.info['label'] = 'S1_Clean_2x1'
    
    # S2: H-passivated
    # We passivate bottom too if needed, but here let's follow user preference
    s2 = s1.copy()
    s2 = passivate_si_surface(s2, h_coverage=1.0, side='top', verbose=verbose)
    # Bottom passivation for symmetry/stability
    s2 = passivate_si_surface(s2, h_coverage=1.0, side='bottom', verbose=verbose)
    s2.info['label'] = 'S2_H_Passivated'
    
    # S3: Oxidized (Clean)
    s3 = s1.copy()
    s3 = oxidize_si_surface(s3, dimer_coverage=0.5, backbond_coverage=0.5, verbose=verbose)
    s3.info['label'] = 'S3_Oxidized'
    
    # S4: Oxidized + H-passivated
    s4 = s3.copy()
    s4 = passivate_si_surface(s4, h_coverage=1.0, side='top', verbose=verbose)
    s4 = passivate_si_surface(s4, h_coverage=1.0, side='bottom', verbose=verbose)
    s4.info['label'] = 'S4_Oxidized_H_Passivated'
    s4.info['label'] = 'S4_Oxidized_H_Passivated'
    
    return [s1, s2, s3, s4]

def get_dangling_bond_info(atoms, idx):

    """Get position and primary dangling bond vector for a surface atom."""
    vecs = get_missing_tetrahedral_vectors(atoms, idx)
    if not vecs: return None
    return {
        'index': idx,
        'pos': atoms.positions[idx],
        'db_vector': vecs[0] # Just take the first one for standard mapping
    }

def find_existing_dimers(atoms, cutoff=2.6):
    """Find dimers that are already present in the structure."""
    dimers, _ = identify_surface_bonds(atoms, cutoff=cutoff)
    return dimers
def get_surface_h_mapping(atoms, cutoff=1.8):
    """Map each surface H atom to its parent Si atom on a passivated surface."""
    h_indices = np.where(atoms.symbols == 'H')[0]
    si_indices = np.where(atoms.symbols == 'Si')[0]
    if len(h_indices) == 0: return {}

    from ase.geometry import get_distances
    # For each H, find the closest Si
    mapping = {}
    for h_idx in h_indices:
        _, d_list = get_distances(atoms.positions[h_idx], atoms.positions[si_indices], 
                                 cell=atoms.cell, pbc=atoms.pbc)
        dists = d_list[0]
        if np.any(dists < cutoff):
            nearest_si_local_idx = np.argmin(dists)
            nearest_si_idx = si_indices[nearest_si_local_idx]
            mapping[nearest_si_idx] = h_idx
            
    return mapping
