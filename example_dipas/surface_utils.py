import numpy as np
from ase import Atoms

def find_surface_indices(atoms, side='top', threshold=1.0, species=None):
    """Find indices of atoms at the top or bottom surface based on Z-coordinates."""
    if species:
        indices = np.where(atoms.symbols == species)[0]
    else:
        indices = np.arange(len(atoms))
    
    if len(indices) == 0: return []
    z_coords = atoms.positions[indices, 2]
    
    z_target = np.max(z_coords) if side == 'top' else np.min(z_coords)
    mask = np.abs(z_coords - z_target) < threshold
    return indices[mask]

def check_overlap(atoms, cutoff=1.2, verbose=False):
    """Check for steric overlaps between atoms using a simple distance threshold."""
    from ase.neighborlist import neighbor_list
    i_list, j_list, dists = neighbor_list('ijd', atoms, cutoff)
    if len(i_list) > 0:
        if verbose:
            print(f"Overlap detected: {len(i_list)//2} pairs closer than {cutoff}A")
        return True
    return False

def get_all_dangling_bonds_general(atoms, valence_map, vector_generator, cutoff=3.1, side='top'):
    """
    Identify missing valences for surface atoms using platform-independent logic.
    valence_map: function(symbol) -> int (target coordination)
    vector_generator: function(atoms, idx, neighbor_data) -> list of vectors
    """
    from ase.neighborlist import neighbor_list
    surface_indices = find_surface_indices(atoms, side=side, threshold=2.0)
    
    # Use a generous cutoff for neighbor list to share with vector generator
    i_list, j_list, D_list = neighbor_list('ijD', atoms, cutoff)
    neighbor_data = (i_list, j_list, D_list)
    
    all_bonds = []
    for idx in surface_indices:
        sym = atoms.symbols[idx]
        target_val = valence_map(sym)
        if target_val <= 0: continue

        mask = (i_list == idx)
        dists = np.linalg.norm(D_list[mask], axis=1)
        # Count only true covalent neighbors (Si-Si is ~2.35, Si-O is ~1.63)
        # Using 2.6A as a safe covalent limit
        num_n = np.sum((dists > 0.1) & (dists < 2.6))
        
        if num_n < target_val:









            vecs = vector_generator(atoms, idx, neighbor_data)
            for v in vecs:
                # Basic directional filter: point into vacuum
                if (side == 'top' and v[2] > -0.1) or (side == 'bottom' and v[2] < 0.1):
                    all_bonds.append({'parent': idx, 'vector': v, 'type': f'{sym}-H'})
    return all_bonds

def passivate_surface_coverage_general(atoms, h_coverage, valence_map, vector_generator, cutoff=3.1, side='top', verbose=False):
    """Uniformly passivate a surface using a greedy max-min distance algorithm."""
    from ase.geometry import get_distances
    
    candidates = get_all_dangling_bonds_general(atoms, valence_map, vector_generator, cutoff, side)
    if not candidates: 
        if verbose: print(f"  [Passivation] No dangling bonds found on {side} surface.")
        return atoms
    
    n_target = int(round(len(candidates) * h_coverage))
    if n_target == 0: return atoms
    
    current_atoms = atoms.copy()
    success = 0
    available = list(candidates)
    
    if verbose: print(f"  [Passivation] Targeting {n_target} sites on {side} surface (out of {len(candidates)} available).")
    
    while success < n_target and available:
        h_indices = [i for i, sym in enumerate(current_atoms.symbols) if sym == 'H']
        ref_indices = h_indices + [i for i, sym in enumerate(current_atoms.symbols) if sym == 'O']
        ref_pos = current_atoms.positions[ref_indices] if ref_indices else []
        
        best_cand_idx = -1
        best_score = -1.0
        
        for i_c, cand in enumerate(available):
            parent_pos = current_atoms.positions[cand['parent']]
            b_len = 1.0 # default H-bond length, could be refined by species
            h_pos_candidate = parent_pos + cand['vector'] * b_len
            
            if len(ref_pos) == 0:
                score = 100.0
            else:
                dists = get_distances(h_pos_candidate, ref_pos, cell=current_atoms.cell, pbc=current_atoms.pbc)[1]
                score = np.min(dists)
            
            if score > best_score:
                # Overlap check using MIC (Periodic Boundary Conditions)
                # Ensure the new H atom isn't too close to ANY existing atom
                _, all_dists_list = get_distances(h_pos_candidate, current_atoms.positions, 
                                                 cell=current_atoms.cell, pbc=current_atoms.pbc)
                all_dists = all_dists_list[0]
                
                # Exclude the parent atom from the overlap check
                mask = np.ones(len(all_dists), dtype=bool)
                mask[cand['parent']] = False
                if np.any(all_dists[mask] < 0.8): continue
                
                best_score = score
                best_cand_idx = i_c
        
        if best_cand_idx != -1:
            cand = available.pop(best_cand_idx)
            # Refine bond length based on type if needed
            b_len = 1.48 if 'Si' in cand['type'] else 0.96 if 'O' in cand['type'] else 1.05
            h_pos = current_atoms.positions[cand['parent']] + cand['vector'] * b_len
            current_atoms += Atoms('H', positions=[h_pos])
            # Ensure the newly added H atom is wrapped correctly within the cell
            current_atoms.wrap()
            success += 1
        else:
            break
            
    if verbose: print(f"  [Passivation] Successfully placed {success}/{n_target} H atoms on {side} surface.")
    return current_atoms
