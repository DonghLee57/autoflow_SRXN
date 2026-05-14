import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment

def match_atoms_geometric(ref: Atoms, target: Atoms, logger=None):
    """
    Finds the optimal permutation of target atoms to match ref atoms' geometry.
    Strictly enforces element count consistency.
    """
    # 1. Strict Consistency Check (Requirement C)
    if len(ref) != len(target):
        msg = f"Atom count mismatch: {len(ref)} (ref) vs {len(target)} (target)"
        if logger: logger.error(msg)
        raise ValueError(msg)
    
    syms_ref = sorted(ref.get_chemical_symbols())
    syms_target = sorted(target.get_chemical_symbols())
    if syms_ref != syms_target:
        msg = "Chemical composition mismatch between reference and target structures."
        if logger: logger.error(msg)
        raise ValueError(msg)

    # 2. Element-wise mapping (Requirement A)
    # We match atoms of the same element to ensure chemical identity is preserved.
    final_indices = np.zeros(len(ref), dtype=int)
    unique_elements = sorted(set(syms_ref))
    
    ref_pos = ref.get_positions()
    target_pos = target.get_positions()
    cell = ref.get_cell()
    pbc = ref.get_pbc()
    
    for elem in unique_elements:
        idx_ref = np.where(np.array(ref.get_chemical_symbols()) == elem)[0]
        idx_target = np.where(np.array(target.get_chemical_symbols()) == elem)[0]
        
        # Distance matrix under PBC
        # cost[i, j] = distance between ref_pos[idx_ref[i]] and target_pos[idx_target[j]]
        n_elem = len(idx_ref)
        cost_matrix = np.zeros((n_elem, n_elem))
        
        for i in range(n_elem):
            diff = target_pos[idx_target] - ref_pos[idx_ref[i]]
            diff_mic, _ = find_mic(diff, cell, pbc)
            cost_matrix[i, :] = np.linalg.norm(diff_mic, axis=1)
            
        # Hungarian algorithm to find minimum cost assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Map target indices to reference indices
        final_indices[idx_ref[row_ind]] = idx_target[col_ind]
        
    return final_indices

def reorder_atoms(target: Atoms, mapping_indices: np.ndarray):
    """Reorders target atoms to match the reference based on mapping_indices."""
    return target[mapping_indices]
