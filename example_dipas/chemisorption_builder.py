import numpy as np
from ase import Atoms
from si_surface_utils import find_existing_dimers, get_surface_h_mapping, get_dangling_bond_info
from ads_workflow_mgr import AdsorptionWorkflowManager

def analyze_surface_reactivity(surface, verbose=True):
    """
    Analyzes the surface geometrically to find reactive sites.
    Returns:
        dimers: list of (idx1, idx2) pairs.
        surface_h: dict mapping si_idx -> h_idx.
    """
    results = {}
    
    # 1. Look for Passivated H configurations (H-exchange candidates)
    h_mapping = get_surface_h_mapping(surface)
    results['surface_h'] = h_mapping
    if verbose and h_mapping:
        print(f"  [Surface Reactivity] Identified {len(h_mapping)} Si-H passivation sites.")
        
    # 2. Look for Dimer configurations (Dissociative chemisorption candidates)
    # We only find existing dimers without mutating the slab.
    dimers = find_existing_dimers(surface)
    results['dimers'] = dimers
    if verbose and dimers:
        print(f"  [Surface Reactivity] Identified {len(dimers)} exposed Si-Si dimers.")
        
    return results


def analyze_molecule_ligands(molecule, center_symbol='Si', verbose=True):
    """
    Algorithmically fragments the precursor molecule to identify reactive ligands.
    Uses AdsorptionWorkflowManager implicitly for the heavy lifting.
    """
    # Create a temporary manager to use its fragmentation logic
    mgr = AdsorptionWorkflowManager(molecule, verbose=verbose)
    c_idx, ligands = mgr.discover_ligands(molecule, center_symbol=center_symbol, verbose=verbose)
    return c_idx, ligands


def build_chemisorption_structures(molecule, center_symbol, surface, rot_steps=8, verbose=True):
    """
    Entry point for algorithmic chemisorption generation based on input molecule and surface.
    Identifies valid mechanisms based on available surface sites.
    """
    if verbose:
        print("\n--- Starting Algorithmic Chemisorption Routing ---")
        
    sites = analyze_surface_reactivity(surface, verbose=verbose)
    c_idx, ligands = analyze_molecule_ligands(molecule, center_symbol=center_symbol, verbose=verbose)
    
    candidates = []
    
    if not ligands:
        if verbose: print("  [Warning] No detachable ligands found. Aborting chemisorption.")
        return candidates
        
    # We instantiate a manager scoped to the current surface for coordinate placement/overlap tests
    mgr = AdsorptionWorkflowManager(surface, verbose=verbose)
    
    # Route to H-Exchange if surface H atoms exist
    if sites.get('surface_h'):
        if verbose: print("  -> Routing to H-Exchange Sequence...")
        h_cands = _execute_h_exchange(mgr, molecule, c_idx, ligands, sites['surface_h'], rot_steps)
        candidates.extend(h_cands)
        
    # Route to Dissociation if dimers exist
    if sites.get('dimers'):
        if verbose: print("  -> Routing to Dissociative Chemisorption Sequence...")
        d_cands = _execute_dissociation(mgr, molecule, c_idx, ligands, sites['dimers'], rot_steps)
        candidates.extend(d_cands)
        
    if verbose:
        print(f"--- Finished Chemisorption Builder. Total Generated: {len(candidates)} ---\n")
        
    return candidates


def _execute_h_exchange(mgr, molecule, c_idx, ligands, h_mapping, rot_steps):
    """ Internal subroutine to execute H-Exchange """
    candidates = []
    stats = {'overlap': 0, 'deduplicated': 0}
    seen_formulas = set()
    
    for l_info in ligands:
        formula = l_info.get('formula', 'Unknown')
        if formula in seen_formulas: 
            stats['deduplicated'] += 1
            continue
        seen_formulas.add(formula)
        
        indices_b = l_info['indices']
        frag_b = molecule[indices_b]
        binding_idx_b = indices_b.index(l_info['binding_atoms'][0])
        
        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)
        
        for si_idx, h_idx in h_mapping.items():
            si_pos = mgr.slab.positions[si_idx]
            h_pos = mgr.slab.positions[h_idx]
            h_vec = h_pos - si_pos
            h_vec_norm = h_vec / np.linalg.norm(h_vec)
            
            for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                p_a = mgr._place_at_dangling_bond(frag_a, binding_idx_a, l_info['bond_vec'], 
                                                   si_pos, h_vec_norm, 2.35, rot_angle=angle)
                
                p_b = mgr._form_byproduct(frag_b, binding_idx_b, -l_info['bond_vec'])
                z_clearance = np.max(mgr.slab.positions[:, 2]) + 4.0
                p_b.translate([si_pos[0], si_pos[1], z_clearance] - p_b.positions[0])
                
                keep_indices = [i for i in range(len(mgr.slab)) if i != h_idx]
                combined = mgr.slab[keep_indices].copy()
                
                for a in p_a: a.tag = 2
                combined += p_a
                for a in p_b: a.tag = 3
                combined += p_b
                
                # Check overlap (since the original length of combined was changed, we check the new combined)
                if not mgr.check_overlap(combined, cutoff=1.2, verbose=False):
                    comp_a = "".join(frag_a.symbols)
                    comp_b = "".join(p_b.symbols)
                    if comp_b == "HH": comp_b = "H2" # Make it more readable
                    
                    combined.info['mechanism'] = f"H-Exchange: {comp_a} on Si_{si_idx}, byproduct={comp_b}, rot={angle:.1f}"
                    # Tag this properly for run_dipas_study
                    combined.info['reaction_type'] = 'h_exchange'
                    candidates.append(combined)
                    break
                else:
                    stats['overlap'] += 1
                    
    return candidates


def _execute_dissociation(mgr, molecule, c_idx, ligands, dimers, rot_steps):
    """ Internal subroutine to execute Dissociative Chemisorption on dimers """
    candidates = []
    stats = {'overlap': 0, 'deduplicated': 0}
    seen_formulas = set()
    
    for l_info in ligands:
        formula = l_info.get('formula', 'Unknown')
        if formula in seen_formulas: 
            stats['deduplicated'] += 1
            continue
        seen_formulas.add(formula)
        
        indices_b = l_info['indices']
        frag_b = molecule[indices_b]
        binding_idx_b = indices_b.index(l_info['binding_atoms'][0])
        
        indices_a = list(set(range(len(molecule))) - set(indices_b))
        frag_a = molecule[indices_a]
        binding_idx_a = indices_a.index(c_idx)
        
        for (idx1, idx2) in dimers:
            db1 = get_dangling_bond_info(mgr.slab, idx1)
            db2 = get_dangling_bond_info(mgr.slab, idx2)
            if not db1 or not db2: continue
            
            for s1, s2 in [(db1, db2), (db2, db1)]:
                best_pose = None
                for angle in np.linspace(0, 360, rot_steps, endpoint=False):
                    p_a = mgr._place_at_dangling_bond(frag_a, binding_idx_a, l_info['bond_vec'], 
                                                       s1['pos'], s1['db_vector'], 2.35, rot_angle=angle)
                    
                    bond_len_b = 1.48 if frag_b.symbols[binding_idx_b] == 'H' else 2.1
                    p_b = mgr._place_at_dangling_bond(frag_b, binding_idx_b, -l_info['bond_vec'], 
                                                       s2['pos'], s2['db_vector'], bond_len_b, rot_angle=0)
                    
                    combined = mgr.slab.copy()
                    for a in p_a: a.tag = 2
                    combined += p_a
                    for a in p_b: a.tag = 3
                    combined += p_b
                    
                    if not mgr.check_overlap(combined, cutoff=1.2, verbose=False):
                        comp_a = "".join(frag_a.symbols)
                        combined.info['mechanism'] = f"Cohesive Chemisorption: {comp_a} on {s1['index']}, {frag_b.symbols[binding_idx_b]} on {s2['index']}, rot={angle:.1f}"
                        combined.info['reaction_type'] = 'chemisorption'
                        best_pose = combined
                        break
                
                if best_pose:
                    candidates.append(best_pose)
                    break
                else:
                    stats['overlap'] += 1
                    
    return candidates
