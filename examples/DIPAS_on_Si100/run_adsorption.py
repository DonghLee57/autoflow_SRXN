import os
import sys
import yaml
import numpy as np
from ase.io import read, write

# Add the src directory to Python path for local development
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ads_workflow_mgr import AdsorptionWorkflowManager
from chemisorption_builder import build_chemisorption_structures
from logger_utils import setup_logger
from surface_utils import create_slab_from_bulk, standardize_vasp_atoms, passivate_surface_coverage_general

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def execute_discovery_stage(slab, mol, config, out_prefix, logger, verbose=True, tag=2, center_target='Si'):
    """
    Common discovery logic for any adsorbate.
    Returns a unified list of physisorption and chemisorption candidates.
    """
    ads_gen_cfg = config.get('adsorbate_generation', config.get('settings', {}))
    run_phy = ads_gen_cfg.get('run_physisorption', ads_gen_cfg.get('do_physisorption', True))
    run_chem = ads_gen_cfg.get('run_chemisorption', ads_gen_cfg.get('do_chemisorption', True))
    rot_steps = ads_gen_cfg.get('rot_steps', 8)
    symprec = ads_gen_cfg.get('symprec', 0.2)
    
    mgr = AdsorptionWorkflowManager(slab, config=config, symprec=symprec, verbose=verbose)
    
    # [Diagnostic] Span Check
    d_mol = mgr.calculate_molecule_lateral_extent(mol)
    a_len = np.linalg.norm(slab.cell[0])
    b_len = np.linalg.norm(slab.cell[1])
    logger.info(f"  DIAGNOSTIC: {mol.get_chemical_formula()} span = {d_mol:.2f} A | Substrate = {a_len:.2f} x {b_len:.2f} A")
    
    padding = 3.0 # Slightly reduced for better reporting
    if a_len < d_mol + padding or b_len < d_mol + padding:
        logger.warning(f"  PBC CONFLICT: Substrate cell ({a_len:.1f}x{b_len:.1f}) might be too small for {mol.get_chemical_formula()} (span={d_mol:.1f}).")

    all_cands = []
    
    # 1. Physisorption
    if run_phy:
        logger.info(f"  Searching for Physisorption (Rigid-body) candidates for {mol.get_chemical_formula()}...")
        phy_height = ads_gen_cfg.get('physisorption_height', 3.5)
        phy_cands = mgr.generate_physisorption_candidates(mol, height=phy_height, n_rot=rot_steps, config=config, tag=tag)
        for c in phy_cands: c.info['mechanism'] = 'physisorption'
        all_cands.extend(phy_cands)

    # 2. Chemisorption
    if run_chem:
        logger.info(f"  Searching for Chemisorption (Mechanistic) candidates for {mol.get_chemical_formula()} (Center={center_target})...")
        chem_cands = build_chemisorption_structures(
            molecule=mol, 
            center_target=center_target, 
            surface=slab, 
            rot_steps=rot_steps, 
            config=config,
            verbose=verbose,
            tag=tag
        )
        for c in chem_cands: c.info['mechanism'] = 'chemisorption'
        all_cands.extend(chem_cands)
        
        # Save isolated byproducts if any
        byproducts = [c.info['isolated_byproduct'] for c in chem_cands if 'isolated_byproduct' in c.info]
        if byproducts:
            write(f'{out_prefix}_byproducts.extxyz', byproducts)
            
    if all_cands:
        write(f'{out_prefix}_all_poses.extxyz', all_cands)
        
    return all_cands

def run_generic_adsorption_study(config_path='config.yaml'):
    config = load_config(config_path)
    ads_gen_cfg = config.get('adsorbate_generation', config.get('settings', {}))
    verbose = ads_gen_cfg.get('verbose', True)
    logger = setup_logger(log_path=ads_gen_cfg.get('log_path', 'workflow.log'), verbose=verbose)
    
    logger.info(f"--- Starting AutoFlow-SRXN Discovery Study ({config_path}) ---")
    
    bulk_file = config['paths'].get('substrate')
    mol_file = config['paths']['molecule']
    inh_file = config['paths'].get('inhibitor')
    out_prefix = config['paths'].get('output_prefix', 'cands_out')
    
    mol = read(mol_file)
    slab = None

    # --- STAGE 0: Substrate Generation & Geometric Passivation ---
    sub_gen_cfg = config.get('substrate_generation', {})
    if sub_gen_cfg.get('run_generation', sub_gen_cfg.get('enabled', False)):
        logger.info("STAGE 0: Generating/Loading Substrate...")
        bulk_atoms = read(sub_gen_cfg['bulk_path'])
        slab = create_slab_from_bulk(
            bulk_atoms=bulk_atoms,
            miller_indices=sub_gen_cfg.get('miller_indices', [1, 0, 0]),
            thickness=sub_gen_cfg.get('thickness', 10.0),
            vacuum=sub_gen_cfg.get('vacuum', 10.0),
            target_area=sub_gen_cfg.get('target_area'),
            supercell_matrix=sub_gen_cfg.get('supercell_matrix'),
            termination=sub_gen_cfg.get('termination'),
            top_termination=sub_gen_cfg.get('top_termination'),
            bottom_termination=sub_gen_cfg.get('bottom_termination'),
            verbose=verbose
        )
        # Standardize and save the raw substrate
        raw_slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
        write('generated_substrate.vasp', raw_slab)
        logger.info("Saved generated raw substrate to 'generated_substrate.vasp'.")
    else:
        slab = read(bulk_file)

    pass_cfg = config.get('passivation', {})
    if pass_cfg.get('run_passivation', pass_cfg.get('enabled', False)):
        logger.info("Applying Geometric Passivation...")
        side = pass_cfg.get('side', 'bottom')
        sides = [side] if side != 'both' else ['top', 'bottom']
        for s in sides:
            slab = passivate_surface_coverage_general(
                slab, h_coverage=pass_cfg.get('coverage', 1.0), 
                valence_map=config.get('ideal_coordination', {}),
                element=pass_cfg.get('element', 'H'), side=s, verbose=verbose
            )
            
        # Final formatting and export of the passivated substrate
        slab = standardize_vasp_atoms(slab, z_min_offset=0.5)
        write('passivated.vasp', slab)
        logger.info("Saved standardized substrate to 'passivated.vasp' (Sorted, z_min=0.5A).")
    
    # --- STAGE 1: Dynamic Inhibitor Discovery (Branching) ---
    base_slabs = [slab]
    if inh_file and os.path.exists(inh_file) and ads_gen_cfg.get('run_pre_inhibition', False):
        logger.info(f"STAGE 1: Dynamic Inhibitor Discovery ({inh_file})")
        inh_mol = read(inh_file)
        inh_center = ads_gen_cfg.get('inhibitor_center', 'O')
        inh_cands = execute_discovery_stage(slab, inh_mol, config, f"{out_prefix}_inh", logger, verbose, tag=2, center_target=inh_center)
        
        limit = ads_gen_cfg.get('branching_limit', 5)
        base_slabs = inh_cands[:limit]
        logger.info(f"  Branching into {len(base_slabs)} inhibited geomorphologies for Stage 2.")
    
    # --- STAGE 2: Main Precursor Discovery ---
    logger.info(f"STAGE 2: Main Precursor Discovery ({mol_file})")
    all_final_results = []
    mol_center = ads_gen_cfg.get('precursor_center', ads_gen_cfg.get('center_target', 'Si'))
    for i, s in enumerate(base_slabs):
        suffix = f"_inh{i}" if len(base_slabs) > 1 else ""
        results = execute_discovery_stage(s, mol, config, f"{out_prefix}{suffix}", logger, verbose, tag=3, center_target=mol_center)
        all_final_results.extend(results)
        
    logger.info(f"--- Study Complete. Total Unique Candidates: {len(all_final_results)} ---")

if __name__ == '__main__':
    # Default to config.yaml in the script's directory if not provided
    default_config = os.path.join(os.path.dirname(__file__), 'config.yaml')
    c_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    run_generic_adsorption_study(c_path)
