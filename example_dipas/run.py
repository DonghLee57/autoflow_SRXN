import os
import sys
import yaml
import numpy as np
from ase.io import read, write

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ads_workflow_mgr import AdsorptionWorkflowManager
from chemisorption_builder import build_chemisorption_structures

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_generic_adsorption_study(config_path='config.yaml'):
    print(f"--- Starting Generic Config-Driven Adsorption Study ({config_path}) ---")
    config = load_config(config_path)
    
    bulk_file = config['paths']['substrate']
    mol_file = config['paths']['molecule']
    out_prefix = config['paths'].get('output_prefix', 'cands_out')
    
    if not os.path.exists(bulk_file):
        print(f"Error: {bulk_file} not found.")
        return
    if not os.path.exists(mol_file):
        print(f"Error: {mol_file} not found.")
        return
        
    slab = read(bulk_file)
    mol = read(mol_file)
    
    # --- [0] Passivation Phase ---
    pass_cfg = config.get('passivation', {})
    if pass_cfg.get('enabled', False):
        from surface_utils import passivate_surface_coverage_general
        
        element = pass_cfg.get('element', 'H')
        side = pass_cfg.get('side', 'bottom')
        coverage = pass_cfg.get('coverage', 1.0)
        output_name = pass_cfg.get('output_name', 'passivated.vasp')
        
        sides_to_passivate = [side] if side != 'both' else ['top', 'bottom']
        valence_map = config.get('ideal_coordination', {})
        
        print(f"\n[0] Applying Surface Passivation (Element: {element}, Side: {side}, Coverage: {coverage})...")
        print("    [Info] Using Generalized VSEPR Engine.")
        
        for s in sides_to_passivate:
            # Utilizing the improved generic logic (no longer requiring Si-specific scripts)
            slab = passivate_surface_coverage_general(slab, h_coverage=coverage, valence_map=valence_map, 
                                                     element=element, side=s, verbose=True)
        
        print(f"    [Output] Saving passivated structure to: {output_name}")
        write(output_name, slab)

    settings = config.get('settings', {})
    center_target = settings.get('center_target', 'Si')
    rot_steps = settings.get('rot_steps', 8)
    symprec = settings.get('symprec', 0.2)
    do_phy = settings.get('do_physisorption', True)
    do_chem = settings.get('do_chemisorption', True)
    
    all_results = []
    
    mgr = AdsorptionWorkflowManager(slab, symprec=symprec, verbose=True)
    
    # 1. Physisorption
    if do_phy:
        print("\n[1] Sampling Physisorption...")
        phy_cands = mgr.generate_physisorption_candidates(mol, height=3.5, n_rot=rot_steps)
        print(f"    -> Generated {len(phy_cands)} physisorption candidates.")
        for c in phy_cands: c.info['mechanism'] = 'physisorption'
        if phy_cands:
            write(f'{out_prefix}_phy.extxyz', phy_cands)
            all_results.extend(phy_cands)

    # 2. Chemisorption
    if do_chem:
        print("\n[2] Sampling Chemisorption...")
        # We pass config into build_chemisorption_structures to use ideal_coordination
        chem_cands = build_chemisorption_structures(
            molecule=mol, 
            center_target=center_target, 
            surface=slab, 
            rot_steps=rot_steps, 
            config=config,
            verbose=True
        )
        print(f"    -> Generated {len(chem_cands)} reactive candidates.")
        if chem_cands:
            write(f'{out_prefix}_chem.extxyz', chem_cands)
            all_results.extend(chem_cands)
        else:
            with open(f'{out_prefix}_chem.extxyz', 'w') as f:
                pass
            
    print(f"\n--- Study Complete. Total candidates: {len(all_results)} ---")

if __name__ == '__main__':
    c_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    run_generic_adsorption_study(c_path)
