import os
import numpy as np
from ase.io import read, write
from si_surface_utils import generate_standard_surfaces
from ads_workflow_mgr import AdsorptionWorkflowManager
from chemisorption_builder import build_chemisorption_structures

def study_dipas_on_multiple_surfaces():
    print("--- Starting Integrated DIPAS Adsorption Study ---")
    
    # 1. Load basic components
    # Assuming run from example_dipas directory
    bulk_file = 'Si.vasp'
    mol_file = 'dipas_3d.extxyz'
    
    if not os.path.exists(bulk_file):
        print(f"Error: {bulk_file} not found.")
        return
    
    bulk_si = read(bulk_file)
    if not os.path.exists(mol_file):
        print(f"Error: {mol_file} not found.")
        return
    dipas = read(mol_file)
    
    # 2. Generate the 4 standardized surfaces
    surfaces = generate_standard_surfaces(bulk_si, verbose=True)
    
    all_results = []
    
    # 3. Iterate through each surface state
    for slab in surfaces:
        label = slab.info.get('label', 'Unnamed')
        print(f"\n>>> Processing Surface: {label} ({len(slab)} atoms)")
        
        # Save the surface itself for reference
        write(f'surface_{label}.extxyz', slab)
        
        mgr = AdsorptionWorkflowManager(slab, verbose=True)
        
        # A. Physisorption (All surfaces)
        print(f"  Sampling Physisorption on {label}...")
        # Reduce n_rot for quick verification if needed, but here using 8
        phy_cands = mgr.generate_physisorption_candidates(dipas, height=3.5, n_rot=8)
        print(f"    -> Generated {len(phy_cands)} physisorption candidates.")
        for i, c in enumerate(phy_cands):
            c.info['surface_state'] = label
            c.info['mechanism'] = 'physisorption'
            write(f'cands_{label}_phy_{i:03d}.extxyz', c)
            all_results.append(c)

        # B. Reactive Adsorption (Algorithmic Routing)
        print(f"  Sampling Chemisorption / H-Exchange on {label}...")
        reactive_cands = build_chemisorption_structures(dipas, center_symbol='Si', surface=slab, rot_steps=8, verbose=True)
        print(f"    -> Generated {len(reactive_cands)} reactive candidates.")
        
        # Separate indexing so we maintain sequence numbers per reaction type
        type_counters = {'chemisorption': 0, 'h_exchange': 0}
        
        for c in reactive_cands:
            c.info['surface_state'] = label
            rtype = c.info.get('reaction_type', 'chemisorption')
            file_type = 'hex' if rtype == 'h_exchange' else 'chem'
            
            idx = type_counters[rtype]
            write(f'cands_{label}_{file_type}_{idx:03d}.extxyz', c)
            type_counters[rtype] += 1
            all_results.append(c)

    print(f"\n--- Study Complete. Total candidates generated: {len(all_results)} ---")
    print("Structures saved as cands_[Label]_[Mechanism]_[Index].extxyz")

if __name__ == "__main__":
    study_dipas_on_multiple_surfaces()
