import os
import numpy as np
from ase.io import read, write
from si_surface_utils import generate_standard_surfaces
from ads_workflow_mgr import AdsorptionWorkflowManager

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
    surfaces = generate_standard_surfaces(bulk_si)
    
    all_results = []
    
    # 3. Iterate through each surface state
    for slab in surfaces:
        label = slab.info.get('label', 'Unnamed')
        print(f"\n>>> Processing Surface: {label} ({len(slab)} atoms)")
        
        # Save the surface itself for reference
        write(f'surface_{label}.extxyz', slab)
        
        mgr = AdsorptionWorkflowManager(slab)
        
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

        # B. Reactive Adsorption
        # Determine if it's passivated
        is_h_passivated = 'H_Passivated' in label
        
        if not is_h_passivated:
            # Clean or Oxidized (unpassivated) -> Chemisorption
            print(f"  Sampling Chemisorption on {label}...")
            chem_cands = mgr.generate_chemisorption_candidates(dipas, rot_steps=8)
            print(f"    -> Generated {len(chem_cands)} chemisorption candidates.")
            for i, c in enumerate(chem_cands):
                c.info['surface_state'] = label
                c.info['mechanism'] = 'chemisorption'
                write(f'cands_{label}_chem_{i:03d}.extxyz', c)
                all_results.append(c)
        else:
            # H-passivated -> H-Exchange
            print(f"  Sampling H-Exchange Adsorption on {label}...")
            hex_cands = mgr.generate_h_exchange_candidates(dipas, rot_steps=8)
            print(f"    -> Generated {len(hex_cands)} H-exchange candidates.")
            for i, c in enumerate(hex_cands):
                c.info['surface_state'] = label
                c.info['mechanism'] = 'h_exchange'
                write(f'cands_{label}_hex_{i:03d}.extxyz', c)
                all_results.append(c)

    print(f"\n--- Study Complete. Total candidates generated: {len(all_results)} ---")
    print("Structures saved as cands_[Label]_[Mechanism]_[Index].extxyz")

if __name__ == "__main__":
    study_dipas_on_multiple_surfaces()
