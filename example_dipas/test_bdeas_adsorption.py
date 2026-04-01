import os
import numpy as np
from ase.io import write
from ase.build import bulk, surface
from surface_utils import reconstruct_2x1_buckled, passivate_slab
from ads_workflow_mgr import AdsorptionWorkflowManager

def test_bdeas_multidentate():
    print("--- Starting BDEAS Multidentate Adsorption Test ---")
    
    # 1. Prepare Standard Surface
    bulk_si = bulk('Si', 'diamond', a=5.43, cubic=True)
    slab = surface(bulk_si, (1, 0, 0), layers=8, vacuum=15.0)
    slab = slab * (4, 4, 1)
    slab.pbc = [True, True, False]
    
    # Tagging before reconstruction
    z_max = slab.positions[:, 2].max()
    for a in slab: 
        if a.position[2] > z_max - 0.5: a.tag = 1
        else: a.tag = 0
            
    reconstruct_2x1_buckled(slab, pattern='checkerboard')
    slab = passivate_slab(slab, side='bottom')
    
    # 2. Setup Manager
    mgr = AdsorptionWorkflowManager(slab)
    
    # 3. Generate BDEAS (2 Si-N bonds)
    bdeas_smiles = "CCN(CC)[SiH2]N(CC)CC" # Bis(diethylamino)silane
    molecule = mgr.generate_rdkit_conformer(bdeas_smiles)
    if molecule is None:
        print("Error: BDEAS generation failed.")
        return
    print(f"BDEAS Conformer Generated: {len(molecule)} atoms.")
    
    # 4. Combinatorial Chemisorption (k=1 to 2)
    # k=1: Single Si-H or Si-N cleavage
    # k=2: Double Si-N cleavage (Tripod or bridging)
    candidates = mgr.generate_multidentate_candidates(molecule, center_symbol='Si', k_max=2)
    
    # 5. Export
    if candidates:
        write('bdeas_multidentate_candidates.extxyz', candidates)
        print(f"Success: {len(candidates)} BDEAS candidates exported to bdeas_multidentate_candidates.extxyz.")
    else:
        print("Warning: No candidates generated.")

if __name__ == "__main__":
    test_bdeas_multidentate()
