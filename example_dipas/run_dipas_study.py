import os
import numpy as np
from ase.io import read, write
from ase.build import bulk, surface
from surface_utils import reconstruct_2x1_buckled, passivate_slab
from ads_workflow_mgr import AdsorptionWorkflowManager
from potential_factory import PotentialFactory

def prepare_si_surface(a=5.43):
    """Generate the standardized 4x4 c(4x2) Si(100) surface."""
    bulk_si = bulk('Si', 'diamond', a=a, cubic=True)
    slab = surface(bulk_si, (1, 0, 0), layers=8, vacuum=15.0)
    slab = slab * (4, 4, 1)
    slab.pbc = [True, True, False]
    
    # Identify surface atoms before reconstruction
    z_max = slab.positions[:, 2].max()
    for a in slab:
        if a.position[2] > z_max - 0.5:
            a.tag = 1 # Mark as surface
            
    dimer_data = reconstruct_2x1_buckled(slab, pattern='checkerboard')
    slab = passivate_slab(slab, side='bottom')
    return slab, dimer_data

def study_dipas_adsorption():
    print("--- Starting DIPAS Adsorption Study ---")
    
    # 1. Prepare Surface
    slab, dimer_data = prepare_si_surface()
    print(f"Surface Prepared: {len(slab)} atoms, {len(dimer_data)} dimers.")
    
    # 2. Setup Manager
    mgr = AdsorptionWorkflowManager(slab)
    pot = PotentialFactory(model_type='emt')
    
    # 3. Generate DIPAS
    dipas_smiles = "CC(C)N(C(C)C)[SiH3]" 
    dipas = mgr.generate_rdkit_conformer(dipas_smiles)
    print(f"DIPAS Conformer Generated: {len(dipas)} atoms.")
    
    # 4. Physisorption Candidates (Symmetry-Reduced)
    print("Sampling Physisorption Candidates...")
    phy_candidates = mgr.generate_physisorption_candidates(dipas, height=3.5, n_rot=16)
    print(f"Generated {len(phy_candidates)} Physisorption candidates.")
    
    # 5. Chemisorption (Dissociative) candidates
    # Using the new combinatorial multidentate engine
    print("Sampling Chemisorption (Dissociative) Candidates...")
    chem_candidates = mgr.generate_multidentate_candidates(dipas, center_symbol='Si', k_max=1)
    print(f"Generated {len(chem_candidates)} Chemisorption (Si-L dissociation) candidates.")
    
    # 6. Screening / Export
    all_candidates = phy_candidates + chem_candidates
    write('dipas_adsorption_candidates.extxyz', all_candidates)
    
    # 7. Generate Mechanistic Log
    with open('adsorption_log.txt', 'w') as f:
        f.write("--- DIPAS Adsorption Candidates Log ---\n")
        for i, atoms in enumerate(all_candidates):
            mech = atoms.info.get('mechanism', 'Unknown')
            f.write(f"Candidate {i:03d}: {mech}\n")
            
    print(f"Total {len(all_candidates)} candidates exported.")
    print("Mechanistic log written to adsorption_log.txt.")

if __name__ == "__main__":
    study_dipas_adsorption()
