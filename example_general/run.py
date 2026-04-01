import os
import numpy as np
from ase.io import read, write
from ads_workflow_mgr import AdsorptionWorkflowManager

def screen_adsorption_sites(slab_path='slab.vasp', mol_path='mol.vasp'):
    print("--- Starting ---")

    slab = read(slab_path)
    mol = read(mol_path)

    mgr = AdsorptionWorkflowManager(slab)

    print("Sampling Physisorption Candidates...")
    phy_candidates = mgr.generate_physisorption_candidates(mol, height=3.5, n_rot=16)
    print(f"Generated {len(phy_candidates)} Physisorption candidates.")
    write('candidates_physisorption.extxyz', phy_candidates)

    all_candidates = phy_candidates #+ chem_candidates
    #write('Adsorption_candidates.extxyz', all_candidates)

    with open('adsorption_log.txt', 'w') as f:
        f.write("--- Adsorption Candidates Log ---\n")
        for i, atoms in enumerate(all_candidates):
            mech = atoms.info.get('mechanism', 'Unknown')
            f.write(f"Candidate {i:03d}: {mech}\n")

    print(f"Total {len(all_candidates)} candidates exported.")
    print("Mechanistic log written to adsorption_log.txt.")

if __name__ == "__main__":
    screen_adsorption_sites()
