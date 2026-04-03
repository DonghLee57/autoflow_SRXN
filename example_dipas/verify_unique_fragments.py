import numpy as np
from ase import Atoms
from ads_workflow_mgr import AdsorptionWorkflowManager
from si_surface_utils import reconstruct_2x1_buckled

def verify_logic():
    # 1. Setup a very simple Si slab
    # We need a passivated surface for some generators, but chemisorption 
    # needs dimers.
    slab = Atoms('Si8', positions=[
        [0,0,0], [2.35,0,0], [0,2.35,0], [2.35,2.35,0],
        [0,0,2.35], [2.35,0,2.35], [0,2.35,2.35], [2.35,2.35,2.35]
    ])
    slab.cell = [5.43, 5.43, 10.0]
    slab.pbc = [True, True, False]
    
    # Force a dimer for the manager to find
    # (Simplified: reconstruct might fail on this tiny slab, so we manually tag)
    for i in range(len(slab)): slab[i].tag = 1
    
    mgr = AdsorptionWorkflowManager(slab)
    
    # 2. Generate DIPAS (3 H's, 1 iPr2N)
    dipas_smiles = "CC(C)N(C(C)C)[SiH3]"
    dipas = mgr.generate_rdkit_conformer(dipas_smiles)
    
    print("\n--- Verifying Unique Fragment Handling ---")
    c_idx, ligands = mgr.discover_ligands(dipas, center_symbol='Si')
    print(f"Total ligands discovered: {len(ligands)}")
    for i, l in enumerate(ligands):
        print(f"  Ligand {i}: Formula={l['formula']}, Hapticity={l['hapticity']}")
    
    # 3. Test Chemisorption Candidate Generation
    # We'll monkey-patch/inspect the number of paths taken
    print("\nTesting generate_chemisorption_candidates...")
    # Note: This might return few candidates because of steric/dimer checks on our dummy slab,
    # but we care about how many dissociation loops it enters.
    
    # We can't easily count internal loops without modification, 
    # so we'll add a temporary counter or print in the code if needed.
    # Actually, let's just run it and see the candidates' mechanisms.
    
    candidates = mgr.generate_chemisorption_candidates(dipas, rot_steps=4)
    print(f"Generated {len(candidates)} chemisorption candidates.")
    
    unique_mechanisms = set()
    for c in candidates:
        mech = c.info.get('mechanism', '')
        # Mechanism string contains the fragment symbols
        unique_mechanisms.add(mech.split(',')[0]) # Get the "Cohesive Chemisorption: [Fragment]" part
        
    print("\nUnique Fragment Dissociation Paths found in results:")
    for mech in sorted(unique_mechanisms):
        print(f"  - {mech}")
        
    # Expectation: 
    # One for Fragment A (remaining after H loss) 
    # One for Fragment A (remaining after N(iPr)2 loss)
    
    # Let's check H-exchange too
    print("\nTesting generate_h_exchange_candidates...")
    # This requires surface H, so let's add one
    slab_h = slab.copy()
    slab_h += Atoms('H', positions=[[0,0,3.0]])
    mgr_h = AdsorptionWorkflowManager(slab_h)
    
    h_candidates = mgr_h.generate_h_exchange_candidates(dipas, rot_steps=4)
    print(f"Generated {len(h_candidates)} H-exchange candidates.")
    
    unique_h_mechs = set()
    for c in h_candidates:
        unique_h_mechs.add(c.info.get('mechanism', '').split(':')[0])
        
    print("\nunique H-Exchange paths found:")
    for mech in sorted(unique_h_mechs):
        print(f"  - {mech}")

if __name__ == "__main__":
    verify_logic()
