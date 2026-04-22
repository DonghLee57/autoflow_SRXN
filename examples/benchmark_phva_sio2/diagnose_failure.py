import os
import sys
import yaml
import numpy as np
from ase.io import read, write

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from ads_workflow_mgr import AdsorptionWorkflowManager

def diagnose():
    slab = read('SiO2_substrate_standard.vasp')
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Slab atoms: {len(slab)}")
    z_max = slab.positions[:, 2].max()
    print(f"Z max: {z_max}")
    
    mgr = AdsorptionWorkflowManager(slab, config=config, verbose=True)
    dipas = read('../../structures/DIPAS.vasp')
    
    candidates = mgr.generate_physisorption_candidates(
        dipas, 
        height=3.5,
        n_rot=16,
        rot_center='com'
    )
    
    print(f"Generated {len(candidates)} candidates.")
    if len(candidates) == 0:
        print("Diagnosis: No candidates generated. Checking overlap cutoff...")
        global_overlap = config.get('adsorbate_generation', {}).get('overlap_cutoff', 3.0)
        print(f"Current overlap cutoff: {global_overlap}")

if __name__ == "__main__":
    diagnose()
