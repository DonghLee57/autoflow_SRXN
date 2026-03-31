import numpy as np
from ase import Atoms
from ase.io import read, write
from surface_utils import passivate_slab, reconstruct_2x1_buckled
import os

def process_poscar_slab(input_path='POSCAR', output_path='POSCAR_reconstructed'):
    """
    Read a Si slab from POSCAR and apply PBC-aware reconstruction and passivation.
    """
    if not os.path.exists(input_path):
        print(f"{input_path} not found. Creating a temporary 2x2 Si(100) slab for demonstration.")
        from ase.build import diamond100
        slab = diamond100('Si', size=(2, 2, 8), vacuum=15.0)
        write(input_path, slab, format='vasp')
        
    # 1. Load Slab from POSCAR
    slab = read(input_path)
    # Ensure standard slab PBC: Periodic in X, Y. Non-periodic in Z.
    slab.pbc = [True, True, False]
    print(f"Loaded slab from {input_path} with lattice {slab.get_cell_lengths_and_angles()[:3]}")
    
    # 2. Apply PBC-Aware Reconstruction
    # (Matches neighbors across periodic boundaries)
    dimer_pairs = reconstruct_2x1_buckled(slab, bond_length=2.30, buckle=0.7)
    
    # 3. Apply H-Passivation (Bottom)
    slab = passivate_slab(slab, species='H', side='bottom')
    
    # 4. Insert O-Bridge
    o_offset = 1.5
    for idx1, idx2, dist_vec in dimer_pairs:
        p1 = slab.positions[idx1]
        p2_eff = p1 + dist_vec # Use MIC-corrected position
        center = (p1 + p2_eff) / 2
        slab += Atoms('O', positions=[center + np.array([0, 0, o_offset])])
        
    # 5. Export to POSCAR and EXTXYZ
    write(output_path, slab, format='vasp')
    write('si100_reconstructed_passivated.extxyz', slab, format='extxyz')
    print(f"Final structure ({len(slab)} atoms) exported to {output_path} and extxyz.")

if __name__ == "__main__":
    process_poscar_slab()
