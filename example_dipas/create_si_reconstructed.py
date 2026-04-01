import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import surface
from surface_utils import passivate_slab, reconstruct_2x1_buckled
import os

def process_poscar_bulk_to_slab_standard(input_path='POSCAR', output_path='POSCAR_reconstructed', pattern='checkerboard'):
    """
    STRICTLY avoids diamond100. Uses ase.build.surface and make_supercell (multiplier).
    """
    if not os.path.exists(input_path):
        print(f"{input_path} not found. Creating a standard Si conventional bulk cell for demonstration.")
        from ase.build import bulk
        bulk_si = bulk('Si', 'diamond', a=5.43, cubic=True)
        write(input_path, bulk_si, format='vasp')
        
    # 1. Load Bulk from POSCAR
    bulk_atoms = read(input_path)
    
    # 2. Standard Surface Cutting (Si-100)
    print(f"Cutting (1,0,0) surface from bulk ({len(bulk_atoms)} atoms)...")
    # Using 'surface' as requested, avoiding specialized builders.
    slab = surface(bulk_atoms, (1, 0, 0), layers=8, vacuum=15.0)
    
    # 3. Supercell Expansion (* multiplier)
    print(f"Expanding slab unit (1x1) to 4x4 supercell...")
    # This corresponds to a 4x4 supercell of the a x a surface unit.
    slab = slab * (4, 4, 1)
    
    # Standard slab PBC
    slab.pbc = [True, True, False]
    
    # 4. Apply Grid-Based 2x1 Reconstruction
    # (The algorithm automatically detects the diagonal pairing axis in the a x a unit)
    dimer_data = reconstruct_2x1_buckled(slab, bond_length=2.30, buckle=0.7, pattern=pattern)
    
    # 5. Apply H-Passivation (Bottom)
    slab = passivate_slab(slab, species='H', side='bottom')
    
    # 5. Insert O-Bridge (Top)
    o_offset = 1.5
    for idx1, idx2, _, _ in dimer_data: # Ignore old dist_vec
        p1 = slab.positions[idx1]
        # Calculate real MIC-corrected vector between buckled atoms
        v12_mic = slab.get_distance(idx1, idx2, vector=True, mic=True)
        center = p1 + v12_mic / 2
        # Place O exactly at the center + Z offset
        slab += Atoms('O', positions=[center + np.array([0, 0, o_offset])])
        
    # 7. Export to POSCAR and EXTXYZ
    write(output_path, slab, format='vasp')
    write('si100_reconstructed_passivated.extxyz', slab, format='extxyz')
    print(f"Final structure ({len(slab)} atoms) exported to {output_path} and extxyz.")
    print(f"Pattern Applied: {pattern}")

if __name__ == "__main__":
    process_poscar_bulk_to_slab_standard(pattern='checkerboard')
