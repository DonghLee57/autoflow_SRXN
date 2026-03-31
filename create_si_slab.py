import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import surface
from surface_utils import passivate_slab, reconstruct_2x1_buckled
import os

def process_poscar_slab(input_path='POSCAR', output_path='POSCAR_reconstructed', pattern='checkerboard'):
    """
    Read a Si slab from POSCAR and apply organized buckled reconstruction (p2x2 or c4x2).
    """
    if not os.path.exists(input_path):
        print(f"{input_path} not found. Creating a temporary 4x4 Si(100) slab for demonstration.")
        from ase.build import diamond100
        slab = diamond100('Si', size=(4, 4, 8), vacuum=15.0)
        write(input_path, slab, format='vasp')
    else:
        size = (4, 4, 3)
        vacuum = 10.0
        tmp = read(input_path)
        slab = surface(tmp, (1, 0, 0), layers=size[2], vacuum=vacuum)
        slab = slab.repeat((size[0], size[1], 1))
        slab.center(axis=2, vacuum=vacuum)
        slab.translate([0,0,-np.min(slab.positions.T[2])])
        slab = slab[slab.numbers.argsort()]

    # Ensure standard slab PBC: Periodic in X, Y. Non-periodic in Z.
    slab.pbc = [True, True, False]
    print(f"Loaded slab from {input_path} with lattice {slab.get_cell_lengths_and_angles()[:3]}")

    # 2. Apply Organized 2x1 Reconstruction (PBC-Aware)
    # Using 'stripe' for p(2x2) or 'checkerboard' for c(4x2)
    dimer_data = reconstruct_2x1_buckled(slab, bond_length=2.30, buckle=0.7, pattern=pattern)

    # 3. Apply H-Passivation (Bottom)
    slab = passivate_slab(slab, species='H', side='bottom')

    # 4. Insert O-Bridge
    o_offset = 1.5
    for idx1, idx2, dist_vec, S in dimer_data:
        p1 = slab.positions[idx1]
        p2_eff = p1 + dist_vec
        center = (p1 + p2_eff) / 2
        slab += Atoms('O', positions=[center + np.array([0, 0, o_offset])])

    # 5. Export to POSCAR and EXTXYZ
    write(output_path, slab, format='vasp')
    write('si100_reconstructed_passivated.extxyz', slab, format='extxyz')
    print(f"Final structure ({len(slab)} atoms) exported to {output_path} and extxyz.")
    print(f"Pattern Applied: {pattern}")

if __name__ == "__main__":
    process_poscar_slab(input_path='POSCAR_Si_unit.vasp')#, pattern='stripe')
