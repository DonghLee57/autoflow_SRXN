import numpy as np
import os
from ase.io import read
from ase.geometry import get_distances

def analyze_overlaps(filename, cutoff=1.5):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    try:
        traj = read(filename, index=':')
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    print(f"\n--- Analyzing {os.path.basename(filename)} ({len(traj)} candidates) ---")
    
    for i, atoms in enumerate(traj):
        # Substrate identification
        # Surface Si are usually tags 1, Adsorbate atoms are tags 2.
        # But CatKit candidates might not have tags. Let's use Z-coordinate or symbols.
        symbols = np.array(atoms.get_chemical_symbols())
        substrate_mask = (symbols == 'Si') | (symbols == 'H')
        substrate_pos = atoms.positions[substrate_mask]
        
        if len(substrate_pos) == 0:
            print(f"  {i:2d} | Error: No Si/H substrate atoms found.")
            continue
            
        z_max_substrate = np.max(substrate_pos[:, 2])
        z_thresh = z_max_substrate - 1.0 # Highest layer of Si
        
        real_substrate_mask = atoms.positions[:, 2] <= z_max_substrate + 0.1
        real_adsorbate_mask = atoms.positions[:, 2] > z_max_substrate + 0.1
        
        substrate = atoms[real_substrate_mask]
        adsorbate = atoms[real_adsorbate_mask]
        
        if len(adsorbate) == 0:
            print(f"  Candidate {i:2d}: No adsorbate found above substrate.")
            continue
            
        # specifically check Si-C overlaps
        # Dists between all adsorbate and all substrate
        _, d = get_distances(adsorbate.positions, substrate.positions, cell=atoms.cell, pbc=atoms.pbc)
        min_dist = np.min(d)
        
        # Check Si(surface)-C(adsorbate) specifically
        si_surf_indices = [idx for idx, s in enumerate(substrate.get_chemical_symbols()) if s == 'Si']
        c_ads_indices = [idx for idx, s in enumerate(adsorbate.get_chemical_symbols()) if s == 'C']
        
        if si_surf_indices and c_ads_indices:
            si_surf_pos = substrate.positions[si_surf_indices]
            c_ads_pos = adsorbate.positions[c_ads_indices]
            _, d_si_c = get_distances(c_ads_pos, si_surf_pos, cell=atoms.cell, pbc=atoms.pbc)
            min_si_c = np.min(d_si_c)
        else:
            min_si_c = 999.0

        name = atoms.info.get('name', atoms.info.get('mechanism', f"Candidate_{i}"))[:35]
        print(f"  {i:2d} | {name:35s} | MinDist: {min_dist:5.2f} | Min Si-C: {min_si_c:5.2f}")

if __name__ == "__main__":
    work_dir = r"c:\Users\user\Downloads\dev_w_antigravity\auto_surface_reaction\example_dipas"
    catkit_file = os.path.join(work_dir, 'catkit_dipas_candidates.extxyz')
    manual_file = os.path.join(work_dir, 'dipas_adsorption_candidates.extxyz')
    
    analyze_overlaps(catkit_file)
    analyze_overlaps(manual_file)
