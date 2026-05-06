import os
import numpy as np
import matplotlib.pyplot as plt
from ase.geometry import get_distances

def plot_site_proximity(surface, all_sites, filtered_sites, prot_idx, cutoff, output_path):
    """
    Generates a 2D map of site proximity considering PBC.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Plot Substrate (all atoms)
    sub_pos = surface.positions
    plt.scatter(sub_pos[:, 0], sub_pos[:, 1], c='lightgray', s=100, label='Substrate', alpha=0.3, zorder=1)
    
    # 2. Plot Inhibitors
    if len(prot_idx) > 0:
        inh_pos = surface.positions[prot_idx]
        plt.scatter(inh_pos[:, 0], inh_pos[:, 1], c='blue', s=80, label='Inhibitor Atoms', edgecolors='white', zorder=5)
        
        # Plot Shaded Regions (Circles around inhibitor atoms with PBC)
        cell_x = surface.cell[0, 0]
        cell_y = surface.cell[1, 1]
        
        for pos in inh_pos:
            for dx in [-cell_x, 0, cell_x]:
                for dy in [-cell_y, 0, cell_y]:
                    circle = plt.Circle((pos[0] + dx, pos[1] + dy), cutoff, color='blue', alpha=0.05, zorder=2)
                    plt.gca().add_patch(circle)

        # Draw cell boundaries
        plt.plot([0, cell_x, cell_x, 0, 0], [0, 0, cell_y, cell_y, 0], 'k-', lw=1, label='Unit Cell', zorder=6)
        plt.xlim(-1, cell_x + 1)
        plt.ylim(-1, cell_y + 1)

    # 3. Plot Candidate Sites
    filtered_indices = [s["index"] for s in filtered_sites]
    outside_sites = [s for s in all_sites if s["index"] not in filtered_indices]
    
    outside_pos = np.array([s["pos"] for s in outside_sites]) if outside_sites else np.empty((0,3))
    filtered_site_pos = np.array([s["pos"] for s in filtered_sites]) if filtered_sites else np.empty((0,3))
    
    if len(outside_pos) > 0:
        plt.scatter(outside_pos[:, 0], outside_pos[:, 1], c='gray', s=50, marker='x', label='Excluded Sites', zorder=3)
    if len(filtered_site_pos) > 0:
        plt.scatter(filtered_site_pos[:, 1] if False else filtered_site_pos[:, 0], filtered_site_pos[:, 1], 
                    c='red', s=80, marker='o', label='Active Sites (Near Inhibitor)', zorder=4)

    plt.xlabel('X (A)')
    plt.ylabel('Y (A)')
    plt.title(f'Chemisorption Site Proximity Map (Cutoff = {cutoff} A)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

