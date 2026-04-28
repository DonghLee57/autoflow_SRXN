import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to sys.path
from autoflow_srxn.vibrational_analyzer import calculate_mac, calculate_atomic_participation

def parse_qpoints(file_path):
    """Parses all modes from a qpoints.yaml file."""
    with open(file_path, 'r') as f:
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        data = yaml.load(f, Loader=Loader)
    
    bands = data['phonon'][0]['band']
    natom = data['natom']
    
    freqs = []
    eigs = [] # List of (3N,) flat vectors
    
    for b in bands:
        freqs.append(b['frequency'])
        # Reconstruct flat mass-weighted eigenvector e_k
        raw_eig = b['eigenvector']
        e_vec = []
        for atom_data in raw_eig:
            # atom_data is [real, imag] for each component if flat, 
            # but usually AutoFlow-SRXN writes - [real, imag] for each k,ll
            # Looking at vibrational_analyzer.py:252: w.write("      - [ %17.14f, %17.14f ]\n" % (float(e_vec[k, ll]), 0.0))
            # So for each atom, we have 3 pairs.
            if isinstance(atom_data[0], list): # Nested [ [r,i], [r,i], [r,i] ]
                for comp in atom_data:
                    e_vec.append(comp[0])
            else: # Flat [r,i] pairs
                e_vec.append(atom_data[0])
        eigs.append(np.array(e_vec))
        
    return np.array(freqs), np.array(eigs).T # eigs: (3N, modes)

def run_visualization():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'outputs_si110'))
    fh_path = os.path.join(out_dir, "qpoints_fhva.yaml")
    ph_path = os.path.join(out_dir, "qpoints_phva.yaml")
    
    if not os.path.exists(fh_path) or not os.path.exists(ph_path):
        print(f"Error: YAML files not found in {out_dir}")
        return

    print("Parsing FHVA...")
    freqs_fh, eigs_fh = parse_qpoints(fh_path)
    print("Parsing PHVA...")
    freqs_ph, eigs_ph = parse_qpoints(ph_path)
    
    n_atoms = eigs_fh.shape[0] // 3
    num_modes_fh = len(freqs_fh)
    num_modes_ph = len(freqs_ph)
    
    # --- 1. Mode Matching & MAC ---
    print(f"Matching {num_modes_ph} PHVA modes to FHVA...")
    best_matches = [] # List of (ph_idx, fh_idx, mac, freq_fh, freq_ph)
    
    # We only match modes with freq > 0.5 THz to avoid noise/translations
    for i in range(num_modes_ph):
        if freqs_ph[i] < 0.5: continue
        
        m_ph = eigs_ph[:, i]
        best_mac = -1.0
        best_fh_idx = -1
        
        for j in range(num_modes_fh):
            m_fh = eigs_fh[:, j]
            mac = calculate_mac(m_ph, m_fh)
            if mac > best_mac:
                best_mac = mac
                best_fh_idx = j
        
        best_matches.append({
            'ph_idx': i,
            'fh_idx': best_fh_idx,
            'mac': best_mac,
            'freq_fh': freqs_fh[best_fh_idx],
            'freq_ph': freqs_ph[i]
        })
        
    # --- 2. Visualization ---
    
    # A. Frequency Parity Plot
    plt.figure(figsize=(6, 6))
    f_fh = [m['freq_fh'] for m in best_matches]
    f_ph = [m['freq_ph'] for m in best_matches]
    plt.scatter(f_fh, f_ph, alpha=0.6, edgecolors='k', label='Matched Modes')
    
    max_f = max(max(f_fh), max(f_ph)) * 1.1
    plt.plot([0, max_f], [0, max_f], 'r--', label='Ideal (y=x)')
    plt.xlabel('FHVA Frequency (THz)')
    plt.ylabel('PHVA Frequency (THz)')
    plt.title('Metric 1: Frequency Parity (Si(110) + DiPAS)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.savefig(os.path.join(out_dir, "plot_freq_parity.png"), dpi=150)
    print(f"Saved: plot_freq_parity.png")

    # B. MAC Distribution
    plt.figure(figsize=(8, 4))
    macs = [m['mac'] for m in best_matches]
    plt.hist(macs, bins=20, color='skyblue', edgecolor='black', alpha=0.8)
    plt.axvline(np.mean(macs), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(macs):.3f}')
    plt.xlabel('Modal Assurance Criterion (MAC)')
    plt.ylabel('Count')
    plt.title('Metric 2: MAC Distribution (PHVA vs FHVA Match)')
    plt.legend()
    plt.savefig(os.path.join(out_dir, "plot_mac_dist.png"), dpi=150)
    print(f"Saved: plot_mac_dist.png")

    # C. Atomic Participation Ratio (Top Mode)
    # Find the mode with highest frequency (likely internal Si-H or C-H)
    top_match = sorted(best_matches, key=lambda x: x['freq_ph'], reverse=True)[0]
    
    p_ph = calculate_atomic_participation(eigs_ph[:, top_match['ph_idx']], n_atoms)
    p_fh = calculate_atomic_participation(eigs_fh[:, top_match['fh_idx']], n_atoms)
    
    plt.figure(figsize=(10, 4))
    x = np.arange(n_atoms)
    width = 0.35
    plt.bar(x - width/2, p_fh, width, label='FHVA', color='gray', alpha=0.5)
    plt.bar(x + width/2, p_ph, width, label='PHVA', color='orange', alpha=0.7)
    plt.xlabel('Atom Index')
    plt.ylabel('Participation Ratio')
    plt.title(f'Metric 3: Participation Ratio (Mode @ {top_match["freq_ph"]:.2f} THz)')
    plt.legend()
    plt.xticks(x[::5]) # Show every 5th index
    plt.savefig(os.path.join(out_dir, "plot_participation.png"), dpi=150)
    print(f"Saved: plot_participation.png")

if __name__ == "__main__":
    run_visualization()
