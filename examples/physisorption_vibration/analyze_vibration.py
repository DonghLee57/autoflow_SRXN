import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

def load_freqs(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    if not data or 'phonon' not in data:
        return None
        
    freqs = []
    for mode in data['phonon'][0]['band']:
        freqs.append(mode['frequency'])
    return np.array(freqs)

def main():
    fhva_path = "results/fhva/qpoints.yaml"
    phva_path = "results/phva/qpoints.yaml"
    
    # Adjust paths if run from example dir
    if not os.path.exists(fhva_path):
        fhva_path = os.path.join("examples", "physisorption_vibration", fhva_path)
        phva_path = os.path.join("examples", "physisorption_vibration", phva_path)

    f_fhva = load_freqs(fhva_path)
    f_phva = load_freqs(phva_path)

    if f_fhva is None or f_phva is None:
        print(f"Error: Could not load qpoints.yaml from {fhva_path} or {phva_path}")
        sys.exit(1)

    # Filter real frequencies (> 0.01 THz)
    f_fhva_real = sorted(f_fhva[f_fhva > 0.01], reverse=True)
    f_phva_real = sorted(f_phva[f_phva > 0.01], reverse=True)

    # Parity match by index (assuming same number of real modes or truncate to shorter)
    n_match = min(len(f_fhva_real), len(f_phva_real))
    f1 = np.array(f_fhva_real[:n_match])
    f2 = np.array(f_phva_real[:n_match])

    # Error metrics
    mae = np.mean(np.abs(f1 - f2))
    rmse = np.sqrt(np.mean((f1 - f2)**2))
    max_err = np.max(np.abs(f1 - f2))
    
    print("=== Frequency Comparison Analysis ===")
    print(f"Total modes compared: {n_match}")
    print(f"MAE: {mae:.4f} THz")
    print(f"RMSE: {rmse:.4f} THz")
    print(f"Max Error: {max_err:.4f} THz")

    # Results dir
    results_dir = os.path.dirname(fhva_path) # e.g. results/fhva -> results
    results_dir = os.path.dirname(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Parity Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(f1, f2, c='blue', alpha=0.6, edgecolors='k', label=f"Modes (N={n_match})")
    
    # 1:1 line
    all_f = np.concatenate([f1, f2])
    lims = [0, max(all_f) * 1.05]
    plt.plot(lims, lims, 'r--', alpha=0.8, label="Ideal Parity (1:1)")
    
    plt.xlabel("FHVA Frequency (THz)", fontsize=12)
    plt.ylabel("PHVA Frequency (THz)", fontsize=12)
    plt.title("Vibrational Frequency Parity: FHVA vs PHVA", fontsize=14)
    plt.text(0.05 * lims[1], 0.9 * lims[1], f"MAE = {mae:.3f} THz\nRMSE = {rmse:.3f} THz", 
             bbox=dict(facecolor='white', alpha=0.5))
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    parity_out = os.path.join(results_dir, "vibration_parity.png")
    plt.savefig(parity_out, dpi=300, bbox_inches='tight')
    print(f"Parity plot saved to {parity_out}")

    # Residual Plot
    plt.figure(figsize=(10, 5))
    plt.stem(range(1, n_match + 1), f2 - f1, basefmt="k-", label="PHVA - FHVA")
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Mode Index (Sorted by Frequency High -> Low)", fontsize=12)
    plt.ylabel("Error (THz)", fontsize=12)
    plt.title("Frequency Residuals (PHVA Deviation)", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    residual_out = os.path.join(results_dir, "vibration_residuals.png")
    plt.savefig(residual_out, dpi=300, bbox_inches='tight')
    print(f"Residual plot saved to {residual_out}")

if __name__ == "__main__":
    main()
