import os
import sys
import yaml
import numpy as np
import argparse

# Add the src directory to Python path for local development
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from thermo_engine import ThermoCalculator, GasThermo, eV_to_J_mol
from qpoint_handler import QPointParser

class AnalyzeThermo:
    """
    Main analyst class for processing Phonopy qpoints.yaml into Gibbs Free Energy.
    """
    def __init__(self, qpoints_yaml, e_elec_ev=0.0):
        self.parser = QPointParser(qpoints_yaml)
        self.e_elec_ev = e_elec_ev
        self.e_elec_j_mol = e_elec_ev * eV_to_J_mol
        
        # Load all frequencies (THz)
        self.freqs = []
        for phon in self.parser.data['phonon']:
            for band in phon['band']:
                self.freqs.append(band['frequency'])
        
    def run_analysis(self, T_range, mode='adsorbent', mass_amu=None, sigma=1, moments=None):
        """
        Calculates thermochemical properties for a given temperature range.
        
        Args:
            mode: 'gas', 'adsorbent', or 'substrate'.
            mass_amu: Molecular mass (required for 'gas').
            sigma: Symmetry number (required for 'gas' entropy).
            moments: Moments of inertia (required for 'gas' rot entropy).
        """
        results = []
        calc = ThermoCalculator(self.freqs)
        
        for T in T_range:
            # Vibrational contribution (Helmholtz free energy includes ZPE)
            s_vib = calc.calculate_vib_entropy(T)
            f_vib = calc.calculate_vib_free_energy(T)
            
            if mode == 'gas':
                if mass_amu is None:
                    raise ValueError("mass_amu is required for 'gas' mode.")
                
                # Detect symmetry for linear/nonlinear logic
                symm = 'nonlinear' if (moments and len(moments) == 3) else 'linear'
                
                # Gas phase corrections: H_corr = H_trans + H_rot
                h_corr = GasThermo.calculate_enthalpy_correction(T, symmetry=symm)
                s_trans = GasThermo.calculate_trans_entropy(mass_amu, T)
                s_rot = GasThermo.calculate_rot_entropy(moments, T, sigma, symmetry=symm) if moments else 0.0
                
                s_total = s_vib + s_trans + s_rot
                # G_gas = E_elec + F_vib + H_corr - T*(S_trans + S_rot)
                g_total = self.e_elec_j_mol + f_vib + h_corr - T * (s_trans + s_rot)
            else:
                # For adsorbents and substrates, G is approx Helmholtz F in the solid phase
                s_total = s_vib
                g_total = self.e_elec_j_mol + f_vib
            
            results.append({
                'T': T,
                'S': s_total,
                'G_kJ_mol': g_total / 1000.0,
                'G_eV': g_total / eV_to_J_mol
            })
            
        return results

def print_table(results, title="Thermochemistry"):
    print(f"\n--- {title} ---")
    print(f"{'T (K)':>8} | {'S (J/mol·K)':>12} | {'G (kJ/mol)':>12} | {'G (eV)':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['T']:8.2f} | {r['S']:12.4f} | {r['G_kJ_mol']:12.4f} | {r['G_eV']:12.6f}")

def main():
    parser = argparse.ArgumentParser(description="Calculate Gibbs Free Energy from qpoints.yaml or config.yaml")
    parser.add_argument("input", help="qpoints.yaml or config.yaml file")
    parser.add_argument("--energy", type=float, help="Electronic energy E_elec (eV)")
    parser.add_argument("--mode", choices=['gas', 'adsorbent', 'substrate'], help="Calculation mode")
    parser.add_argument("--mass", type=float, help="Molecular mass (amu)")
    parser.add_argument("--sigma", type=int, help="Symmetry number")
    parser.add_argument("--moments", type=float, nargs='+', help="Moments of inertia")
    parser.add_argument("--temps", type=float, nargs='+', help="Temperatures to evaluate (K)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return

    # Default settings
    qpoints_file = args.input
    e_elec = args.energy if args.energy is not None else 0.0
    mode = args.mode if args.mode is not None else 'adsorbent'
    mass = args.mass
    sigma = args.sigma if args.sigma is not None else 1
    moments = args.moments
    temps = args.temps if args.temps is not None else [298.15, 400, 500, 600, 700, 800]

    # Check if input is a config.yaml (contains 'thermochemistry' key)
    if args.input.endswith('.yaml'):
        with open(args.input, 'r') as f:
            raw_data = yaml.safe_load(f)
            if raw_data and 'thermochemistry' in raw_data:
                cfg = raw_data['thermochemistry']
                print(f"Loading settings from config: {args.input}")
                
                # Resolve paths relative to config file location
                base_dir = os.path.dirname(os.path.abspath(args.input))
                q_raw = cfg.get('qpoints_file', 'qpoints.yaml')
                qpoints_file = os.path.normpath(os.path.join(base_dir, q_raw))
                
                e_elec = cfg.get('electronic_energy', e_elec)
                mode = cfg.get('mode', mode)
                temps = cfg.get('temperature_range', temps)
                
                if 'gas_properties' in cfg:
                    g_cfg = cfg['gas_properties']
                    mass = g_cfg.get('mass', mass)
                    sigma = g_cfg.get('sigma', sigma)
                    moments = g_cfg.get('moments', moments)

    if not os.path.exists(qpoints_file):
        print(f"Error: qpoints file {qpoints_file} not found.")
        return

    analyzer = AnalyzeThermo(qpoints_file, e_elec)
    
    print(f"Calculation Context:")
    print(f"  Input YAML: {qpoints_file}")
    print(f"  Mode: {mode}")
    print(f"  E_elec: {e_elec:.6f} eV")
    if mode == 'gas':
        print(f"  Gas Prop: mass={mass}, sigma={sigma}, moments={moments}")
    
    try:
        results = analyzer.run_analysis(
            T_range=temps,
            mode=mode,
            mass_amu=mass,
            sigma=sigma,
            moments=moments
        )
        print_table(results, title=f"Results ({mode})")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
