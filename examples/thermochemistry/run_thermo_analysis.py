"""
Standalone thermochemistry analysis tool.

Reads vibrational data from a phonopy qpoints.yaml and computes
Gibbs free energies over a temperature range.

Usage:
    python run_thermo_analysis.py [config.yaml]

Config schema (analysis.thermochemistry):
    paths.adsorbate            — structure file for auto-computing mass/moments/sigma
    analysis.thermochemistry:
        qpoints_file           — path to phonopy qpoints.yaml
        mode                   — "gas" | "adsorbent"
        temperatures_K         — list of temperatures [K]
        gas:
            mass, moments, sigma   — manual overrides (used when paths.adsorbate is null)
        adsorbent:
            electronic_energy      — reference electronic energy [eV]
"""
import os
import sys
import yaml
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from thermo_engine import ThermoCalculator, GasThermo, eV_to_J_mol
from qpoint_handler import QPointParser
from ase.io import read


class AnalyzeThermo:
    """Process a phonopy qpoints.yaml into Gibbs free energies."""

    def __init__(self, qpoints_yaml, e_elec_ev=0.0):
        self.parser      = QPointParser(qpoints_yaml)
        self.e_elec_j_mol = e_elec_ev * eV_to_J_mol
        self.atoms       = None

        self.freqs = [
            band['frequency']
            for phon in self.parser.data['phonon']
            for band in phon['band']
        ]

    def set_atoms(self, atoms):
        self.atoms = atoms

    def run_analysis(self, T_range, mode='adsorbent',
                     mass_amu=None, sigma=None, moments=None):
        """Compute thermochemical properties for each temperature in T_range.

        When self.atoms is set, mass/moments/sigma are auto-derived from the
        structure (via GasThermo.from_atoms). Manual values take precedence
        when explicitly provided (not None).
        """
        calc = ThermoCalculator(self.freqs)

        # Auto-derive gas-phase properties from structure when available
        if mode == 'gas' and self.atoms is not None:
            auto = GasThermo.from_atoms(self.atoms)
            if mass_amu is None:
                mass_amu = auto['mass']
            if moments is None:
                moments  = auto['moments']
            if sigma is None:
                sigma    = auto['sigma']
            symmetry = auto['symmetry']
        else:
            symmetry = 'nonlinear' if (moments and len(moments) == 3) else 'linear'
            if sigma is None:
                sigma = 1

        results = []
        for T in T_range:
            f_vib = calc.calculate_vib_free_energy(T)
            s_vib = calc.calculate_vib_entropy(T)

            if mode == 'gas':
                if mass_amu is None:
                    raise ValueError("mass_amu is required for 'gas' mode (no structure file found).")
                h_corr  = GasThermo.calculate_enthalpy_correction(T, symmetry=symmetry)
                s_trans = GasThermo.calculate_trans_entropy(mass_amu, T)
                s_rot   = GasThermo.calculate_rot_entropy(moments, T, sigma, symmetry=symmetry) if moments else 0.0
                s_total = s_vib + s_trans + s_rot
                g_total = self.e_elec_j_mol + f_vib + h_corr - T * (s_trans + s_rot)
            else:
                s_total = s_vib
                g_total = self.e_elec_j_mol + f_vib

            results.append({
                'T':       T,
                'S':       s_total,
                'G_kJ_mol': g_total / 1000.0,
                'G_eV':    g_total / eV_to_J_mol,
            })

        return results


def print_table(results, title="Thermochemistry"):
    print(f"\n--- {title} ---")
    print(f"{'T (K)':>8} | {'S (J/mol·K)':>12} | {'G (kJ/mol)':>12} | {'G (eV)':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['T']:8.2f} | {r['S']:12.4f} | {r['G_kJ_mol']:12.4f} | {r['G_eV']:12.6f}")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    if not os.path.exists(input_path):
        print(f"Error: config file '{input_path}' not found.")
        print("Usage: python run_thermo_analysis.py [config.yaml]")
        return

    with open(input_path, 'r') as f:
        config = yaml.safe_load(f)

    # Support both new schema (analysis.thermochemistry) and legacy (thermochemistry)
    cfg = config.get('analysis', {}).get('thermochemistry', config.get('thermochemistry', {}))
    if not cfg:
        print("Error: missing 'analysis.thermochemistry' section in config.")
        return

    base_dir = os.path.dirname(os.path.abspath(input_path))

    # ── qpoints file ───────────────────────────────────────────────────────────
    q_rel       = cfg.get('qpoints_file', 'qpoints.yaml')
    qpoints_file = os.path.normpath(os.path.join(base_dir, q_rel))
    if not os.path.exists(qpoints_file):
        print(f"Error: vibrational data '{qpoints_file}' not found.")
        return

    # ── mode and temperatures ──────────────────────────────────────────────────
    mode  = cfg.get('mode', 'adsorbent')
    temps = cfg.get('temperatures_K', cfg.get('temperature_range', [298.15]))

    # ── electronic energy ──────────────────────────────────────────────────────
    e_elec = cfg.get('adsorbent', {}).get(
        'electronic_energy', cfg.get('electronic_energy', 0.0)
    )

    # ── gas-phase property overrides ───────────────────────────────────────────
    gas_cfg = cfg.get('gas', cfg.get('gas_properties', {}))
    mass    = gas_cfg.get('mass')
    sigma   = gas_cfg.get('sigma')    # None → auto-detect
    moments = gas_cfg.get('moments')

    # ── structure file (for auto-computation) ──────────────────────────────────
    struct_rel = (config.get('paths', {}).get('adsorbate')
                  or cfg.get('structure_file'))
    atoms = None
    if struct_rel:
        struct_file = os.path.normpath(os.path.join(base_dir, struct_rel))
        if os.path.exists(struct_file):
            atoms = read(struct_file)
            print(f"  Structure:  {struct_file}")
        else:
            print(f"  Warning: structure file '{struct_file}' not found.")

    # ── run analysis ───────────────────────────────────────────────────────────
    analyzer = AnalyzeThermo(qpoints_file, e_elec)
    if atoms is not None:
        analyzer.set_atoms(atoms)

    print(f"\n{'='*20} AutoFlow-SRXN THERMO ANALYSIS {'='*20}")
    print(f"  Config:   {input_path}")
    print(f"  qpoints:  {qpoints_file}")
    print(f"  Mode:     {mode}")
    print(f"  E_elec:   {e_elec:.6f} eV")
    if mode == 'gas':
        src = "structure" if atoms is not None else "config"
        print(f"  Gas props: mass={mass}, sigma={sigma}, moments={moments}  [source: {src}]")

    try:
        results = analyzer.run_analysis(
            T_range  = temps,
            mode     = mode,
            mass_amu = mass,
            sigma    = sigma,
            moments  = moments,
        )
        print_table(results, title=f"Results ({mode})")
    except Exception as exc:
        print(f"Error during calculation: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
