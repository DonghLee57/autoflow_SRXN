import os
import sys
import yaml
import numpy as np
import time

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from potentials import SimulationEngine
from vibrational_analyzer import VibrationalAnalyzer
from thermo_engine import ThermoCalculator, eV_to_J_mol
from logger_utils import setup_logger

def run_phva_vs_fhva_benchmark():
    logger = setup_logger(log_path="benchmark.log", verbose=True)
    logger.info("="*60)
    logger.info(" PHVA vs FHVA SCIENTIFIC BENCHMARK: DIPAS @ SiO2(001)")
    logger.info("="*60)

    # 1. Setup Engine (MACE-MP-0)
    engine = SimulationEngine(model_type='mace0', device='cpu')
    
    # 2. Load Structure (Optimized DIPAS on SiO2)
    # For benchmark purposes, we will use a representative cluster/slab
    # If the file doesn't exist, we skip or use a mock setup for logic demo
    struct_path = "SiO2_DIPAS_adsorbed.vasp"
    if not os.path.exists(struct_path):
        logger.error(f"Structure {struct_path} not found. Please provide the VASP file.")
        return

    atoms = read(struct_path)
    logger.info(f"Loaded structure with {len(atoms)} atoms.")
    
    # Define active atoms for PHVA (DIPAS + top layer of SiO2)
    # For SiO2(001), let's assume the last 30 atoms are the adsorbate + top surface
    active_indices = list(range(len(atoms)-30, len(atoms)))
    logger.info(f"PHVA Active Set: {len(active_indices)} atoms (Top surface + Adsorbate)")
    logger.info(f"Frozen Set:      {len(atoms)-len(active_indices)} atoms (Support)")

    # 3. Reference: Full Hessian (FHVA)
    logger.info("\n[STAGE 1] Running Reference: Full Hessian (FHVA)...")
    start_fhva = time.time()
    fhva_analyzer = VibrationalAnalyzer(atoms, engine, name="fhva_run")
    freqs_fhva = fhva_analyzer.run_analysis()
    end_fhva = time.time()
    t_fhva = end_fhva - start_fhva
    logger.info(f"FHVA completed in {t_fhva:.2f} seconds.")

    # 4. Test: Partial Hessian (PHVA)
    logger.info("\n[STAGE 2] Running Test: Partial Hessian (PHVA)...")
    start_phva = time.time()
    phva_analyzer = VibrationalAnalyzer(atoms, engine, indices=active_indices, name="phva_run")
    freqs_phva = phva_analyzer.run_analysis()
    end_phva = time.time()
    t_phva = end_phva - start_phva
    logger.info(f"PHVA completed in {t_phva:.2f} seconds.")

    # 5. Scientific Comparison
    logger.info("\n" + "="*60)
    logger.info(" BENCHMARK COMPARISON REPORT")
    logger.info("="*60)
    
    # Matching modes: FHVA has 3N modes, PHVA has 3N_active modes
    # We focus on the intramolecular modes of DIPAS (high frequency)
    fhva_high = sorted([f for f in freqs_fhva if f > 500]) # >500 cm-1
    phva_high = sorted([f for f in freqs_phva if f > 500])
    
    logger.info(f"Efficiency Gain: {t_fhva/t_phva:.1f}x faster")
    
    n_compare = min(len(fhva_high), len(phva_high))
    diffs = np.abs(np.array(fhva_high[-n_compare:]) - np.array(phva_high[-n_compare:]))
    mae = np.mean(diffs)
    max_err = np.max(diffs)
    
    logger.info(f"Frequency Matching (Top {n_compare} modes):")
    logger.info(f"  Mean Absolute Error: {mae:.4f} cm^-1")
    logger.info(f"  Max Absolute Error:  {max_err:.4f} cm^-1")
    
    # Thermochemistry comparison
    thermo_fhva = ThermoCalculator([f/33.356 for f in freqs_fhva]) # cm-1 to THz
    thermo_phva = ThermoCalculator([f/33.356 for f in freqs_phva])
    
    g_fhva = thermo_fhva.calculate_vib_free_energy(298.15) / eV_to_J_mol
    g_phva = thermo_phva.calculate_vib_free_energy(298.15) / eV_to_J_mol
    
    logger.info("\nFree Energy Comparison (T=298.15K):")
    logger.info(f"  G_vib (FHVA): {g_fhva:12.6f} eV")
    logger.info(f"  G_vib (PHVA): {g_phva:12.6f} eV")
    logger.info(f"  Difference:   {abs(g_fhva - g_phva)*1000:12.4f} meV")

    if abs(g_fhva - g_phva) < 0.01:
        logger.info("\n[RESULT] Verification SUCCESS: PHVA reproduces FHVA thermodynamics within 10 meV.")
    else:
        logger.warning("\n[RESULT] Verification WARNING: Discrepancy exceeds 10 meV.")
    logger.info("="*60)

if __name__ == "__main__":
    from ase.io import read
    run_phva_vs_fhva_benchmark()
