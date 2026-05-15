import os
import sys
import numpy as np
from ase.io import read

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.utils.logger_utils import setup_logger
from autoflow_srxn.surface import calculate_gas_energy

def run_physi_chemi_comparison():
    # 1. Setup logging
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logger(log_path=os.path.join(results_dir, "physi_chemi_comparison.log"), verbose=True)
    logger.info("=== Physisorption vs Chemisorption Energy Comparison ===")

    # 2. Load the structures from MACE run
    mace_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TiN_TiCl4_MACE", "results", "clean_on_TiCl4"))
    mace_extxyz_path = os.path.join(mace_results_dir, "stage2_precursor_relaxed.extxyz")
    slab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TiN_TiCl4_MACE", "results", "prepared_slab.extxyz"))
    structures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "structures"))
    mol_path = os.path.join(structures_dir, "TiCl4.vasp")

    if not all(os.path.exists(p) for p in [mace_extxyz_path, slab_path, mol_path]):
        logger.error("Required structure files not found! Please run the MACE example first.")
        return

    # Frame 0: Physisorption, Frame -1: Chemisorption
    physi_struct = read(mace_extxyz_path, "0")
    chemi_struct = read(mace_extxyz_path, "-1")
    slab_struct = read(slab_path, "0")
    mol_struct = read(mol_path)

    # 3. Define Potential Configs
    potentials = [
        {"name": "SevenNet-0", "backend": "sevennet", "model": "7net-0"},
        {"name": "Omni (matpes_r2scan)", "backend": "omni", "modal": "matpes_r2scan"},
        {"name": "Omni (matpes_pbe)", "backend": "omni", "modal": "matpes_pbe"},
    ]

    summary = []

    for pot in potentials:
        logger.info(f"\n--- Testing Potential: {pot['name']} ---")
        config = {"engine": {"potential": pot}, "workflow": {"candidate_relax": True}}
        engine = SimulationEngine(config=config)
        
        try:
            calc = engine.get_calculator()
            
            # E_gas (Using modular core function)
            e_gas = calculate_gas_energy(mol_struct, config, logger)
            
            # E_slab
            slab_struct.calc = calc
            e_slab = slab_struct.get_potential_energy()
            e_ref = e_gas + e_slab
            
            # E_phys
            physi_struct.calc = calc
            e_phys_tot = physi_struct.get_potential_energy()
            e_ads_phys = e_phys_tot - e_ref
            
            # E_chemi
            chemi_struct.calc = calc
            e_chemi_tot = chemi_struct.get_potential_energy()
            e_ads_chemi = e_chemi_tot - e_ref
            
            # Stability: Physi -> Chemi
            delta_e = e_ads_chemi - e_ads_phys
            
            logger.info(f"  E_ads (Physi): {e_ads_phys:.4f} eV")
            logger.info(f"  E_ads (Chemi): {e_ads_chemi:.4f} eV")
            logger.info(f"  Stability (Chemi - Physi): {delta_e:.4f} eV")
            
            summary.append({
                "Potential": pot['name'],
                "E_ads_phys": e_ads_phys,
                "E_ads_chemi": e_ads_chemi,
                "Delta_E": delta_e
            })
        except Exception as e:
            logger.error(f"  Failed for {pot['name']}: {e}")

    # 4. Final Table
    logger.info("\n" + "="*80)
    logger.info(f"{'Potential':<25} | {'E_ads(Phys)':<12} | {'E_ads(Chem)':<12} | {'Stability':<12}")
    logger.info("-" * 80)
    for res in summary:
        logger.info(f"{res['Potential']:<25} | {res['E_ads_phys']:<12.4f} | {res['E_ads_chemi']:<12.4f} | {res['Delta_E']:<12.4f}")
    logger.info("="*80)

if __name__ == "__main__":
    run_physi_chemi_comparison()
