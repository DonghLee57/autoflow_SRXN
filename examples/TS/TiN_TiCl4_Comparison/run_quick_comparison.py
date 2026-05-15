import os
import sys
import numpy as np
from ase.io import read

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.utils.logger_utils import setup_logger
from autoflow_srxn.surface import calculate_gas_energy

def run_quick_comparison():
    # 1. Setup logging
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logger(log_path=os.path.join(results_dir, "quick_comparison.log"), verbose=True)
    logger.info("=== Comprehensive Energy Comparison (Single Point) ===")

    # 2. Load the structures
    mace_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TiN_TiCl4_MACE", "results", "clean_on_TiCl4"))
    mace_struct_path = os.path.join(mace_results_dir, "stage2_precursor_relaxed.extxyz")
    slab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TiN_TiCl4_MACE", "results", "prepared_slab.extxyz"))
    structures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "structures"))
    mol_path = os.path.join(structures_dir, "TiCl4.vasp")

    if not all(os.path.exists(p) for p in [mace_struct_path, slab_path, mol_path]):
        logger.error("Required structure files not found! Please run the MACE example first.")
        return

    ads_struct = read(mace_struct_path, "-1")
    slab_struct = read(slab_path, "0")
    mol_struct = read(mol_path)

    # 3. Define Potential Configs
    potentials = [
        {"name": "SevenNet-0", "backend": "sevennet", "model": "7net-0"},
        {"name": "Omni (matpes_r2scan)", "backend": "omni", "modal": "matpes_r2scan"},
        {"name": "Omni (matpes_pbe)", "backend": "omni", "modal": "matpes_pbe"},
    ]

    results = []

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
            
            # E_total
            ads_struct.calc = calc
            e_total = ads_struct.get_potential_energy()
            
            e_ads = e_total - (e_gas + e_slab)
            
            logger.info(f"  E_gas: {e_gas:.4f}, E_slab: {e_slab:.4f}, E_total: {e_total:.4f}")
            logger.info(f"  ==> E_ads: {e_ads:.4f} eV")
            
            results.append({
                "Potential": pot['name'],
                "E_ads (eV)": e_ads
            })
        except Exception as e:
            logger.error(f"  Failed for {pot['name']}: {e}")

    # 4. Summary Table
    logger.info("\n" + "="*50)
    logger.info(f"{'Potential':<30} | {'E_ads (eV)':<12}")
    logger.info("-" * 50)
    for res in results:
        logger.info(f"{res['Potential']:<30} | {res['E_ads (eV)']:<12.4f}")
    logger.info("="*50)

if __name__ == "__main__":
    run_quick_comparison()
