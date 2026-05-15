import os
import sys
import numpy as np
from ase.io import read, write
from ase.build import add_adsorbate

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.surface import (
    create_slab_from_bulk, 
    standardize_vasp_atoms, 
    calculate_gas_energy
)
from autoflow_srxn.utils.logger_utils import setup_logger

def run_study():
    # 1. Setup logging
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    logger = setup_logger(log_path=os.path.join(results_dir, "comparison.log"), verbose=True)
    logger.info("=== Starting TiCl4 on TiN Adsorption Energy Comparison ===")

    # 2. Load Structures
    structures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "structures"))
    bulk_path = os.path.join(structures_dir, "TiN_bulk.vasp")
    mol_path = os.path.join(structures_dir, "TiCl4.vasp")

    if not os.path.exists(bulk_path) or not os.path.exists(mol_path):
        logger.error("Structure files not found!")
        return

    bulk = read(bulk_path)
    ticl4 = read(mol_path)

    # 3. Define Potential Configs
    potentials = [
        {"name": "SevenNet-0", "backend": "sevennet", "model": "7net-0"},
        {"name": "Omni (matpes_r2scan)", "backend": "omni", "modal": "matpes_r2scan"},
        {"name": "MACE-MP (medium)", "backend": "mace", "model": "medium"}
    ]

    results = []

    for pot in potentials:
        logger.info(f"\n--- Testing Potential: {pot['name']} ---")
        config = {"engine": {"potential": pot}, "workflow": {"candidate_relax": True}}
        engine = SimulationEngine(config=config)
        
        try:
            # A. Gas Phase Relaxation (Using modular core function)
            logger.info(f"  [Gas] Relaxing isolated TiCl4...")
            e_gas = calculate_gas_energy(ticl4, config, logger)
            logger.info(f"    E_gas: {e_gas:.4f} eV")

            # B. Clean Slab Relaxation
            logger.info(f"  [Slab] Generating and relaxing TiN(111) slab...")
            slab = create_slab_from_bulk(bulk, miller_indices=[1, 1, 1], thickness=10.0, vacuum=15.0, supercell_matrix=[[2,0],[0,2]])
            e_slab = engine.relax(slab, fmax=0.03, steps=100, frozen_z_ang=6.0, verbose=False)
            logger.info(f"    E_slab: {e_slab:.4f} eV")

            # C. Adsorbed System Relaxation
            logger.info(f"  [Ads] Relaxing TiCl4 on TiN(111)...")
            combined = slab.copy()
            add_adsorbate(combined, ticl4, height=3.0, position=(0, 0))
            
            e_total = engine.relax(combined, fmax=0.03, steps=150, frozen_z_ang=6.0, verbose=False)
            logger.info(f"    E_total: {e_total:.4f} eV")

            # D. Adsorption Energy
            e_ads = e_total - (e_slab + e_gas)
            logger.info(f"  ==> Adsorption Energy: {e_ads:.4f} eV")

            results.append({
                "Potential": pot['name'],
                "E_ads (eV)": f"{e_ads:.4f}"
            })
            
            # Save structure
            write(os.path.join(results_dir, f"ticl4_tin_{pot['backend']}.vasp"), combined)

        except Exception as e:
            logger.error(f"  Failed for {pot['name']}: {e}")

    # Summary Table
    logger.info("\n" + "="*40)
    logger.info(f"{'Potential':<25} | {'E_ads (eV)':<12}")
    logger.info("-" * 40)
    for res in results:
        logger.info(f"{res['Potential']:<25} | {res['E_ads (eV)']:<12}")
    logger.info("="*40)

if __name__ == "__main__":
    run_study()
