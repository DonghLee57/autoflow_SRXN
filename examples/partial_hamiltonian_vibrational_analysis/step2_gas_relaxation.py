"""
Step 2: Gas-phase molecule relaxation
  1. Load config
  2. Load DIPAS gas molecule
  3. Add 10 Å vacuum cell
  4. Relax
  5. Save to results/
"""
import os
import sys
import yaml
from ase.io import read

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../../src')))

from potentials import SimulationEngine
from surface_utils import write_standardized_vasp
from logger_utils import setup_logger

def run_step2():
    os.chdir(script_dir)

    with open('config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    out_dir = config['paths'].get('output_dir', 'results')
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, 'step2_gas_relaxation.log')
    logger = setup_logger(log_path=log_file, verbose=True, mode='a')

    logger.info("=================================================================")
    logger.info("  Step 2: Gas-phase DIPAS Molecule Relaxation                    ")
    logger.info("=================================================================")

    engine = SimulationEngine(config=config)
    calc   = engine.get_calculator()
    rel    = config['engine']['relaxation']

    gas_path = os.path.join(out_dir, 'DIPAS_gas_relaxed.vasp')

    if os.path.exists(gas_path):
        logger.info(f'[Step 2] Loading pre-relaxed gas molecule from {gas_path}')
        dipas_gas = read(gas_path)
        dipas_gas.calc = calc
    else:
        logger.info('[Step 2] Loading gas-phase DIPAS and adding 10 Å vacuum cell...')
        dipas_gas = read(config['paths']['adsorbate'])
        dipas_gas.center(vacuum=10.0)
        dipas_gas.calc = calc

        logger.info('Relaxing gas-phase DIPAS...')
        engine.relax(dipas_gas, fmax=rel['fmax'], steps=rel['steps'], optimizer=rel['optimizer'])

        write_standardized_vasp(gas_path, dipas_gas)
        logger.info(f'  Saved: {gas_path}')

    E_gas = dipas_gas.get_potential_energy()
    logger.info(f'  Gas Molecule Energy: E_gas = {E_gas:.6f} eV  ({len(dipas_gas)} atoms)')
    logger.info("=================================================================")

if __name__ == '__main__':
    run_step2()
