import argparse
import os
import yaml
import sys
from ase.io import read
from autoflow_srxn.potentials import SimulationEngine
from autoflow_srxn.vibrational_analyzer import VibrationalAnalyzer
from autoflow_srxn.logger_utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Run Vibrational Analysis with specific config.")
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        sys.exit(1)
        
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load structure from config
    paths = config.get("paths", {})
    struct_path = paths.get("input_structure")
    if not struct_path:
        # Fallback to root for legacy support or simpler configs
        struct_path = config.get("input_structure")
        
    if not struct_path:
        print(f"Error: 'paths.input_structure' not specified in {args.config}")
        sys.exit(1)
        
    if not os.path.exists(struct_path):
        # Try relative to config file if not absolute
        config_dir = os.path.dirname(os.path.abspath(args.config))
        alt_path = os.path.join(config_dir, struct_path)
        if os.path.exists(alt_path):
            struct_path = alt_path
        else:
            print(f"Error: Structure file {struct_path} not found.")
            sys.exit(1)
            
    atoms = read(struct_path)
    
    # Run analysis
    vib_cfg = config.get("analysis", {}).get("vibrational", {})
    name_base = vib_cfg.get("name", "results/vib_analysis")
    
    # We want qpoints.yaml in name_base, so we use name_base/cache for VibrationalAnalyzer
    cache_name = os.path.join(name_base, "cache")
    os.makedirs(name_base, exist_ok=True)
    
    # Setup logger
    log_path = os.path.join(name_base, "vibration.log")
    logger = setup_logger(log_path=log_path, verbose=True)
    logger.info(f"Starting vibrational analysis using config: {args.config}")
    logger.info(f"Structure: {struct_path} ({len(atoms)} atoms)")

    # Initialize engine
    engine = SimulationEngine(config=config)
    
    disp = vib_cfg.get("displacement_ang", 0.01)
    
    analyzer = VibrationalAnalyzer(
        atoms=atoms,
        engine=engine,
        displacement=disp,
        name=cache_name
    )
    
    freqs, _ = analyzer.run_analysis(overwrite=True)
    logger.info(f"Analysis complete. Frequencies saved to {os.path.join(name_base, 'qpoints.yaml')}")

if __name__ == "__main__":
    main()
