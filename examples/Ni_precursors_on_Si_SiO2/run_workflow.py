import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoflow_srxn.surface.main_workflow import run_generic_adsorption_study

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    slabs = {
        "Si100": "structures/slabs/Si100_slab.vasp",
        "SiO2_O_term": "structures/slabs/SiO2_O_term_slab.vasp",
        "SiO2_Si_term": "structures/slabs/SiO2_Si_term_slab.vasp"
    }
    
    precursors = {
        "AllylCpNi": "structures/AllylCpNi_relaxed.vasp",
        "NiPF3_4": "structures/NiPF3_4_relaxed.vasp"
    }
    
    inhibitor = "structures/inhibitor_relaxed.vasp"

    base_config = {
        "paths": {
            "substrate_bulk": None,
            "input_structure": None,
            "precursor": None,
            "inhibitor": None,
            "output_prefix": None,
        },
        "workflow": {
            "slab_relax": False,
            "candidate_relax": True,
            "md_equilibrate": False,
            "post_md_relax": False,
        },
        "relaxation": {
            "fmax": 0.05,
            "steps": 1,
            "frozen_z_ang": 5.5,
        },
        "surface_prep": {
            "slab_generation": {
                "enabled": False
            }
        },
        "reaction_search": {
            "symprec": 0.2,
            "mechanisms": {
                "inhibitor": {
                    "enabled": True,
                    "center": "O",
                    "branching_limit": 1,
                    "physisorption": {
                        "enabled": True,
                        "placement_height": 2.5,
                        "height_mode": "clearance"
                    },
                    "chemisorption": {
                        "enabled": False
                    }
                },
                "precursor": {
                    "center": "Ni",
                    "physisorption": {
                        "enabled": True,
                        "placement_height": 3.5,
                        "height_mode": "clearance",
                        "n_rot": 8
                    },
                    "chemisorption": {
                        "enabled": True,
                        "rot_steps": 8,
                        "coordination_analysis": {
                            "bond_slack": 0.45
                        },
                        "proximity_filter": {
                            "enabled": True,
                            "cutoff": 7.0,
                            "visualize": True
                        }
                    },
                    "ts_search": {
                        "enabled": False
                    }
                }
            },
            "candidate_filter": {
                "overlap_scale": 0.60
            }
        },
        "engine": {
            "potential": {
                "backend": "mace",
                "model": "medium",
                "device": "cpu",
                "dtype": "float64",
                "d3": False
            }
        }
    }

    # Iterate over combinations
    for slab_name, slab_path in slabs.items():
        for prec_name, prec_path in precursors.items():
            print(f"\n==================================================")
            print(f"Running {prec_name} on {slab_name} with Inhibitor")
            print(f"==================================================\n")
            
            config = base_config.copy()
            config["paths"] = base_config["paths"].copy()
            
            config["paths"]["input_structure"] = os.path.join(root_dir, slab_path)
            config["paths"]["precursor"] = os.path.join(root_dir, prec_path)
            config["paths"]["inhibitor"] = os.path.join(root_dir, inhibitor)
            
            output_dir = f"results_{slab_name}_{prec_name}"
            config["paths"]["output_prefix"] = output_dir
            
            config_filename = f"config_{slab_name}_{prec_name}.yaml"
            with open(config_filename, "w") as f:
                yaml.dump(config, f, sort_keys=False)
                
            run_generic_adsorption_study(config_filename)

if __name__ == "__main__":
    main()
