# Ni Precursors on Si / SiO2

This example demonstrates how to run a unified batch adsorption study combining multiple substrates (Si, SiO2) and multiple precursors (`AllylCpNi`, `Ni(PF3)4`), co-adsorbed with an inhibitor. 

This single script replaces the manual, multi-script workflow previously split across `phase0` to `phase4`.

## Features
- **Substrates**: Pre-relaxed `Si100`, `SiO2_O_term`, and `SiO2_Si_term` slabs (from the `structures/slabs/` directory).
- **Inhibitor**: Physisorption of the inhibitor molecule to find the best functionalized site.
- **Precursors**: `AllylCpNi` and `Ni(PF3)4` physisorption and chemisorption searches on the inhibited surface.
- **Dynamic Configuration**: Iterates through all combinations programmatically and generates `config.yaml` files on the fly for `autoflow_srxn`.

## Usage
Simply run the script:
```bash
python run_workflow.py
```
This will generate individual `config_<substrate>_<precursor>.yaml` files and `results_<substrate>_<precursor>` output directories for each combination.
