#!/usr/bin/env python
"""
Run config-driven 2D metadynamics and plot the free-energy surface.

Usage:
    python run_metadynamics.py CONFIG.yaml STRUCTURE.vasp [OUTPUT_DIR]

CONFIG.yaml must contain:
    engine:    {potential: {backend: ...}}     # MLIP/EMT backend
    analysis:  {metadynamics: {... CVs ...}}    # see config_full.yaml

The structure's tags should mark the slab (tag < 2) vs the adsorbate
(tag >= 2) so that "Element@substrate" / "Element@adsorbate" CV selectors
resolve correctly (the project's builders set these tags automatically).
"""
import os
import sys
import yaml
from ase.io import read

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.metadynamics import MetadynamicsWorkflow


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    config_path, structure_path = sys.argv[1], sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "metad"

    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    metad_cfg = config.get("analysis", {}).get("metadynamics", {})
    if not metad_cfg.get("enabled", False):
        print("analysis.metadynamics.enabled is false — nothing to do.")
        sys.exit(0)

    atoms = read(structure_path)
    engine = SimulationEngine(config)
    workflow = MetadynamicsWorkflow(engine, config=metad_cfg)

    result = workflow.run(atoms, output_dir=output_dir)
    x, y, fes = result["fes"]
    nx, ny = result["cv_names"][result["plot_dims"][0]], result["cv_names"][result["plot_dims"][1]]
    print(f"Done. 2D FES over ({nx}, {ny}); barrier ≈ {fes.max():.3f} eV.")
    print(f"Outputs written to: {output_dir}/  (fes_2d.png, fes_2d.npz, COLVAR, HILLS)")


if __name__ == "__main__":
    main()
