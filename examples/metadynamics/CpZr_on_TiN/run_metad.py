#!/usr/bin/env python
"""
Run 2D metadynamics for CpZr(NMe2)3 on TiN(100) with SevenNet 7net-0.

Usage:
    python build_system.py            # once, to generate inputs/
    python run_metad.py               # full run (config.yaml: 200k steps)
    python run_metad.py --steps 2000  # quick GPU smoke test (~1-2 min)

Designed for a CUDA GPU server. Performs a short MLIP relaxation of the
physisorbed start (bottom TiN layers frozen), then well-tempered metadynamics,
writing COLVAR / HILLS / fes_2d.png to ./results/.
"""
import os
import sys
import argparse
import numpy as np
import yaml
from ase.io import read
from ase.optimize import FIRE
from ase.constraints import FixAtoms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.metadynamics import MetadynamicsWorkflow

HERE = os.path.dirname(os.path.abspath(__file__))


def auto_freeze_z(atoms):
    """z threshold that freezes the bottom ~half of the TiN slab (tag<2)."""
    sub = atoms.positions[atoms.get_tags() < 2, 2]
    return float(sub.min() + 0.45 * (sub.max() - sub.min()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None, help="override config steps")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--structure", default=os.path.join(HERE, "inputs", "CpZr_on_TiN.extxyz"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--no-relax", action="store_true", help="skip the pre-relaxation")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    metad_cfg = config["analysis"]["metadynamics"]
    if args.steps is not None:
        metad_cfg["steps"] = args.steps

    atoms = read(args.structure)
    engine = SimulationEngine(config)

    # bottom-half slab frozen for both relaxation and metadynamics
    freeze_z = auto_freeze_z(atoms)
    metad_cfg["freeze_below_z"] = freeze_z
    mask = atoms.positions[:, 2] < freeze_z
    print(f"[setup] {len(atoms)} atoms; freezing {int(mask.sum())} atoms below z={freeze_z:.2f} A")

    # pre-relax the physisorbed start (loose) so sampling begins from a minimum
    if not args.no_relax:
        rel = atoms.copy()
        rel.calc = engine.get_calculator()
        rel.set_constraint(FixAtoms(mask=mask))
        print("[relax] FIRE to fmax=0.1 eV/A (max 150 steps)...")
        FIRE(rel, logfile="-").run(fmax=0.1, steps=150)
        atoms = rel

    wf = MetadynamicsWorkflow(engine, config=metad_cfg)
    res = wf.run(atoms, output_dir=args.out)

    x, y, fes = res["fes"]
    fx = fes.min(axis=1); fx -= fx.min()
    print(f"[done] {len(res['bias'].heights)} Gaussians; "
          f"approx. barrier along {res['cv_names'][res['plot_dims'][0]]} "
          f"= {float(fx.max()):.3f} eV")
    print(f"[out ] FES + logs in {args.out}/  (fes_2d.png, fes_2d.npz, COLVAR, HILLS)")


if __name__ == "__main__":
    main()
