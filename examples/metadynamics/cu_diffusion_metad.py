#!/usr/bin/env python
"""
Tier-2 metadynamics validation (physical, literature-comparable).

Cu adatom diffusion on Cu(100) with the EMT potential. The 2-D free-energy
surface over the adatom's lateral (x, y) position is reconstructed by
metadynamics and the hop barrier is compared against:

  * a NEB barrier computed with the SAME EMT potential (rigorous internal
    cross-check — both must agree within the metaD resolution), and
  * the published diffusion-barrier band:
        EMT ~0.4 eV,  DFT ~0.5 eV (hop),  experiment ~0.28-0.4 eV.

Run:
    python cu_diffusion_metad.py [n_steps]

EMT-only, so it runs on CPU in a few minutes. Outputs the 2-D FES plot to
./cu_metad/fes_2d.png.
"""
import os
import sys
import numpy as np
from ase.build import fcc100, add_adsorbate
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE
from ase.mep import NEB

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.metadynamics import MetadynamicsWorkflow


def build(adatom_xy):
    slab = fcc100("Cu", size=(3, 3, 3), vacuum=8.0)
    zmin = slab.positions[:, 2].min()
    slab.set_constraint(FixAtoms(mask=[a.position[2] < zmin + 1.0 for a in slab]))
    add_adsorbate(slab, "Cu", height=1.8, position=adatom_xy)
    slab.set_tags([1] * (len(slab) - 1) + [2])     # substrate=1, adatom=2
    return slab


def neb_barrier():
    s0 = fcc100("Cu", size=(3, 3, 3), vacuum=8.0)
    a = s0.cell[0, 0] / 3.0
    ini, fin = build((a * 1.5, a * 1.5)), build((a * 2.5, a * 1.5))
    for s in (ini, fin):
        s.calc = EMT()
        BFGS(s, logfile=None).run(fmax=0.05, steps=200)
    images = [ini.copy() for _ in range(7)]
    images[-1] = fin.copy()
    for im in images:
        im.calc = EMT()
    neb = NEB(images, climb=True)
    neb.interpolate("idpp", mic=True)
    FIRE(neb, logfile=None).run(fmax=0.05, steps=200)
    e = [im.get_potential_energy() for im in images]
    return max(e) - e[0], a


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60000

    barrier_neb, a = neb_barrier()
    print(f"[NEB ] Cu(100) hop barrier (EMT) = {barrier_neb:.3f} eV")

    atoms = build((a * 1.5, a * 1.5))
    BFGS_atoms = atoms.copy(); BFGS_atoms.calc = EMT()
    BFGS(BFGS_atoms, logfile=None).run(fmax=0.05, steps=200)
    atoms = BFGS_atoms
    ad = len(atoms) - 1

    cfg = {
        "temperature_K": 500.0, "timestep_fs": 2.0, "friction": 0.02,
        "steps": n_steps, "deposition_stride": 20, "colvar_stride": 200,
        "height": 0.03, "bias_factor": 10,
        "cvs": [
            {"name": "x", "type": "coordinate", "atom": ad, "axis": "x",
             "sigma": 0.15, "grid_min": 0.0, "grid_max": 2 * a},
            {"name": "y", "type": "coordinate", "atom": ad, "axis": "y",
             "sigma": 0.15, "grid_min": a * 0.5, "grid_max": a * 2.5},
        ],
        "plot": {"cvs": ["x", "y"], "bins": 100},
    }
    engine = SimulationEngine({"engine": {"potential": {"backend": "emt"}}})
    res = MetadynamicsWorkflow(engine, config=cfg).run(atoms, output_dir="cu_metad")

    x, y, fes = res["fes"]
    fx = fes.min(axis=1); fx -= fx.min()
    barrier_metad = float(fx.max())

    print(f"[MetaD] Cu(100) hop barrier (EMT) = {barrier_metad:.3f} eV "
          f"({len(res['bias'].heights)} Gaussians)")
    print(f"[Cmp ] metaD vs NEB diff = {abs(barrier_metad - barrier_neb):.3f} eV")
    print("[Lit ] EMT ~0.4 eV | DFT ~0.5 eV (hop) | exp ~0.28-0.4 eV")
    print("FES plot: cu_metad/fes_2d.png")


if __name__ == "__main__":
    main()
