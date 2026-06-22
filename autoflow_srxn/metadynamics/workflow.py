"""
autoflow_srxn.metadynamics.workflow
===================================
Config-driven metadynamics runner with 2D free-energy-surface (FES) output.

Reads a ``metadynamics`` config block, builds the requested collective
variables, runs (well-tempered) metadynamics on top of the project's
:class:`SimulationEngine` calculator, and writes:

* ``COLVAR``       — time series of the CVs and bias energy
* ``HILLS``        — deposited Gaussians
* ``metad_traj.extxyz`` — MD trajectory
* ``fes_2d.npz``   — reconstructed 2D FES grid
* ``fes_2d.png``   — 2D FES contour plot over the two selected CVs

Example config (see config_full.yaml for the documented version)::

    metadynamics:
      enabled: true
      temperature_K: 500
      timestep_fs: 1.0
      steps: 20000
      deposition_stride: 50
      height: 0.05
      bias_factor: 10
      cvs:
        - {name: forming_bond,  type: distance,
           center: "Si@adsorbate", partner: "O@substrate", sigma: 0.1}
        - {name: breaking_bond, type: coordination,
           center: "Si@adsorbate", group: "N@adsorbate", r0: 2.2, sigma: 0.1}
      plot: {cvs: [forming_bond, breaking_bond], bins: 120}
"""

from __future__ import annotations

import os
import numpy as np
from ase import Atoms
from ase import units
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.mixing import SumCalculator

from ..utils.logger_utils import get_workflow_logger
from .collective_variables import build_cv
from .md_bias import MetadynamicsBias, ColvarLogger


class MetadynamicsWorkflow:
    def __init__(self, engine, config=None):
        """
        Args:
            engine: SimulationEngine instance (provides the physical calculator).
            config: the ``metadynamics`` config block (dict).
        """
        self.engine = engine
        self.config = config or {}
        self.logger = get_workflow_logger()

    # ------------------------------------------------------------------
    def _build_cvs(self, atoms):
        cv_specs = self.config.get("cvs", [])
        if len(cv_specs) < 2:
            raise ValueError("metadynamics requires at least 2 CVs for a 2D FES.")
        cvs, names = [], []
        for spec in cv_specs:
            cv = build_cv(spec, atoms)
            cvs.append(cv)
            names.append(spec.get("name", cv.label))
        return cvs, names

    def _plot_dims(self, names):
        """Resolve which two CVs span the plotted 2D FES."""
        plot_cfg = self.config.get("plot", {})
        sel = plot_cfg.get("cvs")
        if not sel:
            return (0, 1)
        dims = []
        for item in sel[:2]:
            if isinstance(item, int):
                dims.append(item)
            else:
                dims.append(names.index(item))
        return tuple(dims)

    # ------------------------------------------------------------------
    def run(self, atoms: Atoms, output_dir: str = "metad"):
        os.makedirs(output_dir, exist_ok=True)
        cfg = self.config
        atoms = atoms.copy()

        T = float(cfg.get("temperature_K", 300.0))
        dt = float(cfg.get("timestep_fs", 1.0))
        steps = int(cfg.get("steps", 10000))
        stride = int(cfg.get("deposition_stride", 50))
        colvar_stride = int(cfg.get("colvar_stride", stride))
        height = float(cfg.get("height", 0.05))
        bias_factor = cfg.get("bias_factor", None)
        friction = float(cfg.get("friction", 0.01))

        # --- CVs & bias ---------------------------------------------------
        cvs, names = self._build_cvs(atoms)
        sigmas = [float(s.get("sigma", cv.default_sigma))
                  for s, cv in zip(cfg["cvs"], cvs)]
        bias = MetadynamicsBias(cvs, sigmas=sigmas, height=height,
                                temperature_K=T, bias_factor=bias_factor)

        self.logger.info(
            f"  [MetaD] {len(cvs)} CV(s): {names} | "
            f"{'well-tempered (gamma=%s)' % bias_factor if bias_factor else 'standard'}, "
            f"T={T} K, {steps} steps, deposit every {stride}.")

        # --- optional substrate freezing ---------------------------------
        freeze_z = cfg.get("freeze_below_z", None)
        if freeze_z is not None:
            from ase.constraints import FixAtoms
            mask = atoms.positions[:, 2] < float(freeze_z)
            atoms.set_constraint(FixAtoms(mask=mask))
            self.logger.info(f"  [MetaD] Froze {int(mask.sum())} atoms below z={freeze_z} Å.")

        # --- calculator: physical + bias ---------------------------------
        base = self.engine.get_calculator()
        atoms.calc = SumCalculator([base, bias])

        # --- dynamics -----------------------------------------------------
        MaxwellBoltzmannDistribution(atoms, temperature_K=T)
        dyn = Langevin(atoms, timestep=dt * units.fs,
                       temperature_K=T, friction=friction, logfile="-")

        colvar_path = os.path.join(output_dir, "COLVAR")
        colvar = ColvarLogger(bias, dyn, colvar_path, timestep_fs=dt)
        dyn.attach(colvar, interval=colvar_stride)
        dyn.attach(lambda: bias.deposit(atoms), interval=stride)

        traj_path = os.path.join(output_dir, "metad_traj.extxyz")
        if os.path.exists(traj_path):
            os.remove(traj_path)
        dyn.attach(lambda: write(traj_path, atoms, append=True),
                   interval=cfg.get("traj_stride", colvar_stride))

        dyn.run(steps)
        self.logger.info(f"  [MetaD] MD finished: {len(bias.heights)} Gaussians deposited.")

        # --- outputs ------------------------------------------------------
        bias.write_hills(os.path.join(output_dir, "HILLS"))
        write(os.path.join(output_dir, "metad_final.vasp"), atoms)

        dims = self._plot_dims(names)
        bins = int(cfg.get("plot", {}).get("bins", 120))
        x, y, fes = bias.free_energy_2d(dims=dims, bins=bins)
        np.savez(os.path.join(output_dir, "fes_2d.npz"),
                 x=x, y=y, fes=fes,
                 cv_x=names[dims[0]], cv_y=names[dims[1]])
        self._plot_fes_2d(x, y, fes, cvs[dims[0]], cvs[dims[1]],
                          names[dims[0]], names[dims[1]], output_dir)

        rel = os.path.relpath(output_dir)
        self.logger.info(f"  [MetaD] FES and logs written to {rel}")
        return {"bias": bias, "atoms": atoms, "fes": (x, y, fes),
                "cv_names": names, "plot_dims": dims}

    # ------------------------------------------------------------------
    def _plot_fes_2d(self, x, y, fes, cv_x, cv_y, name_x, name_y, output_dir):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            X, Y = np.meshgrid(x, y, indexing="ij")
            fig, ax = plt.subplots(figsize=(7.5, 6))
            levels = np.linspace(0, np.nanmax(fes), 30)
            cf = ax.contourf(X, Y, fes, levels=levels, cmap="viridis")
            ax.contour(X, Y, fes, levels=levels, colors="k", linewidths=0.3, alpha=0.4)
            cbar = fig.colorbar(cf, ax=ax)
            cbar.set_label("Free energy (eV)", fontsize=12)

            xl = f"{name_x}" + (f" ({cv_x.unit})" if cv_x.unit else "")
            yl = f"{name_y}" + (f" ({cv_y.unit})" if cv_y.unit else "")
            ax.set_xlabel(xl, fontsize=12)
            ax.set_ylabel(yl, fontsize=12)
            ax.set_title("Metadynamics 2D Free Energy Surface", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "fes_2d.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            self.logger.warning(f"  [MetaD] Could not generate FES plot: {e}")
