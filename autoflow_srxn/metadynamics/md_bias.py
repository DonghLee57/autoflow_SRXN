"""
autoflow_srxn.metadynamics.md_bias
==================================
PLUMED-free metadynamics for ASE.

``MetadynamicsBias`` is an ASE :class:`~ase.calculators.calculator.Calculator`
that returns the history-dependent bias energy and forces along an arbitrary
set of collective variables (CVs). Combine it with the physical (MLIP/EMT)
calculator via :class:`~ase.calculators.mixing.SumCalculator` and drive any
ASE MD integrator; deposit a Gaussian every ``stride`` steps by attaching
:meth:`deposit` as an observer.

Bias potential (N CVs):

    V(s) = Σ_k h_k · exp( -Σ_d (s_d - c_{k,d})² / (2 σ_d²) )

Well-tempered deposition scales each new Gaussian height by
``exp(-V(s)/(k_B ΔT))`` with ΔT = (γ-1)·T.

Free-energy surface (FES) reconstruction:

    standard       : F(s) = -V(s)
    well-tempered  : F(s) = -(γ/(γ-1)) · V(s)
"""

from __future__ import annotations

import os
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes


class MetadynamicsBias(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, cvs, sigmas=None, height=0.05, temperature_K=300.0,
                 bias_factor=None, **kwargs):
        """
        Parameters
        ----------
        cvs : list of CollectiveVariable
            Active CVs spanning the biased space.
        sigmas : sequence of float, optional
            Gaussian width per CV. Defaults to each CV's ``default_sigma``.
        height : float
            Initial Gaussian height h0 (eV).
        temperature_K : float
            Simulation temperature (used for the well-tempered factor).
        bias_factor : float or None
            Well-tempered bias factor γ (> 1). ``None`` -> standard metaD.
        """
        super().__init__(**kwargs)
        self.cvs = list(cvs)
        self.ncv = len(self.cvs)
        if sigmas is None:
            sigmas = [cv.default_sigma for cv in self.cvs]
        self.sigmas = np.asarray(sigmas, dtype=float)
        self.h0 = float(height)
        self.kT = units.kB * float(temperature_K)
        self.gamma = None if bias_factor in (None, 0) else float(bias_factor)
        # ΔT = (γ-1) T  ->  k_B ΔT in eV
        self._kB_dT = None if self.gamma is None else (self.gamma - 1.0) * self.kT

        self.centers = np.empty((0, self.ncv))   # deposited Gaussian centers
        self.heights = np.empty((0,))            # deposited Gaussian heights
        self._last_cv = None                     # cache for logging

    # -- bias evaluation ----------------------------------------------------

    def _cv_values_and_grads(self, atoms):
        s = np.zeros(self.ncv)
        grads = []
        for d, cv in enumerate(self.cvs):
            val, grad = cv.value_and_grad(atoms)
            s[d] = val
            grads.append(grad)
        return s, grads

    def _bias_and_dVds(self, s):
        """Bias energy V(s) and dV/ds_d at a single CV point."""
        if len(self.heights) == 0:
            return 0.0, np.zeros(self.ncv)
        diff = (s[None, :] - self.centers) / self.sigmas[None, :]      # (nhills, ncv)
        g = self.heights * np.exp(-0.5 * np.sum(diff**2, axis=1))      # (nhills,)
        V = float(g.sum())
        dVds = -np.sum((g[:, None] * diff / self.sigmas[None, :]), axis=0)
        return V, dVds

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        s, grads = self._cv_values_and_grads(atoms)
        self._last_cv = s
        V, dVds = self._bias_and_dVds(s)
        forces = np.zeros((len(atoms), 3))
        for d in range(self.ncv):
            forces -= dVds[d] * grads[d]            # F = -dV/ds · ds/dR
        self.results["energy"] = V
        self.results["forces"] = forces

    # -- deposition ---------------------------------------------------------

    def deposit(self, atoms):
        """Add one Gaussian at the current CV position (call periodically)."""
        s, _ = self._cv_values_and_grads(atoms)
        if self.gamma is not None:
            V, _ = self._bias_and_dVds(s)
            h = self.h0 * np.exp(-V / self._kB_dT)
        else:
            h = self.h0
        self.centers = np.vstack([self.centers, s])
        self.heights = np.append(self.heights, h)
        return s, h

    # -- I/O ----------------------------------------------------------------

    def write_hills(self, path):
        """Write deposited Gaussians (PLUMED-like HILLS file)."""
        labels = [cv.label for cv in self.cvs]
        header = "# " + "  ".join(labels + [f"sigma_{l}" for l in labels] + ["height"])
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + "\n")
            for c, h in zip(self.centers, self.heights):
                row = list(c) + list(self.sigmas) + [h]
                fh.write("  ".join(f"{v: .6f}" for v in row) + "\n")

    # -- free-energy reconstruction ----------------------------------------

    def _fes_scale(self):
        return 1.0 if self.gamma is None else self.gamma / (self.gamma - 1.0)

    def grid_axes(self, bins=100, margin=0.1):
        """Per-CV grid axis, using configured grid_min/max or sampled range."""
        axes = []
        for d, cv in enumerate(self.cvs):
            gmin, gmax = cv.grid_min, cv.grid_max
            if gmin is None or gmax is None:
                col = self.centers[:, d] if len(self.centers) else np.array([0.0, 1.0])
                lo, hi = float(col.min()), float(col.max())
                span = max(hi - lo, 1e-3)
                gmin = lo - margin * span if gmin is None else gmin
                gmax = hi + margin * span if gmax is None else gmax
            axes.append(np.linspace(gmin, gmax, bins))
        return axes

    def free_energy_grid(self, bins=100):
        """Return (axes, FES) on the full N-D grid (FES in eV, min shifted to 0)."""
        axes = self.grid_axes(bins=bins)
        mesh = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([m.ravel() for m in mesh], axis=1)          # (npts, ncv)
        V = np.zeros(pts.shape[0])
        for c, h in zip(self.centers, self.heights):
            diff = (pts - c) / self.sigmas[None, :]
            V += h * np.exp(-0.5 * np.sum(diff**2, axis=1))
        fes = -self._fes_scale() * V
        fes = fes.reshape([len(a) for a in axes])
        fes -= fes.min()
        return axes, fes

    def free_energy_2d(self, dims=(0, 1), bins=100):
        """2D FES over CV ``dims``. Extra CVs are marginalised out via
        F(a,b) = -kT log Σ_others exp(-F/kT)."""
        axes, fes = self.free_energy_grid(bins=bins)
        if self.ncv == 2:
            return axes[dims[0]], axes[dims[1]], fes if dims == (0, 1) else fes.T
        other = tuple(d for d in range(self.ncv) if d not in dims)
        p = np.exp(-(fes - fes.min()) / self.kT)
        p_marg = p.sum(axis=other)
        # restore the plotted-axis ordering to (dims[0], dims[1])
        kept = [d for d in range(self.ncv) if d in dims]
        if kept != list(dims):
            p_marg = p_marg.T
        fes2d = -self.kT * np.log(p_marg + 1e-300)
        fes2d -= fes2d.min()
        return axes[dims[0]], axes[dims[1]], fes2d


class ColvarLogger:
    """Observer that appends the current CV values and bias energy to a file."""

    def __init__(self, bias: MetadynamicsBias, dyn, path: str, timestep_fs: float):
        self.bias = bias
        self.dyn = dyn
        self.path = path
        self.dt = timestep_fs
        labels = [cv.label for cv in bias.cvs]
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("# step  time_fs  " + "  ".join(labels) + "  V_bias_eV\n")

    def __call__(self):
        atoms = self.dyn.atoms
        s, _ = self.bias._cv_values_and_grads(atoms)
        V, _ = self.bias._bias_and_dVds(s)
        step = self.dyn.get_number_of_steps()
        with open(self.path, "a", encoding="utf-8") as fh:
            row = [step, step * self.dt] + list(s) + [V]
            fh.write("  ".join(f"{v: .6f}" if isinstance(v, float) else f"{v}"
                               for v in row) + "\n")
