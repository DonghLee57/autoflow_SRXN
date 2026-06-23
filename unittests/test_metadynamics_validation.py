"""
Validation of the metadynamics engine against references with KNOWN answers.

Two tiers:

  Tier 1 (algorithm correctness)
    * FES reconstruction + well-tempered scaling — exact, deterministic, no MD.
    * Double-well analytic potential — metadynamics must recover the analytic
      barrier. This is the canonical enhanced-sampling sanity benchmark
      (model potential with an exactly known barrier height).

  Tier 2 (physical, literature-comparable) is provided as a runnable example,
  examples/metadynamics/cu_diffusion_metad.py, which compares the metadynamics
  barrier for Cu adatom diffusion against a NEB barrier computed with the SAME
  EMT potential (and the published EMT/DFT diffusion-barrier band). It is kept
  out of the unit suite because a converged surface-diffusion FES needs longer
  sampling than is appropriate for CI.
"""

import numpy as np
import unittest

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixedPlane
from ase.calculators.mixing import SumCalculator
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from autoflow_srxn.metadynamics import CoordinateCV, MetadynamicsBias


class TestCoordinateCV(unittest.TestCase):
    def test_value_and_grad(self):
        atoms = Atoms("H2", positions=[[0.3, 1.1, -0.4], [2.0, 0.0, 0.5]])
        cv = CoordinateCV(1, "y")
        s, grad = cv.value_and_grad(atoms)
        self.assertAlmostEqual(s, 0.0)               # y of atom 1
        expected = np.zeros((2, 3))
        expected[1, 1] = 1.0
        np.testing.assert_allclose(grad, expected)


class TestFESReconstruction(unittest.TestCase):
    """Deterministic check of the FES math (no MD)."""

    def _single_hill_depth(self, bias_factor):
        cv = CoordinateCV(0, "x", grid_min=-2.0, grid_max=2.0)
        bias = MetadynamicsBias([cv], sigmas=[0.2], height=0.1,
                                temperature_K=300.0, bias_factor=bias_factor)
        # place one Gaussian of height 0.1 at the origin
        bias.centers = np.array([[0.0]])
        bias.heights = np.array([0.1])
        axes, fes = bias.free_energy_grid(bins=201)
        # F = -scale * V, shifted to min 0 -> a well of depth scale*h
        return float(fes.max()), float(fes[np.argmin(np.abs(axes[0]))])

    def test_standard_reconstruction(self):
        depth, center = self._single_hill_depth(bias_factor=None)
        self.assertAlmostEqual(depth, 0.1, places=3)      # well depth == h
        self.assertAlmostEqual(center, 0.0, places=4)     # bottom at the hill

    def test_welltempered_scaling(self):
        gamma = 10.0
        depth, center = self._single_hill_depth(bias_factor=gamma)
        scale = gamma / (gamma - 1.0)
        self.assertAlmostEqual(depth, scale * 0.1, places=3)  # depth == γ/(γ-1)·h
        self.assertAlmostEqual(center, 0.0, places=4)


# --- Analytic double well: V = h0*((x/a)^2 - 1)^2 + 0.5*k*y^2 + walls ---------
_A, _H, _K = 1.5, 0.20, 4.0
_XW, _YW, _KW = 2.0, 0.7, 50.0          # harmonic walls keep the walker bounded


class _DoubleWell(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        x, y = atoms.positions[0, 0], atoms.positions[0, 1]
        V = _H * ((x / _A) ** 2 - 1) ** 2 + 0.5 * _K * y ** 2
        fx = -(4 * _H * x * ((x / _A) ** 2 - 1) / _A ** 2)
        fy = -(_K * y)
        for val, w, is_x in ((x, _XW, True), (y, _YW, False)):
            d = abs(val) - w
            if d > 0:
                V += 0.5 * _KW * d ** 2
                F = -_KW * d * np.sign(val)
                if is_x:
                    fx += F
                else:
                    fy += F
        f = np.zeros((1, 3))
        f[0, 0], f[0, 1] = fx, fy
        self.results["energy"] = float(V)
        self.results["forces"] = f


class TestDoubleWellBarrier(unittest.TestCase):
    """End-to-end: metadynamics must recover the analytic barrier height."""

    def test_barrier_recovery(self):
        np.random.seed(0)
        atoms = Atoms("H", positions=[[-_A, 0.0, 0.0]])
        atoms.set_masses([20.0])
        atoms.set_constraint(FixedPlane(0, [0, 0, 1]))   # confine to xy-plane

        bias = MetadynamicsBias(
            [CoordinateCV(0, "x", grid_min=-_XW, grid_max=_XW),
             CoordinateCV(0, "y", grid_min=-_YW, grid_max=_YW)],
            sigmas=[0.10, 0.08], height=0.015,
            temperature_K=400.0, bias_factor=8.0,
        )
        atoms.calc = SumCalculator([_DoubleWell(), bias])
        MaxwellBoltzmannDistribution(atoms, temperature_K=400.0)
        dyn = Langevin(atoms, timestep=2.0 * units.fs, temperature_K=400.0,
                       friction=0.02, fixcm=False, logfile=None)
        dyn.attach(lambda: bias.deposit(atoms), interval=10)
        dyn.run(35000)

        axes, fes = bias.free_energy_grid(bins=100)
        f_x = fes.min(axis=1)                 # 1D profile along x (min over y)
        f_x -= f_x.min()
        x = axes[0]
        barrier = f_x[np.argmin(np.abs(x))]   # FES at the x=0 saddle

        # metaD underestimates slightly at finite sampling; assert it recovers
        # the analytic barrier (0.20 eV) within a generous band — the buggy /
        # unconverged regimes give < 0.10 eV and would fail this.
        self.assertGreater(barrier, 0.13)
        self.assertLess(barrier, 0.27)

        # the two minima must sit at x ≈ ±a
        left = x[np.argmin(np.where(x < 0, f_x, 9.9))]
        right = x[np.argmin(np.where(x > 0, f_x, 9.9))]
        self.assertLess(abs(abs(left) - _A), 0.4)
        self.assertLess(abs(abs(right) - _A), 0.4)


if __name__ == "__main__":
    unittest.main()
