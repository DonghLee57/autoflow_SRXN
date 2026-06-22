import os
import shutil
import unittest

import numpy as np
from ase import Atoms
from ase.build import fcc111, molecule

from autoflow_srxn.simulation.potentials import SimulationEngine
from autoflow_srxn.metadynamics import (
    DistanceCV, CoordinationCV, ProtonTransferCV, build_cv,
    MetadynamicsBias, MetadynamicsWorkflow,
)


def _num_grad(cv, atoms, eps=1e-5):
    """Central-difference gradient of a CV w.r.t. all coordinates."""
    g = np.zeros((len(atoms), 3))
    for a in range(len(atoms)):
        for c in range(3):
            p = atoms.copy()
            p.positions[a, c] += eps
            sp = cv.value(p)
            p.positions[a, c] -= 2 * eps
            sm = cv.value(p)
            g[a, c] = (sp - sm) / (2 * eps)
    return g


class TestCVGradients(unittest.TestCase):
    def setUp(self):
        # A small non-symmetric cluster so gradients are non-trivial
        self.atoms = Atoms(
            "OHNCl",
            positions=[[0, 0, 0], [0.3, 0.9, 0.1],
                       [1.6, 0.2, 0.0], [3.0, 0.1, 0.2]],
        )

    def test_distance_grad(self):
        cv = DistanceCV(0, 2)
        ana = cv.value_and_grad(self.atoms)[1]
        np.testing.assert_allclose(ana, _num_grad(cv, self.atoms), atol=1e-5)

    def test_coordination_grad(self):
        cv = CoordinationCV(0, [1, 2, 3], r0=1.5, n=6, m=12)
        ana = cv.value_and_grad(self.atoms)[1]
        np.testing.assert_allclose(ana, _num_grad(cv, self.atoms), atol=1e-4)

    def test_proton_transfer_grad(self):
        cv = ProtonTransferCV(donor=0, acceptor=2, proton=1)
        ana = cv.value_and_grad(self.atoms)[1]
        np.testing.assert_allclose(ana, _num_grad(cv, self.atoms), atol=1e-5)


class TestBiasForceConsistency(unittest.TestCase):
    def test_bias_force_matches_energy(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0]])
        bias = MetadynamicsBias([DistanceCV(0, 1)], sigmas=[0.2],
                                height=0.1, bias_factor=None)
        # deposit a couple of hills away from current position
        for d in (0.6, 1.3):
            p = atoms.copy(); p.positions[1, 2] = d
            bias.deposit(p)
        atoms.calc = bias
        f_ana = atoms.get_forces()
        # numerical force from bias energy
        f_num = np.zeros_like(f_ana)
        eps = 1e-5
        for a in range(len(atoms)):
            for c in range(3):
                p = atoms.copy(); p.positions[a, c] += eps; p.calc = bias
                ep = p.get_potential_energy()
                p.positions[a, c] -= 2 * eps
                em = p.get_potential_energy()
                f_num[a, c] = -(ep - em) / (2 * eps)
        np.testing.assert_allclose(f_ana, f_num, atol=1e-4)


class TestMetadWorkflowEMT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SimulationEngine({"engine": {"potential": {"backend": "emt"}}})
        cls.out = "test_metad_out"

    def test_2d_fes_run(self):
        # Cu(111) slab + a Cu adatom -> two distance CVs to surface atoms
        slab = fcc111("Cu", size=(2, 2, 3), vacuum=6.0)
        slab.set_tags([1] * len(slab))                # substrate
        adatom = Atoms("Cu", positions=[slab.positions[-1] + [0.0, 0.0, 2.2]])
        adatom.set_tags([2])                          # adsorbate
        atoms = slab + adatom
        ad_idx = len(atoms) - 1

        cfg = {
            "temperature_K": 300.0,
            "timestep_fs": 1.0,
            "steps": 40,
            "deposition_stride": 5,
            "height": 0.05,
            "bias_factor": 8.0,
            "cvs": [
                {"name": "cv_a", "type": "distance",
                 "center": ad_idx, "partner": 0, "sigma": 0.15},
                {"name": "cv_b", "type": "coordination",
                 "center": ad_idx, "group": "Cu@substrate", "r0": 3.0, "sigma": 0.1},
            ],
            "plot": {"cvs": ["cv_a", "cv_b"], "bins": 40},
        }
        wf = MetadynamicsWorkflow(self.engine, config=cfg)
        res = wf.run(atoms, output_dir=self.out)

        x, y, fes = res["fes"]
        self.assertEqual(fes.shape, (40, 40))
        self.assertTrue(np.isfinite(fes).all())
        self.assertAlmostEqual(float(fes.min()), 0.0, places=6)
        for fname in ["COLVAR", "HILLS", "fes_2d.npz", "fes_2d.png", "metad_traj.extxyz"]:
            self.assertTrue(os.path.exists(os.path.join(self.out, fname)), fname)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.out):
            shutil.rmtree(cls.out)


if __name__ == "__main__":
    unittest.main()
