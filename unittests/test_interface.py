"""
Unit tests for autoflow_srxn.interface
=======================================
Split into two suites:

1. ``TestLatticeMatchNumpy`` – pure-NumPy logic (always runs).
2. ``TestInterfaceBuilderPymatgen`` – pymatgen-dependent tests
   (skipped automatically when pymatgen is absent).
"""

import math
import unittest

import numpy as np

from autoflow_srxn.interface import _HAS_PYMATGEN
from autoflow_srxn.interface.builder import (
    POLAR_SG,
    find_coincidences,
    iter_hnf_2d,
    miller_polar_inplane,
    polar_axis_for_sg,
    strain_from_F,
)


# ===========================================================================
# Suite 1: pure-NumPy tests
# ===========================================================================

class TestLatticeMatchNumpy(unittest.TestCase):
    """Tests for core lattice matching logic in builder.py."""

    # -----------------------------------------------------------------------
    # iter_hnf_2d
    # -----------------------------------------------------------------------

    def test_hnf_det1(self):
        """det=1 yields exactly one matrix: [[1,0],[0,1]]."""
        mats = list(iter_hnf_2d(1))
        self.assertEqual(len(mats), 1)
        np.testing.assert_array_equal(mats[0], [[1, 0], [0, 1]])

    def test_hnf_det2(self):
        """det=2 yields exactly two HNF matrices."""
        mats = list(iter_hnf_2d(2))
        dets = [round(abs(np.linalg.det(m))) for m in mats]
        # det==1 plus det==2 matrices
        self.assertIn(1, dets)
        self.assertIn(2, dets)

    def test_hnf_count_up_to_4(self):
        """Known HNF count for max_det=4 is 15."""
        mats = list(iter_hnf_2d(4))
        self.assertEqual(len(mats), 15)

    def test_hnf_lower_triangular(self):
        """All returned matrices are lower-triangular."""
        for m in iter_hnf_2d(6):
            self.assertEqual(m[0, 1], 0, msg=f"Upper off-diagonal nonzero in {m}")

    def test_hnf_positive_diagonal(self):
        """Diagonal entries must be positive integers."""
        for m in iter_hnf_2d(6):
            self.assertGreater(m[0, 0], 0)
            self.assertGreater(m[1, 1], 0)

    # -----------------------------------------------------------------------
    # strain_from_F
    # -----------------------------------------------------------------------

    def test_strain_identity_zero(self):
        """Identity deformation -> zero strain."""
        I = np.eye(2)
        eps1, eps2, vm = strain_from_F(I, I)
        self.assertAlmostEqual(vm, 0.0, places=10)

    def test_strain_known_uniaxial(self):
        """Uniform 5% biaxial stretch should give vm ~ 0.05."""
        A_sub = np.eye(2)
        A_film = np.eye(2) * (1.0 / 1.05)   # film lattice shrunk
        eps1, eps2, vm = strain_from_F(A_sub, A_film)
        self.assertAlmostEqual(vm, 0.05, places=3)

    def test_strain_singular_returns_one(self):
        """Singular film matrix should safely return vm=1."""
        A_sub = np.eye(2)
        A_film = np.zeros((2, 2))
        eps1, eps2, vm = strain_from_F(A_sub, A_film)
        self.assertEqual(vm, 1.0)

    def test_strain_symmetry(self):
        """eps1 <= eps2 (sorted output)."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            A_sub = rng.uniform(2.0, 5.0, (2, 2))
            A_film = rng.uniform(2.0, 5.0, (2, 2))
            eps1, eps2, vm = strain_from_F(A_sub, A_film)
            self.assertLessEqual(eps1, eps2 + 1e-12)

    # -----------------------------------------------------------------------
    # polar_axis_for_sg
    # -----------------------------------------------------------------------

    def test_polar_axis_c_for_hexagonal(self):
        """Hexagonal SG (e.g. 186 P6_3mc) polar along c."""
        axis = polar_axis_for_sg(186)
        np.testing.assert_array_equal(axis, [0, 0, 1])

    def test_polar_axis_none_for_Fm3m(self):
        """Cubic Fm-3m (SG 225) is non-polar."""
        axis = polar_axis_for_sg(225)
        self.assertIsNone(axis)

    def test_polar_axis_b_for_monoclinic(self):
        """Monoclinic P2 (SG 3) is polar along b."""
        axis = polar_axis_for_sg(3)
        np.testing.assert_array_equal(axis, [0, 1, 0])

    def test_polar_sg_count(self):
        """The registry contains exactly 70 polar space groups."""
        self.assertEqual(len(POLAR_SG), 70)

    # -----------------------------------------------------------------------
    # miller_polar_inplane
    # -----------------------------------------------------------------------

    def test_polar_inplane_true_when_nonpolar(self):
        """Non-polar material (axis=None) always returns True."""
        self.assertTrue(miller_polar_inplane((1, 0, 0), None))
        self.assertTrue(miller_polar_inplane((0, 0, 1), None))

    def test_polar_along_c_001_is_NOT_inplane(self):
        """(0,0,1) surface cuts the c-axis polar direction -> not in-plane."""
        axis = np.array([0, 0, 1])
        self.assertFalse(miller_polar_inplane((0, 0, 1), axis))

    def test_polar_along_c_100_IS_inplane(self):
        """(1,0,0) surface: c-axis is in-plane -> True."""
        axis = np.array([0, 0, 1])
        self.assertTrue(miller_polar_inplane((1, 0, 0), axis))

    # -----------------------------------------------------------------------
    # find_coincidences
    # -----------------------------------------------------------------------

    def test_find_coincidences_returns_list(self):
        """Basic smoke test: square lattices -> at least one match."""
        A = np.eye(2) * 3.0
        results = find_coincidences(A, A, max_det=3, strain_cutoff=0.01)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_find_coincidences_sorted_ascending_vm(self):
        """Results must be sorted in ascending vm order (within float epsilon)."""
        A = np.eye(2) * 3.0
        results = find_coincidences(A, A, max_det=4, strain_cutoff=0.1)
        vms = [r["vm"] for r in results]
        for i in range(len(vms) - 1):
            self.assertLessEqual(vms[i], vms[i + 1] + 1e-12,
                                 msg=f"vm not sorted at index {i}: {vms[i]} > {vms[i+1]}")

    def test_find_coincidences_exact_match_vm_zero(self):
        """Identical lattices with identity HNFs -> vm=0."""
        A = np.eye(2) * 4.0
        results = find_coincidences(A, A, max_det=1, strain_cutoff=0.01)
        self.assertGreater(len(results), 0)
        self.assertAlmostEqual(results[0]["vm"], 0.0, places=10)

    def test_find_coincidences_cutoff_respected(self):
        """No result should exceed the strain cutoff."""
        A_sub = np.eye(2) * 3.0
        A_film = np.eye(2) * 4.0
        cutoff = 0.03
        results = find_coincidences(A_sub, A_film, max_det=6, strain_cutoff=cutoff)
        for r in results:
            self.assertLessEqual(r["vm"], cutoff + 1e-9)

    def test_find_coincidences_incommensurate_empty(self):
        """Highly incommensurate lattices with tiny cutoff -> empty list."""
        A_sub = np.eye(2) * 3.0
        A_film = np.eye(2) * math.pi   # irrational ratio
        results = find_coincidences(A_sub, A_film, max_det=3, strain_cutoff=0.001)
        self.assertEqual(results, [])

    def test_find_coincidences_keys(self):
        """Each result dict has the required keys."""
        A = np.eye(2) * 3.0
        results = find_coincidences(A, A, max_det=2, strain_cutoff=0.05)
        required = {"Na", "Nb", "det_Na", "det_Nb", "eps1", "eps2", "vm", "area_ratio"}
        for r in results:
            self.assertTrue(required.issubset(r.keys()), msg=f"Missing keys in {r.keys()}")


# ===========================================================================
# Suite 2: pymatgen-dependent tests
# ===========================================================================

@unittest.skipUnless(_HAS_PYMATGEN, "pymatgen not installed – skipping builder tests")
class TestInterfaceBuilderPymatgen(unittest.TestCase):
    """Tests that require pymatgen to be installed."""

    def _make_si_structure(self):
        """Return a minimal Si diamond-cubic bulk structure."""
        from pymatgen.core import Structure, Lattice

        a = 5.43
        latt = Lattice.cubic(a)
        return Structure(
            latt,
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        )

    def _make_sio2_structure(self):
        """Return a minimal SiO2 (cristobalite) bulk structure."""
        from pymatgen.core import Structure, Lattice

        a = 7.16
        latt = Lattice.cubic(a)
        species = ["Si", "Si", "O", "O", "O", "O"]
        coords = [
            [0.00, 0.00, 0.00],
            [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
        ]
        return Structure(latt, species, coords)

    # -----------------------------------------------------------------------
    # get_surface_lattice_2d
    # -----------------------------------------------------------------------

    def test_surface_lattice_shape(self):
        """get_surface_lattice_2d returns a 2x2 matrix."""
        from autoflow_srxn.interface.builder import get_surface_lattice_2d

        si = self._make_si_structure()
        A = get_surface_lattice_2d(si, (0, 0, 1))
        self.assertEqual(A.shape, (2, 2))

    def test_surface_lattice_positive_area(self):
        """Surface unit cell must have positive area."""
        from autoflow_srxn.interface.builder import get_surface_lattice_2d

        si = self._make_si_structure()
        A = get_surface_lattice_2d(si, (1, 1, 0))
        self.assertGreater(abs(np.linalg.det(A)), 0)

    # -----------------------------------------------------------------------
    # get_slab_atom_count
    # -----------------------------------------------------------------------

    def test_slab_atom_count_positive(self):
        """A reasonable slab should have at least 1 atom."""
        from autoflow_srxn.interface.builder import get_slab_atom_count

        si = self._make_si_structure()
        n = get_slab_atom_count(si, (0, 0, 1), min_thickness_ang=6.0)
        self.assertGreater(n, 0)

    def test_slab_atom_count_scales_with_det(self):
        """HNF supercell should scale the atom count by det(HNF)."""
        from autoflow_srxn.interface.builder import get_slab_atom_count

        si = self._make_si_structure()
        n_base = get_slab_atom_count(si, (0, 0, 1), min_thickness_ang=6.0)
        HNF_2x2 = np.array([[2, 0], [0, 2]])
        n_super = get_slab_atom_count(
            si, (0, 0, 1), min_thickness_ang=6.0, HNF=HNF_2x2
        )
        self.assertEqual(n_super, 4 * n_base)

    # -----------------------------------------------------------------------
    # build_symmetric_slab
    # -----------------------------------------------------------------------

    def test_build_symmetric_slab_returns_atoms(self):
        """build_symmetric_slab should return an ASE Atoms object."""
        import ase
        from autoflow_srxn.interface.builder import build_symmetric_slab

        si = self._make_si_structure()
        atoms = build_symmetric_slab(si, (0, 0, 1), min_thickness_ang=6.0)
        self.assertIsInstance(atoms, ase.Atoms)

    def test_build_symmetric_slab_has_vacuum(self):
        """The slab cell should be taller than the slab itself (vacuum present)."""
        from autoflow_srxn.interface.builder import build_symmetric_slab

        si = self._make_si_structure()
        vac = 12.0
        atoms = build_symmetric_slab(si, (0, 0, 1), min_thickness_ang=6.0, vacuum_ang=vac)
        z_pos = atoms.get_positions()[:, 2]
        slab_height = z_pos.max() - z_pos.min()
        cell_c = atoms.cell[2, 2]
        self.assertGreater(cell_c, slab_height + vac * 0.9)

    # -----------------------------------------------------------------------
    # InterfaceWorkflow end-to-end
    # -----------------------------------------------------------------------

    def test_workflow_screen_returns_sorted_candidates(self):
        """InterfaceWorkflow.screen() returns a non-empty sorted list."""
        from autoflow_srxn.interface import InterfaceWorkflow

        si = self._make_si_structure()
        wf = InterfaceWorkflow(
            sub_structure=si,
            film_structure=si,
            sub_millers=[(0, 0, 1)],
            film_millers=[(0, 0, 1)],
            max_det=3,
            strain_cutoff=0.01,
        )
        candidates = wf.screen()
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        vms = [c.vm for c in candidates]
        self.assertEqual(vms, sorted(vms))

    def test_workflow_screen_identical_lattice_vm_zero(self):
        """Two identical Si structures should yield at least one vm~0 candidate."""
        from autoflow_srxn.interface import InterfaceWorkflow

        si = self._make_si_structure()
        wf = InterfaceWorkflow(
            sub_structure=si,
            film_structure=si,
            sub_millers=[(0, 0, 1)],
            film_millers=[(0, 0, 1)],
            max_det=1,
            strain_cutoff=0.01,
        )
        candidates = wf.screen()
        self.assertAlmostEqual(candidates[0].vm, 0.0, places=10)

    def test_workflow_build_returns_two_atoms(self):
        """InterfaceWorkflow.build() returns (sub_slab, film_slab) tuple."""
        import ase
        from autoflow_srxn.interface import InterfaceWorkflow

        si = self._make_si_structure()
        wf = InterfaceWorkflow(
            sub_structure=si,
            film_structure=si,
            sub_millers=[(0, 0, 1)],
            film_millers=[(0, 0, 1)],
            max_det=1,
            strain_cutoff=0.01,
            min_slab_thickness=6.0,
        )
        candidates = wf.screen()
        sub_slab, film_slab = wf.build(candidates[0])
        self.assertIsInstance(sub_slab, ase.Atoms)
        self.assertIsInstance(film_slab, ase.Atoms)

    def test_workflow_summary_is_string(self):
        """InterfaceWorkflow.summary() returns a non-empty string."""
        from autoflow_srxn.interface import InterfaceWorkflow

        si = self._make_si_structure()
        wf = InterfaceWorkflow(
            sub_structure=si,
            film_structure=si,
            sub_millers=[(0, 0, 1)],
            film_millers=[(0, 0, 1)],
            max_det=2,
            strain_cutoff=0.05,
        )
        candidates = wf.screen()
        summary = wf.summary(candidates)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)


if __name__ == "__main__":
    unittest.main()
