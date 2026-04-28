import unittest
import os
import sys
import numpy as np
from ase.build import molecule
from ase.calculators.emt import EMT

from autoflow_srxn.potentials import SimulationEngine

class TestPotentials(unittest.TestCase):
    def test_engine_init_emt(self):
        """Test SimulationEngine with EMT backend."""
        config = {
            'engine': {
                'potential': {
                    'backend': 'emt',
                    'device': 'cpu'
                }
            }
        }
        engine = SimulationEngine(config)
        calc = engine.get_calculator()
        self.assertIsInstance(calc, EMT)

    def test_engine_init_mace_fallback(self):
        """Test if SimulationEngine handles missing MACE gracefully."""
        config = {
            'engine': {
                'potential': {
                    'backend': 'mace',
                    'model': 'invalid_path_to_force_failure',
                    'device': 'cpu'
                }
            }
        }
        engine = SimulationEngine(config)
        self.assertEqual(engine.backend, 'mace')

    def test_engine_relaxation(self):
        """Test basic relaxation wrapper using EMT."""
        config = {
            'engine': {
                'potential': {'backend': 'emt'},
                'relaxation': {
                    'fmax': 0.05,
                    'steps': 10,
                    'optimizer': 'BFGS'
                }
            }
        }
        engine = SimulationEngine(config)
        atoms = molecule('H2')
        atoms.calc = engine.get_calculator()
        
        # Move one atom slightly to force relaxation
        atoms.positions[1, 2] += 0.1
        
        initial_fmax = (atoms.get_forces()**2).sum(axis=1).max()**0.5
        engine.relax(atoms)
        final_fmax = (atoms.get_forces()**2).sum(axis=1).max()**0.5
        
        self.assertLess(final_fmax, initial_fmax)

    def test_engine_md(self):
        """Test NVT MD wrapper using EMT."""
        config = {
            'engine': {
                'potential': {'backend': 'emt'},
                'md': {
                    'temperature_K': 300,
                    'timestep_fs': 1.0,
                    'md_steps': 5
                }
            }
        }
        engine = SimulationEngine(config)
        atoms = molecule('H2')
        atoms.calc = engine.get_calculator()
        
        initial_pos = atoms.positions.copy()
        engine.run_md(atoms)
        final_pos = atoms.positions
        
        # Atoms should have moved
        self.assertFalse(np.allclose(initial_pos, final_pos))

class TestZBLCalculator(unittest.TestCase):
    """Unit tests for the pure-Python ZBL screened Coulomb calculator."""

    def _make_h2(self, bond_length=0.74):
        """Two-hydrogen molecule at given bond length (Å).

        Atom 0 is placed at the origin so the bond length equals the
        inter-atomic distance exactly, regardless of ASE's default centering.
        """
        from ase.build import molecule
        atoms = molecule('H2')
        atoms.positions = [[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]]
        atoms.pbc = False
        return atoms

    def test_zbl_energy_positive(self):
        """ZBL energy is always positive (repulsive)."""
        from autoflow_srxn.potentials import ZBLCalculator
        atoms = self._make_h2(bond_length=0.5)
        calc  = ZBLCalculator(cutoff_inner=0.3, cutoff_outer=2.0)
        atoms.calc = calc
        e = atoms.get_potential_energy()
        self.assertGreater(e, 0.0)

    def test_zbl_energy_zero_beyond_cutoff(self):
        """ZBL energy is zero when all atoms are beyond cutoff_outer."""
        from autoflow_srxn.potentials import ZBLCalculator
        atoms = self._make_h2(bond_length=4.0)
        calc  = ZBLCalculator(cutoff_inner=1.0, cutoff_outer=2.5)
        atoms.calc = calc
        e = atoms.get_potential_energy()
        self.assertAlmostEqual(e, 0.0, places=10)

    def test_zbl_forces_newton_third(self):
        """ZBL forces satisfy Newton's third law (sum ≈ 0)."""
        from autoflow_srxn.potentials import ZBLCalculator
        atoms = self._make_h2(bond_length=0.6)
        calc  = ZBLCalculator(cutoff_inner=0.4, cutoff_outer=2.0)
        atoms.calc = calc
        forces = atoms.get_forces()
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)

    def test_zbl_forces_repulsive_direction(self):
        """ZBL forces push atoms apart (repulsive along bond axis)."""
        from autoflow_srxn.potentials import ZBLCalculator
        atoms = self._make_h2(bond_length=0.6)   # < H-H pair outer (0.8 Å) so force is active
        calc  = ZBLCalculator(cutoff_inner=0.5, cutoff_outer=2.0)
        atoms.calc = calc
        forces = atoms.get_forces()
        # Atom 0 is at origin, atom 1 is at +z → atom 0 should be pushed in -z
        self.assertLess(forces[0, 2], 0.0)
        # Atom 1 should be pushed in +z
        self.assertGreater(forces[1, 2], 0.0)

    def test_zbl_energy_decreases_with_distance(self):
        """ZBL energy decreases monotonically as H-H atoms separate.

        Distances are kept below the H-H per-pair outer cutoff (~0.8 Å) so
        the energy is non-zero and strictly decreasing throughout the range.
        """
        from autoflow_srxn.potentials import ZBLCalculator
        calc = ZBLCalculator(cutoff_inner=0.3, cutoff_outer=2.5)
        energies = []
        for r in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75]:
            atoms = self._make_h2(bond_length=r)
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
        for i in range(len(energies) - 1):
            self.assertGreater(energies[i], energies[i + 1])

    def test_zbl_with_emt_sum_calculator(self):
        """ZBL + EMT SumCalculator produces a finite energy and forces."""
        from ase.calculators.mixing import SumCalculator
        from autoflow_srxn.potentials import ZBLCalculator
        atoms = self._make_h2(bond_length=0.74)
        zbl   = ZBLCalculator(cutoff_inner=0.5, cutoff_outer=2.0)
        emt   = EMT()
        combined = SumCalculator([emt, zbl])
        atoms.calc = combined
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        self.assertTrue(np.isfinite(e))
        self.assertTrue(np.all(np.isfinite(f)))

    def test_engine_zbl_emt(self):
        """SimulationEngine with EMT + ZBL returns finite energy."""
        from ase.build import molecule
        from autoflow_srxn.potentials import SimulationEngine
        config = {
            'engine': {
                'potential': {
                    'backend': 'emt',
                    'zbl': {
                        'enabled': True,
                        'cutoff_inner': 0.5,
                        'cutoff_outer': 2.0,
                    }
                }
            }
        }
        engine = SimulationEngine(config)
        atoms  = molecule('H2')
        atoms.calc = engine.get_calculator()
        e = atoms.get_potential_energy()
        self.assertTrue(np.isfinite(e))

    def test_engine_zbl_disabled(self):
        """SimulationEngine with zbl.enabled=false returns EMT calculator directly."""
        from autoflow_srxn.potentials import SimulationEngine
        config = {
            'engine': {
                'potential': {
                    'backend': 'emt',
                    'zbl': {'enabled': False}
                }
            }
        }
        engine = SimulationEngine(config)
        calc   = engine.get_calculator()
        self.assertIsInstance(calc, EMT)


if __name__ == '__main__':
    unittest.main()
