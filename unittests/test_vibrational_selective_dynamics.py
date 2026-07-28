import os
import tempfile
import unittest

from ase import Atoms
from ase.constraints import FixAtoms, FixScaled
from ase.io import read, write

from autoflow_srxn.vibrational.mode_following import _load_structure
from autoflow_srxn.vibrational.vibrational_analyzer import VibrationalAnalyzer


class _DummyEngine:
    def __init__(self, config):
        self.all_config = config

    def get_calculator(self):
        return None


class _ListLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class TestVibrationalSelectiveDynamics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=".")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_poscar(self, selective_dynamics):
        atoms = Atoms(
            "H3",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        )
        if selective_dynamics:
            atoms.set_constraint(
                [
                    FixScaled(0, mask=[True, False, False]),
                    FixAtoms(indices=[1]),
                ]
            )

        path = os.path.abspath(os.path.join(self.temp_dir.name, "POSCAR"))
        write(path, atoms, format="vasp", direct=True)
        return path

    @staticmethod
    def _config(path, *, phva_enabled, use_selective_dynamics=True):
        return {
            "paths": {"input_structure": path},
            "analysis": {
                "vibrational": {
                    "phva": {
                        "enabled": phva_enabled,
                        "use_selective_dynamics": use_selective_dynamics,
                    }
                }
            },
        }

    @staticmethod
    def _selective_constraints(atoms):
        return [
            constraint
            for constraint in atoms.constraints
            if isinstance(constraint, (FixAtoms, FixScaled))
        ]

    def test_fhva_uses_all_atoms_with_or_without_selective_dynamics(self):
        for has_selective_dynamics in (False, True):
            with self.subTest(selective_dynamics=has_selective_dynamics):
                path = self._write_poscar(has_selective_dynamics)
                config = self._config(path, phva_enabled=False)
                engine = _DummyEngine(config)

                atoms, frozen_idx = _load_structure(
                    config, self.temp_dir.name, engine, _ListLogger()
                )
                analyzer = VibrationalAnalyzer(atoms, engine)

                self.assertIsNone(frozen_idx)
                self.assertEqual(self._selective_constraints(atoms), [])
                self.assertEqual(self._selective_constraints(analyzer.atoms), [])
                self.assertIsNone(analyzer.indices)

    def test_direct_fhva_ignores_constraints_without_mutating_input(self):
        path = self._write_poscar(True)
        atoms = read(path)
        config = self._config(path, phva_enabled=False)

        analyzer = VibrationalAnalyzer(atoms, _DummyEngine(config))

        self.assertEqual(len(self._selective_constraints(atoms)), 2)
        self.assertEqual(self._selective_constraints(analyzer.atoms), [])
        self.assertIsNone(analyzer.indices)

    def test_phva_uses_selective_dynamics_when_enabled(self):
        for has_selective_dynamics in (False, True):
            with self.subTest(selective_dynamics=has_selective_dynamics):
                path = self._write_poscar(has_selective_dynamics)
                config = self._config(
                    path,
                    phva_enabled=True,
                    use_selective_dynamics=True,
                )
                engine = _DummyEngine(config)

                atoms, frozen_idx = _load_structure(
                    config, self.temp_dir.name, engine, _ListLogger()
                )
                analyzer = VibrationalAnalyzer(atoms, engine)

                expected_constraint_count = 2 if has_selective_dynamics else 0
                expected_indices = [0, 2] if has_selective_dynamics else [0, 1, 2]
                self.assertIsNone(frozen_idx)
                self.assertEqual(
                    len(self._selective_constraints(atoms)),
                    expected_constraint_count,
                )
                self.assertEqual(
                    len(self._selective_constraints(analyzer.atoms)),
                    expected_constraint_count,
                )
                self.assertEqual(analyzer.indices, expected_indices)

    def test_phva_defaults_to_using_selective_dynamics(self):
        path = self._write_poscar(True)
        config = self._config(path, phva_enabled=True)
        del config["analysis"]["vibrational"]["phva"]["use_selective_dynamics"]
        engine = _DummyEngine(config)

        atoms, frozen_idx = _load_structure(
            config, self.temp_dir.name, engine, _ListLogger()
        )
        analyzer = VibrationalAnalyzer(atoms, engine)

        self.assertIsNone(frozen_idx)
        self.assertEqual(len(self._selective_constraints(atoms)), 2)
        self.assertEqual(analyzer.indices, [0, 2])

    def test_phva_can_ignore_selective_dynamics(self):
        path = self._write_poscar(True)
        config = self._config(
            path,
            phva_enabled=True,
            use_selective_dynamics=False,
        )
        engine = _DummyEngine(config)

        atoms, frozen_idx = _load_structure(
            config, self.temp_dir.name, engine, _ListLogger()
        )
        analyzer = VibrationalAnalyzer(atoms, engine)

        self.assertIsNone(frozen_idx)
        self.assertEqual(self._selective_constraints(atoms), [])
        self.assertEqual(self._selective_constraints(analyzer.atoms), [])
        self.assertEqual(analyzer.indices, [0, 1, 2])

    def test_phva_frozen_z_applies_only_when_selective_dynamics_is_disabled(self):
        for use_selective_dynamics in (True, False):
            with self.subTest(use_selective_dynamics=use_selective_dynamics):
                path = self._write_poscar(True)
                config = self._config(
                    path,
                    phva_enabled=True,
                    use_selective_dynamics=use_selective_dynamics,
                )
                config["analysis"]["vibrational"]["phva"]["frozen_z_ang"] = 0.5
                engine = _DummyEngine(config)

                atoms, frozen_idx = _load_structure(
                    config, self.temp_dir.name, engine, _ListLogger()
                )
                analyzer = VibrationalAnalyzer(atoms, engine)

                if use_selective_dynamics:
                    self.assertIsNone(frozen_idx)
                    self.assertEqual(len(self._selective_constraints(atoms)), 2)
                    self.assertEqual(analyzer.indices, [0, 2])
                else:
                    self.assertEqual(frozen_idx, [0])
                    self.assertEqual(len(self._selective_constraints(atoms)), 1)
                    self.assertEqual(analyzer.indices, [1, 2])

    def test_phva_radius_is_read_from_nested_config_when_selective_is_disabled(self):
        atoms = Atoms(
            "H4",
            positions=[
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.5, 0.0, 0.0],
            ],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        )
        atoms.set_tags([0, 0, 0, 1])
        config = self._config(
            "unused",
            phva_enabled=True,
            use_selective_dynamics=False,
        )
        config["analysis"]["vibrational"]["phva"]["phva_radius_ang"] = 0.75

        analyzer = VibrationalAnalyzer(atoms, _DummyEngine(config))

        self.assertEqual(analyzer.indices, [2, 3])


if __name__ == "__main__":
    unittest.main()
