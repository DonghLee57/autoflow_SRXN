import unittest
import os
import sys
import numpy as np
import yaml
import shutil
from ase.build import molecule
from ase.calculators.emt import EMT

from autoflow_srxn.vibrational_analyzer import VibrationalAnalyzer
from autoflow_srxn.qpoint_handler import QPointParser
from autoflow_srxn.thermo_engine import ThermoCalculator

class TestVibrationAndThermo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock Engine for VibrationalAnalyzer
        class MockEngine:
            def __init__(self):
                self.all_config = {}
            def get_calculator(self):
                return EMT()
        cls.engine = MockEngine()
        cls.atoms = molecule('H2O')
        cls.atoms.set_cell([10, 10, 10])
        cls.atoms.center()
        cls.atoms.calc = EMT()

    def test_qpoints_nested_format(self):
        """Verify that qpoints.yaml is written in the standardized nested format."""
        # Use a unique name to avoid conflicts with existing 'vib_analysis' folders
        vib_name = "test_vib_h2o"
        if os.path.exists(vib_name):
            shutil.rmtree(vib_name)
            
        va = VibrationalAnalyzer(self.atoms, self.engine, name=vib_name)
        va.run_analysis(overwrite=True)
        
        test_file = 'test_qpoints_nested.yaml'
        va.generate_qpoints_file(test_file)
        
        try:
            with open(test_file, 'r') as f:
                data = yaml.safe_load(f)
            
            self.assertEqual(data['natom'], 3)
            self.assertIn('reciprocal_lattice', data)
            
            # Check nested structure of eigenvector
            band1 = data['phonon'][0]['band'][0]
            eig = band1['eigenvector']
            
            # Should be [ [ [ux,ix], [uy,iy], [uz,iz] ], ... ]
            self.assertEqual(len(eig), 3) # 3 atoms
            self.assertEqual(len(eig[0]), 3) # 3 components per atom
            
            # Test parser can read it
            parser = QPointParser(test_file)
            modes = parser.get_filtered_modes()
            self.assertGreater(len(modes), 0)
            self.assertEqual(modes[0]['eigenvector'].shape, (3, 3))
            
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)
            if os.path.exists(vib_name):
                shutil.rmtree(vib_name)

    def test_thermo_calculator(self):
        """Test vibrational free energy calculation."""
        # ThermoCalculator.__init__ takes (frequencies_thz: List[float])
        freqs_thz = [10.0, 20.0, 30.0]
        
        calc = ThermoCalculator(freqs_thz)
        f_vib = calc.calculate_vib_free_energy(T=298.15)
        
        self.assertIsInstance(f_vib, float)
        # ZPE should be positive for real frequencies
        self.assertGreater(calc.calculate_zpe(), 0)

if __name__ == '__main__':
    unittest.main()
