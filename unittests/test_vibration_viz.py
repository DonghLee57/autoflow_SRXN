import unittest
import sys
import os
import numpy as np
from ase.build import molecule

# Add src to sys.path
from autoflow_srxn.potentials import SimulationEngine
from autoflow_srxn.vibrational_analyzer import VibrationalAnalyzer, MultiModeFollower

class TestVibrationViz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup output directory
        cls.output_dir = "test_mode_anims"
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)
            
        cls.config = {
            'engine': {
                'potential': {'backend': 'emt'},
            },
            'analysis': {
                'vibrational': {
                    'qpoints_file': 'test_qpoints.yaml',
                    'mode_refinement': {
                        'fmax': 0.1,
                        'freq_threshold_thz': 100.0,  # Catch all modes for testing
                        'max_modes': 2,
                        'perturbation_alpha': 0.2,
                        'max_displacement': 0.5,
                    },
                    'visualization': {
                        'enabled': True,
                        'n_frames': 5,
                        'output_traj': os.path.join(cls.output_dir, 'relaxation.extxyz'),
                    }
                }
            }
        }

    def test_animation_generation(self):
        """Test if MultiModeFollower correctly generates extxyz animation files."""
        # 1. Create a simple molecule and generate a qpoints.yaml
        atoms = molecule('H2')
        atoms.set_cell([10, 10, 10])
        atoms.center()
        
        engine = SimulationEngine(config=self.config)
        analyzer = VibrationalAnalyzer(atoms, engine)
        analyzer.generate_qpoints_file(filename='test_qpoints.yaml')
        
        self.assertTrue(os.path.exists('test_qpoints.yaml'))
        
        # 2. Run MultiModeFollower
        follower = MultiModeFollower(engine, self.config)
        # Note: In real life, optimize also runs relaxation which might change positions.
        # But for test, we just want to see if files are written.
        final_atoms = follower.optimize(atoms)
        
        # 3. Check if files exist in output_dir
        # MultiModeFollower should have saved mode_1_refinement.extxyz etc.
        files = os.listdir(self.output_dir)
        self.assertGreater(len(files), 0, "No animation files were generated.")
        
        has_extxyz = any(f.endswith('.extxyz') for f in files)
        self.assertTrue(has_extxyz, "No .extxyz files found in output directory.")
        
        # 4. Verify content (specifically 'forces' array as requested)
        from ase.io import read
        test_file = os.path.join(self.output_dir, files[0])
        frames = read(test_file, index=':')
        self.assertGreater(len(frames), 0)
        
        # In ASE's extxyz reader, 'forces' property is mapped to calc.results['forces']
        has_forces = False
        if 'forces' in frames[0].arrays:
            has_forces = True
        elif frames[0].calc is not None and 'forces' in frames[0].calc.results:
            has_forces = True
            
        self.assertTrue(has_forces, "Eigenvectors missing from 'forces' column in extxyz.")

    @classmethod
    def tearDownClass(cls):
        # Cleanup
        if os.path.exists('test_qpoints.yaml'):
            os.remove('test_qpoints.yaml')
        if os.path.exists(cls.output_dir):
            import shutil
            shutil.rmtree(cls.output_dir)

if __name__ == '__main__':
    unittest.main()
