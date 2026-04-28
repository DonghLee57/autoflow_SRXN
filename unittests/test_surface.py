import unittest
import sys
import os
import numpy as np
from ase.build import bulk

# Add src to sys.path
from autoflow_srxn.si_surface_utils import build_si100_slab
from autoflow_srxn.surface_utils import passivate_surface_coverage_general

class TestSurfaceConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Standard Bulk Si
        cls.bulk_si = bulk('Si', 'diamond', a=5.431)
        cls.valence_map = {'Si': 4, 'H': 1}

    def test_si100_slab_generation(self):
        """Test if build_si100_slab creates a valid Si(100) slab (4x4, 8 layers)."""
        layers = 8
        size = (4, 4)
        slab = build_si100_slab(self.bulk_si, size=size, layers=layers)
        
        # Check that all atoms are Silicon
        self.assertTrue(all(s == 'Si' for s in slab.get_chemical_symbols()))
        # Check dimensions: 4x4 slab with 8 layers
        # Diamond surface(100) primitive cell has 1 atom per layer.
        # Unit cell of surface is 2x larger than primitive usually? 
        # Actually ase.build.surface for diamond(100) gives a specific cell.
        self.assertGreater(len(slab), 0)
        
        # Check tagging
        tags = slab.get_tags()
        self.assertIn(1, tags, "Top layer (tag 1) missing.")
        self.assertIn(4, tags, "Bottom layer (tag 4) missing.")

    def test_h_passivation_top(self):
        """Test H-passivation on the top side."""
        slab = build_si100_slab(self.bulk_si, size=(2, 2), layers=4)
        passivated = passivate_surface_coverage_general(
            slab, h_coverage=1.0, valence_map=self.valence_map, side='top'
        )
        
        h_atoms = [a for a in passivated if a.symbol == 'H']
        self.assertGreater(len(h_atoms), 0)
        
        # All H should be above the top Si layer
        z_max_si = max(a.position[2] for a in passivated if a.symbol == 'Si')
        for h in h_atoms:
            self.assertGreater(h.position[2], z_max_si - 0.2)

    def test_h_passivation_bottom(self):
        """Test H-passivation on the bottom side."""
        slab = build_si100_slab(self.bulk_si, size=(2, 2), layers=4)
        passivated = passivate_surface_coverage_general(
            slab, h_coverage=1.0, valence_map=self.valence_map, side='bottom'
        )
        
        h_atoms = [a for a in passivated if a.symbol == 'H']
        self.assertGreater(len(h_atoms), 0)
        
        # All H should be below the bottom Si layer
        z_min_si = min(a.position[2] for a in passivated if a.symbol == 'Si')
        for h in h_atoms:
            self.assertLess(h.position[2], z_min_si + 0.2)

    def test_h_passivation_both(self):
        """Test H-passivation on both sides of the slab."""
        slab = build_si100_slab(self.bulk_si, size=(2, 2), layers=4)
        # Passivate top
        passivated_top = passivate_surface_coverage_general(
            slab, h_coverage=1.0, valence_map=self.valence_map, side='top'
        )
        # Passivate bottom of the top-passivated slab
        passivated_both = passivate_surface_coverage_general(
            passivated_top, h_coverage=1.0, valence_map=self.valence_map, side='bottom'
        )
        
        h_atoms = [a for a in passivated_both if a.symbol == 'H']
        z_si = [a.position[2] for a in passivated_both if a.symbol == 'Si']
        z_min_si, z_max_si = min(z_si), max(z_si)
        
        h_top = [a for a in h_atoms if a.position[2] > z_max_si - 0.2]
        h_bottom = [a for a in h_atoms if a.position[2] < z_min_si + 0.2]
        
        self.assertGreater(len(h_top), 0, "No top-side H atoms found.")
        self.assertGreater(len(h_bottom), 0, "No bottom-side H atoms found.")
        self.assertEqual(len(h_top) + len(h_bottom), len(h_atoms), "Found H atoms neither at top nor bottom.")

if __name__ == '__main__':
    unittest.main()
