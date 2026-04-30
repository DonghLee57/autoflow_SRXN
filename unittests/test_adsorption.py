import unittest

from ase.build import bulk

from autoflow_srxn.ads_workflow_mgr import AdsorptionWorkflowManager
from autoflow_srxn.chemisorption_builder import (
    analyze_surface_reactivity,
    build_chemisorption_structures,
)

# Add src to sys.path
from autoflow_srxn.si_surface_utils import build_si100_slab


class TestAdsorption(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bulk_si = bulk("Si", "diamond", a=5.431)
        cls.dipas_smiles = "CC(C)N(C(C)C)[SiH3]"
        cls.config = {
            "reaction_search": {
                "candidate_filter": {
                    "overlap_cutoff": 1.2,
                    "symprec": 0.2,
                    "max_pair_dist": 5.0,
                },
                "mechanisms": {
                    "protector_exchange": {
                        "enabled": True,
                        "reactive_leaves": ["H"],
                        "heuristic": "tag",
                        "target_tags": [2, 10],
                    }
                },
            },
            "surface_prep": {
                "surface_analysis": {
                    "ideal_coordination": {"Si": 4, "H": 1},
                }
            },
        }

    def test_dipas_physisorption(self):
        """Test physisorption of DIPAS on a clean Si(100) surface."""
        slab = build_si100_slab(self.bulk_si, size=(2, 2), layers=4)
        mgr = AdsorptionWorkflowManager(slab, config=self.config)

        # Generate DIPAS molecule
        dipas = mgr.generate_rdkit_conformer(self.dipas_smiles)
        self.assertIsNotNone(dipas)

        # Run physisorption candidate generation
        candidates = mgr.generate_physisorption_candidates(dipas, height=2.5, n_rot=4, tag=2)

        self.assertGreater(len(candidates), 0, "No physisorption candidates generated.")
        # Check if the generated structure has higher number of atoms than slab
        self.assertGreater(len(candidates[0]), len(slab))
        # Check tags (adsorbate should be tag 2)
        tags = candidates[0].get_tags()
        self.assertIn(2, tags)

    def test_chemisorption_on_passivated_surface(self):
        """Test if chemisorption builder identifies exchange sites on H-passivated Si."""
        from autoflow_srxn.surface_utils import passivate_surface_coverage_general

        slab = build_si100_slab(self.bulk_si, size=(2, 2), layers=4)
        # Passivate top with H
        passivated = passivate_surface_coverage_general(slab, h_coverage=1.0, valence_map={"Si": 4, "H": 1}, side="top")
        # Manually tag H atoms as 10 to be picked up by 'tag' heuristic
        tags = passivated.get_tags()
        for i, atom in enumerate(passivated):
            if atom.symbol == "H":
                tags[i] = 10
        passivated.set_tags(tags)

        # Analyze reactivity - should find 'exchange' sites because 'H' is in reactive_leaves
        sites = analyze_surface_reactivity(passivated, self.config)

        self.assertIn("exchange", sites)
        self.assertGreater(len(sites["exchange"]), 0, "No exchange sites found on H-passivated surface.")

        # Verify exchange site data structure
        site = sites["exchange"][0]
        self.assertEqual(site["sym"], "H")
        self.assertIn("db_vector", site)

    def test_chemisorption_logic_branching(self):
        """Test the branching logic in build_chemisorption_structures."""
        slab = build_si100_slab(self.bulk_si, size=(2, 2), layers=4)
        mgr = AdsorptionWorkflowManager(slab, config=self.config)
        dipas = mgr.generate_rdkit_conformer(self.dipas_smiles)

        # Case 1: Clean surface (should trigger generic dissociation)
        candidates_clean = build_chemisorption_structures(dipas, center_target="Si", surface=slab, config=self.config)

        # Case 2: Passivated surface (should trigger protector exchange)
        from autoflow_srxn.surface_utils import passivate_surface_coverage_general

        passivated = passivate_surface_coverage_general(slab, h_coverage=1.0, valence_map={"Si": 4, "H": 1}, side="top")
        # Tag H atoms as 10 for 'tag' heuristic
        tags = passivated.get_tags()
        for i, atom in enumerate(passivated):
            if atom.symbol == "H":
                tags[i] = 10
        passivated.set_tags(tags)

        candidates_pass = build_chemisorption_structures(
            dipas, center_target="Si", surface=passivated, config=self.config
        )

        # Both should potentially generate candidates if geometry allows
        # We just check if at least one candidate is generated in either case to verify routing
        total_cands = len(candidates_clean) + len(candidates_pass)
        self.assertGreater(total_cands, 0, "No chemisorption candidates generated at all.")


if __name__ == "__main__":
    unittest.main()
