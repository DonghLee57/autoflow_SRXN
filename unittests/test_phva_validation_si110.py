import os
import shutil
import unittest

import numpy as np
from ase.build import add_adsorbate, bulk
from ase.io import read, write
from mace.calculators import MACECalculator

from autoflow_srxn.si_surface_utils import SI_VALENCE_MAP
from autoflow_srxn.surface_utils import create_slab_from_bulk, passivate_surface_coverage_general
from autoflow_srxn.vibrational_analyzer import VibrationalAnalyzer, calculate_mac


class EngineMock:
    def __init__(self, calc, config=None):
        self.calc = calc
        self.all_config = config or {}

    def get_calculator(self):
        return self.calc


class TestPHVAVariationSi110(unittest.TestCase):
    """Validation of PHVA for DiPAS on Si(110) H-passivated surface.
    Follows autoflow_SRXN internal standards.
    """

    @classmethod
    def setUpClass(cls):
        # 1. Setup Output Directory
        cls.out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs_si110"))
        if not os.path.exists(cls.out_dir):
            os.makedirs(cls.out_dir)

        # 2. Build Si(110) Slab using internal tools
        bulk_si = bulk("Si", "diamond", a=5.431)
        # Use create_slab_from_bulk for consistent surface geometry
        # thickness=8.0A (~4-5 layers), target_area=80A^2 for reasonable size
        cls.slab = create_slab_from_bulk(bulk_si, (1, 1, 0), thickness=8.0, vacuum=12.0, target_area=80.0, verbose=True)

        # 3. Passivate Both Sides
        cls.slab = passivate_surface_coverage_general(
            cls.slab, h_coverage=1.0, valence_map=SI_VALENCE_MAP, side="top", verbose=True
        )
        cls.slab = passivate_surface_coverage_general(
            cls.slab, h_coverage=1.0, valence_map=SI_VALENCE_MAP, side="bottom", verbose=True
        )

        # 4. Load DIPAS Adsorbate
        dipas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "structures", "DIPAS.vasp"))
        if not os.path.exists(dipas_path):
            raise FileNotFoundError(f"DIPAS structure not found at {dipas_path}")
        cls.dipas = read(dipas_path)

        # 5. Place Adsorbate
        # Use explicit coordinates to avoid ASE metadata issues
        add_adsorbate(cls.slab, cls.dipas, height=3.0, position=(0.0, 0.0))

        # 6. Setup Calculator
        model_path = "c:/Users/user/Downloads/dev_w_antigravity/auto_surface_reaction/MLALD/mace_model.model"
        cls.calc = MACECalculator(model_paths=model_path, device="cpu", default_dtype="float32")
        cls.slab.calc = cls.calc
        cls.engine = EngineMock(cls.calc)

        # 7. Pre-relax (Briefly)
        from ase.optimize import BFGS

        print("Optimizing test system...")
        dyn = BFGS(cls.slab)
        dyn.run(fmax=0.05, steps=50)  # Moderate relaxation for validation

        write(os.path.join(cls.out_dir, "si110_dipas_relaxed.extxyz"), cls.slab)

    def test_phva_validation_si110(self):
        """Perform FHVA and PHVA comparison on Si(110) with H-passivation."""
        atoms = self.slab
        n_atoms = len(atoms)

        # STAGE 1: FHVA
        print("\n[Validation] Running FHVA...")
        analyzer_fh = VibrationalAnalyzer(atoms, self.engine, name=os.path.join(self.out_dir, "fhva_cache"))
        analyzer_fh.indices = None  # Full
        freqs_fh, modes_fh = analyzer_fh.run_analysis(overwrite=True)
        analyzer_fh.generate_qpoints_file(os.path.join(self.out_dir, "qpoints_fhva.yaml"))

        # STAGE 2: PHVA
        # Define PHVA set: Adsorbate + Top Passivation + Top 2 Si layers
        z_max = atoms.positions[:, 2].max()
        # Active atoms: Z > z_max - 5.5 A
        ph_indices = [i for i, p in enumerate(atoms.positions) if p[2] > z_max - 5.5]
        print(f"[Validation] PHVA Active Count: {len(ph_indices)}")

        analyzer_ph = VibrationalAnalyzer(atoms, self.engine, name=os.path.join(self.out_dir, "phva_cache"))
        analyzer_ph.indices = ph_indices
        freqs_ph, modes_ph = analyzer_ph.run_analysis(overwrite=True)
        analyzer_ph.generate_qpoints_file(os.path.join(self.out_dir, "qpoints_phva.yaml"))

        # STAGE 3: METRICS
        # Sorted indices (descending)
        sorted_fh = np.argsort(freqs_fh)[::-1]
        sorted_ph = np.argsort(freqs_ph)[::-1]

        print(f"Max Freq FHVA: {freqs_fh[sorted_fh[0]]:.2f} THz")
        print(f"Max Freq PHVA: {freqs_ph[sorted_ph[0]]:.2f} THz")

        # MAC Analysis for top 5 modes
        active_mask = np.zeros(3 * n_atoms, dtype=bool)
        for idx in ph_indices:
            active_mask[3 * idx : 3 * idx + 3] = True

        mac_scores = []
        for i in range(5):
            m_ph = modes_ph[:, sorted_ph[i]]
            v_ph_l = m_ph[active_mask]

            best_mac = 0.0
            for j in range(modes_fh.shape[1]):
                m_fh = modes_fh[:, j]
                v_fh_l = m_fh[active_mask]
                mac = calculate_mac(v_ph_l, v_fh_l)
                if mac > best_mac:
                    best_mac = mac
            mac_scores.append(best_mac)
            print(f"  Mode {i + 1} Best MAC: {best_mac:.4f}")

        avg_mac = np.mean(mac_scores)
        print(f"Average MAC (Top 5): {avg_mac:.4f}")

        # Validation
        self.assertGreater(avg_mac, 0.90, "PHVA MAC score is too low for Si(110) top modes.")

        # Clean up cache dirs but keep yaml/extxyz as requested
        for d in ["fhva_cache", "phva_cache"]:
            path = os.path.join(self.out_dir, d)
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
