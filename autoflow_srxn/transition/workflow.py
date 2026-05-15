import os
import numpy as np
from ase import Atoms
from ase.io import write

from ..utils.logger_utils import get_workflow_logger
from .engine import NEBSearcher, ARTSearcher
from ..vibrational.vibrational_analyzer import VibrationalAnalyzer

class TransitionStateWorkflow:
    """
    Automated workflow for transition state search.
    Connects physisorption and chemisorption states using NEB and ARTn.
    """

    def __init__(self, engine, config=None):
        """
        Args:
            engine: SimulationEngine instance.
            config: Workflow configuration dictionary.
        """
        self.engine = engine
        self.config = config or {}
        self.logger = get_workflow_logger()

    def align_states(self, initial: Atoms, final: Atoms) -> Atoms:
        """
        Aligns the initial (physisorption) state to the final (chemisorption) state 
        using deterministic index mapping stored in the final state's metadata.
        Handles PBC via Minimum Image Convention (MIC).
        """
        if "index_mapping" not in final.info:
            self.logger.warning("  [TS Workflow] 'index_mapping' not found in final state. Falling back to identity mapping.")
            return initial.copy()

        mapping = final.info["index_mapping"]
        n_slab_init = len(np.where(initial.get_tags() < 2)[0])
        
        if "protector_idx" in mapping:
             prot_idx = mapping["protector_idx"]
             new_indices = [i for i in range(n_slab_init) if i != prot_idx]
        else:
             new_indices = list(range(n_slab_init))
        
        frag_a_orig = mapping["frag_a"]
        frag_b_orig = mapping["frag_b"]
        
        for orig_idx in frag_a_orig:
            new_indices.append(n_slab_init + orig_idx)
        for orig_idx in frag_b_orig:
            new_indices.append(n_slab_init + orig_idx)
            
        if "protector_idx" in mapping:
            new_indices.append(mapping["protector_idx"])

        if len(new_indices) != len(final):
            self.logger.error(f"  [TS Workflow] Total atom count mismatch: aligned={len(new_indices)}, final={len(final)}")
            raise ValueError(f"Atom count mismatch between states.")

        # --- Robust Geometric Alignment & Mapping ---
        from ..utils.mapping import match_atoms_geometric, reorder_atoms
        
        try:
            map_indices = match_atoms_geometric(final, initial, logger=self.logger)
            aligned_initial = reorder_atoms(initial, map_indices)
        except ValueError as e:
            self.logger.error(f"  [TS Workflow] Aborting due to structure inconsistency: {e}")
            raise
        
        aligned_initial.set_cell(final.get_cell())
        aligned_initial.set_pbc(final.get_pbc())
        
        # --- Robust MIC Position Alignment ---
        from ase.geometry import find_mic
        diff = aligned_initial.get_positions() - final.get_positions()
        diff_mic, _ = find_mic(diff, final.get_cell(), final.get_pbc())
        aligned_initial.set_positions(final.get_positions() + diff_mic)
            
        return aligned_initial

    def run_ts_search(self, initial: Atoms, final: Atoms,
                      n_images: int = 7,
                      fmax_neb: float = 0.05,
                      steps_neb: int = 100,
                      interpolate: str = "idpp",
                      climbing_image: bool = False,
                      fmax_art: float = 0.05,
                      steps_art: int = 200,
                      displacement_ang: float = 0.1,
                      output_dir: str = "ts_search"):
        """Full pipeline: Alignment -> NEB -> ARTn Refinement -> Verification."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        rel_output = os.path.relpath(output_dir)
        self.logger.info(f"--- Starting Automated TS Search: {rel_output} ---")
        
        # Handle Protector Exchange
        if "isolated_byproduct" in final.info and len(final) < len(initial):
            self.logger.info("  [TS Workflow] Merging byproduct into final state for NEB.")
            byproduct = final.info["isolated_byproduct"]
            merged_final = final.copy()
            z_top = final.positions[:, 2].max()
            shift = [0, 0, z_top + 4.0] - byproduct.positions.mean(axis=0)
            byproduct.translate(shift)
            merged_final += byproduct
            final = merged_final

        try:
            aligned_initial = self.align_states(initial, final)
        except Exception as e:
            self.logger.error(f"  [TS Workflow] Alignment failed: {e}")
            return None

        write(os.path.join(output_dir, "init_aligned.vasp"), aligned_initial)
        write(os.path.join(output_dir, "final_state.vasp"), final)

        # NEB Phase
        neb_searcher = NEBSearcher(self.engine)
        images = neb_searcher.run(
            aligned_initial, final,
            n_images=n_images,
            fmax=fmax_neb,
            steps=steps_neb,
            interpolate=interpolate,
            climbing_image=climbing_image,
            trajectory=os.path.join(output_dir, "neb_path.extxyz"),
        )
        
        # Read energies from SinglePointCalculators attached by NEBSearcher.run()
        # — no new calculator calls needed (shared-calculator cache already invalid)
        energies = []
        for img in images:
            calc = getattr(img, 'calc', None)
            if calc is not None and hasattr(calc, 'results') and 'energy' in calc.results:
                energies.append(calc.results['energy'])
            else:
                energies.append(img.get_potential_energy())
        ts_idx = np.argmax(energies)
        ts_candidate = images[ts_idx].copy()
        ts_candidate.calc = self.engine.get_calculator()
        
        # ARTn Phase
        self.logger.info("  [TS Workflow] Starting ARTn refinement from NEB peak...")
        art_searcher = ARTSearcher(self.engine)
        artn_freqs = None
        try:
            ts_refined, artn_freqs, _ = art_searcher.find_saddle(
                ts_candidate, fmax=fmax_art, steps=steps_art,
                displacement_ang=displacement_ang,
            )
            write(os.path.join(output_dir, "ts_refined.vasp"), ts_refined)
        except Exception as e:
            self.logger.error(f"  [TS Workflow] ARTn refinement failed: {e}")
            ts_refined = ts_candidate

        barrier = ts_refined.get_potential_energy() - energies[0]
        self.logger.info(f"  [TS Workflow] Completed. Barrier: {barrier:.4f} eV")

        # Vibrational Verification — optional
        if self.config.get("verification", True):
            self.verify_transition_state(ts_refined, output_dir, freqs=artn_freqs)
        return ts_refined

    def verify_transition_state(self, ts_atoms: Atoms, output_dir: str, freqs=None):
        """Confirm exactly one imaginary frequency at the saddle point.

        Parameters
        ----------
        freqs : array-like or None
            Pre-computed frequency array (e.g. from ARTn vib run).  When
            provided the vibrational calculation is skipped and these values
            are used directly, avoiding a redundant Hessian evaluation.
        """
        self.logger.info("  [TS Workflow] Starting Vibrational Verification...")

        try:
            if freqs is not None:
                self.logger.info("  [VibAnalyzer] Reusing frequencies from ARTn (skipping duplicate Hessian).")
            else:
                vib_dir = os.path.join(output_dir, "vibrations")
                vib = VibrationalAnalyzer(ts_atoms, self.engine, name=vib_dir)
                freqs, _ = vib.run_analysis()

            imag_freqs = [f for f in freqs if f < -0.1]
            n_imag = len(imag_freqs)

            status = "SUCCESS" if n_imag == 1 else "FAILED"
            msg = (
                f"Exactly 1 imaginary frequency found ({imag_freqs[0]:.2f} THz)."
                if n_imag == 1
                else f"{n_imag} imaginary modes found."
            )
            self.logger.info(f"  [TS Workflow] VERIFICATION {status}: {msg}")

            with open(os.path.join(output_dir, "verification.log"), "w") as f:
                f.write(f"STATUS: {status}\nMODES: {n_imag}\n")
                if n_imag > 0:
                    f.write(f"IMAG_FREQS: {[float(f) for f in imag_freqs]}\n")
            return n_imag == 1
        except Exception as e:
            self.logger.error(f"  [TS Workflow] Vibrational verification failed: {e}")
            return False
