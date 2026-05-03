import json
import os
import sys

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.data import chemical_symbols as _CHEM_SYMS
from ase.optimize import BFGS, FIRE, LBFGS, GPMin
from ase.optimize.sciopt import SciPyFminCG

from .logger_utils import get_workflow_logger

# Absolute path to the per-pair ZBL outer cutoff database
_ZBL_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zbl_pairs.json")


class ZBLCalculator(Calculator):
    """Ziegler-Biersack-Littmark (ZBL) screened Coulomb repulsion.

    Adds a short-range repulsive correction on top of MACE or SevenNet to
    prevent unphysical atomic overlaps.  The ZBL contribution is smoothly
    switched off between *cutoff_inner* and a pair-specific *cutoff_outer*
    looked up from ``src/zbl_pairs.json`` (falls back to the global
    *cutoff_outer* parameter for element pairs not in the database).

    Per-pair outer cutoffs are set to approximately the equilibrium bond
    length (R_cov_i + R_cov_j) so that ZBL is essentially inactive at
    normal bonding distances and only activates at sub-bonding close contacts.

    Usage (standalone):
        calc = ZBLCalculator(cutoff_inner=0.5, cutoff_outer=2.5)

    Usage (combined with MACE / SevenNet):
        from ase.calculators.mixing import SumCalculator
        combined = SumCalculator([mace_calc, ZBLCalculator()])
    """

    implemented_properties = ["energy", "forces"]

    # Universal ZBL screening-function coefficients (ZBL 1985)
    _C = np.array([0.1818, 0.5099, 0.2802, 0.02817])
    _ETA = np.array([3.2000, 0.9423, 0.4029, 0.2016])
    # e² / (4πε₀) in eV·Å
    _KE = 14.3996

    @classmethod
    def _load_pair_db(cls) -> dict:
        """Load per-pair outer cutoffs from zbl_pairs.json (skips _meta block)."""
        try:
            with open(_ZBL_DB_PATH, encoding="utf-8") as fh:
                raw = json.load(fh)
            return {k: float(v) for k, v in raw.items() if not k.startswith("_")}
        except FileNotFoundError:
            return {}

    def __init__(self, cutoff_inner: float = 0.5, cutoff_outer: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        if cutoff_inner >= cutoff_outer:
            raise ValueError("cutoff_inner must be strictly less than cutoff_outer")
        self.cutoff_inner = cutoff_inner
        self.cutoff_outer = cutoff_outer  # global fallback for pairs absent from DB

        self._pair_db = self._load_pair_db()
        # NeighborList must cover all known pair cutoffs so no pair is missed
        all_r_out = list(self._pair_db.values()) + [cutoff_outer]
        self._nl_cutoff = max(all_r_out)

        self._a_cache: dict = {}

    # ------------------------------------------------------------------
    # Internal ZBL helpers
    # ------------------------------------------------------------------

    def _pair_outer_cutoff(self, Z1: int, Z2: int) -> float:
        """Return the outer switching cutoff for a given element pair.

        Looks up ``src/zbl_pairs.json`` using the alphabetically sorted
        key convention (e.g. ``"H-Si"``).  Falls back to ``self.cutoff_outer``
        for pairs not present in the database.
        """
        key = "-".join(sorted([_CHEM_SYMS[Z1], _CHEM_SYMS[Z2]]))
        return self._pair_db.get(key, self.cutoff_outer)

    def _screening_length(self, Z1: int, Z2: int) -> float:
        """Universal ZBL screening length a (Å)."""
        key = (min(Z1, Z2), max(Z1, Z2))
        if key not in self._a_cache:
            self._a_cache[key] = 0.4685 / (float(Z1) ** 0.23 + float(Z2) ** 0.23)
        return self._a_cache[key]

    def _phi_and_dphi(self, x: float):
        """Screening function Φ(x) and dΦ/dx."""
        ex = np.exp(-self._ETA * x)
        phi = float(np.dot(self._C, ex))
        dphi = float(-np.dot(self._C * self._ETA, ex))
        return phi, dphi

    def _switch(self, r: float, r_in: float, r_out: float):
        """Smooth switch S(r): 1 for r ≤ r_in, 0 for r ≥ r_out, cubic in between.

        Returns (S, dS/dr).  Accepts explicit r_in / r_out so each pair can
        use its own outer cutoff from the per-pair database.
        """
        if r_out <= r_in:
            return 0.0, 0.0
        if r <= r_in:
            return 1.0, 0.0
        if r >= r_out:
            return 0.0, 0.0
        t = (r - r_in) / (r_out - r_in)
        # smoothstep (1→0): 1 - 3t² + 2t³
        s = 1.0 - t * t * (3.0 - 2.0 * t)
        ds_dr = -6.0 * t * (1.0 - t) / (r_out - r_in)
        return s, ds_dr

    # ------------------------------------------------------------------
    # ASE Calculator interface
    # ------------------------------------------------------------------

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        from ase.neighborlist import NeighborList

        nums = atoms.get_atomic_numbers()
        pos = atoms.positions
        cell = atoms.get_cell()
        n = len(atoms)

        # Use the maximum pair cutoff as the NL radius so every potentially
        # active pair is found; per-pair filtering happens inside the loop.
        radii = [self._nl_cutoff * 0.5] * n
        nl = NeighborList(radii, skin=0.0, self_interaction=False, bothways=False)
        nl.update(atoms)

        energy = 0.0
        forces = np.zeros((n, 3))

        for i in range(n):
            indices, offsets = nl.get_neighbors(i)
            if len(indices) == 0:
                continue

            # Vectorised distance computation for neighbours of i
            r_vecs = pos[indices] + np.dot(offsets, cell) - pos[i]
            rs = np.linalg.norm(r_vecs, axis=1)

            for k in range(len(indices)):
                j = indices[k]
                r = rs[k]

                if r < 1e-10:
                    continue

                Z1 = int(nums[i])
                Z2 = int(nums[j])
                r_out = self._pair_outer_cutoff(Z1, Z2)

                if r >= r_out:
                    continue

                a = self._screening_length(Z1, Z2)
                x = r / a

                phi, dphi = self._phi_and_dphi(x)
                s, ds_dr = self._switch(r, self.cutoff_inner, r_out)

                if s == 0.0 and ds_dr == 0.0:
                    continue

                # ZBL pair energy and its radial derivative
                V = self._KE * Z1 * Z2 / r * phi
                dV_dr = self._KE * Z1 * Z2 * (-phi / r**2 + dphi / (r * a))

                energy += V * s

                # dE/dr = dV/dr·S + V·dS/dr
                dE_dr = dV_dr * s + V * ds_dr
                r_hat = r_vecs[k] / r

                # Force on i (repulsive → away from j)
                forces[i] += dE_dr * r_hat
                # Newton's 3rd law
                forces[j] -= dE_dr * r_hat

        self.results["energy"] = energy
        self.results["forces"] = forces


# ---------------------------------------------------------------------------
# Explosion Monitoring
# ---------------------------------------------------------------------------

class ExplosionMonitor:
    """Monitors normalized energy during dynamics to detect system instability."""
    def __init__(self, atoms, threshold_ev_per_atom=10.0, logger=None):
        self.atoms = atoms
        self.n_atoms = len(atoms)
        self.threshold_per_atom = threshold_ev_per_atom
        self.logger = logger
        try:
            self.initial_e = atoms.get_potential_energy()
        except:
            self.initial_e = None
            
    def __call__(self):
        try:
            e = self.atoms.get_potential_energy()
            if self.initial_e is None:
                self.initial_e = e
                return
            
            # 1. Normalize by atom count
            e_per_atom = e / self.n_atoms
            delta_per_atom = (e - self.initial_e) / self.n_atoms
            
            # 2. Check criteria
            # - Energy becomes positive (> 1 eV/atom is usually a broken MLIP state)
            # - Energy jumps by more than 10 eV/atom (unphysical explosion)
            # - Total energy changes by more than an order of magnitude relative to initial
            is_unstable = False
            reason = ""
            
            if e_per_atom > 0.0:
                is_unstable = True
                reason = f"Positive energy ({e_per_atom:.2f} eV/atom)"
            elif abs(delta_per_atom) > self.threshold_per_atom:
                is_unstable = True
                reason = f"Massive energy jump ({delta_per_atom:.2f} eV/atom)"
            elif self.initial_e != 0 and abs(e / self.initial_e) > 10.0:
                is_unstable = True
                reason = f"Order-of-magnitude shift (E_curr/E_init = {e/self.initial_e:.1f})"

            if is_unstable:
                msg = f"  [ExplosionDetector] {reason}. Terminating calculation."
                if self.logger: self.logger.error(msg)
                raise RuntimeError(msg)
                
        except (RuntimeError, ValueError) as ex:
            raise ex
        except Exception:
            pass 

# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------


class SimulationEngine:
    """ASE-based simulation engine supporting MACE, SevenNet, and EMT backends.

    ZBL repulsion can be layered on top of any backend by setting
    ``engine.potential.zbl.enabled: true`` in the configuration.  The ZBL
    contribution is active only at very short range (below *cutoff_outer*,
    default 2.5 Å) so that it does not interfere with the MLIP at normal
    bonding distances.

    All settings are read from config['engine']['potential'].
    Relaxation and MD parameters are passed as arguments to each method,
    allowing per-stage override of the defaults defined in config['engine']['relaxation']
    and config['engine']['md'].

    Supported potential.backend values
    ------------------------------------
    * ``"mace"``      – MACE-MP foundation model (mace_mp)
    * ``"sevennet"``  – SevenNet calculator
    * ``"emt"``       – ASE built-in EMT (fast, light-element only)

    ZBL configuration example (config.yaml)
    ----------------------------------------
    engine:
      potential:
        backend: "mace"          # or "sevennet"
        device:  "cpu"
        dtype:   "float64"
        model:   null
        zbl:
          enabled:       true
          cutoff_inner:  1.0     # Å  – ZBL fully active below this
          cutoff_outer:  2.5     # Å  – ZBL switched off above this
    """

    def __init__(self, config=None):
        self.all_config = config or {}
        engine_cfg = self.all_config.get("engine", {})
        pot_cfg = engine_cfg.get("potential", {})

        self.backend = pot_cfg.get("backend", "mace").lower()
        self.device = pot_cfg.get("device", "cpu")
        self.dtype = pot_cfg.get("dtype", "float64")
        self.model = pot_cfg.get("model", None)
        self.modal = pot_cfg.get("modal", None)
        self.d3 = pot_cfg.get("d3", False)
        self.enable_cueq = pot_cfg.get("enable_cueq", False)
        self.enable_flash = pot_cfg.get("enable_flash", False)

        # ZBL settings
        zbl_cfg = pot_cfg.get("zbl", {})
        self.zbl_enabled = bool(zbl_cfg.get("enabled", False))
        self.zbl_cutoff_inner = float(zbl_cfg.get("cutoff_inner", 0.5))
        self.zbl_cutoff_outer = float(zbl_cfg.get("cutoff_outer", 2.5))

        self._calculator = None

    # ------------------------------------------------------------------
    # Calculator construction
    # ------------------------------------------------------------------

    def _build_base_calculator(self, logger):
        """Construct the bare MACE / SevenNet / EMT calculator."""
        if self.backend == "emt":
            logger.info("  [Engine] Loaded EMT calculator.")
            return EMT()

        if self.backend == "mace":
            return self._build_mace(logger)

        if self.backend == "sevennet":
            return self._build_sevennet(logger)

        logger.warning(f"  [Engine] Unknown backend '{self.backend}'. Falling back to EMT.")
        return EMT()

    def _build_mace(self, logger):
        try:
            from mace.calculators import mace_mp

            model = self.model or "medium"
            calc = mace_mp(
                model=model,
                device=self.device,
                default_dtype=self.dtype,
                dispersion=self.d3,
            )
            label = f"model={model}, dtype={self.dtype}"
            if self.d3:
                label += ", D3=True"
            logger.info(f"  [Engine] Loaded MACE-MP calculator ({label}).")
            return calc
        except Exception as e:
            logger.warning(f"  [Engine] MACE loading failed ({type(e).__name__}: {e}). Falling back to EMT.")
            return EMT()

    def _build_sevennet(self, logger):
        try:
            model = self.model or "7net-0"

            # Resolve local checkpoint path
            if isinstance(model, str) and model.endswith(".pth") and os.path.isfile(model):
                model = os.path.abspath(model)
                logger.info(f"  [Engine] Detected local SevenNet checkpoint: {os.path.relpath(model)}")

            # Build kwargs; only pass optional flags when explicitly set to
            # preserve compatibility across SevenNet versions.
            sn_kwargs: dict = {"model": model, "device": self.device}
            if self.modal is not None:
                sn_kwargs["modal"] = self.modal
            if self.enable_cueq:
                sn_kwargs["enable_cueq"] = True
            if self.enable_flash:
                sn_kwargs["enable_flash"] = True

            if self.d3:
                from sevenn.calculator import SevenNetD3Calculator

                calc = SevenNetD3Calculator(**sn_kwargs)
                logger.info(f"  [Engine] Loaded SevenNet+D3 (model={model}, modal={self.modal}).")
            else:
                from sevenn.calculator import SevenNetCalculator

                calc = SevenNetCalculator(**sn_kwargs)
                logger.info(f"  [Engine] Loaded SevenNet (model={model}, modal={self.modal}).")
            return calc

        except Exception as e:
            logger.warning(f"  [Engine] SevenNet loading failed ({type(e).__name__}: {e}). Falling back to EMT.")
            return EMT()

    def get_calculator(self):
        if self._calculator is not None:
            return self._calculator

        logger = get_workflow_logger()
        base_calc = self._build_base_calculator(logger)

        if self.zbl_enabled:
            from ase.calculators.mixing import SumCalculator

            zbl_calc = ZBLCalculator(
                cutoff_inner=self.zbl_cutoff_inner,
                cutoff_outer=self.zbl_cutoff_outer,
            )
            self._calculator = SumCalculator([base_calc, zbl_calc])
            logger.info(
                f"  [Engine] Added ZBL repulsion "
                f"(r_inner={self.zbl_cutoff_inner} Å, r_outer={self.zbl_cutoff_outer} Å)."
            )
        else:
            self._calculator = base_calc

        return self._calculator

    # ------------------------------------------------------------------
    # Internal constraint helper
    # ------------------------------------------------------------------

    def _apply_constraints(self, atoms, frozen_z_ang, fix_atom_indices, config_section):
        """Merge FixAtoms constraints into *atoms* from explicit args and/or config defaults.

        Priority: explicit kwarg > engine.<config_section> config > no-op.
        New constraints are *added* to (not replacing) any already present on the object.
        """
        from ase.constraints import FixAtoms

        section_cfg = self.all_config.get("engine", {}).get(config_section, {})
        z_ang = frozen_z_ang if frozen_z_ang is not None else section_cfg.get("frozen_z_ang")
        idx = fix_atom_indices if fix_atom_indices is not None else section_cfg.get("fix_atom_indices")

        if z_ang is None and not idx:
            return

        to_fix = set()
        for c in atoms.constraints:
            if isinstance(c, FixAtoms):
                to_fix.update(c.index.tolist())

        if z_ang is not None:
            z_min = atoms.positions[:, 2].min()
            to_fix.update(np.where(atoms.positions[:, 2] < z_min + z_ang)[0].tolist())

        if idx:
            to_fix.update(idx)

        other = [c for c in atoms.constraints if not isinstance(c, FixAtoms)]
        atoms.set_constraint(other + ([FixAtoms(sorted(to_fix))] if to_fix else []))

    # ------------------------------------------------------------------
    # Public simulation methods
    # ------------------------------------------------------------------

    def relax(
        self,
        atoms,
        fmax=None,
        steps=None,
        optimizer=None,
        verbose=True,
        frozen_z_ang=None,
        fix_atom_indices=None,
        **kwargs,
    ):
        """Structural relaxation using BFGS, FIRE, or two-stage CG_FIRE.

        Arguments (fmax, steps, optimizer) default to ``config['engine']['relaxation']``
        when None. ``frozen_z_ang`` and ``fix_atom_indices`` are merged with any
        FixAtoms constraints already on *atoms*.
        """
        relax_cfg = self.all_config.get("engine", {}).get("relaxation", {})
        fmax = fmax if fmax is not None else relax_cfg.get("fmax", 0.05)
        steps = steps if steps is not None else relax_cfg.get("steps", 200)
        optimizer = optimizer if optimizer is not None else relax_cfg.get("optimizer", "BFGS")

        self._apply_constraints(atoms, frozen_z_ang, fix_atom_indices, "relaxation")
        atoms.calc = self.get_calculator()

        trajectory = kwargs.get("trajectory")
        logger = get_workflow_logger()
        monitor = ExplosionMonitor(atoms, logger=logger)

        try:
            if optimizer.upper() == "CG_FIRE":
                fmax_cg = max(fmax * 10, 0.05)
                if verbose:
                    print(f"  [Relax] Stage 1: SciPyFminCG (fmax={fmax_cg})")
                dyn_cg = SciPyFminCG(atoms, logfile=sys.stdout if verbose else None)
                dyn_cg.attach(monitor)
                dyn_cg.run(fmax=fmax_cg, steps=steps // 2)
                if verbose:
                    print(f"  [Relax] Stage 2: FIRE (fmax={fmax})")
                dyn_fire = FIRE(atoms, logfile=sys.stdout if verbose else None, trajectory=trajectory)
                dyn_fire.attach(monitor)
                dyn_fire.run(fmax=fmax, steps=steps)
            elif optimizer.upper() == "GPMIN":
                dyn = GPMin(atoms, logfile=sys.stdout if verbose else None, trajectory=trajectory)
                dyn.attach(monitor)
                dyn.run(fmax=fmax, steps=steps)
            else:
                opt_map = {"BFGS": BFGS, "FIRE": FIRE, "LBFGS": LBFGS}
                opt_class = opt_map.get(optimizer.upper(), BFGS)
                dyn = opt_class(atoms, logfile=sys.stdout if verbose else None, trajectory=trajectory)
                dyn.attach(monitor)
                dyn.run(fmax=fmax, steps=steps)
        except RuntimeError as e:
            if "ExplosionDetector" in str(e):
                if verbose: print(f"  [Relax] Halted: {e}")
            else:
                raise e

        return atoms.get_potential_energy()

    def run_md(
        self,
        atoms,
        temp_K=None,
        md_steps=None,
        damping=None,
        timestep_fs=None,
        random_seed=12345,
        frozen_z_ang=None,
        fix_atom_indices=None,
    ):
        """NVT Langevin MD. ASE FixAtoms constraints are honoured natively.

        Arguments (temp_K, md_steps, damping, timestep_fs) default to
        ``config['engine']['md']`` when None.
        """
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        md_cfg = self.all_config.get("engine", {}).get("md", {})
        temp_K = temp_K if temp_K is not None else md_cfg.get("temperature_K", 300.0)
        md_steps = md_steps if md_steps is not None else md_cfg.get("md_steps", 1000)
        damping = damping if damping is not None else md_cfg.get("damping", 100.0)
        timestep_fs = timestep_fs if timestep_fs is not None else md_cfg.get("timestep_fs", 1.0)

        self._apply_constraints(atoms, frozen_z_ang, fix_atom_indices, "md")

        calc = self.get_calculator()
        atoms.calc = calc
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_K)
        dyn = Langevin(
            atoms,
            timestep_fs * units.fs,
            temperature_K=temp_K,
            friction=1.0 / (damping * units.fs),
        )
        
        logger = get_workflow_logger()
        monitor = ExplosionMonitor(atoms, logger=logger)
        dyn.attach(monitor)
        
        try:
            dyn.run(md_steps)
        except RuntimeError as e:
            if "ExplosionDetector" in str(e):
                print(f"  [MD] Halted: {e}")
            else:
                raise e

    def get_forces(self, atoms):
        calc = self.get_calculator()
        atoms.calc = calc
        return atoms.get_forces()
