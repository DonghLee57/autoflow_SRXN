import sys
import os
import numpy as np
from ase.optimize import BFGS, FIRE
from ase.optimize.sciopt import SciPyFminCG
from ase.calculators.emt import EMT
from logger_utils import get_workflow_logger


class SimulationEngine:
    """
    ASE-based simulation engine supporting MACE, SevenNet, and EMT backends.

    All settings are read from config['engine']['potential'].
    Relaxation and MD parameters are passed as arguments to each method,
    allowing per-stage override of the defaults defined in config['engine']['relaxation']
    and config['engine']['md'].
    """

    def __init__(self, config=None):
        self.all_config = config or {}
        engine_cfg = self.all_config.get('engine', {})
        pot_cfg = engine_cfg.get('potential', {})

        self.backend = pot_cfg.get('backend', 'mace').lower()
        self.device  = pot_cfg.get('device', 'cpu')
        self.dtype   = pot_cfg.get('dtype', 'float64')
        self.model   = pot_cfg.get('model', None)
        self.modal   = pot_cfg.get('modal', None)
        self.d3      = pot_cfg.get('d3', False)
        self.enable_cueq  = pot_cfg.get('enable_cueq', False)
        self.enable_flash = pot_cfg.get('enable_flash', False)

        self._calculator = None

    def get_calculator(self):
        if self._calculator is not None:
            return self._calculator

        logger = get_workflow_logger()

        if self.backend == 'emt':
            self._calculator = EMT()
            logger.info("  [Engine] Loaded EMT calculator.")

        elif self.backend == 'mace':
            try:
                from mace.calculators import mace_mp
                model = self.model or 'medium'
                self._calculator = mace_mp(
                    model=model, device=self.device, default_dtype=self.dtype
                )
                logger.info(f"  [Engine] Loaded MACE-MP calculator (model={model}, dtype={self.dtype}).")
            except ImportError:
                logger.warning("  [Engine] MACE not installed. Falling back to EMT.")
                self._calculator = EMT()

        elif self.backend == 'sevennet':
            try:
                import os
                model = self.model or '7net-0'
                
                # Check if model is a local checkpoint file
                is_checkpoint = isinstance(model, str) and model.endswith('.pth')
                if is_checkpoint and os.path.isfile(model):
                    model = os.path.abspath(model)
                    logger.info(f"  [Engine] Detected local SevenNet checkpoint: {os.path.relpath(model)}")

                kwargs = {
                    'model': model,
                    'device': self.device,
                    'modal': self.modal,
                    'enable_cueq': self.enable_cueq,
                    'enable_flash': self.enable_flash
                }
                
                if self.d3:
                    from sevenn.calculator import SevenNetD3Calculator
                    self._calculator = SevenNetD3Calculator(**kwargs)
                    logger.info(f"  [Engine] Loaded SevenNet+D3 (model={model}, modal={self.modal}).")
                else:
                    from sevenn.calculator import SevenNetCalculator
                    self._calculator = SevenNetCalculator(**kwargs)
                    logger.info(f"  [Engine] Loaded SevenNet (model={model}, modal={self.modal}).")
            except ImportError:
                logger.warning("  [Engine] SevenNet (sevenn) not installed. Falling back to EMT.")
                self._calculator = EMT()

        else:
            logger.warning(f"  [Engine] Unknown backend '{self.backend}'. Falling back to EMT.")
            self._calculator = EMT()

        return self._calculator

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _apply_constraints(self, atoms, frozen_z_ang, fix_atom_indices, config_section):
        """Merge FixAtoms constraints into *atoms* from explicit args and/or config defaults.

        Priority: explicit kwarg > engine.<config_section> config > no-op.
        New constraints are *added* to (not replacing) any already present on the object.
        """
        from ase.constraints import FixAtoms

        section_cfg = self.all_config.get('engine', {}).get(config_section, {})
        z_ang = frozen_z_ang    if frozen_z_ang    is not None else section_cfg.get('frozen_z_ang')
        idx   = fix_atom_indices if fix_atom_indices is not None else section_cfg.get('fix_atom_indices')

        if z_ang is None and not idx:
            return  # Nothing to add

        # Collect indices from existing FixAtoms + new specs
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

    def relax(self, atoms, fmax=None, steps=None, optimizer=None, verbose=True,
              frozen_z_ang=None, fix_atom_indices=None, **kwargs):
        """Structural relaxation using BFGS, FIRE, or two-stage CG_FIRE.

        Arguments (fmax, steps, optimizer) default to ``config['engine']['relaxation']``
        when None. ``frozen_z_ang`` and ``fix_atom_indices`` are merged with any
        FixAtoms constraints already on *atoms*.
        """
        # Fetch defaults from config
        relax_cfg = self.all_config.get('engine', {}).get('relaxation', {})
        fmax      = fmax      if fmax      is not None else relax_cfg.get('fmax', 0.05)
        steps     = steps     if steps     is not None else relax_cfg.get('steps', 200)
        optimizer = optimizer if optimizer is not None else relax_cfg.get('optimizer', 'BFGS')

        self._apply_constraints(atoms, frozen_z_ang, fix_atom_indices, 'relaxation')
        calc = self.get_calculator()
        atoms.calc = calc

        trajectory = kwargs.get('trajectory')

        if optimizer.upper() == 'CG_FIRE':
            fmax_cg = max(fmax * 10, 0.05)
            if verbose:
                print(f"  [Relax] Stage 1: SciPyFminCG (fmax={fmax_cg})")
            dyn_cg = SciPyFminCG(atoms, logfile=sys.stdout if verbose else None)
            dyn_cg.run(fmax=fmax_cg, steps=steps // 2)
            if verbose:
                print(f"  [Relax] Stage 2: FIRE (fmax={fmax})")
            dyn_fire = FIRE(atoms, logfile=sys.stdout if verbose else None, trajectory=trajectory)
            dyn_fire.run(fmax=fmax, steps=steps)
        else:
            opt_class = BFGS if optimizer.upper() == 'BFGS' else FIRE
            dyn = opt_class(atoms, logfile=sys.stdout if verbose else None, trajectory=trajectory)
            dyn.run(fmax=fmax, steps=steps)

        return atoms.get_potential_energy()

    def run_md(self, atoms, temp_K=None, md_steps=None, damping=None, timestep_fs=None,
               random_seed=12345, frozen_z_ang=None, fix_atom_indices=None):
        """NVT Langevin MD. ASE FixAtoms constraints are honoured natively.

        Arguments (temp_K, md_steps, damping, timestep_fs) default to
        ``config['engine']['md']`` when None.
        """
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase import units

        # Fetch defaults from config
        md_cfg = self.all_config.get('engine', {}).get('md', {})
        temp_K      = temp_K      if temp_K      is not None else md_cfg.get('temperature_K', 300.0)
        md_steps    = md_steps    if md_steps    is not None else md_cfg.get('md_steps', 1000)
        damping     = damping     if damping     is not None else md_cfg.get('damping', 100.0)
        
        # Get timestep (fs)
        if timestep_fs is None:
            timestep_fs = md_cfg.get('timestep_fs', 1.0)

        self._apply_constraints(atoms, frozen_z_ang, fix_atom_indices, 'md')

        calc = self.get_calculator()
        atoms.calc = calc
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_K)
        dyn = Langevin(
            atoms, timestep_fs * units.fs,
            temperature_K=temp_K,
            friction=1.0 / (damping * units.fs),
        )
        dyn.run(md_steps)

    def get_forces(self, atoms):
        calc = self.get_calculator()
        atoms.calc = calc
        return atoms.get_forces()
