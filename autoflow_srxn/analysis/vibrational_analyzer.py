import os
import shutil

import numpy as np
from ase.io import write

# NOTE: ase.optimize.dimer is intentionally excluded due to environment import issues.
# TSSearcher uses a self-contained Hessian-based gradient flipping strategy instead.
from ase.optimize import FIRE
from ase.vibrations import Vibrations

from ..utils.knowledge_engine import chem_kb
from ..utils.logger_utils import get_workflow_logger
from ..simulation.qpoint_handler import QPointParser

# ---------------------------------------------------------------------------
# Unit conversion: sqrt(eV / (amu * Å²)) → THz
#   ω(THz) = sqrt(λ[eV/amu/Å²]) × _EV_PER_AMU_ANG2_TO_THZ
# Derivation:
#   ω² = λ × eV_to_J / (amu_to_kg × Å²_to_m²)
#   ω  = sqrt(λ) × sqrt(1.60218e-19 / (1.66054e-27 × 1e-20))  [rad s⁻¹]
#   f  = ω / (2π × 10¹²)  [THz]
# Numerically: ~15.633 THz per sqrt(eV/amu/Å²)
# ---------------------------------------------------------------------------
_EV_PER_AMU_ANG2_TO_THZ: float = float(
    np.sqrt(1.60218e-19 / (1.66054e-27 * 1e-20)) / (2.0 * np.pi * 1e12)
)


def _resolve_phva_center(atoms, ads_idx, center_cfg):
    """Resolve which adsorbate atom index/indices to use as the focal point(s)
    for the ``phva.radius_ang`` sphere.

    Parameters
    ----------
    atoms       : ASE Atoms
    ads_idx     : list/set of int — adsorbate atom indices
    center_cfg  : str | int | None
        None / ``"adsorbate"``  → all adsorbate atoms (default, backward-compat)
        element symbol str      → adsorbate atoms whose element matches, e.g. ``"Si"``
        int                     → one specific atom by its absolute index in *atoms*
        ``"com"``               → return ``None`` as a sentinel; caller uses COM-based
                                  distance instead of a neighbour-list

    Returns
    -------
    list[int] | None
        Atom indices to use as focal centres, or ``None`` signalling COM mode.
    """
    ads_list = list(ads_idx)

    if center_cfg is None or str(center_cfg).lower() in ("adsorbate", "all"):
        return ads_list

    if str(center_cfg).lower() == "com":
        return None  # handled separately in caller

    # Integer index (passed directly or as a string)
    if isinstance(center_cfg, int):
        return [center_cfg]
    try:
        return [int(center_cfg)]
    except (ValueError, TypeError):
        pass

    # Element symbol: filter adsorbate atoms to the matching element
    symbols = atoms.get_chemical_symbols()
    matched = [i for i in ads_list if symbols[i] == str(center_cfg)]
    if matched:
        return matched

    # Element not found — fall back to all adsorbate atoms (warn in caller)
    return None


class VibrationalAnalyzer:
    """Handles vibrational frequency analysis using ASE Vibrations (supporting PHVA)
    or Phonopy.
    """

    def __init__(self, atoms, engine, indices=None, displacement=0.01, name="vib_analysis"):
        """Args:
        atoms: ASE Atoms object.
        engine: SimulationEngine (ASE-compatible).
        indices: List of atomic indices to include in the Partial Hessian.
                 If None, it will be automatically determined from config.
        displacement: Finite difference displacement (A).
        name: Name for the vibration log directory.
        """
        self.atoms = atoms
        self.engine = engine
        self._indices = indices
        self.displacement = displacement
        self.name = name
        self.logger = get_workflow_logger()

        # Attach calculator
        self.atoms.calc = self.engine.get_calculator()
        self._freqs_thz = None
        self._eigs = None

    @property
    def indices(self):
        """Returns the active atom indices for the Hessian calculation.

        Resolution logic
        ----------------
        1. Explicit override: if ``self._indices`` was set at construction, use it.
        2. Config-driven:
           ``analysis.vibrational.phva.enabled: false``  →  Full Hessian (all atoms,
               return ``None`` so ASE Vibrations uses every atom).
           ``analysis.vibrational.phva.enabled: true``   →  Partial Hessian:
               a. ``phva.frozen_z_ang``  – exclude atoms whose z < z_min + threshold.
               b. ``phva.radius_ang``    – if set, further restrict to adsorbate atoms
                  (identified via tag/protector detection) plus all neighbours within
                  the given radius.  ``phva.center`` controls the focal atom(s).
        """
        if self._indices is not None:
            return self._indices

        config = self.engine.all_config
        vib_cfg = config.get("analysis", {}).get("vibrational", {})
        phva_cfg = vib_cfg.get("phva", {})

        # phva.enabled = false (or block absent) → Full Hessian
        if not phva_cfg.get("enabled", False):
            return None

        # ── Resolve PHVA parameters ──────────────────────────────────────────
        frozen_z = phva_cfg.get("frozen_z_ang")
        # Legacy fallback: surface_prep.equilibration.frozen_z_ang
        if frozen_z is None:
            frozen_z = config.get("surface_prep", {}).get("equilibration", {}).get("frozen_z_ang")

        radius     = phva_cfg.get("radius_ang")
        center_cfg = phva_cfg.get("center", None)

        indices_set = set(range(len(self.atoms)))

        # 1. Height-based frozen-atom exclusion
        if frozen_z is not None:
            z_min = self.atoms.positions[:, 2].min()
            mask  = self.atoms.positions[:, 2] >= z_min + float(frozen_z)
            indices_set &= set(np.where(mask)[0])
            self.logger.info(
                f"  [VibAnalyzer] PHVA frozen_z_ang={frozen_z} Ang  "
                f"({len(indices_set)} atoms above threshold)"
            )

        # 2. Radius-based restriction around adsorbate
        if radius is not None:
            from ..surface.surface_utils import identify_protectors
            _, ads_idx_arr = identify_protectors(self.atoms, config)
            ads_idx = set(ads_idx_arr.tolist())

            if len(ads_idx) > 0:
                center_atom_indices = _resolve_phva_center(self.atoms, ads_idx, center_cfg)

                if center_atom_indices is None and str(center_cfg).lower() == "com":
                    ads_pos = self.atoms.positions[list(ads_idx)]
                    com     = ads_pos.mean(axis=0)
                    dists   = np.linalg.norm(self.atoms.positions - com, axis=1)
                    neighbor_set = set(int(i) for i in np.where(dists < radius)[0])
                    self.logger.info(
                        f"  [VibAnalyzer] PHVA center=com  "
                        f"({len(neighbor_set)} atoms within {radius} Ang of COM)"
                    )
                elif center_atom_indices is None:
                    self.logger.warning(
                        f"  [VibAnalyzer] PHVA center='{center_cfg}' not found in adsorbate — "
                        f"falling back to all adsorbate atoms."
                    )
                    center_atom_indices = list(ads_idx)
                    from ase.neighborlist import neighbor_list
                    i_list, j_list = neighbor_list("ij", self.atoms, radius)
                    neighbor_set = set()
                    for a_idx in center_atom_indices:
                        neighbor_set.update(int(j) for j in j_list[i_list == a_idx])
                else:
                    from ase.neighborlist import neighbor_list
                    i_list, j_list = neighbor_list("ij", self.atoms, radius)
                    neighbor_set = set()
                    for a_idx in center_atom_indices:
                        neighbor_set.update(int(j) for j in j_list[i_list == a_idx])
                    symbols      = self.atoms.get_chemical_symbols()
                    center_desc  = (
                        f"atom {center_atom_indices[0]} ({symbols[center_atom_indices[0]]})"
                        if len(center_atom_indices) == 1
                        else f"{len(center_atom_indices)} '{center_cfg}' atoms"
                    )
                    self.logger.info(
                        f"  [VibAnalyzer] PHVA radius={radius} Ang, center={center_desc}  "
                        f"({len(neighbor_set)} atoms in neighborhood)"
                    )

                phva_set = set(ads_idx) | neighbor_set
                indices_set &= phva_set
            else:
                self.logger.warning(
                    "  [VibAnalyzer] phva.radius_ang set but no adsorbate found — "
                    "using height-filtered selection only."
                )

        return sorted(list(indices_set))

    @indices.setter
    def indices(self, value):
        self._indices = value

    @property
    def min_freq(self):
        """Returns the minimum frequency in THz."""
        if self._freqs_thz is None:
            return None
        return float(np.min(self._freqs_thz))

    @property
    def modes(self):
        """Returns the list of modes (freq and eigenvector) for refinement."""
        if self._freqs_thz is None:
            return []

        modes_list = []
        n_atoms = len(self.atoms)
        mass_sqrt = np.sqrt(self.atoms.get_masses())

        for i, freq in enumerate(self._freqs_thz):
            u_vec = self._eigs[:, i].reshape(n_atoms, 3)
            # Standardizing to mass-weighted eigenvector (e = u * sqrt(m))
            e_vec = u_vec * mass_sqrt[:, np.newaxis]
            norm = np.linalg.norm(e_vec)
            if norm > 1e-10:
                e_vec /= norm

            modes_list.append({"frequency": float(freq), "eigenvector": e_vec.tolist()})
        return modes_list

    def run_analysis(self, overwrite=False):
        """Performs (Partial) Hessian Vibrational Analysis.

        By default, uses ASE ``Vibrations`` with its built-in JSON cache.
        When ``analysis.vibrational.cache_format: "lammps_dump"`` is set in the
        config, a custom finite-difference loop is executed instead and every
        displaced configuration's forces are written to a LAMMPS custom dump file
        (columns: ``id type x y z fx fy fz``) inside the *name* directory.

        Cache lifecycle
        ---------------
        ``overwrite`` (parameter or ``analysis.vibrational.overwrite`` in config):
            ``True``  → delete all existing cache files first, then recompute.
            ``False`` → reuse cached files that already exist (fast restart).

        ``analysis.vibrational.save_cache``:
            ``false`` (default) → delete the cache directory after the run,
                                   matching the original behaviour.
            ``true``  → preserve the cache directory for post-processing
                        (e.g. phonopy comparison via the LAMMPS dump files).

        Returns
        -------
        freqs_thz : ndarray
            Frequencies in THz (negative values denote imaginary modes).
        eigs : ndarray, shape (3*N_total, n_modes)
            Mode eigenvectors padded to the full system size.
        """
        vib_cfg = self.engine.all_config.get("analysis", {}).get("vibrational", {})
        cache_format = vib_cfg.get("cache_format", None)
        save_cache   = vib_cfg.get("save_cache", False)

        n_active = len(self.indices) if self.indices else len(self.atoms)
        self.logger.info(
            f"  [VibAnalyzer] Starting PHVA/FHVA  "
            f"(active atoms: {n_active}, cache: {cache_format or 'ase_json'}, "
            f"save_cache: {save_cache})"
        )

        if cache_format == "lammps_dump":
            freqs_thz, eigs = self._run_lammps_dump_analysis(overwrite, save_cache)
        else:
            freqs_thz, eigs = self._run_ase_json_analysis(overwrite, save_cache)

        n_imag = sum(1 for f in freqs_thz if f < -0.01)
        self.logger.info(
            f"  [VibAnalyzer] Analysis complete.  "
            f"Total modes: {len(freqs_thz)}, Imaginary: {n_imag}"
        )

        self._freqs_thz = np.array(freqs_thz)
        self._eigs = eigs

        parent_dir = os.path.dirname(self.name) if os.path.dirname(self.name) else "."
        self.generate_qpoints_file(os.path.join(parent_dir, "qpoints.yaml"))

        return self._freqs_thz, self._eigs

    # ------------------------------------------------------------------
    # Private: ASE JSON backend (original behaviour)
    # ------------------------------------------------------------------

    def _run_ase_json_analysis(self, overwrite: bool, save_cache: bool):
        """Run analysis using ASE Vibrations (JSON cache)."""
        if overwrite and os.path.exists(self.name):
            self._robust_rmtree(self.name)

        vib = Vibrations(self.atoms, indices=self.indices, name=self.name, delta=self.displacement)
        vib.run()

        freqs_raw = vib.get_frequencies()
        freqs_thz = []
        for f in freqs_raw:
            cf = complex(f)
            if abs(cf.imag) > abs(cf.real):
                freqs_thz.append(-abs(cf.imag) / 33.3564)
            else:
                freqs_thz.append(cf.real / 33.3564)

        vib_data = vib.get_vibrations()
        modes = vib_data.get_modes()          # (n_modes, n_active, 3)

        N_total  = len(self.atoms)
        n_modes  = modes.shape[0]
        indices  = self.indices if self.indices is not None else list(range(N_total))
        eigs     = np.zeros((3 * N_total, n_modes))
        for i in range(n_modes):
            mode_3d          = np.zeros((N_total, 3))
            mode_3d[indices] = modes[i]
            eigs[:, i]       = mode_3d.ravel()

        if not save_cache and os.path.exists(self.name):
            self._robust_rmtree(self.name)

        return freqs_thz, eigs

    # ------------------------------------------------------------------
    # Private: LAMMPS dump backend
    # ------------------------------------------------------------------

    @staticmethod
    def _specorder(atoms) -> list:
        """Alphabetically sorted unique element list — the LAMMPS specorder.

        Type index in the dump file = position in this list + 1.
        Consistent with ``ase.io.lammpsdata.write_lammps_data`` convention.
        """
        return sorted(set(atoms.get_chemical_symbols()))

    def _write_lammps_dump(
        self,
        filepath: str,
        atoms,
        forces: np.ndarray,
        timestep: int = 0,
        specorder: list = None,
    ) -> None:
        """Write one LAMMPS custom dump snapshot: id type x y z fx fy fz.

        Cell → LAMMPS box conversion is delegated to ``ase.io.lammpsdata.Prism``
        (handles both orthogonal and triclinic cells).  Positions are transformed
        with ``Prism.vector_to_lammps()`` to match the LAMMPS coordinate frame.
        Forces are written in ASE native units (eV/Å); positions in Å.

        Parameters
        ----------
        specorder : list, optional
            Element ordering that defines the type integers (type 1 = specorder[0],
            type 2 = specorder[1], …).  Falls back to alphabetical if omitted.
        """
        from ase.io.lammpsdata import Prism
        from ase.calculators.singlepoint import SinglePointCalculator

        if specorder is None:
            specorder = self._specorder(atoms)
        type_of  = {elem: i + 1 for i, elem in enumerate(specorder)}
        symbols  = atoms.get_chemical_symbols()

        # Attach forces so the Atoms object carries them (good ASE practice)
        atoms_wr = atoms.copy()
        atoms_wr.calc = SinglePointCalculator(atoms_wr, forces=forces, energy=0.0)

        prism    = Prism(atoms_wr.cell, atoms_wr.pbc)
        lp       = prism.get_lammps_prism()   # [xhi, yhi, zhi, xy, xz, yz]
        xhi, yhi, zhi, xy, xz, yz = lp

        # Transform all positions to the LAMMPS frame
        pos_lammps = np.array(
            [prism.vector_to_lammps(p) for p in atoms_wr.get_positions()]
        )

        with open(filepath, "w") as fout:
            fout.write(f"ITEM: TIMESTEP\n{timestep}\n")
            fout.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms_wr)}\n")
            if prism.is_skewed():
                fout.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
                fout.write(f"0.0 {xhi:.10f} {xy:.10f}\n")
                fout.write(f"0.0 {yhi:.10f} {xz:.10f}\n")
                fout.write(f"0.0 {zhi:.10f} {yz:.10f}\n")
            else:
                fout.write("ITEM: BOX BOUNDS pp pp pp\n")
                fout.write(f"0.0 {xhi:.10f}\n")
                fout.write(f"0.0 {yhi:.10f}\n")
                fout.write(f"0.0 {zhi:.10f}\n")
            fout.write("ITEM: ATOMS id type x y z fx fy fz\n")
            for i in range(len(atoms_wr)):
                px, py, pz = pos_lammps[i]
                fx, fy, fz = forces[i]
                fout.write(
                    f"{i + 1} {type_of[symbols[i]]}"
                    f" {px:.10f} {py:.10f} {pz:.10f}"
                    f" {fx:.10f} {fy:.10f} {fz:.10f}\n"
                )

    def _read_lammps_dump_forces(
        self, filepath: str, n_atoms: int, specorder: list
    ) -> np.ndarray:
        """Read forces from a LAMMPS custom dump file using ``ase.io.read``.

        ASE's reader interprets the ``fx fy fz`` columns and stores the result
        in a ``SinglePointCalculator`` attached to the returned Atoms object;
        ``atoms.get_forces()`` retrieves them in eV/Å.

        Parameters
        ----------
        specorder : list
            Element ordering used when writing (type 1 = specorder[0], …).
            Required so ASE maps integer type codes back to element symbols.
        """
        from ase.io import read as ase_read

        at     = ase_read(filepath, format="lammps-dump-text", specorder=specorder)
        forces = at.get_forces()

        if len(forces) != n_atoms:
            raise ValueError(
                f"  [VibAnalyzer] Expected {n_atoms} atoms in {filepath}, "
                f"got {len(forces)}"
            )
        return forces

    def _run_lammps_dump_analysis(self, overwrite: bool, save_cache: bool):
        """Custom finite-difference Hessian with LAMMPS dump cache.

        For each displaced configuration (±δ per active atom per Cartesian
        component) forces are computed and saved as a LAMMPS custom dump file:

            {name}/disp_0000.dump   — undisplaced reference
            {name}/disp_0001.dump   — 1st displacement (+δ on first active atom, x)
            {name}/disp_0002.dump   — 2nd displacement (−δ on first active atom, x)
            …
            {name}/disp_manifest.txt — frame index → (atom, direction, ±δ) mapping

        Dump files are written via ``_write_lammps_dump``, which uses ASE's
        ``Prism`` for box/coordinate conversion and ``SinglePointCalculator``
        to attach forces to the Atoms object before writing.  Forces are read
        back via ``ase.io.read(..., format='lammps-dump-text', specorder=...)``.

        When ``overwrite=False`` an existing dump file is reused without a
        new MLIP evaluation — enabling fast restarts of interrupted runs.
        """
        from pathlib import Path

        cache_dir = self.name
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        if overwrite:
            self._robust_rmtree(cache_dir)
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        n_atoms        = len(self.atoms)
        active_indices = self.indices if self.indices is not None else list(range(n_atoms))
        n_active       = len(active_indices)
        delta          = self.displacement
        specorder      = self._specorder(self.atoms)   # e.g. ['H', 'N', 'Si']
        dir_names      = ["x", "y", "z"]

        # ------------------------------------------------------------------
        # Metadata file (types.map)
        # ------------------------------------------------------------------
        with open(os.path.join(cache_dir, "types.map"), "w") as f:
            f.write("# AutoFlow-SRXN vibrational FD cache\n")
            f.write(f"# displacement_ang  {delta}\n")
            f.write(f"# n_active_atoms    {n_active}\n")
            f.write(f"# force_units       eV/Ang  (ASE native)\n")
            f.write(f"# position_units    Ang\n")
            f.write("# LAMMPS type -> element (specorder)\n")
            for i, elem in enumerate(specorder):
                f.write(f"{i + 1}  {elem}\n")
            f.write(f"# active_indices  {active_indices}\n")

        # ------------------------------------------------------------------
        # Reference snapshot  (disp_0000.dump, timestep 0)
        # ------------------------------------------------------------------
        ref_path  = os.path.join(cache_dir, "disp_0000.dump")
        pos0      = self.atoms.get_positions().copy()
        if overwrite or not os.path.exists(ref_path):
            ref_forces = self.atoms.get_forces()
            self._write_lammps_dump(
                ref_path, self.atoms, ref_forces, timestep=0, specorder=specorder
            )
            self.atoms.set_positions(pos0)

        # ------------------------------------------------------------------
        # Finite-difference loop — sequential frame numbering starting at 1
        # ------------------------------------------------------------------
        forces_plus  = {}   # (atom_i, cart) -> ndarray(n_atoms, 3)
        forces_minus = {}

        n_disp     = 2 * n_active * 3
        n_computed = 0
        n_cached   = 0
        frame      = 1          # frame 0 = reference

        manifest_rows = []      # collected for writing at the end

        for atom_i in active_indices:
            for cart in range(3):
                dir_name = dir_names[cart]

                for sign, pm_label, store in [
                    (+1, "+", forces_plus),
                    (-1, "-", forces_minus),
                ]:
                    dump_path = os.path.join(cache_dir, f"disp_{frame:04d}.dump")

                    if not overwrite and os.path.exists(dump_path):
                        forces = self._read_lammps_dump_forces(
                            dump_path, n_atoms, specorder
                        )
                        n_cached += 1
                    else:
                        pos                = pos0.copy()
                        pos[atom_i, cart] += sign * delta
                        self.atoms.set_positions(pos)
                        forces             = self.atoms.get_forces()
                        self._write_lammps_dump(
                            dump_path, self.atoms, forces,
                            timestep=frame, specorder=specorder,
                        )
                        n_computed += 1

                    store[(atom_i, cart)] = forces
                    manifest_rows.append(
                        f"{frame:04d}  {atom_i:6d}  {dir_name}  {pm_label}{delta:.4f}"
                    )
                    frame += 1

        # Restore undisplaced geometry
        self.atoms.set_positions(pos0)

        # ------------------------------------------------------------------
        # Displacement manifest file
        # ------------------------------------------------------------------
        manifest_path = os.path.join(cache_dir, "disp_manifest.txt")
        with open(manifest_path, "w") as f:
            f.write("# AutoFlow-SRXN displacement manifest\n")
            f.write("# frame  atom_idx  direction  displacement_ang\n")
            f.write("# 0000   ---       ---        reference (undisplaced)\n")
            for row in manifest_rows:
                f.write(row + "\n")

        self.logger.info(
            f"  [VibAnalyzer] FD displacements: {n_computed} computed, "
            f"{n_cached} loaded from cache  (total {n_disp})"
        )

        # ------------------------------------------------------------------
        # Build partial Hessian  H[row, col]  (n_active*3 × n_active*3)
        # ------------------------------------------------------------------
        n_dof     = 3 * n_active
        H_partial = np.zeros((n_dof, n_dof))

        for ci, atom_i in enumerate(active_indices):
            for cart_i in range(3):
                col                = 3 * ci + cart_i
                df                 = (forces_plus[(atom_i, cart_i)]
                                      - forces_minus[(atom_i, cart_i)])
                df_active          = df[active_indices]      # (n_active, 3)
                H_partial[:, col]  = -df_active.ravel() / (2.0 * delta)

        H_partial = 0.5 * (H_partial + H_partial.T)          # symmetrise

        # ------------------------------------------------------------------
        # Dynamical matrix  D = H / sqrt(m_i * m_j)  →  eigenvalue problem
        # ------------------------------------------------------------------
        masses     = self.atoms.get_masses()
        mass_vec   = np.repeat(masses[active_indices], 3)
        D          = H_partial / np.sqrt(np.outer(mass_vec, mass_vec))

        eigenvalues, eigenvectors = np.linalg.eigh(D)

        # ------------------------------------------------------------------
        # Convert eigenvalues (eV / amu / Å²) → THz
        # ------------------------------------------------------------------
        freqs_thz = [
            float(np.sqrt(ev) * _EV_PER_AMU_ANG2_TO_THZ) if ev >= 0.0
            else float(-np.sqrt(-ev) * _EV_PER_AMU_ANG2_TO_THZ)
            for ev in eigenvalues
        ]

        # ------------------------------------------------------------------
        # Expand eigenvectors to full 3N space (zero-pad inactive atoms)
        # ------------------------------------------------------------------
        n_modes = len(eigenvalues)
        eigs    = np.zeros((3 * n_atoms, n_modes))
        for mode_i in range(n_modes):
            mode_active = eigenvectors[:, mode_i].reshape(n_active, 3)
            mode_full   = np.zeros((n_atoms, 3))
            for ai, gi in enumerate(active_indices):
                mode_full[gi] = mode_active[ai]
            eigs[:, mode_i] = mode_full.ravel()

        # ------------------------------------------------------------------
        # Cache lifecycle
        # ------------------------------------------------------------------
        n_files = 1 + n_disp   # ref + displacements
        if not save_cache:
            self._robust_rmtree(cache_dir)
        else:
            self.logger.info(
                f"  [VibAnalyzer] Dump cache preserved → {os.path.abspath(cache_dir)}"
                f"  ({n_files} dump files + types.map + disp_manifest.txt)"
            )

        return freqs_thz, eigs

    def _robust_rmtree(self, path):
        """Robustly remove a directory, retrying on failure (common on Windows)."""
        import time

        for i in range(3):
            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
                return
            except Exception:
                time.sleep(0.5)
        # Last resort: ignore errors
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)

    def generate_qpoints_file(self, filename="qpoints.yaml"):
        """Write a phonopy-compatible qpoints.yaml at *filename* using
        manual formatting to match Phonopy's exact style.
        """
        if self._freqs_thz is None or self._eigs is None:
            self.run_analysis()

        n_total_atoms = len(self.atoms)
        masses = self.atoms.get_masses()
        mass_sqrt = np.sqrt(masses)
        num_modes = len(self._freqs_thz)
        lattice = self.atoms.cell

        with open(filename, "w", encoding="utf-8") as w:
            # Header
            w.write("nqpoint: %-7d\n" % 1)
            w.write("natom:   %-7d\n" % n_total_atoms)

            # Reciprocal lattice
            if lattice.volume > 1e-6:
                rec_lattice = np.linalg.inv(lattice)  # column vectors
                w.write("reciprocal_lattice:\n")
                for vec, axis in zip(rec_lattice.T, ("a*", "b*", "c*"), strict=True):
                    w.write("- [ %12.8f, %12.8f, %12.8f ] # %2s\n" % (tuple(vec) + (axis,)))

            w.write("phonon:\n")
            # Q-point (Gamma only)
            w.write("- q-position: [ %12.7f, %12.7f, %12.7f ]\n" % (0.0, 0.0, 0.0))
            w.write("  band:\n")

            for j in range(num_modes):
                freq = float(self._freqs_thz[j])
                w.write("  - # %d\n" % (j + 1))
                w.write("    frequency: %15.10f\n" % freq)

                if self._eigs is not None:
                    # Reconstruct mass-weighted eigenvector (Phonopy convention)
                    # e = u * sqrt(m)
                    u_vec = self._eigs[:, j].reshape(n_total_atoms, 3)
                    e_vec = u_vec * mass_sqrt[:, np.newaxis]
                    norm = np.linalg.norm(e_vec)
                    if norm > 1e-10:
                        e_vec = e_vec / norm

                    w.write("    eigenvector:\n")
                    for k in range(n_total_atoms):
                        w.write("    - # atom %d\n" % (k + 1))
                        for ll in (0, 1, 2):
                            # [real, imag] pair
                            w.write("      - [ %17.14f, %17.14f ]\n" % (float(e_vec[k, ll]), 0.0))
            w.write("\n")

        self.logger.info(f"  [VibAnalyzer] {os.path.relpath(filename)} written in Phonopy-style ({num_modes} modes).")


def calculate_thermo(freqs_thz, T):
    """Calculates vibrational free energy and ZPE given THz frequencies."""
    from ..simulation.thermo_engine import ThermoCalculator, eV_to_J_mol

    thermo = ThermoCalculator(freqs_thz)
    G_vib_J = thermo.calculate_vib_free_energy(T)
    ZPE_J = thermo.calculate_zpe()
    return float(G_vib_J / eV_to_J_mol), float(ZPE_J / eV_to_J_mol)


def build_phva_active_indices(atoms, n_precursor, cutoff_angstrom):
    from ase.neighborlist import neighbor_list

    n_total = len(atoms)
    pre_set = set(range(n_total - n_precursor, n_total))
    i_arr, j_arr = neighbor_list("ij", atoms, cutoff_angstrom)
    slab_neighbors = {int(j_arr[k]) for k, i in enumerate(i_arr) if i in pre_set and j_arr[k] not in pre_set}
    return sorted(pre_set | slab_neighbors)



class MultiModeFollower:
    """Advanced stability refinement using linear combination of imaginary modes."""

    def __init__(self, engine, config):
        self.engine = engine
        self.all_config = config
        # Navigate to analysis.vibrational in the full config tree
        self.vib_config = config.get("analysis", {}).get("vibrational", {})
        self.config = self.vib_config.get("mode_refinement", {})
        self.viz_config = self.vib_config.get("visualization", {})
        self.logger = get_workflow_logger()

    def optimize(self, atoms, modes=None, **kwargs):
        """Refines structure using linear combination of unstable modes.

        Args:
            atoms: ASE Atoms object.
            modes: Optional list of modes (dicts with 'frequency' and 'eigenvector').
            **kwargs: Passed to engine.relax (e.g. fmax, steps).
        """
        # 1. Selection — filter imaginary modes
        if modes is None:
            qpath = self.vib_config.get("qpoints_file") or "qpoints.yaml"
            if not os.path.exists(qpath):
                self.logger.error(f"  [MultiMode] qpoints file not found at '{qpath}'")
                return atoms
            parser = QPointParser(qpath)
            modes = [b for phon in parser.data["phonon"] for b in phon["band"]]

        threshold = self.config.get("freq_threshold_thz", -0.1)
        max_modes = self.config.get("max_modes", 3)
        target_modes = [m for m in modes if m["frequency"] < threshold][:max_modes]

        if not target_modes:
            self.logger.info("  [MultiMode] No imaginary modes found below threshold. Skipping.")
            return atoms

        # 2. Combine displacements
        n_atoms = len(atoms)
        masses = atoms.get_masses()
        m_sqrt = np.sqrt(masses)

        # Resultant displacement vector in Cartesian coordinates
        total_u = np.zeros((n_atoms, 3))

        for mode in target_modes:
            # eigenvector is usually [real, imag] pairs.
            e_raw = np.array(mode["eigenvector"])
            if e_raw.size == 2 * n_atoms * 3:
                # Handle both [[r,i],...] and [r,i,r,i,...] formats
                e_vec = e_raw.reshape(-1, 2)[:, 0].reshape(n_atoms, 3)
            else:
                e_vec = e_raw.reshape(n_atoms, 3)
            # u = e / sqrt(m)
            u_vec = e_vec / m_sqrt[:, np.newaxis]
            total_u += u_vec

        # 3. Apply perturbation scale (alpha)
        alpha = self.config.get("perturbation_alpha", 0.1)
        total_u *= alpha

        # 4. Enforce max_displacement constraint globally
        max_d = np.linalg.norm(total_u, axis=1).max()
        limit = self.config.get("max_displacement", 0.5)
        if max_d > limit:
            scale = limit / max_d
            self.logger.warning(
                f"  [MultiMode] Combined max displacement {max_d:.3f} > limit {limit:.3f}. "
                f"Scaling entire vector by {scale:.3f}"
            )
            total_u *= scale

        # 5. Backup initial positions for interpolation
        initial_atoms = atoms.copy()

        # 6. Perturb and Relax
        current_atoms = atoms.copy()
        current_atoms.set_positions(current_atoms.get_positions() + total_u)

        self.logger.info(f"  [MultiMode] Combined {len(target_modes)} modes. Starting single relaxation...")

        # Ensure 'modes' is NOT in kwargs when calling relax
        relax_kwargs = kwargs.copy()
        if "modes" in relax_kwargs:
            relax_kwargs.pop("modes")
        if "trajectory" in relax_kwargs:
            relax_kwargs.pop("trajectory")

        self.engine.relax(current_atoms, **relax_kwargs)

        # 7. Visualization: Interpolated Animation
        if self.viz_config.get("enabled", False):
            n_frames = self.viz_config.get("n_frames", 10)
            traj_name = self.viz_config.get("output_traj", "relaxation.extxyz")
            self.logger.info(
                f"  [MultiMode] Generating {n_frames} interpolation frames -> {os.path.relpath(traj_name)}"
            )

            final_pos = current_atoms.get_positions()
            start_pos = initial_atoms.get_positions()

            # Prepare the 'forces' array to store displacement vectors for visualization
            # This allows tools like OVITO/VESTA to show arrows for the mode direction.
            viz_forces = total_u.copy()

            animation = []
            for i in range(n_frames):
                # Linear interpolation: t from 0 to 1
                t = i / (n_frames - 1) if n_frames > 1 else 1.0
                frame = initial_atoms.copy()
                frame.set_positions((1.0 - t) * start_pos + t * final_pos)

                # Store displacement vectors in the 'forces' array
                # In extxyz, this maps to FX, FY, FZ columns
                from ase.calculators.singlepoint import SinglePointCalculator

                frame.calc = SinglePointCalculator(frame, forces=viz_forces)

                animation.append(frame)

            write(traj_name, animation)

        return current_atoms


# ---------------------------------------------------------------------------
# Internal exceptions
# ---------------------------------------------------------------------------


class _OvershotError(Exception):
    """Raised by the FIRE observer when the tracked bond exceeds max_bond_dist.

    args: (bond_dist_A: float, energy_eV: float)
    """


# ---------------------------------------------------------------------------
# Gradient Flipping Calculator
# ---------------------------------------------------------------------------

from ase.calculators.calculator import Calculator, all_changes


class GradientFlippingCalculator(Calculator):
    """Custom ASE Calculator that modifies the force vector returned by an
    underlying calculator according to the climbing-image gradient-flipping rule:

    Physics / Algorithm
    -------------------
    Given the true force vector  **g** = -∇E  and the unit eigenvector **v_TS**
    corresponding to the target transition-mode, the modified force is:

        **f_mod** = **g** - 2 (g · v_TS) v_TS          [units: eV/A]

    This inverts the force component along **v_TS** so that the FIRE optimizer
    *climbs* the PES along the TS direction while relaxing in all perpendicular
    directions, driving the structure to a 1st-order saddle point.

    Reference: Henkelman & Jónsson, J. Chem. Phys. 111, 7010 (1999).
    DOI: 10.1063/1.480097
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, v_ts: np.ndarray, **kwargs):
        """Args:
        base_calc: Any ASE-compatible Calculator (e.g. MACE).
        v_ts:      Normalised 3N eigenvector of the target TS mode  (units: dimensionless).
        """
        super().__init__(**kwargs)
        self.base_calc = base_calc
        # Flatten and normalise defensively
        self.v_ts = v_ts.ravel() / np.linalg.norm(v_ts)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        # Delegate to the underlying calculator directly — do NOT set atoms.calc,
        # because that would overwrite this wrapper on the atoms object and break
        # gradient flipping on every step after the first.
        self.base_calc.calculate(atoms, properties, system_changes)

        energy = self.base_calc.results["energy"]  # units: eV
        g = self.base_calc.results["forces"].ravel()  # units: eV/A, shape (3N,)

        # Gradient-flipping: f_mod = g - 2(g · v_TS) v_TS
        overlap = np.dot(g, self.v_ts)  # scalar projection  [eV/A]
        f_mod = g - 2.0 * overlap * self.v_ts  # modified force     [eV/A]

        self.results["energy"] = energy
        self.results["forces"] = f_mod.reshape(atoms.positions.shape)


# ---------------------------------------------------------------------------
# Adaptive Gradient Flipping Calculator
# ---------------------------------------------------------------------------


class AdaptiveGradientFlippingCalculator(Calculator):
    """Gradient-flipping calculator whose climbing direction is recomputed at
    every force evaluation as the current unit vector along the bond being
    broken (central_idx -> ligand_idx).

    Motivation
    ----------
    A fixed v_TS (from the Hessian at the initial geometry) stops tracking
    the actual bond-stretch direction once the structure deforms.  Keeping
    v_TS aligned with the live bond vector ensures:

    • The climbing force always acts along the correct coordinate.
    • After the structure crosses the saddle (Si-N force reverses sign),
      the gradient-flip automatically creates a restoring force that
      drives FIRE to converge *at* the TS rather than overshooting to
      the fully-dissociated limit.

    Only the two bond atoms have their forces modified; all other atoms
    experience unmodified MACE forces and relax normally.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, central_idx: int, ligand_idx: int, **kwargs):
        """Args:
        base_calc:    Any ASE-compatible Calculator (e.g. MACE).
        central_idx:  Index of the central atom (Si).
        ligand_idx:   Index of the ligand atom (N).
        """
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.c_idx = central_idx
        self.l_idx = ligand_idx
        self._last_energy = float("nan")  # updated every calculate(); read by FIRE observer

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        # --- Dynamic climbing direction: current Si->N unit vector in 3N space ---
        r_c = atoms.positions[self.c_idx]
        r_l = atoms.positions[self.l_idx]
        v_dir = r_l - r_c  # units: A
        v_hat = v_dir / np.linalg.norm(v_dir)  # dimensionless unit vector

        n = len(atoms)
        v_ts = np.zeros((n, 3))
        v_ts[self.l_idx] = v_hat  # ligand atom moves away from central
        v_ts[self.c_idx] = -v_hat  # central atom moves away from ligand
        v_ts = v_ts.ravel()
        v_ts /= np.linalg.norm(v_ts)  # norm = 1/sqrt(2) * sqrt(2) = 1

        # --- Base forces (do NOT set atoms.calc — see GradientFlippingCalculator) ---
        self.base_calc.calculate(atoms, properties, system_changes)
        energy = self.base_calc.results["energy"]  # units: eV
        g = self.base_calc.results["forces"].ravel()  # units: eV/A

        # Cache energy so the FIRE observer can read it reliably via closure.
        # (self.atoms.calc.results is not guaranteed to be populated when the
        # observer fires between FIRE steps.)
        self._last_energy = float(energy)

        # --- Gradient-flipping: f_mod = g - 2(g · v_TS) v_TS ---
        overlap = np.dot(g, v_ts)
        f_mod = g - 2.0 * overlap * v_ts

        self.results["energy"] = energy
        self.results["forces"] = f_mod.reshape(atoms.positions.shape)


# ---------------------------------------------------------------------------
# TSSearcher  (Hessian-Based Gradient Flipping)
# ---------------------------------------------------------------------------


class TSSearcher:
    """Transition State Searcher using a Simplified Hessian-Based Gradient
    Flipping strategy.

    Algorithm Overview
    ------------------
    1. Compute the full molecular Hessian at the initial geometry via
       VibrationalAnalyzer (finite-difference, δ = 0.01 A).
    2. Identify the ligand fragment attached to bond_indices[1] through
       graph partitioning of the covalent-bond adjacency matrix (bond to
       be broken is excluded from the graph).
    3. Compute the dissociation direction:
           V_dir = COM(Ligand) - r(central_atom)          [units: A]
           V_hat = V_dir / |V_dir|                        [dimensionless]
    4. Select the target TS eigenvector **v_TS** by maximum dot-product
       overlap with V_hat:
           k* = argmax_k |v_k · V_hat_3N|
    5. Apply an initial perturbation along **v_TS** and relax with FIRE
       using GradientFlippingCalculator.
    6. Return the converged structure (1st-order saddle point candidate).
    """

    def __init__(self, engine, atoms, config: dict | None = None):
        """Args:
        engine: SimulationEngine with a MACE (or other ASE-compatible) backend.
        atoms:  ASE Atoms object (initial geometry, already relaxed).
        config: Optional dict from config.yaml ts_search block.
        """
        self.engine = engine
        self.atoms = atoms.copy()
        self.config = config or {}
        self.logger = get_workflow_logger()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _identify_ligand_fragment(self, atoms, bond_indices: list[int]) -> list[int]:
        """Graph-partitions the covalent bond network, excluding the target
        bond, and returns the indices belonging to the ligand fragment.

        Args:
            atoms:        ASE Atoms object.
            bond_indices: [central_idx, ligand_start_idx]

        Returns:
            Sorted list of atom indices in the ligand fragment.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        c_idx, l_idx = bond_indices
        n_atoms = len(atoms)
        adj = np.zeros((n_atoms, n_atoms), dtype=int)
        pos = atoms.positions

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Exclude the bond being broken
                if (i == c_idx and j == l_idx) or (i == l_idx and j == c_idx):
                    continue
                dist = np.linalg.norm(pos[i] - pos[j])
                cutoff = chem_kb.get_radius(atoms.symbols[i], "covalent") + chem_kb.get_radius(atoms.symbols[j], "covalent") + 0.3  # units: A
                if dist < cutoff:
                    adj[i, j] = adj[j, i] = 1

        _, labels = connected_components(csr_matrix(adj), directed=False)
        ligand_label = labels[l_idx]
        return sorted(np.where(labels == ligand_label)[0].tolist())

    def _compute_hessian_eigensystem(self, atoms, displacement: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Computes the full (3N × 3N) Hessian via finite differences and
        returns its eigensystem.

        Physics
        -------
        The Hessian element is approximated by central differences:

            H_{ij} = [ F_i(+δ_j) - F_i(-δ_j) ] / (2δ)     [units: eV/A2]

        Diagonalisation:  H v_k = λ_k v_k

        Args:
            atoms:       ASE Atoms object with calculator attached.
            displacement: Finite-difference step δ  (units: A, default 0.01 A).

        Returns:
            eigenvalues:  shape (3N,)    [units: eV/A2]
            eigenvectors: shape (3N, 3N) [dimensionless], columns are modes
        """
        self.logger.info(
            f"  [TSSearch] Computing full Hessian "
            f"(δ={displacement} A, N={len(atoms)} atoms, "
            f"{2 * 3 * len(atoms)} MACE evaluations)..."
        )

        n_atoms = len(atoms)
        dof = 3 * n_atoms
        H = np.zeros((dof, dof))  # units: eV/A2
        pos0 = atoms.get_positions().copy()

        for atom_i in range(n_atoms):
            for cart in range(3):  # x, y, z
                col = 3 * atom_i + cart

                # Forward displacement
                pos_fwd = pos0.copy()
                pos_fwd[atom_i, cart] += displacement  # units: A
                atoms.set_positions(pos_fwd)
                f_fwd = atoms.get_forces().ravel()  # units: eV/A

                # Backward displacement
                pos_bwd = pos0.copy()
                pos_bwd[atom_i, cart] -= displacement  # units: A
                atoms.set_positions(pos_bwd)
                f_bwd = atoms.get_forces().ravel()  # units: eV/A

                # Central-difference gradient of force = -Hessian column
                H[:, col] = -(f_fwd - f_bwd) / (2.0 * displacement)  # units: eV/A2

        # Restore original positions
        atoms.set_positions(pos0)

        # Symmetrise to suppress numerical asymmetry
        H = 0.5 * (H + H.T)

        eigenvalues, eigenvectors = np.linalg.eigh(H)
        self.logger.info("  [TSSearch] Hessian diagonalisation complete.")
        return eigenvalues, eigenvectors  # cols of eigenvectors are modes

    def _select_ts_mode(
        self,
        eigenvectors: np.ndarray,
        direction_3n: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """Selects the eigenvector v_TS with the highest overlap with the
        3N-dimensional dissociation direction vector V_hat:

            k* = argmax_k |v_k · V_hat|,   V_hat = V_dir / |V_dir|

        Args:
            eigenvectors: (3N × 3N) matrix, columns are mode eigenvectors.
            direction_3n: Raw 3N dissociation vector (need not be normalised).

        Returns:
            (k_star, v_ts)  —  selected mode index and unit eigenvector.
        """
        v_hat = direction_3n / np.linalg.norm(direction_3n)  # dimensionless
        overlaps = np.abs(eigenvectors.T @ v_hat)  # shape (3N,)
        k_star = int(np.argmax(overlaps))
        v_ts = eigenvectors[:, k_star]  # unit eigenvector
        self.logger.info(f"  [TSSearch] TS mode selected: index {k_star}, overlap = {overlaps[k_star]:.4f}")
        return k_star, v_ts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_transition_state(
        self,
        bond_indices: list[int],
        fmax: float = 0.05,
        steps: int = 200,
        trajectory: str | None = None,
    ):
        """Finds the 1st-order saddle point (Transition State) for the bond
        specified by bond_indices using Hessian-based gradient flipping.

        Physics / Algorithm
        -------------------
        See class docstring for full derivation.

        Args:
            bond_indices: [central_atom_idx, ligand_start_idx]  (e.g. [Si, N]).
            fmax:         Force convergence threshold   (units: eV/A).
            steps:        Maximum FIRE optimizer steps.
            trajectory:   Optional path to save the optimisation trajectory.

        Returns:
            ASE Atoms object at the TS geometry.
        """
        c_idx, l_idx = bond_indices
        n_atoms = len(self.atoms)
        displacement = self.config.get("hessian_displacement", 0.01)  # units: A

        # --- Step 1: Attach calculator and compute Hessian ---
        self.atoms.calc = self.engine.get_calculator()
        eigenvalues, eigenvectors = self._compute_hessian_eigensystem(self.atoms, displacement=displacement)

        # --- Step 2: Identify ligand fragment ---
        self.logger.info(f"  [TSSearch] Identifying ligand fragment (central={c_idx}, ligand_start={l_idx})...")
        ligand_indices = self._identify_ligand_fragment(self.atoms, bond_indices)
        self.logger.info(f"  [TSSearch] Fragment: {len(ligand_indices)} atoms -> indices {ligand_indices}")

        # --- Step 3: Build 3N dissociation direction vector ---
        # Use only the two bond atoms (central and ligand-start) moving in opposite
        # directions. Distributing the direction across the whole ligand fragment gives
        # a large net-drift vector that inadvertently overlaps with translational
        # zero-modes instead of the actual bond-stretching mode.
        pos_central = self.atoms.positions[c_idx]  # units: A
        pos_ligstart = self.atoms.positions[l_idx]  # units: A
        v_dir_3d = pos_ligstart - pos_central  # units: A
        v_hat = v_dir_3d / np.linalg.norm(v_dir_3d)  # unit vector

        v_dir_3n = np.zeros((n_atoms, 3))  # units: A
        v_dir_3n[l_idx] = v_hat  # ligand-start atom moves away from central
        v_dir_3n[c_idx] = -v_hat  # central atom moves away from ligand-start
        v_dir_3n = v_dir_3n.ravel()

        self.logger.info(
            f"  [TSSearch] Dissociation direction |V_dir| = {np.linalg.norm(v_dir_3d):.3f} A (Si-N bond vector)"
        )

        # --- Step 4: Select TS mode by maximum overlap ---
        k_star, v_ts = self._select_ts_mode(eigenvectors, v_dir_3n)

        n_imag_init = int(np.sum(eigenvalues < 0.0))
        self.logger.info(
            f"  [TSSearch] Initial Hessian: {n_imag_init} negative eigenvalue(s). "
            f"Selected mode eigenvalue λ_{k_star} = {eigenvalues[k_star]:.4f} eV/A2"
        )

        # --- Step 5: Apply initial perturbation along v_TS ---
        disp_ang = self.config.get("displacement_ang", 0.2)  # units: A
        perturbation = v_ts.reshape(n_atoms, 3) * np.sign(np.dot(v_ts, v_dir_3n))  # align sign with direction
        self.atoms.set_positions(
            self.atoms.positions + disp_ang * perturbation  # units: A
        )
        self.logger.info(f"  [TSSearch] Applied perturbation: {disp_ang} A along v_TS.")

        # --- Step 6: Adaptive Gradient-Flipping optimisation with FIRE ---
        # AdaptiveGradientFlippingCalculator recomputes the climbing direction
        # at every force evaluation using the current bond vector, so the
        # climbing always follows the actual Si-N axis regardless of molecular
        # rotation or other conformational changes during the trajectory.
        base_calc = self.engine.get_calculator()
        gf_calc = AdaptiveGradientFlippingCalculator(base_calc=base_calc, central_idx=c_idx, ligand_idx=l_idx)
        self.atoms.calc = gf_calc

        max_bond_dist = self.config.get("max_bond_dist", 3.5)  # A
        log_interval = self.config.get("log_interval", 10)  # steps

        r_init = float(np.linalg.norm(self.atoms.positions[l_idx] - self.atoms.positions[c_idx]))

        self.logger.info(
            f"  [TSSearch] Starting FIRE with adaptive gradient flipping "
            f"(fmax={fmax} eV/A, max_steps={steps}, "
            f"max_bond_dist={max_bond_dist:.2f} A)..."
        )

        # --- Overshoot monitor ---
        # Called every `log_interval` FIRE steps via dyn.attach().
        # Raises _OvershotError when the bond length exceeds max_bond_dist,
        # which propagates through dyn.run() and is caught below.
        step_log = []  # list of (step, bond_dist_A, energy_eV)

        dyn = FIRE(self.atoms, trajectory=trajectory, logfile="-")

        def _check_overshoot():
            r = float(np.linalg.norm(self.atoms.positions[l_idx] - self.atoms.positions[c_idx]))
            e = gf_calc._last_energy  # cached by AdaptiveGradientFlippingCalculator.calculate()
            step_log.append((dyn.nsteps, r, e))
            if r > max_bond_dist:
                raise _OvershotError(r, e)

        dyn.attach(_check_overshoot, interval=log_interval)

        overshoot_detected = False
        try:
            converged = dyn.run(fmax=fmax, steps=steps)
        except _OvershotError as exc:
            converged = False
            overshoot_detected = True
            r_over, e_over = exc.args

        # --- Report ---
        if overshoot_detected:
            e_start = step_log[0][2] if step_log else float("nan")
            r_final = step_log[-1][1] if step_log else float("nan")
            e_final = step_log[-1][2] if step_log else float("nan")

            self.logger.warning("  [TSSearch] *** BARRIERLESS DISSOCIATION DETECTED ***")
            self.logger.warning(
                f"  [TSSearch]   Bond length exceeded max_bond_dist = {max_bond_dist:.2f} A during gradient-flipping."
            )
            self.logger.warning(
                "  [TSSearch]   The Si-N PES appears to have no classical barrier with this potential/model."
            )
            self.logger.warning(f"  [TSSearch]   Initial : bond = {r_init:.3f} A, E = {e_start:.4f} eV")
            self.logger.warning(
                f"  [TSSearch]   At stop : bond = {r_final:.3f} A, "
                f"E = {e_final:.4f} eV  "
                f"(dE = {e_final - e_start:+.3f} eV)"
            )
            self.logger.warning(f"  [TSSearch]   Bond / energy profile (every {log_interval} steps):")
            for step, r, e in step_log:
                self.logger.warning(
                    f"  [TSSearch]     step {step:4d}:  bond = {r:.3f} A,  E = {e:.4f} eV  (dE = {e - e_start:+.3f} eV)"
                )
            self.logger.warning(
                "  [TSSearch]   Recommendation: use a surface slab model so that adsorption energy stabilises the TS."
            )
        elif converged:
            self.logger.info("  [TSSearch] Gradient-flipping optimisation converged.")
        else:
            self.logger.warning(
                f"  [TSSearch] Did not converge within {steps} steps. "
                "Consider increasing 'steps' or checking the initial geometry."
            )

        return self.atoms


def calculate_mac(eig_a: np.ndarray, eig_b: np.ndarray) -> float:
    """Computes the Modal Assurance Criterion (MAC) between two eigenvectors.
    MAC = |(a^T * b)|^2 / ((a^T * a) * (b^T * b))

    Physics: MAC measures the degree of consistency between two modal vectors.
    A value of 1.0 indicates a perfect match.
    """
    a = eig_a.flatten()
    b = eig_b.flatten()
    # Normalize
    norm_a = np.dot(a, a)
    norm_b = np.dot(b, b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    overlap = np.dot(a, b)
    return (overlap**2) / (norm_a * norm_b)


def calculate_atomic_participation(eig: np.ndarray, n_atoms: int) -> np.ndarray:
    """Calculates the normalized displacement contribution per atom.
    Returns an array of shape (n_atoms,).
    """
    mode_3d = eig.reshape(n_atoms, 3)
    # Sum of squares of x,y,z displacements per atom
    participation = np.sum(mode_3d**2, axis=1)
    # Normalize so that sum across all atoms = 1.0
    total = np.sum(participation)
    if total < 1e-12:
        return participation
    return participation / total


