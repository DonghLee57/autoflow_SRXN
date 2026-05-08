"""
autoflow_srxn/vibrational/mode_participation_analyzer.py
=======================================================
Physical comparison of vibrational modes between PHVA and FHVA calculations,
or analysis of a single qpoints.yaml.

Three quantities are computed from the normalised mass-weighted eigenvectors
stored in qpoints.yaml files produced by ``VibrationalAnalyzer``:

  P_j(ν)  — Energy Participation Ratio  (Gemini §1)
             P_j = Σ_α |e^mw_{j,α}|²
             Σ_j P_j = 1  (holds because e^mw is a unit vector by construction)
             Physical meaning: fraction of mode ν's total harmonic energy
             carried by atom j.  For a Si–H stretch mode: P_H ≈ 0.97
             (momentum conservation → u_H/u_Si ≈ m_Si/m_H = 28).

  IPR(ν)  — Inverse Participation Ratio  (Gemini §2)
             IPR = Σ_j P_j²,  range [1/N, 1]
             N_eff ≡ 1/IPR  — effective number of atoms participating.
             IPR → 1: single-atom localisation.  IPR → 1/N: collective mode.

  MAC(i,j) — Modal Assurance Criterion
             MAC = |⟨e_i, e_j⟩|² / (‖e_i‖² ‖e_j‖²),  range [0, 1]
             Identifies physically corresponding modes between two calculations
             without relying on frequency ordering, which is unreliable across
             different Hessian truncations (PHVA vs FHVA).

Notes on PHVA eigenvectors
--------------------------
In PHVA the Hessian is restricted to ``n_active`` atoms; the remaining atoms
are zero-padded in the output eigenvectors.  As a result:
  • Inactive atom P_j = 0 by construction — not a physical zero.
  • PHVA IPR > FHVA IPR by construction (energy concentrates on fewer atoms).
  • MAC-based matching is still valid: the dot product naturally vanishes for
    modes that are delocalized onto inactive atoms, giving a low MAC score and
    correctly flagging the mode as poorly represented by PHVA.

Typical usage
-------------
Single-file analysis (P_j + IPR for every mode)::

    from autoflow_srxn.vibrational.mode_participation_analyzer import analyze_single
    report = analyze_single("run/qpoints.yaml")
    report.print_summary(top_atoms=3)
    report.save_csv("participation.csv")

PHVA vs FHVA comparison::

    from autoflow_srxn.vibrational.mode_participation_analyzer import compare_phva_fhva
    result = compare_phva_fhva("run_phva/qpoints.yaml", "run_fhva/qpoints.yaml")
    result.print_summary(mac_threshold=0.6)
    result.save_csv("mode_pairs.csv")
    result.save_participation_csv("participation.csv")
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml_fast(path: str) -> dict:
    """Load a YAML file using CLoader when available (faster for large files)."""
    with open(path, encoding="utf-8") as fh:
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        return yaml.load(fh, Loader=Loader)


def _eig_from_band(raw_eig, n_atoms: int) -> np.ndarray:
    """Extract the **real** part of one band's eigenvector.

    Parameters
    ----------
    raw_eig :
        Raw value of the ``eigenvector`` key from a yaml band entry.
        Two layouts are supported:

        Nested (AutoFlow-SRXN generator, len = n_atoms)::

            raw_eig[atom_idx][coord_idx] = [real, imag]

        Flat (some Phonopy outputs, len = 3*n_atoms)::

            raw_eig[3*atom_idx + coord_idx] = [real, imag]

    n_atoms : int
        Expected atom count (used for zero-initialisation when the block
        is missing or shorter than expected).

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
        Real components only (imaginary part is 0 at Γ for real-space
        systems).
    """
    eig = np.zeros((n_atoms, 3))
    if not raw_eig:
        return eig

    # Distinguish nested from flat by inspecting the first element.
    # Nested: raw_eig[0] is a list of 3 items, each a [real, imag] pair.
    # Flat:   raw_eig[0] is a [real, imag] pair directly.
    is_nested = isinstance(raw_eig[0][0], (list, tuple))

    if is_nested:
        n = min(len(raw_eig), n_atoms)
        for j in range(n):
            for alpha in range(3):
                eig[j, alpha] = float(raw_eig[j][alpha][0])
    else:
        n_dof = min(len(raw_eig), 3 * n_atoms)
        for k in range(n_dof):
            j, alpha = divmod(k, 3)
            eig[j, alpha] = float(raw_eig[k][0])

    return eig


# ---------------------------------------------------------------------------
# Core physics functions (standalone, importable independently)
# ---------------------------------------------------------------------------

def atomic_participation(eig: np.ndarray) -> np.ndarray:
    """Compute the Energy Participation Ratio P_j for one mode.

    Parameters
    ----------
    eig : ndarray, shape (n_atoms, 3)
        Normalised mass-weighted eigenvector for one mode
        (as stored in qpoints.yaml).

    Returns
    -------
    P_j : ndarray, shape (n_atoms,)
        P_j = Σ_α |e^mw_{j,α}|².  Σ_j P_j = 1 iff ‖eig‖ = 1.
        If ‖eig‖ = 0 (zero mode), returns all zeros.
    """
    mode_3d = np.asarray(eig).reshape(-1, 3)  # (n_atoms, 3)
    P = np.sum(mode_3d ** 2, axis=1)          # Σ_α |e_{j,α}|²
    total = P.sum()
    return P / total if total > 1e-12 else P


def ipr(P_j: np.ndarray) -> float:
    """Inverse Participation Ratio from a P_j array.

    Parameters
    ----------
    P_j : ndarray, shape (n_atoms,)
        Energy Participation Ratio per atom (Σ P_j = 1).

    Returns
    -------
    float
        IPR = Σ_j P_j²,  range [1/N, 1].
        ``1/IPR`` gives the effective number of participating atoms.
    """
    return float(np.sum(P_j ** 2))


def mac_matrix(eigs_a: np.ndarray, eigs_b: np.ndarray) -> np.ndarray:
    """Vectorized Modal Assurance Criterion matrix.

    Parameters
    ----------
    eigs_a : ndarray, shape (n_a, n_atoms, 3) or (n_a, D)
    eigs_b : ndarray, shape (n_b, n_atoms, 3) or (n_b, D)

    Returns
    -------
    MAC : ndarray, shape (n_a, n_b)
        MAC[i, j] = |⟨e_i^a, e_j^b⟩|² / (‖e_i^a‖² · ‖e_j^b‖²).
        Rows with ‖e_i^a‖ < 1e-12 (zero modes) get MAC = 0.
    """
    a = np.asarray(eigs_a, dtype=float).reshape(len(eigs_a), -1)  # (n_a, D)
    b = np.asarray(eigs_b, dtype=float).reshape(len(eigs_b), -1)  # (n_b, D)

    norm_a = np.einsum("id,id->i", a, a)   # (n_a,)  — squared norms
    norm_b = np.einsum("jd,jd->j", b, b)   # (n_b,)

    cross = a @ b.T                         # (n_a, n_b) — dot products

    denom = np.outer(norm_a, norm_b)
    denom = np.where(denom > 1e-24, denom, 1.0)  # guard division by zero

    mac = (cross ** 2) / denom

    # Zero out rows/cols corresponding to null eigenvectors
    mac[norm_a < 1e-12, :] = 0.0
    mac[:, norm_b < 1e-12] = 0.0

    return np.clip(mac, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class QPointsData:
    """Parsed content of one qpoints.yaml file (Gamma-point only)."""

    source: str                  # absolute file path
    n_atoms: int                 # total number of atoms (including zero-padded)
    n_modes: int                 # number of phonon bands
    freqs_thz: np.ndarray        # (n_modes,) signed frequencies in THz
    eigs: np.ndarray             # (n_modes, n_atoms, 3) real, mass-weighted, normalised
    active_mask: np.ndarray      # (n_atoms,) bool — False for zero-padded (PHVA inactive)

    @property
    def active_indices(self) -> List[int]:
        """Indices of atoms with non-zero eigenvector components (auto-detected)."""
        return list(np.where(self.active_mask)[0])

    @property
    def n_active(self) -> int:
        return int(self.active_mask.sum())


@dataclass
class SingleModeRecord:
    """Participation data for one mode from a single qpoints.yaml."""
    mode_idx: int         # 0-based
    freq_thz: float
    P_j: np.ndarray       # (n_atoms,)
    ipr_val: float
    n_eff: float          # 1/IPR


@dataclass
class SingleAnalysisResult:
    """Result of ``analyze_single``: participation data for every mode."""
    source: QPointsData
    records: List[SingleModeRecord]    # sorted by freq descending (real first)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_summary(
        self,
        top_atoms: int = 3,
        symbols: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = None,
    ) -> None:
        """Print a tabular summary of all modes.

        Parameters
        ----------
        top_atoms : int
            Number of highest-P_j atoms to list per mode.
        symbols : sequence of str, optional
            Element symbols in atom-index order.  When provided, the top-atom
            column shows ``El(idx):P_j`` instead of ``idx:P_j``.
        max_rows : int, optional
            Cap number of printed rows (useful for large systems).
        """
        src = os.path.relpath(self.source.source)
        n_active = self.source.n_active
        n_total = self.source.n_atoms
        print(f"\n{'='*80}")
        print(f" Single-mode participation report")
        print(f"   Source  : {src}")
        print(f"   Atoms   : {n_active} active / {n_total} total")
        print(f"   Modes   : {len(self.records)}")
        print(f"{'='*80}")
        header = (
            f"{'#':>4}  {'freq(THz)':>10}  {'IPR':>7}  {'N_eff':>6}"
            f"  Top-{top_atoms} atoms (P_j)"
        )
        print(header)
        print("-" * len(header))

        rows = self.records if max_rows is None else self.records[:max_rows]
        for rec in rows:
            freq_str = f"{rec.freq_thz:+10.4f}"
            ipr_str = f"{rec.ipr_val:7.4f}"
            neff_str = f"{rec.n_eff:6.1f}"
            top_str = _format_top_atoms(rec.P_j, top_atoms, symbols)
            print(f"{rec.mode_idx+1:>4}  {freq_str}  {ipr_str}  {neff_str}  {top_str}")

        if max_rows is not None and max_rows < len(self.records):
            print(f"  ... ({len(self.records) - max_rows} more rows omitted)")
        print()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_csv(self, path: str, symbols: Optional[Sequence[str]] = None) -> None:
        """Save mode-level summary (freq, IPR, N_eff) to CSV.

        Parameters
        ----------
        symbols : sequence of str, optional
            Element symbols; when given, adds a ``top3_atoms`` column with
            element-labelled contributions.
        """
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            header = ["mode_idx", "freq_thz", "ipr", "n_eff"]
            n = self.source.n_atoms
            header += [f"P_j_{i}" for i in range(n)]
            if symbols:
                header.append("top3_atoms")
            w.writerow(header)
            for rec in self.records:
                row = [
                    rec.mode_idx + 1,
                    f"{rec.freq_thz:.6f}",
                    f"{rec.ipr_val:.6f}",
                    f"{rec.n_eff:.4f}",
                ] + [f"{v:.6f}" for v in rec.P_j]
                if symbols:
                    row.append(_format_top_atoms(rec.P_j, 3, symbols))
                w.writerow(row)

        print(f"  [ModeParticipation] Saved mode summary → {os.path.relpath(path)}")


# ---------------------------------------------------------------------------

@dataclass
class MatchedMode:
    """One physically corresponding mode pair identified by MAC."""

    phva_mode_idx: int        # 0-based index in PHVA yaml
    phva_freq_thz: float
    phva_P_j: np.ndarray      # (n_atoms,) — inactive atoms are 0.0
    phva_ipr: float
    phva_n_eff: float         # 1/IPR (atoms)

    fhva_mode_idx: int        # 0-based index in FHVA yaml
    fhva_freq_thz: float
    fhva_P_j: np.ndarray      # (n_atoms,)
    fhva_ipr: float
    fhva_n_eff: float

    mac_score: float          # [0, 1]; higher = more similar mode shapes
    freq_delta_thz: float     # phva_freq − fhva_freq  [THz]
    freq_delta_pct: float     # 100 * (phva − fhva) / |fhva|  [%]
    ambiguous: bool = False   # True when another PHVA mode also matched this FHVA mode


@dataclass
class ModeComparisonResult:
    """Full PHVA ↔ FHVA comparison produced by ``compare_phva_fhva``."""

    phva: QPointsData
    fhva: QPointsData
    matched: List[MatchedMode]      # all PHVA modes, ordered by fhva_freq descending
    mac_mat: np.ndarray             # (n_phva_modes, n_fhva_modes)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_summary(
        self,
        mac_threshold: float = 0.5,
        top_atoms: int = 3,
        symbols: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = None,
    ) -> None:
        """Print a tabular PHVA ↔ FHVA mode comparison.

        Parameters
        ----------
        mac_threshold : float
            Modes with MAC < threshold are flagged with ``!`` in the table.
        top_atoms : int
            Number of highest-P_j atoms shown per mode (FHVA side).
        symbols : sequence of str, optional
            Element symbols in atom-index order for labelled atom columns.
        max_rows : int, optional
            Cap on printed rows.
        """
        phva_src = os.path.relpath(self.phva.source)
        fhva_src = os.path.relpath(self.fhva.source)
        n_good = sum(1 for m in self.matched if m.mac_score >= mac_threshold)

        print(f"\n{'='*100}")
        print(f" PHVA ↔ FHVA Mode Comparison")
        print(f"   PHVA : {phva_src}  ({self.phva.n_active} active / {self.phva.n_atoms} atoms, {self.phva.n_modes} modes)")
        print(f"   FHVA : {fhva_src}  ({self.fhva.n_atoms} atoms, {self.fhva.n_modes} modes)")
        print(f"   Pairs with MAC ≥ {mac_threshold:.2f} : {n_good} / {len(self.matched)}")
        print(f"{'='*100}")

        col_w = 100
        header = (
            f"{'':1}{'PHVA#':>5}  {'PHVA freq':>10}  "
            f"{'FHVA#':>5}  {'FHVA freq':>10}  "
            f"{'Δfreq%':>7}  {'MAC':>6}  "
            f"{'IPR_P':>7}  {'IPR_F':>7}  {'Neff_F':>6}  "
            f"Top-{top_atoms} atoms [FHVA P_j]"
        )
        print(header)
        print("-" * min(len(header), col_w))

        rows = self.matched if max_rows is None else self.matched[:max_rows]
        for m in rows:
            flag = "!" if m.mac_score < mac_threshold else " "
            amb = "*" if m.ambiguous else " "
            top_str = _format_top_atoms(m.fhva_P_j, top_atoms, symbols)
            print(
                f"{flag}{amb}"
                f"{m.phva_mode_idx+1:>4}  {m.phva_freq_thz:>+10.4f}  "
                f"{m.fhva_mode_idx+1:>5}  {m.fhva_freq_thz:>+10.4f}  "
                f"{m.freq_delta_pct:>+7.2f}  {m.mac_score:>6.4f}  "
                f"{m.phva_ipr:>7.4f}  {m.fhva_ipr:>7.4f}  {m.fhva_n_eff:>6.1f}  "
                f"{top_str}"
            )

        if max_rows is not None and max_rows < len(self.matched):
            print(f"  ... ({len(self.matched) - max_rows} rows omitted)")

        print("\n  Legend:  ! = MAC below threshold,  * = ambiguous match (shared FHVA mode)")
        print()

    # ------------------------------------------------------------------
    # I/O — mode-level summary
    # ------------------------------------------------------------------

    def save_csv(self, path: str, symbols: Optional[Sequence[str]] = None) -> None:
        """Save mode-pair summary to CSV (one row per PHVA mode).

        Columns
        -------
        phva_mode, phva_freq_thz, fhva_mode, fhva_freq_thz,
        delta_freq_thz, delta_freq_pct, mac_score,
        ipr_phva, n_eff_phva, ipr_fhva, n_eff_fhva, ambiguous
        """
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([
                "phva_mode", "phva_freq_thz",
                "fhva_mode", "fhva_freq_thz",
                "delta_freq_thz", "delta_freq_pct",
                "mac_score",
                "ipr_phva", "n_eff_phva",
                "ipr_fhva", "n_eff_fhva",
                "ambiguous",
            ])
            for m in self.matched:
                w.writerow([
                    m.phva_mode_idx + 1,
                    f"{m.phva_freq_thz:.6f}",
                    m.fhva_mode_idx + 1,
                    f"{m.fhva_freq_thz:.6f}",
                    f"{m.freq_delta_thz:.6f}",
                    f"{m.freq_delta_pct:.4f}",
                    f"{m.mac_score:.6f}",
                    f"{m.phva_ipr:.6f}",
                    f"{m.phva_n_eff:.4f}",
                    f"{m.fhva_ipr:.6f}",
                    f"{m.fhva_n_eff:.4f}",
                    int(m.ambiguous),
                ])
        print(f"  [ModeParticipation] Saved mode pairs    → {os.path.relpath(path)}")

    # ------------------------------------------------------------------
    # I/O — per-atom participation
    # ------------------------------------------------------------------

    def save_participation_csv(
        self,
        path: str,
        symbols: Optional[Sequence[str]] = None,
        mac_threshold: float = 0.0,
    ) -> None:
        """Save per-atom P_j data for every matched mode pair to CSV.

        Each matched pair contributes ``n_atoms`` rows, giving full atom-
        resolved participation for both PHVA and FHVA sides.

        Columns
        -------
        phva_mode, fhva_mode, mac_score, atom_idx, [element,]
        P_j_phva, P_j_fhva, delta_P_j
        """
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            cols = ["phva_mode", "fhva_mode", "mac_score", "atom_idx"]
            if symbols:
                cols.append("element")
            cols += ["P_j_phva", "P_j_fhva", "delta_P_j"]
            w.writerow(cols)

            for m in self.matched:
                if m.mac_score < mac_threshold:
                    continue
                for j in range(self.phva.n_atoms):
                    row = [
                        m.phva_mode_idx + 1,
                        m.fhva_mode_idx + 1,
                        f"{m.mac_score:.6f}",
                        j,
                    ]
                    if symbols and j < len(symbols):
                        row.append(symbols[j])
                    delta = float(m.phva_P_j[j]) - float(m.fhva_P_j[j])
                    row += [
                        f"{m.phva_P_j[j]:.8f}",
                        f"{m.fhva_P_j[j]:.8f}",
                        f"{delta:.8f}",
                    ]
                    w.writerow(row)

        print(f"  [ModeParticipation] Saved participation → {os.path.relpath(path)}")

    # ------------------------------------------------------------------
    # I/O — full YAML
    # ------------------------------------------------------------------

    def save_yaml(self, path: str) -> None:
        """Save the complete comparison result (including P_j arrays) to YAML.

        The output can be loaded with ``yaml.safe_load`` for further analysis
        without re-running the comparison.
        """
        def _arr(a):
            return [round(float(v), 8) for v in a]

        doc = {
            "phva_source": self.phva.source,
            "fhva_source": self.fhva.source,
            "n_atoms": self.phva.n_atoms,
            "n_phva_modes": self.phva.n_modes,
            "n_fhva_modes": self.fhva.n_modes,
            "phva_active_indices": self.phva.active_indices,
            "matched_modes": [
                {
                    "phva_mode": m.phva_mode_idx + 1,
                    "phva_freq_thz": round(m.phva_freq_thz, 6),
                    "fhva_mode": m.fhva_mode_idx + 1,
                    "fhva_freq_thz": round(m.fhva_freq_thz, 6),
                    "delta_freq_thz": round(m.freq_delta_thz, 6),
                    "delta_freq_pct": round(m.freq_delta_pct, 4),
                    "mac_score": round(m.mac_score, 6),
                    "ipr_phva": round(m.phva_ipr, 6),
                    "n_eff_phva": round(m.phva_n_eff, 4),
                    "ipr_fhva": round(m.fhva_ipr, 6),
                    "n_eff_fhva": round(m.fhva_n_eff, 4),
                    "ambiguous": bool(m.ambiguous),
                    "P_j_phva": _arr(m.phva_P_j),
                    "P_j_fhva": _arr(m.fhva_P_j),
                }
                for m in self.matched
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(doc, fh, default_flow_style=False, allow_unicode=True)

        print(f"  [ModeParticipation] Saved full result   → {os.path.relpath(path)}")


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------

def _format_top_atoms(
    P_j: np.ndarray,
    top_n: int,
    symbols: Optional[Sequence[str]],
) -> str:
    """Return a compact string listing the top-N atoms by P_j.

    Example output:  ``Si(12):0.472  N(0):0.308  H(5):0.118``
    """
    order = np.argsort(P_j)[::-1][:top_n]
    parts = []
    for idx in order:
        label = (
            f"{symbols[idx]}({idx})" if symbols and idx < len(symbols)
            else f"atom{idx}"
        )
        parts.append(f"{label}:{P_j[idx]:.3f}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Public API: parse one qpoints.yaml
# ---------------------------------------------------------------------------

def parse_qpoints(path: str) -> QPointsData:
    """Parse a qpoints.yaml file into a :class:`QPointsData` container.

    Parameters
    ----------
    path : str
        Path to the qpoints.yaml file.

    Returns
    -------
    QPointsData
    """
    path = os.path.abspath(path)
    raw = _load_yaml_fast(path)

    if "phonon" not in raw:
        raise ValueError(f"[ModeParticipation] No 'phonon' key found in {path}")

    n_atoms: int = int(raw.get("natom", 0))
    bands: list = raw["phonon"][0].get("band", [])
    n_modes = len(bands)

    freqs = np.zeros(n_modes)
    eigs = np.zeros((n_modes, n_atoms, 3))

    for i, band in enumerate(bands):
        freqs[i] = float(band.get("frequency", 0.0))
        raw_eig = band.get("eigenvector")
        if raw_eig is not None:
            eigs[i] = _eig_from_band(raw_eig, n_atoms)

    # Auto-detect inactive (zero-padded) atoms:
    # An atom is active if ‖eig[mode, atom, :]‖ > 0 for at least one mode.
    atom_max_norm = np.sqrt(np.max(np.sum(eigs ** 2, axis=2), axis=0))  # (n_atoms,)
    active_mask = atom_max_norm > 1e-10

    return QPointsData(
        source=path,
        n_atoms=n_atoms,
        n_modes=n_modes,
        freqs_thz=freqs,
        eigs=eigs,
        active_mask=active_mask,
    )


# ---------------------------------------------------------------------------
# Public API: single-file analysis
# ---------------------------------------------------------------------------

def analyze_single(
    path: str,
    sort_by: str = "freq_desc",
) -> SingleAnalysisResult:
    """Compute P_j and IPR for every mode in one qpoints.yaml.

    Parameters
    ----------
    path : str
        Path to qpoints.yaml.
    sort_by : {"freq_desc", "freq_asc", "ipr_desc", "none"}
        Ordering of the returned records.

    Returns
    -------
    SingleAnalysisResult
    """
    data = parse_qpoints(path)

    records: List[SingleModeRecord] = []
    for i in range(data.n_modes):
        P_j = atomic_participation(data.eigs[i])
        ipr_val = ipr(P_j)
        n_eff = 1.0 / ipr_val if ipr_val > 1e-12 else float("inf")
        records.append(SingleModeRecord(
            mode_idx=i,
            freq_thz=float(data.freqs_thz[i]),
            P_j=P_j,
            ipr_val=ipr_val,
            n_eff=n_eff,
        ))

    _sort_records(records, sort_by)
    return SingleAnalysisResult(source=data, records=records)


def _sort_records(records: List[SingleModeRecord], sort_by: str) -> None:
    if sort_by == "freq_desc":
        records.sort(key=lambda r: r.freq_thz, reverse=True)
    elif sort_by == "freq_asc":
        records.sort(key=lambda r: r.freq_thz)
    elif sort_by == "ipr_desc":
        records.sort(key=lambda r: r.ipr_val, reverse=True)
    # "none": leave in original (mode-index) order


# ---------------------------------------------------------------------------
# Public API: PHVA ↔ FHVA comparison
# ---------------------------------------------------------------------------

def compare_phva_fhva(
    phva_yaml: str,
    fhva_yaml: str,
    mac_threshold: float = 0.0,
    sort_by: str = "fhva_freq_desc",
) -> ModeComparisonResult:
    """Compare PHVA and FHVA modes using MAC-based matching.

    For every PHVA mode the FHVA mode with the highest MAC score is
    selected as the physical counterpart.  Frequency ordering is not used.

    Parameters
    ----------
    phva_yaml : str
        Path to qpoints.yaml produced by a PHVA (partial Hessian) run.
    fhva_yaml : str
        Path to qpoints.yaml produced by a FHVA (full Hessian) run.
    mac_threshold : float
        Pairs with MAC < threshold are still included but flagged.
        Set to 0.0 to include all pairs regardless of quality.
    sort_by : {"fhva_freq_desc", "fhva_freq_asc", "mac_desc", "phva_mode"}
        Ordering of matched pairs in the result.

    Returns
    -------
    ModeComparisonResult
    """
    phva = parse_qpoints(phva_yaml)
    fhva = parse_qpoints(fhva_yaml)

    _check_atom_counts(phva, fhva)

    # ----------------------------------------------------------------
    # MAC matrix  (n_phva × n_fhva)
    # ----------------------------------------------------------------
    mac_mat = mac_matrix(phva.eigs, fhva.eigs)  # (n_phva, n_fhva)

    # ----------------------------------------------------------------
    # Greedy best-match: for each PHVA mode, pick the FHVA mode with
    # highest MAC.  Track which FHVA modes are claimed more than once.
    # ----------------------------------------------------------------
    best_fhva = np.argmax(mac_mat, axis=1)  # (n_phva,) — best FHVA idx per PHVA mode
    best_mac  = mac_mat[np.arange(len(best_fhva)), best_fhva]

    # Flag ambiguous pairs (multiple PHVA modes pointing to same FHVA mode)
    from collections import Counter
    claim_count = Counter(best_fhva.tolist())

    matched: List[MatchedMode] = []
    for pi in range(phva.n_modes):
        fi = int(best_fhva[pi])
        mac_score = float(best_mac[pi])

        pf = float(phva.freqs_thz[pi])
        ff = float(fhva.freqs_thz[fi])
        delta_thz = pf - ff
        delta_pct = (
            100.0 * delta_thz / abs(ff) if abs(ff) > 1e-6 else float("nan")
        )

        P_phva = atomic_participation(phva.eigs[pi])
        P_fhva = atomic_participation(fhva.eigs[fi])

        ipr_p = ipr(P_phva)
        ipr_f = ipr(P_fhva)

        matched.append(MatchedMode(
            phva_mode_idx=pi,
            phva_freq_thz=pf,
            phva_P_j=P_phva,
            phva_ipr=ipr_p,
            phva_n_eff=1.0 / ipr_p if ipr_p > 1e-12 else float("inf"),
            fhva_mode_idx=fi,
            fhva_freq_thz=ff,
            fhva_P_j=P_fhva,
            fhva_ipr=ipr_f,
            fhva_n_eff=1.0 / ipr_f if ipr_f > 1e-12 else float("inf"),
            mac_score=mac_score,
            freq_delta_thz=delta_thz,
            freq_delta_pct=delta_pct,
            ambiguous=(claim_count[fi] > 1),
        ))

    _sort_matched(matched, sort_by)

    return ModeComparisonResult(
        phva=phva,
        fhva=fhva,
        matched=matched,
        mac_mat=mac_mat,
    )


def _check_atom_counts(phva: QPointsData, fhva: QPointsData) -> None:
    """Warn (not error) when n_atoms differs between the two yamls."""
    if phva.n_atoms != fhva.n_atoms:
        import warnings
        warnings.warn(
            f"[ModeParticipation] Atom count mismatch: "
            f"PHVA has {phva.n_atoms}, FHVA has {fhva.n_atoms}.  "
            f"MAC computation uses the smaller dimension (zero-padding assumed).",
            UserWarning,
            stacklevel=3,
        )


def _sort_matched(matched: List[MatchedMode], sort_by: str) -> None:
    if sort_by == "fhva_freq_desc":
        matched.sort(key=lambda m: m.fhva_freq_thz, reverse=True)
    elif sort_by == "fhva_freq_asc":
        matched.sort(key=lambda m: m.fhva_freq_thz)
    elif sort_by == "mac_desc":
        matched.sort(key=lambda m: m.mac_score, reverse=True)
    elif sort_by == "phva_mode":
        matched.sort(key=lambda m: m.phva_mode_idx)


# ---------------------------------------------------------------------------
# Convenience: MAC matrix for external use
# ---------------------------------------------------------------------------

def build_mac_matrix(phva_yaml: str, fhva_yaml: str) -> Tuple[np.ndarray, QPointsData, QPointsData]:
    """Return the raw MAC matrix without constructing MatchedMode objects.

    Useful when you want to inspect the full matrix (e.g. via a heatmap)
    before committing to a particular matching strategy.

    Returns
    -------
    mac_mat : ndarray, shape (n_phva_modes, n_fhva_modes)
    phva    : QPointsData
    fhva    : QPointsData
    """
    phva = parse_qpoints(phva_yaml)
    fhva = parse_qpoints(fhva_yaml)
    return mac_matrix(phva.eigs, fhva.eigs), phva, fhva
