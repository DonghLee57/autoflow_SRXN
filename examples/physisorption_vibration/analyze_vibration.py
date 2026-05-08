"""
analyze_vibration.py  —  PHVA vs FHVA post-analysis and visualization
=======================================================================
Run from examples/physisorption_vibration/ after run_vibration.py:

    python analyze_vibration.py

Reads
-----
    results/fhva/qpoints.yaml
    results/phva/qpoints.yaml
    dipas_sio2_relaxed.vasp  (for element symbols)

Outputs (all saved to results/)
---------------------------------
    mode_pairs.csv              — mode-level summary (MAC, freq, IPR, …)
    participation.csv           — per-atom P_j for each matched pair
    comparison_full.yaml        — full result including P_j arrays

    fig1_parity.png             — Frequency parity plot (MAC-based coloring)
    fig2_mac_heatmap.png        — Raw MAC matrix heatmap
    fig3_residuals.png          — Frequency residuals (PHVA–FHVA) vs FHVA freq
    fig4_ipr_comparison.png     — IPR scatter (PHVA vs FHVA) + N_eff vs freq
    fig5_localization_spectrum.png — IPR vs frequency for both methods
    fig6_element_participation.png — Element-resolved P_j per mode
    fig7_top_mode_Pj.png        — Per-atom P_j bar chart for best-MAC modes
    fig8_mac_score_distribution.png — MAC score + Δfreq distribution
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

# Ensure package root is on sys.path when run directly
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ase.io import read as ase_read
from autoflow_srxn.analysis.mode_participation_analyzer import (
    compare_phva_fhva,
    analyze_single,
    ModeComparisonResult,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
os.chdir(_here)

FHVA_YAML    = os.path.join("results", "fhva", "qpoints.yaml")
PHVA_YAML    = os.path.join("results", "phva", "qpoints.yaml")
STRUCT_PATH  = "dipas_sio2_relaxed.vasp"
OUT_DIR      = "results"

ELEM_COLORS = {
    "H":  "#a8d8ea",   # light blue
    "C":  "#555555",   # dark gray
    "N":  "#3d85c8",   # blue
    "O":  "#e06c75",   # red-pink
    "Si": "#f4a261",   # orange
}

# Good MAC threshold: modes below this are "poorly captured by PHVA"
MAC_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_files():
    missing = [p for p in (FHVA_YAML, PHVA_YAML) if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing result file(s):")
        for p in missing:
            print(f"  {p}")
        print("Run  python run_vibration.py  first.")
        sys.exit(1)


def _elem_color(sym: str) -> str:
    return ELEM_COLORS.get(sym, "#999999")


def _save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.relpath(path)}")


def _apply_style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.8)


# ---------------------------------------------------------------------------
# Figure 1 — Frequency parity plot (MAC-based pairing)
# ---------------------------------------------------------------------------

def fig1_parity(result: ModeComparisonResult, out_dir: str):
    """Scatter: FHVA freq (x) vs PHVA freq (y), coloured by MAC score."""
    matched = result.matched

    fhva_f = np.array([m.fhva_freq_thz for m in matched])
    phva_f = np.array([m.phva_freq_thz for m in matched])
    mac    = np.array([m.mac_score     for m in matched])

    # Separate real from imaginary (any mode with FHVA freq < 0)
    real_mask = fhva_f > 0.5
    imag_mask = ~real_mask

    fig, ax = plt.subplots(figsize=(7, 6.5))

    cmap = cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    sc = ax.scatter(
        fhva_f[real_mask], phva_f[real_mask],
        c=mac[real_mask], cmap=cmap, norm=norm,
        s=30, edgecolors="none", alpha=0.85, zorder=3,
        label="Real modes"
    )
    if imag_mask.any():
        ax.scatter(
            fhva_f[imag_mask], phva_f[imag_mask],
            c=mac[imag_mask], cmap=cmap, norm=norm,
            s=40, marker="v", edgecolors="k", linewidths=0.5, alpha=0.8, zorder=3,
            label="Imaginary modes"
        )

    # Parity line
    all_f = np.concatenate([fhva_f[real_mask], phva_f[real_mask]])
    if len(all_f):
        lim = [-2, all_f.max() * 1.05]
        ax.plot(lim, lim, "k--", lw=1.2, alpha=0.7, label="y = x  (perfect)")
        ax.set_xlim(lim); ax.set_ylim(lim)

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("MAC score", fontsize=10)

    # Stats annotation (real modes only)
    good = [m for m in matched if m.fhva_freq_thz > 0.5 and m.mac_score >= MAC_THRESHOLD]
    if good:
        delta = np.array([m.freq_delta_thz for m in good])
        mae  = np.mean(np.abs(delta))
        rmse = np.sqrt(np.mean(delta**2))
        ax.text(
            0.04, 0.96,
            f"MAC ≥ {MAC_THRESHOLD:.1f}:  {len(good)} / {real_mask.sum()} real modes\n"
            f"MAE  = {mae:.3f} THz\nRMSE = {rmse:.3f} THz",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.4"),
        )

    _apply_style(ax,
        xlabel="FHVA Frequency (THz)",
        ylabel="PHVA Frequency (THz)",
        title="Frequency Parity: PHVA vs FHVA  (MAC-based pairing)"
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig1_parity.png")


# ---------------------------------------------------------------------------
# Figure 2 — Raw MAC matrix heatmap
# ---------------------------------------------------------------------------

def fig2_mac_heatmap(result: ModeComparisonResult, out_dir: str):
    """Heatmap of the full (n_phva × n_fhva) MAC matrix."""
    mac_mat = result.mac_mat   # (n_phva, n_fhva)
    n_p, n_f = mac_mat.shape

    # Sort modes by frequency (high → low) for display
    phva_order = np.argsort(result.phva.freqs_thz)[::-1]
    fhva_order = np.argsort(result.fhva.freqs_thz)[::-1]
    mat_sorted = mac_mat[np.ix_(phva_order, fhva_order)]

    # Subsample if very large (> 200 modes per axis)
    step_p = max(1, n_p // 150)
    step_f = max(1, n_f // 150)
    mat_show = mat_sorted[::step_p, ::step_f]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(
        mat_show, aspect="auto", origin="upper",
        cmap="hot", vmin=0, vmax=1,
        interpolation="nearest",
    )
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("MAC score", fontsize=10)

    # Mark the best-match column for each PHVA mode (argmax per row)
    best_fhva = np.argmax(mac_mat, axis=1)  # in original indexing
    # Map to sorted/subsampled display coords
    phva_rank_of = {orig: rank for rank, orig in enumerate(phva_order)}
    fhva_rank_of = {orig: rank for rank, orig in enumerate(fhva_order)}

    xs, ys = [], []
    for pi, fi in enumerate(best_fhva):
        yr = phva_rank_of[pi] // step_p
        xr = fhva_rank_of[fi] // step_f
        xs.append(xr); ys.append(yr)
    ax.scatter(xs, ys, s=6, c="cyan", alpha=0.5, zorder=4, label="Best match")

    ax.set_xlabel("FHVA mode index (freq↓)", fontsize=11)
    ax.set_ylabel("PHVA mode index (freq↓)", fontsize=11)
    ax.set_title("Raw MAC Matrix  (PHVA × FHVA)", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig2_mac_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 3 — Frequency residuals
# ---------------------------------------------------------------------------

def fig3_residuals(result: ModeComparisonResult, out_dir: str):
    """Δfreq = PHVA – FHVA vs FHVA freq.  Colour = MAC score."""
    matched = result.matched
    fhva_f  = np.array([m.fhva_freq_thz  for m in matched])
    delta   = np.array([m.freq_delta_thz  for m in matched])
    delta_p = np.array([m.freq_delta_pct  for m in matched])
    mac     = np.array([m.mac_score       for m in matched])

    real_mask = fhva_f > 0.5
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.RdYlGn

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: absolute Δfreq ---
    ax = axes[0]
    sc = ax.scatter(
        fhva_f[real_mask], delta[real_mask],
        c=mac[real_mask], cmap=cmap, norm=norm,
        s=28, edgecolors="none", alpha=0.8
    )
    ax.axhline(0, color="k", lw=1, ls="--")
    # Shade ±0.5 THz band
    ax.axhspan(-0.5, 0.5, color="green", alpha=0.06, label="±0.5 THz")
    _apply_style(ax,
        xlabel="FHVA Frequency (THz)",
        ylabel="PHVA − FHVA  (THz)",
        title="Frequency Residuals"
    )
    ax.legend(fontsize=9)

    # --- Right: relative Δfreq % ---
    ax = axes[1]
    ax.scatter(
        fhva_f[real_mask], delta_p[real_mask],
        c=mac[real_mask], cmap=cmap, norm=norm,
        s=28, edgecolors="none", alpha=0.8
    )
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.axhspan(-5, 5, color="green", alpha=0.06, label="±5 %")
    _apply_style(ax,
        xlabel="FHVA Frequency (THz)",
        ylabel="(PHVA − FHVA) / |FHVA|  (%)",
        title="Relative Frequency Residuals"
    )
    ax.legend(fontsize=9)

    cb = fig.colorbar(sc, ax=axes, pad=0.02, fraction=0.02)
    cb.set_label("MAC score", fontsize=10)

    fig.suptitle("PHVA Frequency Accuracy vs FHVA Reference", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig3_residuals.png")


# ---------------------------------------------------------------------------
# Figure 4 — IPR comparison scatter + N_eff vs frequency
# ---------------------------------------------------------------------------

def fig4_ipr(result: ModeComparisonResult, out_dir: str):
    """Two-panel: (left) PHVA IPR vs FHVA IPR; (right) N_eff vs frequency."""
    matched = result.matched
    fhva_f   = np.array([m.fhva_freq_thz for m in matched])
    ipr_p    = np.array([m.phva_ipr       for m in matched])
    ipr_f    = np.array([m.fhva_ipr       for m in matched])
    neff_p   = np.array([m.phva_n_eff     for m in matched])
    neff_f   = np.array([m.fhva_n_eff     for m in matched])
    mac      = np.array([m.mac_score      for m in matched])
    real     = fhva_f > 0.5

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.RdYlGn

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: IPR scatter PHVA vs FHVA ---
    ax = axes[0]
    sc = ax.scatter(
        ipr_f[real], ipr_p[real],
        c=mac[real], cmap=cmap, norm=norm,
        s=30, edgecolors="none", alpha=0.8
    )
    lim = [0, max(ipr_p[real].max(), ipr_f[real].max()) * 1.08]
    ax.plot(lim, lim, "k--", lw=1.2, alpha=0.7, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    _apply_style(ax,
        xlabel="IPR (FHVA)",
        ylabel="IPR (PHVA)",
        title="IPR Comparison\n(PHVA tends to overestimate localisation)"
    )
    ax.legend(fontsize=9)

    # Annotation: fraction of modes where PHVA IPR > FHVA IPR
    frac_over = (ipr_p[real] > ipr_f[real]).mean()
    ax.text(
        0.04, 0.96,
        f"PHVA IPR > FHVA IPR: {frac_over:.0%} of real modes",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.4"),
    )

    # --- Right: N_eff vs frequency ---
    ax = axes[1]
    ax.scatter(fhva_f[real], neff_f[real],
               s=25, alpha=0.7, color="#2196f3", edgecolors="none",
               label="FHVA  N$_{eff}$", zorder=3)
    ax.scatter(fhva_f[real], neff_p[real],
               s=25, alpha=0.7, color="#ff7043", edgecolors="none", marker="s",
               label="PHVA  N$_{eff}$", zorder=3)
    _apply_style(ax,
        xlabel="FHVA Frequency (THz)",
        ylabel="N$_{eff}$ = 1 / IPR  (atoms)",
        title="Effective Number of Participating Atoms"
    )
    ax.legend(fontsize=9)

    cb = fig.colorbar(sc, ax=axes[0], pad=0.02)
    cb.set_label("MAC score", fontsize=10)

    fig.suptitle("Localisation: IPR and N$_{eff}$  (PHVA vs FHVA)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig4_ipr_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5 — Localisation spectrum (IPR vs frequency)
# ---------------------------------------------------------------------------

def fig5_localization_spectrum(result: ModeComparisonResult, out_dir: str):
    """IPR vs frequency for all modes in FHVA (background) and PHVA (overlay)."""
    fhva_data = result.fhva
    phva_data = result.phva
    matched   = result.matched

    from autoflow_srxn.analysis.mode_participation_analyzer import atomic_participation, ipr

    # Compute IPR for every FHVA and PHVA mode (all modes, not just matched)
    fhva_freqs = fhva_data.freqs_thz
    fhva_iprs  = np.array([ipr(atomic_participation(fhva_data.eigs[i]))
                            for i in range(fhva_data.n_modes)])
    phva_freqs = phva_data.freqs_thz
    phva_iprs  = np.array([ipr(atomic_participation(phva_data.eigs[i]))
                            for i in range(phva_data.n_modes)])

    # MAC score for each PHVA mode (best match)
    mac_per_phva = np.zeros(phva_data.n_modes)
    for m in matched:
        mac_per_phva[m.phva_mode_idx] = m.mac_score

    fig, ax = plt.subplots(figsize=(10, 5))

    # FHVA: all modes
    real_f = fhva_freqs > 0.5
    ax.scatter(fhva_freqs[real_f], fhva_iprs[real_f],
               s=20, alpha=0.5, color="#2196f3", edgecolors="none",
               label=f"FHVA  ({real_f.sum()} real modes)", zorder=2)

    # PHVA: colour by MAC quality
    real_p = phva_freqs > 0.5
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.RdYlGn
    sc = ax.scatter(
        phva_freqs[real_p], phva_iprs[real_p],
        c=mac_per_phva[real_p], cmap=cmap, norm=norm,
        s=35, edgecolors="k", linewidths=0.4, marker="D", alpha=0.9, zorder=3,
        label=f"PHVA  ({real_p.sum()} real modes,  coloured by MAC)"
    )

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Best-match MAC (PHVA→FHVA)", fontsize=10)

    # Annotate 1/N_active for PHVA
    n_active = phva_data.n_active
    if n_active > 0:
        ax.axhline(1.0 / n_active, color="orange", lw=1, ls="--",
                   label=f"1/N$_{{active}}$ = 1/{n_active}  (PHVA max delocal.)")

    _apply_style(ax,
        xlabel="Frequency (THz)",
        ylabel="IPR = Σ P$_j²$",
        title="Localisation Spectrum: IPR vs Frequency"
    )
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig5_localization_spectrum.png")


# ---------------------------------------------------------------------------
# Figure 6 — Element-resolved participation per mode
# ---------------------------------------------------------------------------

def fig6_element_participation(
    result: ModeComparisonResult,
    symbols: list,
    out_dir: str,
):
    """Stacked bar: element-summed P_j vs FHVA frequency for matched modes."""
    elements = sorted(set(symbols))
    elem_idx: dict[str, list[int]] = {
        el: [j for j, s in enumerate(symbols) if s == el]
        for el in elements
    }
    n_atoms = result.fhva.n_atoms

    matched = result.matched
    real    = [m for m in matched if m.fhva_freq_thz > 0.5]
    # Sort by FHVA frequency (descending)
    real.sort(key=lambda m: m.fhva_freq_thz, reverse=True)

    # Subsample if too many modes
    step = max(1, len(real) // 80)
    real = real[::step]

    fhva_f = np.array([m.fhva_freq_thz for m in real])

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for ax_idx, (ax, side, P_key) in enumerate(zip(
        axes,
        ["FHVA", "PHVA"],
        ["fhva_P_j", "phva_P_j"],
    )):
        bottom = np.zeros(len(real))
        for el in elements:
            indices = elem_idx[el]
            if not indices:
                continue
            # Sum P_j over all atoms of this element for each matched mode
            vals = np.array([
                sum(getattr(m, P_key)[j] for j in indices if j < n_atoms)
                for m in real
            ])
            ax.bar(
                range(len(real)), vals, bottom=bottom,
                color=_elem_color(el), label=el, width=1.0, align="center",
            )
            bottom += vals

        ax.set_ylabel(f"Σ P_j  ({side})", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{side} element-resolved participation", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9, ncol=len(elements))

        # x-tick: every ~10th mode labelled with FHVA freq
        tick_step = max(1, len(real) // 15)
        ax.set_xticks(range(0, len(real), tick_step))
        ax.set_xticklabels(
            [f"{fhva_f[i]:.1f}" for i in range(0, len(real), tick_step)],
            rotation=45, ha="right",
        )

    axes[-1].set_xlabel("FHVA Frequency (THz)", fontsize=11)
    fig.suptitle(
        "Element-resolved Energy Participation  (top = FHVA ref, bottom = PHVA)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig6_element_participation.png")


# ---------------------------------------------------------------------------
# Figure 7 — Per-atom P_j bar chart for best-matched modes
# ---------------------------------------------------------------------------

def fig7_top_mode_Pj(
    result: ModeComparisonResult,
    symbols: list,
    n_modes: int = 8,
    out_dir: str = "",
):
    """Side-by-side PHVA vs FHVA P_j bar charts for the highest-MAC real modes."""
    matched = result.matched
    real    = [m for m in matched
               if m.fhva_freq_thz > 0.5 and m.mac_score >= MAC_THRESHOLD]
    real.sort(key=lambda m: m.mac_score, reverse=True)
    top     = real[:n_modes]

    if not top:
        print("  [fig7] No modes with MAC ≥ threshold — skipping.")
        return

    ncols = 2
    nrows = len(top)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.4 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    n_atoms = result.fhva.n_atoms
    atom_idx = np.arange(n_atoms)
    colors   = [_elem_color(s) for s in symbols]

    for row, m in enumerate(top):
        for col, (P_j, label) in enumerate([
            (m.fhva_P_j, f"FHVA  f={m.fhva_freq_thz:.2f} THz"),
            (m.phva_P_j, f"PHVA  f={m.phva_freq_thz:.2f} THz"),
        ]):
            ax = axes[row, col]
            # Only show atoms with P_j > 0.001
            mask = P_j > 0.001
            if mask.any():
                ax.bar(atom_idx[mask], P_j[mask],
                       color=[colors[i] for i in atom_idx[mask]],
                       edgecolor="none", width=1.0)
            ax.set_ylabel("P_j", fontsize=9)
            ax.set_title(f"{label}  |  MAC={m.mac_score:.3f}", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.set_xlim(-1, n_atoms)
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))

    # Legend patches
    from matplotlib.patches import Patch
    seen = list(dict.fromkeys(symbols))
    handles = [Patch(facecolor=_elem_color(el), label=el) for el in seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(seen),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Per-atom P_j for top-{len(top)} MAC-matched modes  (left=FHVA, right=PHVA)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    _save(fig, "fig7_top_mode_Pj.png")


# ---------------------------------------------------------------------------
# Figure 8 — MAC score & frequency error distributions
# ---------------------------------------------------------------------------

def fig8_distributions(result: ModeComparisonResult, out_dir: str):
    """Histograms: MAC score distribution and Δfreq distribution."""
    matched = result.matched
    real    = [m for m in matched if m.fhva_freq_thz > 0.5]
    if not real:
        return

    mac    = np.array([m.mac_score     for m in real])
    delta  = np.array([m.freq_delta_thz for m in real])
    delta_p= np.array([m.freq_delta_pct for m in real])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- MAC histogram ---
    ax = axes[0]
    bins = np.linspace(0, 1, 26)
    ax.hist(mac, bins=bins, color="#3d85c8", edgecolor="white", linewidth=0.5)
    ax.axvline(MAC_THRESHOLD, color="red", lw=1.5, ls="--",
               label=f"threshold={MAC_THRESHOLD}")
    ax.text(MAC_THRESHOLD + 0.02, ax.get_ylim()[1] * 0.95,
            f"{(mac >= MAC_THRESHOLD).mean():.0%} good",
            fontsize=9, color="red", va="top")
    _apply_style(ax,
        xlabel="MAC score", ylabel="Count",
        title="MAC Score Distribution"
    )
    ax.legend(fontsize=9)

    # --- Δfreq (THz) histogram ---
    ax = axes[1]
    lim = np.abs(delta).max() * 1.1
    bins_d = np.linspace(-lim, lim, 41)
    ax.hist(delta, bins=bins_d, color="#f4a261", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="k", lw=1.2, ls="--")
    ax.text(0.04, 0.96,
            f"MAE  = {np.mean(np.abs(delta)):.3f} THz\n"
            f"RMSE = {np.sqrt(np.mean(delta**2)):.3f} THz",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.4"))
    _apply_style(ax,
        xlabel="PHVA − FHVA  (THz)", ylabel="Count",
        title="Frequency Residual Distribution"
    )

    # --- Δfreq (%) histogram ---
    ax = axes[2]
    lim_p = min(np.nanpercentile(np.abs(delta_p), 98) * 1.3, 50)
    bins_p = np.linspace(-lim_p, lim_p, 41)
    ax.hist(np.clip(delta_p, -lim_p, lim_p), bins=bins_p,
            color="#e06c75", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="k", lw=1.2, ls="--")
    finite = delta_p[np.isfinite(delta_p)]
    if len(finite):
        ax.text(0.04, 0.96,
                f"Median = {np.median(finite):+.2f} %\n"
                f"σ = {np.std(finite):.2f} %",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.4"))
    _apply_style(ax,
        xlabel="(PHVA − FHVA) / |FHVA|  (%)", ylabel="Count",
        title="Relative Frequency Error Distribution"
    )

    fig.suptitle(
        "Mode Matching Quality: MAC and Frequency Accuracy",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "fig8_mac_score_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _check_files()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load element symbols from structure
    print(f"\nLoading structure:  {STRUCT_PATH}")
    atoms = ase_read(STRUCT_PATH)
    symbols = atoms.get_chemical_symbols()

    # ----------------------------------------------------------------
    # Run MAC-based comparison
    # ----------------------------------------------------------------
    print(f"Comparing:\n  FHVA: {FHVA_YAML}\n  PHVA: {PHVA_YAML}\n")
    result = compare_phva_fhva(PHVA_YAML, FHVA_YAML)
    result.print_summary(mac_threshold=MAC_THRESHOLD, top_atoms=4, symbols=symbols, max_rows=30)

    # ----------------------------------------------------------------
    # Save tabular data
    # ----------------------------------------------------------------
    result.save_csv(os.path.join(OUT_DIR, "mode_pairs.csv"), symbols=symbols)
    result.save_participation_csv(
        os.path.join(OUT_DIR, "participation.csv"),
        symbols=symbols,
        mac_threshold=MAC_THRESHOLD,
    )
    result.save_yaml(os.path.join(OUT_DIR, "comparison_full.yaml"))

    # Also run single-file analysis for both
    print("\n  Single-mode analysis (FHVA) …")
    fhva_single = analyze_single(FHVA_YAML)
    fhva_single.save_csv(os.path.join(OUT_DIR, "fhva_participation.csv"), symbols=symbols)

    print("  Single-mode analysis (PHVA) …")
    phva_single = analyze_single(PHVA_YAML)
    phva_single.save_csv(os.path.join(OUT_DIR, "phva_participation.csv"), symbols=symbols)

    # ----------------------------------------------------------------
    # Generate all figures
    # ----------------------------------------------------------------
    print("\nGenerating figures …")
    fig1_parity(result, OUT_DIR)
    fig2_mac_heatmap(result, OUT_DIR)
    fig3_residuals(result, OUT_DIR)
    fig4_ipr(result, OUT_DIR)
    fig5_localization_spectrum(result, OUT_DIR)
    fig6_element_participation(result, symbols, OUT_DIR)
    fig7_top_mode_Pj(result, symbols, n_modes=8, out_dir=OUT_DIR)
    fig8_distributions(result, OUT_DIR)

    print(f"\nAll outputs saved to  {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
