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
from matplotlib.collections import LineCollection

# Ensure package root is on sys.path when run directly
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ase.io import read as ase_read
from autoflow_srxn.vibrational.mode_participation_analyzer import (
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
# Figure 2 — Raw MAC matrix heatmap
# ---------------------------------------------------------------------------

def fig1_mac_heatmap(result: ModeComparisonResult, out_dir: str):
    """Heatmap of the full (n_phva × n_fhva) MAC matrix."""
    mac_mat = result.mac_mat   # (n_phva, n_fhva)
    n_p, n_f = mac_mat.shape

    # 2. Raw MAC matrix heatmap
    # Sort modes by frequency (high → low) for display
    phva_order = np.argsort(result.phva.freqs_thz)[::-1]
    fhva_order = np.argsort(result.fhva.freqs_thz)[::-1]
    mat_sorted = mac_mat[np.ix_(phva_order, fhva_order)]

    # Do NOT subsample destructively — we want to see every potential match.
    # Instead, we use a wide figure to accommodate FHVA's high dimensionality.
    fig, ax = plt.subplots(figsize=(15, 6))
    cmap_hm = cm.RdYlGn
    
    # Use aspect='auto' because n_f >> n_p
    im = ax.imshow(
        mat_sorted, aspect="auto", origin="upper",
        cmap=cmap_hm, vmin=0, vmax=1,
        interpolation="nearest",
    )

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.015)
    cb.set_label("MAC score", fontsize=10)

    # Axis limits: strictly 0 to N-1
    ax.set_xlim(-0.5, n_f - 0.5)
    ax.set_ylim(n_p - 0.5, -0.5)

    # Mark the best-match FHVA column for each PHVA row
    best_indices = np.argmax(mat_sorted, axis=1)
    ax.scatter(best_indices, np.arange(n_p), color="cyan", s=8, alpha=0.7, edgecolors="none", label="Best match")

    # FHVA axis on top
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("FHVA mode index (freq↓)", fontsize=11, labelpad=8)
    ax.set_ylabel("PHVA mode index (freq↓)", fontsize=11)
    ax.set_title("Raw MAC Matrix  (PHVA × FHVA)", fontsize=12, fontweight="bold", pad=40)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig1_mac_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 2 — Frequency parity plot (MAC-based pairing)
# ---------------------------------------------------------------------------

def fig2_parity(result: ModeComparisonResult, out_dir: str):
    """Scatter: FHVA freq (x) vs PHVA freq (y), coloured by MAC score."""
    matched = result.matched

    fhva_f = np.array([m.fhva_freq_thz for m in matched])
    phva_f = np.array([m.phva_freq_thz for m in matched])
    mac    = np.array([m.mac_score     for m in matched])
    delta  = np.array([m.freq_delta_thz for m in matched])

    # Standardize frequency mask (> 0.1 THz) for MAE calculation
    real_mask = fhva_f > 0.1
    mae = np.mean(np.abs(delta[real_mask])) if np.any(real_mask) else 0.0

    fig, ax = plt.subplots(figsize=(7, 6.5))

    cmap = cm.RdYlGn
    norm = mcolors.Normalize(vmin=0, vmax=1)

    sc = ax.scatter(
        fhva_f[real_mask], phva_f[real_mask],
        c=mac[real_mask], cmap=cmap, norm=norm,
        s=30, edgecolors="none", alpha=0.85, zorder=3,
        label="Real modes"
    )

    # Parity line
    all_f = np.concatenate([fhva_f[real_mask], phva_f[real_mask]])
    if len(all_f):
        lim = [-2, all_f.max() * 1.05]
        ax.plot(lim, lim, "k--", lw=1.2, alpha=0.7, label="y = x  (perfect)")
        ax.set_xlim(lim); ax.set_ylim(lim)

    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("MAC score", fontsize=10)

    # Calculate R2 for positive frequencies
    f_ref_r = fhva_f[real_mask]
    f_test_r = phva_f[real_mask]
    if len(f_ref_r) > 1:
        corr_matrix = np.corrcoef(f_ref_r, f_test_r)
        r2 = corr_matrix[0, 1]**2
    else:
        r2 = 0.0

    if real_mask.any():
        d_real = np.array([m.freq_delta_thz for m in matched if m.fhva_freq_thz > 0.1])
        rmse = np.sqrt(np.mean(d_real**2))
        ax.text(
            0.04, 0.96,
            f"MAE  = {mae:.3f} THz\n"
            f"RMSE = {rmse:.3f} THz\n"
            f"R$^2$   = {r2:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.4"),
        )

    _apply_style(ax,
        xlabel="FHVA Frequency (THz)",
        ylabel="PHVA Frequency (THz)",
        title=f"Frequency Parity (MAE = {mae:.3f} THz)"
    )
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig2_parity.png")


# ---------------------------------------------------------------------------
# Figure 3 — Frequency residuals
# ---------------------------------------------------------------------------

def fig3_residuals(result: ModeComparisonResult, out_dir: str):
    """Δfreq = PHVA – FHVA vs FHVA freq.  Colour = MAC score."""
    matched = result.matched
    fhva_f  = np.array([m.fhva_freq_thz  for m in matched])
    delta   = np.array([m.freq_delta_thz  for m in matched])
    mac     = np.array([m.mac_score       for m in matched])

    real_mask = fhva_f > 0.1
    mae = np.mean(np.abs(delta[real_mask])) if np.any(real_mask) else 0.0
    
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.RdYlGn

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        fhva_f[real_mask], delta[real_mask],
        c=mac[real_mask], cmap=cmap, norm=norm,
        s=35, edgecolors="k", lw=0.5, alpha=0.8
    )
    ax.axhline(0, color="k", lw=1, ls="--")
    # Shade ±0.5 THz band
    ax.axhspan(-0.5, 0.5, color="green", alpha=0.06, label="±0.5 THz")
    
    _apply_style(ax,
        xlabel="FHVA Frequency (THz)",
        ylabel="PHVA − FHVA (THz)",
        title=f"Frequency Residuals (MAE = {mae:.3f} THz)"
    )
    ax.legend(fontsize=9, loc="upper left")
    
    # Colorbar
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("MAC score", fontsize=10)
    
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
    real     = fhva_f > 0.1

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.RdYlGn

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: IPR scatter PHVA vs FHVA ---
    ax = axes[0]
    sc = ax.scatter(
        ipr_f[real], ipr_p[real],
        c=mac[real], cmap=cmap, norm=norm,
        s=30, edgecolors="k", lw=0.5, alpha=0.8
    )
    lim = [0, max(ipr_p[real].max(), ipr_f[real].max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=1.2, alpha=0.7, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    _apply_style(ax,
        xlabel="IPR (FHVA)",
        ylabel="IPR (PHVA)",
        title="IPR Comparison"
    )
    
    # Calculate R2 for IPR
    ipr_f_r = ipr_f[real]
    ipr_p_r = ipr_p[real]
    if len(ipr_f_r) > 1:
        corr_matrix = np.corrcoef(ipr_f_r, ipr_p_r)
        r2 = corr_matrix[0, 1]**2
    else:
        r2 = 0.0
    
    ax.text(
        0.04, 0.96,
        f"R$^2$ = {r2:.3f}",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.4"),
    )
    ax.legend(fontsize=9, loc="lower right")

    # --- Right: N_eff vs frequency ---
    ax = axes[1]
    ax.scatter(fhva_f[real], neff_f[real],
               s=30, alpha=0.7, color="#2196f3", edgecolors="k", lw=0.5,
               label="FHVA  N$_{eff}$", zorder=3)
    ax.scatter(fhva_f[real], neff_p[real],
               s=30, alpha=0.7, color="#ff7043", edgecolors="k", lw=0.5, marker="s",
               label="PHVA  N$_{eff}$", zorder=3)
    
    # Set axis limits with margin
    max_f = fhva_f[real].max()
    max_neff = max(neff_f[real].max(), neff_p[real].max())
    ax.set_xlim(0, max_f * 1.05)
    ax.set_ylim(0, max_neff * 1.1)
    
    _apply_style(ax,
        xlabel="FHVA Frequency (THz)",
        ylabel="N$_{eff}$ = 1 / IPR  (atoms)",
        title="Effective Number of Participating Atoms"
    )
    ax.legend(fontsize=9, loc="upper right")

    cb = fig.colorbar(sc, ax=axes[0], pad=0.02)
    cb.set_label("MAC score", fontsize=10)

    fig.tight_layout()
    _save(fig, "fig4_ipr_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5 — Localisation spectrum (IPR vs frequency)
# ---------------------------------------------------------------------------

def fig5_localization_spectrum(result: ModeComparisonResult, out_dir: str, sigma: float = 0.3):
    """
    Advanced 'Spectral Localization Density' plot.
    1. Gaussian smearing of IPR vs Frequency to show density.
    2. PHVA curve color-coded by MAC score.
    3. Cumulative error panel to justify HTST reliability.
    """
    from autoflow_srxn.vibrational.mode_participation_analyzer import atomic_participation, ipr

    # --- Data Extraction ---
    fhva_data = result.fhva
    phva_data = result.phva
    matched   = result.matched

    f_fhva = fhva_data.freqs_thz
    i_fhva = np.array([ipr(atomic_participation(fhva_data.eigs[i])) for i in range(fhva_data.n_modes)])
    
    f_phva = phva_data.freqs_thz
    i_phva = np.array([ipr(atomic_participation(phva_data.eigs[i])) for i in range(phva_data.n_modes)])
    
    # Map MAC scores to PHVA modes (best match per PHVA mode)
    mac_phva = np.zeros(phva_data.n_modes)
    for m in matched:
        mac_phva[m.phva_mode_idx] = m.mac_score

    # Only consider real modes
    mask_f = f_fhva > 0.5
    mask_p = f_phva > 0.5
    f_fhva, i_fhva = f_fhva[mask_f], i_fhva[mask_f]
    f_phva, i_phva, m_phva = f_phva[mask_p], i_phva[mask_p], mac_phva[mask_p]

    # --- Gaussian Smearing ---
    f_min = min(f_fhva.min(), f_phva.min()) - 5
    f_max = max(f_fhva.max(), f_phva.max()) + 5
    grid = np.linspace(f_min, f_max, 1000)

    def get_smeared_ipr(freqs, iprs, macs=None):
        # Resulting density curve
        density = np.zeros_like(grid)
        # Resulting weighted average MAC curve
        weighted_mac = np.zeros_like(grid)
        
        for f, val, *m in zip(freqs, iprs, macs if macs is not None else [None]*len(freqs)):
            kernel = np.exp(-0.5 * ((grid - f) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
            contrib = val * kernel
            density += contrib
            if m[0] is not None:
                weighted_mac += m[0] * kernel # We'll normalize this by density later
        
        if macs is not None:
            # Avoid division by zero where density is very low
            safe_den = np.where(density > 1e-6, density, 1.0)
            # Actually, normalize by the 'sum of kernels' to get avg MAC, not 'IPR-weighted' density
            sum_kernels = np.zeros_like(grid)
            avg_mac = np.zeros_like(grid)
            for f, m in zip(freqs, macs):
                kernel = np.exp(-0.5 * ((grid - f) / sigma)**2)
                sum_kernels += kernel
                avg_mac += m * kernel
            weighted_mac = np.where(sum_kernels > 1e-3, avg_mac / sum_kernels, 0.0)

        return density, weighted_mac

    dens_fhva, _ = get_smeared_ipr(f_fhva, i_fhva)
    dens_phva, mac_grid = get_smeared_ipr(f_phva, i_phva, m_phva)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Main Spectrum Panel
    # FHVA Background
    ax.plot(grid, dens_fhva, color="#999999", lw=1.5, ls="--", alpha=0.6, label="FHVA (Ref) IPR Density")
    ax.fill_between(grid, dens_fhva, color="#999999", alpha=0.1)

    # PHVA Color-coded Curve
    points = np.array([grid, dens_phva]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = mcolors.Normalize(vmin=0, vmax=1) 
    cmap = cm.RdYlGn
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5, zorder=3)
    lc.set_array(mac_grid)
    ax.add_collection(lc)
    
    # Representative line for legend
    ax.plot([], [], color=cmap(1.0), lw=2.5, label="PHVA (MAC-weighted)")

    # Colorbar for MAC
    cb = fig.colorbar(lc, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("Weighted Avg MAC Score", fontsize=10)

    # Labels and Style
    ax.set_ylabel("Smeared IPR Density", fontsize=11)
    ax.set_title(f"Localization Spectrum (Gaussian $\sigma$ = {sigma} THz)", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(f_min + 2, f_max - 2)
    ax.set_ylim(0, max(dens_fhva.max(), dens_phva.max()) * 1.15)

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
    # Sort by FHVA frequency (ascending: low → high)
    real.sort(key=lambda m: m.fhva_freq_thz)

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

        ax.set_ylabel(r"$\sum P_j$ (" + side + ")", fontsize=11)
        ax.set_ylim(0, 1.1)
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
    fig.tight_layout()
    _save(fig, "fig6_element_participation.png")


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
    fig1_mac_heatmap(result, OUT_DIR)
    fig2_parity(result, OUT_DIR)
    fig3_residuals(result, OUT_DIR)
    fig4_ipr(result, OUT_DIR)
    
    # Generate Final Fig 5
    print("  Generating Fig 5 (sigma=0.3 THz) ...")
    fig5_localization_spectrum(result, OUT_DIR, sigma=0.3)

    fig6_element_participation(result, symbols, OUT_DIR)
    
    print("\nAll outputs saved to ", os.path.abspath(OUT_DIR))


if __name__ == "__main__":
    main()
