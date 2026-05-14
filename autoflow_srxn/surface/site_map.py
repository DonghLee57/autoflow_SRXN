"""Surface adsorption site map — top-view visualization.

Public API
----------
plot_adsorption_site_map(slab, sites, output_path, ...)
    Draw a top-view map of unique adsorption sites on any surface slab.
    Works for any element composition and cell geometry.

generate_and_plot_site_map(slab, symprec, output_path, ...)
    Convenience wrapper: runs site generation + deduplication + plot in
    one call.  Mirrors what AdsorptionWorkflowManager does internally so
    results are identical to what the workflow uses.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Site-type classifier
# ---------------------------------------------------------------------------

def _classify_site(xy: np.ndarray, surf_xy: np.ndarray) -> str:
    """Return 'top', 'bridge', or 'hollow' based on distances to surface atoms.

    Parameters
    ----------
    xy : (2,) array — Cartesian XY of the candidate site.
    surf_xy : (N, 2) array — Cartesian XY of all surface atoms.
    """
    dists = np.linalg.norm(surf_xy - xy, axis=1)
    dists_sorted = np.sort(dists)
    d1, d2 = dists_sorted[0], dists_sorted[1] if len(dists_sorted) > 1 else 999.

    if d1 < 0.15:
        return "top"
    # Bridge: two nearest neighbours at nearly the same distance
    if d2 < 2.5 and abs(d1 - d2) / max(d1, 1e-9) < 0.08:
        return "bridge"
    if d2 < 2.5:
        return "bridge"
    return "hollow"


# ---------------------------------------------------------------------------
# Sublattice colouring — group surface atoms by element and Z height
# ---------------------------------------------------------------------------

def _assign_sublattice_colors(
    surf_pos: np.ndarray,
    surf_syms: list[str],
    palette: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Assign one colour per unique (element, z-bucket) group.

    Returns
    -------
    atom_colors : list[str]  length == len(surf_pos)
    legend_labels : list[str] unique labels in order of first appearance
    """
    if palette is None:
        palette = [
            "#aaaaaa", "#f0d090", "#90c8f0", "#f09090",
            "#90f090", "#c890f0", "#f0c090", "#90f0c8",
        ]

    z_vals = surf_pos[:, 2]
    # Bucket Z into layers (0.5 Å tolerance)
    z_min, z_max = z_vals.min(), z_vals.max()
    buckets = np.round((z_vals - z_min) / 0.5).astype(int)

    groups: dict[tuple, int] = {}
    atom_colors: list[str] = []
    legend_labels: list[str] = []

    for sym, bk in zip(surf_syms, buckets):
        key = (sym, int(bk))
        if key not in groups:
            idx = len(groups)
            groups[key] = idx
            z_center = z_min + int(bk) * 0.5
            legend_labels.append(f"{sym} (z≈{z_center:.1f} Å)")
        atom_colors.append(palette[groups[key] % len(palette)])

    return atom_colors, legend_labels


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_adsorption_site_map(
    slab,
    sites: list,
    output_path: str,
    *,
    title: str | None = None,
    site_labels: list[str] | None = None,
    show_delaunay: bool = True,
    show_cell: bool = True,
    margin_ang: float = 1.5,
    figsize: tuple = (10, 10),
    dpi: int = 150,
) -> None:
    """Draw a top-view (XY) map of unique adsorption sites on a surface slab.

    The plot is self-contained and surface-agnostic: surface atoms are coloured
    by (element, Z-layer) group, sites are coloured by type (top/bridge/hollow),
    and pairwise distances < 1.5 Å are annotated automatically.

    Parameters
    ----------
    slab : ASE Atoms
        Substrate slab.  Atoms with tag >= 2 are ignored (adsorbate tags).
    sites : list of array-like (3,)
        Unique adsorption site Cartesian coordinates (e.g. from
        ``AdsorptionWorkflowManager.get_unique_coordinates``).
    output_path : str
        File path for the saved figure (PNG/PDF/SVG — inferred from extension).
    title : str, optional
        Figure title.  Auto-generated if not supplied.
    site_labels : list[str], optional
        Override auto-generated site labels (must match len(sites)).
    show_delaunay : bool
        Draw Delaunay triangulation of surface atoms (default True).
    show_cell : bool
        Draw unit-cell boundary (default True).
    margin_ang : float
        Extra margin around the view window in Å (default 1.5).
    figsize, dpi
        Matplotlib figure parameters.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from scipy.spatial import Delaunay
    from .surface_utils import find_surface_indices

    # --- Substrate atoms only ---
    tags = slab.get_tags()
    sub_mask = tags < 2
    sub = slab[sub_mask]

    surf_idx = find_surface_indices(sub, side="top")
    surf_pos = sub.positions[surf_idx]
    surf_syms = [sub.get_chemical_symbols()[i] for i in surf_idx]
    surf_xy = surf_pos[:, :2]

    cell = sub.get_cell()

    # --- Classify sites ---
    sites_arr = [np.asarray(s, dtype=float) for s in sites]
    types = [_classify_site(s[:2], surf_xy) for s in sites_arr]

    type_style: dict[str, dict] = {
        "top":    {"color": "#e03030", "marker": "D", "size": 220, "label": "top"},
        "bridge": {"color": "#2060c0", "marker": "^", "size": 200, "label": "bridge"},
        "hollow": {"color": "#20a020", "marker": "o", "size": 200, "label": "hollow"},
    }

    # --- Auto site labels ---
    type_count: dict[str, int] = {}
    auto_labels: list[str] = []
    for t in types:
        n = type_count.get(t, 0)
        auto_labels.append(f"{t[0].upper()}{n if n else ''}")
        type_count[t] = n + 1
    if site_labels is not None and len(site_labels) == len(sites_arr):
        display_labels = site_labels
    else:
        display_labels = auto_labels

    # --- Sublattice colours for surface atoms ---
    atom_colors, atom_legend = _assign_sublattice_colors(surf_pos, surf_syms)

    # --- View window ---
    x_lo = min(surf_xy[:, 0].min(), *(s[0] for s in sites_arr)) - margin_ang
    x_hi = max(surf_xy[:, 0].max(), *(s[0] for s in sites_arr)) + margin_ang
    y_lo = min(surf_xy[:, 1].min(), *(s[1] for s in sites_arr)) - margin_ang
    y_hi = max(surf_xy[:, 1].max(), *(s[1] for s in sites_arr)) + margin_ang

    fig, ax = plt.subplots(figsize=figsize)

    # --- Delaunay triangles ---
    if show_delaunay and len(surf_xy) >= 3:
        try:
            tri = Delaunay(surf_xy)
            for s in tri.simplices:
                pts = surf_xy[s]
                cx, cy = pts.mean(axis=0)
                if x_lo < cx < x_hi and y_lo < cy < y_hi:
                    triangle = plt.Polygon(
                        pts, fill=False, edgecolor="#aad4f0", lw=0.8, alpha=0.7, zorder=1
                    )
                    ax.add_patch(triangle)
                    ax.plot(cx, cy, "+", c="#aad4f0", ms=5, alpha=0.5, zorder=2)
        except Exception:
            pass

    # --- Surface atoms ---
    seen_labels: set[str] = set()
    for (x, y, _), col, sym in zip(surf_pos, atom_colors, surf_syms):
        lbl = f"{sym}" if sym not in seen_labels else None
        ax.scatter(x, y, s=600, c=col, edgecolors="black", lw=1.5, zorder=3)
        ax.text(x, y, sym[0], ha="center", va="center",
                fontsize=7, fontweight="bold", zorder=4, color="black")
        if lbl:
            seen_labels.add(sym)

    # --- Adsorption sites ---
    for i, (s, t, lbl) in enumerate(zip(sites_arr, types, display_labels)):
        st = type_style[t]
        ax.scatter(s[0], s[1], s=st["size"] + 80, c=st["color"],
                   marker=st["marker"], edgecolors="black", lw=1.2,
                   zorder=6, alpha=0.9)
        # Label placement: offset to avoid overlap with atom markers
        x_off, y_off = 10, 6
        if s[0] > (x_lo + x_hi) / 2:
            x_off = -8
        ax.annotate(
            lbl, (s[0], s[1]),
            textcoords="offset points", xytext=(x_off, y_off),
            fontsize=8, color=st["color"], fontweight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      alpha=0.80, ec=st["color"], lw=1.0),
        )

    # --- Annotate close pairs (< 1.5 Å) ---
    for i in range(len(sites_arr)):
        for j in range(i + 1, len(sites_arr)):
            d = np.linalg.norm(sites_arr[i][:2] - sites_arr[j][:2])
            if d < 1.5:
                mid = (sites_arr[i][:2] + sites_arr[j][:2]) / 2
                ax.annotate(
                    "", xy=sites_arr[j][:2], xytext=sites_arr[i][:2],
                    arrowprops=dict(arrowstyle="<->", color="magenta",
                                   lw=1.8, mutation_scale=14),
                    zorder=5,
                )
                ax.text(
                    mid[0] + 0.1, mid[1], f"{d:.2f} Å",
                    fontsize=7.5, color="magenta", fontweight="bold",
                    ha="left", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.85, ec="magenta", lw=0.9),
                )

    # --- Cell boundary ---
    if show_cell:
        a, b = cell[0, :2], cell[1, :2]
        corners = np.array([
            [0, 0], a, a + b, b, [0, 0]
        ])
        ax.plot(corners[:, 0], corners[:, 1], "k:", lw=0.9, alpha=0.5, zorder=0)

    # --- Legend ---
    legend_handles = []
    # Atom sublattice patches
    seen_cols: dict[str, str] = {}
    for col, lbl in zip(atom_colors, [f"{s}" for s in surf_syms]):
        if lbl not in seen_cols:
            seen_cols[lbl] = col
    for lbl, col in seen_cols.items():
        legend_handles.append(
            mpatches.Patch(facecolor=col, edgecolor="black", label=lbl)
        )
    # Site-type markers
    for t, st in type_style.items():
        legend_handles.append(
            Line2D([0], [0], marker=st["marker"], color="w",
                   markerfacecolor=st["color"], markeredgecolor="black",
                   markersize=9, label=st["label"])
        )
    ax.legend(handles=legend_handles, loc="upper left",
              fontsize=8, framealpha=0.92, borderpad=0.6)

    # --- Axes ---
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)", fontsize=11)
    ax.set_ylabel("y (Å)", fontsize=11)
    ax.grid(True, alpha=0.2)

    n_top    = types.count("top")
    n_bridge = types.count("bridge")
    n_hollow = types.count("hollow")
    default_title = (
        f"Adsorption site map — {len(sites_arr)} unique sites "
        f"({n_top} top, {n_bridge} bridge, {n_hollow} hollow)\n"
        f"surface: {sub.get_chemical_formula()} | cell: "
        f"{np.linalg.norm(cell[0]):.2f} × {np.linalg.norm(cell[1]):.2f} Å"
    )
    ax.set_title(title or default_title, fontsize=10)

    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_and_plot_site_map(
    slab,
    output_path: str,
    *,
    symprec: float = 0.2,
    title: str | None = None,
    **plot_kwargs,
) -> list:
    """Generate unique adsorption sites from *slab* and plot them.

    Reproduces exactly the site set used by
    :class:`~autoflow_srxn.surface.ads_workflow_mgr.AdsorptionWorkflowManager`
    during physisorption candidate generation.

    Parameters
    ----------
    slab : ASE Atoms
        Surface slab (substrate atoms only, or with adsorbate tags ≥ 2).
    output_path : str
        Where to save the PNG/PDF figure.
    symprec : float
        spglib symmetry precision (default 0.2).
    title : str, optional
        Custom figure title.
    **plot_kwargs
        Forwarded to :func:`plot_adsorption_site_map`.

    Returns
    -------
    sites : list of ndarray (3,)
        Unique site Cartesian coordinates (same as used by the workflow).
    """
    from .ads_workflow_mgr import AdsorptionWorkflowManager
    from .surface_utils import find_surface_indices
    from scipy.spatial import Delaunay

    tags = slab.get_tags()
    sub_mask = tags < 2
    sub = slab[sub_mask]

    # Reproduce _generate_surface_sites
    surf_idx = find_surface_indices(sub, side="top")
    pos = sub.positions[surf_idx]
    z_ref = float(pos[:, 2].max())

    raw_sites = [np.array([p[0], p[1], z_ref]) for p in pos]
    if len(surf_idx) >= 3:
        try:
            tri = Delaunay(pos[:, :2])
            seen_edges: set = set()
            for s in tri.simplices:
                for a, b in [(0, 1), (1, 2), (0, 2)]:
                    key = tuple(sorted((int(s[a]), int(s[b]))))
                    if key not in seen_edges:
                        seen_edges.add(key)
                        mid = (pos[s[a]] + pos[s[b]]) / 2
                        raw_sites.append(np.array([mid[0], mid[1], z_ref]))
                centroid = pos[s].mean(axis=0)
                raw_sites.append(np.array([centroid[0], centroid[1], z_ref]))
        except Exception:
            pass

    mgr = AdsorptionWorkflowManager(sub, symprec=symprec, verbose=False)
    sites = mgr.get_unique_coordinates(sub, raw_sites, symprec=symprec)

    plot_adsorption_site_map(slab, sites, output_path, title=title, **plot_kwargs)
    return sites
