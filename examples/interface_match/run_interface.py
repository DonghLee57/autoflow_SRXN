"""
Interface Lattice-Match Search — standalone example script
===========================================================
Uses :class:`~autoflow_srxn.interface.InterfaceWorkflow` for 2D ZSL screening
and slab construction.

Output (written to ``output_dir`` defined in the config)
---------------------------------------------------------
* ``interface_summary.txt``      — plain-text candidate ranking table
* ``candidates.json``            — full candidate list (JSON, schema-compatible
                                   with interface_tool.py)
* ``candidates.html``            — interactive Plotly dashboard (scatter + table)
* ``sub_slab_<N>.extxyz``        — substrate slab for candidate N
* ``film_slab_<N>.extxyz``       — film slab for candidate N

Usage
-----
    python run_interface.py              # reads config.yaml in this directory
    python run_interface.py config.yaml  # explicit config path

Requirements
------------
    pip install pymatgen ase plotly
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from ase.io import write as ase_write


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_path: str | None = None) -> logging.Logger:
    logger = logging.getLogger("interface_match")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%H:%M:%S")
    # Use errors='replace' so non-ASCII characters don't crash on Windows cp949
    ch = logging.StreamHandler(
        open(sys.stdout.fileno(), mode="w", encoding="utf-8",
             errors="replace", closefd=False)
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ("sub_path", "film_path"):
        val = cfg.get(key, "")
        if val and not os.path.isabs(val):
            candidate = os.path.join(config_dir, val)
            if os.path.exists(candidate):
                cfg[key] = candidate
    return cfg


# ---------------------------------------------------------------------------
# Candidate post-processing helpers
# ---------------------------------------------------------------------------

def _polar_ok(cand) -> bool:
    """Return True when neither polar-termination warning is present."""
    return not any(
        w in cand.notes
        for w in ("WARN:sub_polar_termination", "WARN:film_polar_termination")
    )


def _large_cell_warn(cand) -> bool:
    return any(w.startswith("WARN:large_cell") for w in cand.notes)


def _mark_recommended(candidates) -> list[bool]:
    """
    Return a parallel list of booleans: True for the first (lowest-vm)
    candidate per film_miller group -- same logic as interface_tool.py's
    ``is_recommended`` flag.
    """
    seen: set = set()
    flags = []
    for c in candidates:
        key = c.film_miller
        if key not in seen:
            seen.add(key)
            flags.append(True)
        else:
            flags.append(False)
    return flags


# ---------------------------------------------------------------------------
# Miller index resolution
# ---------------------------------------------------------------------------

def _resolve_millers(
    explicit: list | None,
    max_m: int | None,
    structure,
    mode: str = "distinct",
) -> tuple[list[tuple], str]:
    """Resolve Miller indices from config, returning (miller_list, source_label).

    Priority
    --------
    1. ``explicit``  — non-empty list provided directly in config
    2. ``max_m``     — auto-enumerate with ``max(|h|,|k|,|l|) <= max_m`` (inclusive)
    3. fallback      — [(0,0,1), (1,1,0), (1,1,1)]

    Parameters
    ----------
    explicit : list or None
        Raw list from config (e.g. [[0,0,1],[1,1,0]]).  Empty list treated as None.
    max_m : int or None
        Maximum Miller index value (inclusive).
    structure : pymatgen Structure
        Required only for ``mode="distinct"``.
    mode : "distinct" | "raw"
        "distinct" uses pymatgen's symmetry reduction;
        "raw" enumerates all (h,k,l) with h,k,l in [0,max_m], gcd=1.

    Returns
    -------
    millers : list of (h, k, l) tuples
    source  : human-readable label for logging
    """
    # 1. Explicit list wins
    if explicit:
        millers = [tuple(int(x) for x in m) for m in explicit]
        return millers, f"explicit ({len(millers)} faces)"

    # 2. Auto-enumerate
    if max_m is not None:
        if mode == "distinct":
            from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
            millers = [
                tuple(m)
                for m in get_symmetrically_distinct_miller_indices(structure, max_m)
            ]
            return millers, f"distinct  max_miller={max_m}  ({len(millers)} faces)"
        else:  # raw
            from math import gcd
            millers = []
            for h in range(0, max_m + 1):
                for k in range(0, max_m + 1):
                    for l in range(0, max_m + 1):
                        if h == k == l == 0:
                            continue
                        if gcd(gcd(h, k), l) == 1:
                            millers.append((h, k, l))
            return millers, f"raw  max_miller={max_m}  ({len(millers)} faces)"

    # 3. Default
    default = [(0, 0, 1), (1, 1, 0), (1, 1, 1)]
    return default, "default (3 faces)"


# ---------------------------------------------------------------------------
# Interface stacking
# ---------------------------------------------------------------------------

def _stack_interface(
    sub_slab,
    film_slab,
    gap_ang: float = 2.5,
    vacuum_ang: float = 15.0,
) -> "ase.Atoms":  # noqa: F821
    """Stack substrate and film slabs into a single interface Atoms object.

    The substrate's in-plane cell is used as the reference.  Film atom
    positions are mapped to the substrate cell via a 2D affine transform
    (this absorbs the small in-plane mismatch captured by vm_strain).

    Tags
    ----
    0 — substrate atoms
    1 — film atoms

    Parameters
    ----------
    sub_slab, film_slab : ase.Atoms
        Output of InterfaceWorkflow.build(), both with vacuum applied.
    gap_ang : float
        Rigid gap inserted between the top of the sub slab and the
        bottom of the film slab (Angstrom).
    vacuum_ang : float
        Vacuum added above the film slab in the combined cell (Angstrom).
    """
    from ase import Atoms

    sub_pos  = sub_slab.get_positions().copy()
    film_pos = film_slab.get_positions().copy()
    sub_cell  = np.array(sub_slab.cell)
    film_cell = np.array(film_slab.cell)

    # Shift both so atom z starts at 0
    sub_pos[:,  2] -= sub_pos[:,  2].min()
    film_pos[:, 2] -= film_pos[:, 2].min()

    sub_thickness  = sub_pos[:,  2].max()
    film_thickness = film_pos[:, 2].max()

    # Map film in-plane (x,y) to substrate cell via fractional coordinate
    # transform: frac = cart @ inv(cell_2d),  cart_new = frac @ sub_cell_2d
    film_cell_2d = film_cell[:2, :2]   # [[ax,ay],[bx,by]] of film
    sub_cell_2d  = sub_cell[:2,  :2]   # [[ax,ay],[bx,by]] of sub
    try:
        T = np.linalg.inv(film_cell_2d) @ sub_cell_2d
    except np.linalg.LinAlgError:
        T = np.eye(2)

    film_xy_new  = film_pos[:, :2] @ T
    film_z_new   = film_pos[:,  2] + sub_thickness + gap_ang
    film_pos_new = np.column_stack([film_xy_new, film_z_new])

    # Combine
    all_pos     = np.vstack([sub_pos, film_pos_new])
    all_symbols = (list(sub_slab.get_chemical_symbols()) +
                   list(film_slab.get_chemical_symbols()))
    all_tags    = [0] * len(sub_slab) + [1] * len(film_slab)

    # New cell: in-plane from substrate, c tall enough to hold both + vacuum
    new_cell = sub_cell.copy()
    new_cell[2] = [0.0, 0.0, sub_thickness + gap_ang + film_thickness + vacuum_ang]

    return Atoms(
        symbols  = all_symbols,
        positions= all_pos,
        cell     = new_cell,
        pbc      = [True, True, True],
        tags     = all_tags,
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def save_json(
    candidates,
    recommended_flags: list[bool],
    sub_name: str,
    film_name: str,
    out_path: str,
) -> None:
    """Serialise candidates to JSON, schema-compatible with interface_tool.py."""

    def _conv(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return str(o)

    records = []
    for c, is_rec in zip(candidates, recommended_flags):
        records.append({
            "sub_name":       sub_name,
            "film_name":      film_name,
            "sub_miller":     list(c.sub_miller),
            "film_miller":    list(c.film_miller),
            "Na":             c.Na.tolist(),
            "Nb":             c.Nb.tolist(),
            "eps1":           float(c.eps1),
            "eps2":           float(c.eps2),
            "vm_strain":      float(c.vm),       # key kept for schema compatibility
            "n_total":        int(c.n_atoms),    # key kept for schema compatibility
            "polar_ok":       _polar_ok(c),
            "is_recommended": is_rec,
            "notes":          list(c.notes),
        })

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(records, fp, default=_conv, indent=2)


# ---------------------------------------------------------------------------
# Plotly HTML dashboard
# ---------------------------------------------------------------------------

def save_html_dashboard(
    candidates,
    recommended_flags: list[bool],
    sub_name: str,
    film_name: str,
    sub_sg: str,
    film_sg: str,
    sub_polar: bool,
    film_polar: bool,
    out_path: str,
) -> None:
    """
    Generate an interactive HTML dashboard identical in structure to
    interface_tool.py's plot() / _make_html() output.
    Requires plotly.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  [HTML] plotly not installed -- skipping dashboard. "
              "Install with:  pip install plotly")
        return

    if not candidates:
        return

    PALETTE = [
        "#2563EB", "#06B6D4", "#F97316", "#DC2626",
        "#16A34A", "#9333EA", "#CA8A04", "#0EA5E9",
        "#E11D48", "#14B8A6", "#F59E0B", "#6366F1",
    ]
    SYMBOLS = ["circle", "square", "diamond", "cross", "x",
               "triangle-up", "triangle-down", "pentagon",
               "star", "hexagram"]

    film_millers_unique = sorted(set(c.film_miller for c in candidates), key=str)
    sub_millers_unique  = sorted(set(c.sub_miller  for c in candidates), key=str)
    film_color  = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(film_millers_unique)}
    sub_symbol  = {m: SYMBOLS[i % len(SYMBOLS)]  for i, m in enumerate(sub_millers_unique)}

    # Attach recommended flag to each candidate for easy lookup
    rec_set: set[int] = {i for i, f in enumerate(recommended_flags) if f}

    title = f"{sub_name} // {film_name} interface candidates"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "All candidates  (VM strain vs atom count)",
            "Zoom: strain < 10 %",
        ),
        horizontal_spacing=0.10,
    )

    for panel, strain_max in [(1, None), (2, 0.10)]:
        for fm in film_millers_unique:
            subset_idx = [
                i for i, c in enumerate(candidates)
                if c.film_miller == fm
                and (strain_max is None or c.vm <= strain_max)
            ]
            normal_idx  = [i for i in subset_idx if i not in rec_set]
            starred_idx = [i for i in subset_idx if i in rec_set]

            for group_idx, marker_extra in [
                (normal_idx,  {}),
                (starred_idx, {"symbol": "star", "size": 16,
                               "line": {"width": 2, "color": "black"}}),
            ]:
                if not group_idx:
                    continue
                group = [candidates[i] for i in group_idx]

                xs = [c.n_atoms    for c in group]
                ys = [c.vm * 100   for c in group]

                hovers = []
                for i_g, c in zip(group_idx, group):
                    star_txt  = " BEST" if i_g in rec_set else ""
                    polar_txt = "polar OK" if _polar_ok(c) else "polar UNSAFE"
                    note_txt  = (f"<br>Notes: {', '.join(c.notes)}"
                                 if c.notes else "")
                    hovers.append(
                        f"<b>{sub_name}{c.sub_miller} // "
                        f"{film_name}{c.film_miller}{star_txt}</b>"
                        f"<br>Na=[[{c.Na[0,0]},{c.Na[0,1]}],[{c.Na[1,0]},{c.Na[1,1]}]]"
                        f"  Nb=[[{c.Nb[0,0]},{c.Nb[0,1]}],[{c.Nb[1,0]},{c.Nb[1,1]}]]"
                        f"<br>eps1={c.eps1*100:+.2f}%  eps2={c.eps2*100:+.2f}%"
                        f"<br><b>VM strain = {c.vm*100:.2f}%</b>"
                        f"<br>det(Na)={c.Na[0,0]*c.Na[1,1]}  "
                        f"det(Nb)={c.Nb[0,0]*c.Nb[1,1]}"
                        f"<br><b>Atoms (est) = {c.n_atoms}</b>"
                        f"<br>Polar: {polar_txt}"
                        f"{note_txt}"
                    )

                mk = dict(
                    size=10 if not marker_extra else 16,
                    color=film_color[fm],
                    symbol=[sub_symbol[c.sub_miller] for c in group],
                    opacity=0.85,
                    line=dict(width=1, color="rgba(0,0,0,0.4)"),
                )
                mk.update(marker_extra)

                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys,
                        mode="markers",
                        name=f"film{fm}" + (" ★" if marker_extra else ""),
                        legendgroup=str(fm) + str(bool(marker_extra)),
                        showlegend=(panel == 1),
                        marker=mk,
                        text=hovers,
                        hovertemplate="%{text}<extra></extra>",
                    ),
                    row=1, col=panel,
                )

    # Threshold lines
    for panel in (1, 2):
        for thresh, lbl, col in [
            (2,  "2% (DFT grade)",  "rgba(34,197,94,0.5)"),
            (5,  "5% (acceptable)", "rgba(234,179,8,0.5)"),
            (10, "10% (max)",       "rgba(239,68,68,0.4)"),
        ]:
            fig.add_hline(
                y=thresh, line_dash="dash", line_color=col,
                annotation_text=lbl,
                annotation_position="top right" if panel == 1 else "top left",
                row=1, col=panel,
            )

    # Shaded recommended zone
    fig.add_shape(
        type="rect", x0=0, x1=3000, y0=0, y1=5,
        fillcolor="rgba(34,197,94,0.05)",
        line=dict(color="rgba(34,197,94,0.3)", width=1, dash="dot"),
        row=1, col=1,
    )

    fig.update_layout(
        title=dict(
            text=(f"{title}<br>"
                  f"<sup>Separate symmetric slabs  |  HNF 2D ZSL  |  "
                  f"autoflow_srxn.interface</sup>"),
            font=dict(size=16),
        ),
        height=620, width=1300,
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="#f8fafc",
        legend=dict(
            title="Film Miller",
            itemsizing="constant",
            borderwidth=1,
            bordercolor="#cbd5e1",
            bgcolor="rgba(255,255,255,0.9)",
        ),
        font=dict(family="Inter, Arial, sans-serif", size=12),
    )
    for col in (1, 2):
        fig.update_xaxes(title_text="Estimated total atoms",
                         gridcolor="#e2e8f0",
                         showline=True, linecolor="#94a3b8",
                         row=1, col=col)
    fig.update_yaxes(title_text="von Mises strain (%)",
                     gridcolor="#e2e8f0", showline=True,
                     linecolor="#94a3b8",
                     zeroline=True, zerolinecolor="#94a3b8",
                     row=1, col=1)
    fig.update_yaxes(title_text="von Mises strain (%)",
                     gridcolor="#e2e8f0", range=[-0.3, 10.3],
                     showline=True, linecolor="#94a3b8",
                     row=1, col=2)

    # ── Build HTML ────────────────────────────────────────────────────────────
    # Best per film miller (for summary table)
    best_per_film: dict = {}
    for i, c in enumerate(candidates):
        fm = c.film_miller
        if fm not in best_per_film or c.vm < best_per_film[fm][1].vm:
            best_per_film[fm] = (i, c)

    table_rows = ""
    for fm, (i_best, c) in sorted(best_per_film.items(),
                                   key=lambda kv: kv[1][1].vm):
        bg = ("#f0fdf4" if c.vm < 0.02 else
              "#fefce8" if c.vm < 0.05 else "#fff1f2")
        star   = " ★" if i_best in rec_set else ""
        p_txt  = "OK"   if _polar_ok(c)        else "WARN"
        lc_txt = " (large)" if _large_cell_warn(c) else ""
        table_rows += f"""
        <tr style="background:{bg}">
          <td><b>{sub_name}{c.sub_miller}</b></td>
          <td><b>{film_name}{c.film_miller}</b>{star}</td>
          <td>[[{c.Na[0,0]},{c.Na[0,1]}],[{c.Na[1,0]},{c.Na[1,1]}]]</td>
          <td>[[{c.Nb[0,0]},{c.Nb[0,1]}],[{c.Nb[1,0]},{c.Nb[1,1]}]]</td>
          <td>{c.eps1*100:+.2f}%</td>
          <td>{c.eps2*100:+.2f}%</td>
          <td><b>{c.vm*100:.2f}%</b></td>
          <td>{c.n_atoms}{lc_txt}</td>
          <td>{p_txt}</td>
        </tr>"""

    syms_html = " &nbsp; ".join(f"<code>{m}</code>" for m in sub_millers_unique)

    plot_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"scrollZoom": True, "displayModeBar": True},
    )

    n_lt2  = sum(1 for c in candidates if c.vm < 0.02)
    n_lt5  = sum(1 for c in candidates if c.vm < 0.05)
    n_lt10 = sum(1 for c in candidates if c.vm < 0.10)
    n_polar_ok = sum(1 for c in candidates if _polar_ok(c))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body  {{ font-family: Inter, Arial, sans-serif; margin: 0; padding: 20px;
           background: #f1f5f9; color: #1e293b; }}
  h1    {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .sub  {{ color: #64748b; font-size: 0.9rem; margin-bottom: 20px; }}
  .card {{ background: white; border-radius: 12px; padding: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 20px; }}
  .stat {{ display: inline-block; margin-right: 20px; font-size: 0.88rem; }}
  .stat b {{ font-size: 1.1rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
  th    {{ background: #1e293b; color: white; padding: 8px 12px; text-align: left; }}
  td    {{ padding: 7px 12px; border-bottom: 1px solid #e2e8f0; }}
</style>
</head>
<body>

<h1>{title}</h1>
<p class="sub">
  Substrate: <b>{sub_name}</b> (SG {sub_sg}, polar={'YES' if sub_polar else 'NO'})
  &nbsp;|&nbsp;
  Film: <b>{film_name}</b> (SG {film_sg}, polar={'YES' if film_polar else 'NO'})
  &nbsp;|&nbsp; HNF 2D ZSL &nbsp;|&nbsp; ★ = best per film orientation
</p>

<div class="card">
  <span class="stat">Total candidates <b>{len(candidates)}</b></span>
  <span class="stat">Polar OK <b>{n_polar_ok}</b></span>
  <span class="stat">VM &lt; 2% <b style="color:#16a34a">{n_lt2}</b></span>
  <span class="stat">VM &lt; 5% <b style="color:#ca8a04">{n_lt5}</b></span>
  <span class="stat">VM &lt; 10% <b style="color:#dc2626">{n_lt10}</b></span>
</div>

<div class="card">
  <div style="overflow-x:auto">{plot_html}</div>
  <p style="margin-top:10px; font-size:0.83rem; color:#64748b;">
    <b>Colour</b> = film Miller index &nbsp;|&nbsp;
    <b>Symbol</b> = substrate Miller index: {syms_html}<br>
    Hover for details &nbsp;|&nbsp; Scroll to zoom, drag to pan
  </p>
</div>

<div class="card">
  <h2 style="font-size:1.1rem; margin-top:0">Best candidate per film orientation</h2>
  <table>
    <tr>
      <th>Substrate</th><th>Film</th>
      <th>Na (sub HNF)</th><th>Nb (film HNF)</th>
      <th>eps1</th><th>eps2</th>
      <th>VM strain</th><th>Est. atoms</th><th>Polar</th>
    </tr>
    {table_rows}
  </table>
  <p style="font-size:0.8rem; color:#64748b; margin-top:10px;">
    VM = &radic;(0.5&thinsp;(eps1&sup2; + eps2&sup2; + (eps1&minus;eps2)&sup2;))
    &nbsp;|&nbsp; atoms = estimated combined sub+film slab count
  </p>
</div>

<div class="card" style="font-size:0.82rem; color:#64748b;">
  Generated by <b>run_interface.py</b> using
  <b>autoflow_srxn.interface.InterfaceWorkflow</b>
</div>

</body>
</html>"""

    Path(out_path).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_interface_search(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)

    # ── Output directory ─────────────────────────────────────────────────────
    out_dir = cfg.get("output_dir") or "."
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(config_path)), out_dir
        )
    os.makedirs(out_dir, exist_ok=True)

    logger = _setup_logger(os.path.join(out_dir, "interface_match.log"))

    # ── Check pymatgen ────────────────────────────────────────────────────────
    try:
        from pymatgen.core import Structure
        from autoflow_srxn.interface import InterfaceWorkflow
    except ImportError as exc:
        logger.error(
            "pymatgen is required for interface lattice-match search.\n"
            f"  Install with:  pip install pymatgen\n  ({exc})"
        )
        sys.exit(1)

    # ── Validate input bulk crystal paths ────────────────────────────────────
    sub_path  = cfg.get("sub_path", "")
    film_path = cfg.get("film_path", "")
    for label, path in [("Substrate", sub_path), ("Film", film_path)]:
        if not os.path.exists(path):
            logger.error(f"{label} bulk file not found: {path!r}")
            sys.exit(1)

    # ── Load structures ───────────────────────────────────────────────────────
    logger.info("Loading substrate bulk: %s", sub_path)
    sub_struct  = Structure.from_file(sub_path)
    logger.info("Loading film bulk:      %s", film_path)
    film_struct = Structure.from_file(film_path)

    sub_sg_sym, sub_sg_num   = sub_struct.get_space_group_info()
    film_sg_sym, film_sg_num = film_struct.get_space_group_info()

    # Human-readable names (config override, else formula)
    sub_name  = cfg.get("sub_name")  or sub_struct.formula.replace(" ", "")
    film_name = cfg.get("film_name") or film_struct.formula.replace(" ", "")

    # Polarity flags (for HTML header only — workflow computes them internally)
    from autoflow_srxn.interface.lattice_match import POLAR_SG
    sub_polar  = sub_sg_num  in POLAR_SG
    film_polar = film_sg_num in POLAR_SG

    logger.info("  Substrate: %s  SG#%d %s  polar=%s",
                sub_name, sub_sg_num, sub_sg_sym, sub_polar)
    logger.info("  Film:      %s  SG#%d %s  polar=%s",
                film_name, film_sg_num, film_sg_sym, film_polar)

    # ── Miller index resolution ───────────────────────────────────────────────
    # Priority per material: explicit list > *_max_miller > default
    # sub_max_miller / film_max_miller: max(|h|,|k|,|l|) <= N  (inclusive)
    # miller_mode: "distinct" (symmetry-reduced, default) | "raw" (all gcd=1)
    mode = cfg.get("miller_mode", "distinct")

    sub_millers, sub_src = _resolve_millers(
        explicit = cfg.get("sub_millers")  or None,
        max_m    = cfg.get("sub_max_miller"),
        structure= sub_struct,
        mode     = mode,
    )
    film_millers, film_src = _resolve_millers(
        explicit = cfg.get("film_millers") or None,
        max_m    = cfg.get("film_max_miller"),
        structure= film_struct,
        mode     = mode,
    )

    logger.info("  Sub  Miller indices [%s]: %s", sub_src,  sub_millers)
    logger.info("  Film Miller indices [%s]: %s", film_src, film_millers)

    # ── Build workflow ────────────────────────────────────────────────────────
    wf = InterfaceWorkflow(
        sub_structure      = sub_struct,
        film_structure     = film_struct,
        sub_millers        = sub_millers,
        film_millers       = film_millers,
        max_det            = cfg.get("max_det",            6),
        strain_cutoff      = cfg.get("strain_cutoff",      0.05),
        max_atoms          = cfg.get("max_atoms",          400),
        min_slab_thickness = cfg.get("min_slab_thickness", 12.0),
        vacuum             = cfg.get("vacuum",             15.0),
    )

    # ── Screen ────────────────────────────────────────────────────────────────
    logger.info(
        "Screening  max_det=%d  strain_cutoff=%.3f  miller_mode=%s ...",
        cfg.get("max_det", 6), cfg.get("strain_cutoff", 0.05), mode,
    )
    candidates = wf.screen()

    if not candidates:
        logger.warning(
            "No candidates found -- try increasing max_det or strain_cutoff."
        )
        return

    logger.info("Found %d candidate(s).", len(candidates))

    n_lt2  = sum(1 for c in candidates if c.vm < 0.02)
    n_lt5  = sum(1 for c in candidates if c.vm < 0.05)
    n_lt10 = sum(1 for c in candidates if c.vm < 0.10)
    logger.info("  VM < 2%%: %d   VM < 5%%: %d   VM < 10%%: %d",
                n_lt2, n_lt5, n_lt10)

    # Derive per-candidate extras used by JSON / HTML outputs
    recommended_flags = _mark_recommended(candidates)

    # ── Plain-text summary ────────────────────────────────────────────────────
    summary = wf.summary(candidates, top_n=20)
    print("\n" + summary + "\n")
    summary_path = os.path.join(out_dir, "interface_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary)
    logger.info("Summary written: %s", summary_path)

    # ── JSON export ───────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "candidates.json")
    save_json(candidates, recommended_flags, sub_name, film_name, json_path)
    logger.info("JSON  written: %s  (%d records)", json_path, len(candidates))

    # ── HTML dashboard ────────────────────────────────────────────────────────
    html_path = os.path.join(out_dir, "candidates.html")
    save_html_dashboard(
        candidates        = candidates,
        recommended_flags = recommended_flags,
        sub_name          = sub_name,
        film_name         = film_name,
        sub_sg            = f"#{sub_sg_num} {sub_sg_sym}",
        film_sg           = f"#{film_sg_num} {film_sg_sym}",
        sub_polar         = sub_polar,
        film_polar        = film_polar,
        out_path          = html_path,
    )
    logger.info("HTML  written: %s", html_path)

    # ── Build top-k interface structures ─────────────────────────────────────
    build_top_k  = int(cfg.get("build_top_k",   1))
    verbose      = bool(cfg.get("verbose",       False))
    interface_gap = float(cfg.get("interface_gap", 2.5))
    vacuum_ang    = float(cfg.get("vacuum",        15.0))

    if build_top_k <= 0:
        logger.info("build_top_k=0 -- skipping slab construction.")
        return

    top_k = min(build_top_k, len(candidates))
    logger.info(
        "Building interface structures for top %d candidate(s)  "
        "[verbose=%s  gap=%.1f A] ...",
        top_k, verbose, interface_gap,
    )

    for idx, (cand, is_rec) in enumerate(
        zip(candidates[:top_k], recommended_flags[:top_k])
    ):
        star = " (*)" if is_rec else ""
        logger.info(
            "  [%d/%d] sub%s | film%s  vm=%.4f  eps1=%+.4f  eps2=%+.4f"
            "  n_atoms=%d%s",
            idx + 1, top_k,
            cand.sub_miller, cand.film_miller,
            cand.vm, cand.eps1, cand.eps2, cand.n_atoms, star,
        )
        if cand.notes:
            logger.warning("      Notes: %s", "  ".join(cand.notes))

        try:
            sub_slab, film_slab = wf.build(cand)
        except Exception as exc:
            logger.error("      Build failed: %s", exc)
            continue

        # Shared metadata tag
        id_tag = (f"{sub_name}_{''.join(map(str, cand.sub_miller))}_"
                  f"{film_name}_{''.join(map(str, cand.film_miller))}_"
                  f"Na{cand.Na[0,0]}x{cand.Na[1,1]}_"
                  f"Nb{cand.Nb[0,0]}x{cand.Nb[1,1]}")

        # ── Default output: stacked interface structure ───────────────────
        interface = _stack_interface(
            sub_slab, film_slab,
            gap_ang=interface_gap,
            vacuum_ang=vacuum_ang,
        )
        interface.info.update({
            "vm_strain":  float(cand.vm),
            "eps1":       float(cand.eps1),
            "eps2":       float(cand.eps2),
            "sub_miller": str(cand.sub_miller),
            "film_miller":str(cand.film_miller),
            "id_tag":     id_tag,
        })
        iface_out = os.path.join(out_dir, f"interface_{idx}.extxyz")
        ase_write(iface_out, interface)
        logger.info(
            "      interface_%d.extxyz: %d atoms  "
            "(sub=%d tag0 + film=%d tag1)",
            idx, len(interface), len(sub_slab), len(film_slab),
        )

        # ── Verbose output: individual substrate and film slabs ───────────
        if verbose:
            for slab, role, miller in [
                (sub_slab,  "substrate", cand.sub_miller),
                (film_slab, "film",      cand.film_miller),
            ]:
                slab.info.update({
                    "vm_strain": float(cand.vm),
                    "eps1":      float(cand.eps1),
                    "eps2":      float(cand.eps2),
                    "role":      role,
                    "miller":    str(miller),
                    "id_tag":    id_tag,
                })
            sub_out  = os.path.join(out_dir, f"sub_slab_{idx}.extxyz")
            film_out = os.path.join(out_dir, f"film_slab_{idx}.extxyz")
            ase_write(sub_out,  sub_slab)
            ase_write(film_out, film_slab)
            logger.info(
                "      [verbose] sub_slab_%d.extxyz (%d atoms)  "
                "film_slab_%d.extxyz (%d atoms)",
                idx, len(sub_slab), idx, len(film_slab),
            )

    logger.info("Done.  All output in: %s", out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not os.path.exists(config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, "config.yaml")
    run_interface_search(config_file)
