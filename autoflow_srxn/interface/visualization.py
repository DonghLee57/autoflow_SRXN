"""
autoflow_srxn.interface.visualization
=====================================
Visualization and export utilities for interface candidates.
"""

from __future__ import annotations
import json
import os
import numpy as np

def save_candidates_json(
    candidates: list,
    recommended_flags: list[bool],
    sub_name: str,
    film_name: str,
    out_path: str
):
    """Export screened candidates to a JSON file."""
    data = {
        "substrate": sub_name,
        "film": film_name,
        "candidates": []
    }
    for cand, is_rec in zip(candidates, recommended_flags):
        data["candidates"].append({
            "sub_miller": list(cand.sub_miller),
            "film_miller": list(cand.film_miller),
            "vm": float(cand.vm),
            "eps1": float(cand.eps1),
            "eps2": float(cand.eps2),
            "det_Na": int(round(abs(np.linalg.det(cand.Na)))),
            "det_Nb": int(round(abs(np.linalg.det(cand.Nb)))),
            "n_atoms": int(cand.n_atoms),
            "recommended": bool(is_rec),
            "notes": list(cand.notes)
        })

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def save_candidates_html(
    candidates: list,
    recommended_flags: list[bool],
    sub_name: str,
    film_name: str,
    sub_sg: str,
    film_sg: str,
    sub_polar: bool,
    film_polar: bool,
    out_path: str
):
    """Export screened candidates to an interactive Plotly HTML dashboard."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        # Fallback if plotly is missing
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("<html><body><h1>Plotly not installed.</h1></body></html>")
        return

    # Extract data for plotting
    vms = [c.vm * 100 for c in candidates]
    atoms = [c.n_atoms for c in candidates]
    colors = ["#636EFA" if not r else "#EF553B" for r in recommended_flags]
    symbols = ["circle" if not r else "star" for r in recommended_flags]
    names = [f"Sub{c.sub_miller} Film{c.film_miller}" for c in candidates]

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(f"{sub_name} // {film_name} Interface Screening",)
    )

    fig.add_trace(
        go.Scatter(
            x=vms, y=atoms,
            mode='markers',
            marker=dict(size=10, color=colors, symbol=symbols, line=dict(width=1, color='DarkSlateGrey')),
            text=names,
            hovertemplate="<b>%{text}</b><br>Strain: %{x:.2f}%<br>Atoms: %{y}<extra></extra>"
        )
    )

    fig.update_layout(
        title=f"Interface Match: {sub_name} ({sub_sg}) vs {film_name} ({film_sg})",
        xaxis_title="Von Mises Strain (%)",
        yaxis_title="Total Atom Count",
        template="plotly_white",
        showlegend=False
    )

    # Save to HTML
    fig.write_html(out_path, include_plotlyjs='cdn')
