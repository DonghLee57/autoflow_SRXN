"""Generate Phase 1 top/bridge/hollow adsorption site maps.

Outputs are written to structures/slabs/site_maps/:
  - <slab>_site_map.png
  - <slab>_sites.csv
  - site_maps_summary.txt
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from ase.io import read

from autoflow_srxn.surface.site_map import generate_and_plot_site_map
from autoflow_srxn.surface.site_map import _classify_site
from autoflow_srxn.surface.surface_utils import find_surface_indices


SLABS = [
    ("Si100", ROOT / "structures/slabs/Si100_slab.vasp"),
    ("SiO2_O_term", ROOT / "structures/slabs/SiO2_O_term_slab.vasp"),
    ("SiO2_Si_term", ROOT / "structures/slabs/SiO2_Si_term_slab.vasp"),
]

OUT_DIR = ROOT / "structures/slabs/site_maps"
SYMPREC = 1.5


def classify_sites(slab, sites):
    tags = slab.get_tags()
    sub = slab[tags < 2]
    surf_idx = find_surface_indices(sub, side="top")
    surf_xy = sub.positions[surf_idx, :2]
    return [_classify_site(np.asarray(site)[:2], surf_xy) for site in sites]


def write_sites_csv(path, sites, site_types):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["site_id", "type", "x_A", "y_A", "z_A"])
        counts = {}
        for site, site_type in zip(sites, site_types):
            ordinal = counts.get(site_type, 0)
            label = f"{site_type[0].upper()}{ordinal if ordinal else ''}"
            counts[site_type] = ordinal + 1
            writer.writerow(
                [
                    label,
                    site_type,
                    f"{site[0]:.6f}",
                    f"{site[1]:.6f}",
                    f"{site[2]:.6f}",
                ]
            )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = ["Phase 1 Adsorption Site Maps", "=" * 70]

    for name, slab_path in SLABS:
        slab = read(str(slab_path))
        png_path = OUT_DIR / f"{name}_site_map.png"
        csv_path = OUT_DIR / f"{name}_sites.csv"

        sites = generate_and_plot_site_map(
            slab,
            str(png_path),
            symprec=SYMPREC,
            title=f"{name} adsorption site map",
        )
        site_types = classify_sites(slab, sites)
        write_sites_csv(csv_path, sites, site_types)

        counts = {site_type: site_types.count(site_type) for site_type in ("top", "bridge", "hollow")}
        line = (
            f"{name:16s} sites={len(sites):3d}  "
            f"top={counts['top']:2d}  bridge={counts['bridge']:2d}  hollow={counts['hollow']:2d}"
        )
        print(line)
        summary.append(line)
        summary.append(f"  png: {png_path.relative_to(ROOT)}")
        summary.append(f"  csv: {csv_path.relative_to(ROOT)}")

    summary.insert(2, f"symprec = {SYMPREC:.2f} A")
    summary_path = OUT_DIR / "site_maps_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n")
    print(f"\nSummary written to: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
