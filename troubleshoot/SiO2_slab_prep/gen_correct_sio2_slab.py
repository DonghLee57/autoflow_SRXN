"""
Regenerate correct_SiO2_slab.vasp using the user's config.yaml (tilt removed).
Verifies H atoms are correctly attached to bottom O atoms.
"""
import sys, os
sys.path.insert(0, r"c:\Users\user\Downloads\dev_w_antigravity\auto_surface_reaction\autoflow_SRXN")

import yaml
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list
from autoflow_srxn.surface.surface_utils import (
    create_slab_from_bulk,
    passivate_surface_coverage_general,
)

CONFIG_PATH = r"c:\Users\user\Downloads\dev_w_antigravity\auto_surface_reaction\autoflow_SRXN\troubleshoot\SiO2_slab_prep\config.yaml"
OUT_PATH = r"c:\Users\user\Downloads\dev_w_antigravity\auto_surface_reaction\autoflow_SRXN\troubleshoot\SiO2_slab_prep\correct_SiO2_slab.vasp"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

sp_cfg       = config["surface_prep"]
sub_gen_cfg  = sp_cfg["slab_generation"]
pass_cfg     = sp_cfg["passivation"]
valence_map  = sp_cfg.get("surface_analysis", {}).get("ideal_coordination", {})

bulk = read(r"c:\Users\user\Downloads\dev_w_antigravity\auto_surface_reaction\autoflow_SRXN\troubleshoot\SiO2_slab_prep\POSCAR_SiO2.vasp")

print("Generating O-terminated slab...")
slab = create_slab_from_bulk(
    bulk,
    miller_indices=sub_gen_cfg["miller"],
    thickness=sub_gen_cfg["thickness_ang"],
    vacuum=sub_gen_cfg["vacuum_ang"],
    target_area=sub_gen_cfg.get("target_area_ang2"),
    top_termination=sub_gen_cfg.get("top_termination"),
    bottom_termination=sub_gen_cfg.get("bottom_termination"),
    verbose=True,
)
print(f"Bare slab: {slab.symbols}  ({len(slab)} atoms)")

print("\nPassivating bottom surface with H...")
passivated = passivate_surface_coverage_general(
    slab,
    coverage=pass_cfg.get("coverage", 1.0),
    valence_map=valence_map,
    element=pass_cfg.get("element", "H"),
    side=pass_cfg.get("side", "bottom"),
    verbose=True,
)
print(f"Passivated slab: {passivated.symbols}  ({len(passivated)} atoms)")

# --- Verification ---
syms = np.array(passivated.get_chemical_symbols())
pos  = passivated.positions
h_idx = np.where(syms == "H")[0]
print(f"\nH atoms: {len(h_idx)}")
print(f"H z-range: [{pos[h_idx, 2].min():.3f}, {pos[h_idx, 2].max():.3f}] Ang")

# Check O-H bond lengths via neighbor list
i_list, j_list, d_list = neighbor_list("ijd", passivated, 1.5)
oh_bonds = [(i, j, d) for i, j, d in zip(i_list, j_list, d_list)
            if syms[i] == "O" and syms[j] == "H"]
print(f"O-H bonds found: {len(oh_bonds)}")
if oh_bonds:
    dists = [d for _, _, d in oh_bonds]
    print(f"  O-H distance range: [{min(dists):.3f}, {max(dists):.3f}] Ang  (expected ~0.96 Ang)")

if len(h_idx) == len(oh_bonds) and len(h_idx) > 0:
    print("\nRESULT: [OK] All H atoms correctly bonded to O atoms")
    write(OUT_PATH, passivated)
    print(f"Written: {OUT_PATH}")
else:
    print("\nRESULT: [FAIL] H-O bonding mismatch")
