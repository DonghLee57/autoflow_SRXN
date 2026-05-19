"""
Diagnostic script: trace full passivation execution path using the user's config.yaml.
This directly simulates what prepare_slab_stage() does, step by step.
"""
import sys
sys.path.insert(0, r"c:\Users\user\Downloads\dev_w_antigravity\auto_surface_reaction\autoflow_SRXN")

import yaml
import numpy as np
from ase.io import read
from autoflow_srxn.surface.surface_utils import (
    create_slab_from_bulk,
    passivate_surface_coverage_general,
    standardize_vasp_atoms,
    get_all_dangling_bonds_general,
)

CONFIG_PATH = r"c:\Users\user\Downloads\dev_w_antigravity\auto_surface_reaction\autoflow_SRXN\troubleshoot\SiO2_slab_prep\config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

sp_cfg       = config.get("surface_prep", {})
sub_gen_cfg  = sp_cfg.get("slab_generation", {})
pass_cfg     = sp_cfg.get("passivation", {})
valence_map  = sp_cfg.get("surface_analysis", {}).get("ideal_coordination", {})

print("=" * 60)
print("STEP 1: Config values parsed from config.yaml")
print(f"  slab_generation.enabled  : {sub_gen_cfg.get('enabled', False)}")
print(f"  slab_generation.miller   : {sub_gen_cfg.get('miller')}")
print(f"  slab_generation.top_term : {sub_gen_cfg.get('top_termination')}")
print(f"  slab_generation.bot_term : {sub_gen_cfg.get('bottom_termination')}")
print(f"  passivation.enabled      : {pass_cfg.get('enabled', False)}")
print(f"  passivation.element      : {pass_cfg.get('element', 'H')}")
print(f"  passivation.side         : {pass_cfg.get('side', 'bottom')}")
print(f"  passivation.coverage     : {pass_cfg.get('coverage', 1.0)}")
print(f"  valence_map              : {valence_map}")

print()
print("=" * 60)
print("STEP 2: Generating bare slab from bulk...")
bulk_path = r"c:\Users\user\Downloads\dev_w_antigravity\auto_surface_reaction\autoflow_SRXN\troubleshoot\SiO2_slab_prep\POSCAR_SiO2.vasp"
bulk = read(bulk_path)
slab = create_slab_from_bulk(
    bulk,
    miller_indices=sub_gen_cfg.get("miller", [1, 0, 0]),
    thickness=sub_gen_cfg.get("thickness_ang", 10.0),
    vacuum=sub_gen_cfg.get("vacuum_ang", 10.0),
    target_area=sub_gen_cfg.get("target_area_ang2"),
    top_termination=sub_gen_cfg.get("top_termination"),
    bottom_termination=sub_gen_cfg.get("bottom_termination"),
    verbose=True,
)
print(f"  Bare slab composition: {slab.symbols}   ({len(slab)} atoms)")

print()
print("=" * 60)
print("STEP 3: Detecting dangling bonds (before passivation)...")
dangling = get_all_dangling_bonds_general(
    slab, valence_map,
    side=pass_cfg.get("side", "bottom"),
    bond_slack=0.45,
)
print(f"  Dangling bonds found: {len(dangling)}")
for db in dangling:
    z = slab.positions[db['parent'], 2]
    print(f"    Parent idx={db['parent']:3d} ({db['parent_sym']}), z={z:.3f} Ang, "
          f"vec_z={db['vector'][2]:.3f}")

print()
print("=" * 60)
print("STEP 4: Running passivation...")
if pass_cfg.get("enabled", False):
    passivated = passivate_surface_coverage_general(
        slab,
        coverage=pass_cfg.get("coverage", 1.0),
        valence_map=valence_map,
        element=pass_cfg.get("element", "H"),
        side=pass_cfg.get("side", "bottom"),
        verbose=True,
    )
    h_syms = [s for s in passivated.get_chemical_symbols() if s == "H"]
    print(f"  Passivated slab: {passivated.symbols}  ({len(passivated)} atoms)")
    print(f"  H atoms added  : {len(h_syms)}")
    if len(h_syms) > 0:
        z_H = passivated.positions[[i for i,s in enumerate(passivated.get_chemical_symbols()) if s=="H"], 2]
        print(f"  H z-range      : [{z_H.min():.3f}, {z_H.max():.3f}] Ang")
        print()
        print("  RESULT: [OK] PASSIVATION SUCCESSFUL")
    else:
        print()
        print("  RESULT: [FAIL] PASSIVATION FAILED -- No H atoms placed!")
else:
    print("  passivation.enabled is False — skipped.")
