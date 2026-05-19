# Troubleshooting: SiO2 Slab Prep — Termination & Passivation

This directory contains the files used to diagnose and resolve issues with the slab
termination and hydrogen passivation of silica ($\text{SiO}_2$) surfaces.

---

## 1. Issue Description

When attempting to generate a slab from `POSCAR_SiO2.vasp` with both top and bottom
surfaces terminated by oxygen (`top_termination: "O"`, `bottom_termination: "O"`) and
the bottom surface passivated with hydrogen (`element: "H"`), the passivation appeared
to have no effect.

### A. Parameter Omission in Slab Generator (Root Cause)

In the original codebase, `main_workflow.py` called `create_slab_from_bulk` but did
**not** forward the `top_termination` and `bottom_termination` parameters from
`config.yaml`:

```python
# BEFORE (broken): termination parameters silently ignored
slab = create_slab_from_bulk(
    bulk_atoms=read(paths["substrate_bulk"]),
    miller_indices=sub_gen_cfg.get("miller", [1, 0, 0]),
    thickness=sub_gen_cfg.get("thickness_ang", 10.0),
    # top_termination and bottom_termination missing!
    ...
)
```

As a result, the slab was cut at the default bulk plane, producing a
**silicon-terminated** bottom surface (`O144Si72`) instead of the requested
oxygen-terminated one. The passivation algorithm then attempted to satisfy Si valences
(coordination = 4) downward, forming geometrically strained Si-H species that detached
during relaxation — which made it appear as if passivation was not applied.

### B. Diagnosis: Passivation Logic Was Correct

The `passivate_surface_coverage_general` function itself was operating correctly.
The problem was entirely upstream: the wrong surface was being exposed before
passivation was called. Confirmed by `diag_passivation.py`:

- With the correct O-terminated slab: **18 dangling O bonds detected**, **18 H atoms placed**.
- H z-range: `[0.500, 1.272] Ang` — all H atoms correctly below the slab.

---

## 2. Fix Applied

Updated `autoflow_srxn/surface/main_workflow.py` — `prepare_slab_stage()` — to
forward the termination configuration keys:

```python
# AFTER (fixed)
slab = create_slab_from_bulk(
    bulk_atoms=read(paths["substrate_bulk"]),
    miller_indices=sub_gen_cfg.get("miller", [1, 0, 0]),
    thickness=sub_gen_cfg.get("thickness_ang", 10.0),
    vacuum=sub_gen_cfg.get("vacuum_ang", 10.0),
    target_area=sub_gen_cfg.get("target_area_ang2"),
    supercell_matrix=sub_gen_cfg.get("supercell_matrix"),
    bulk_shift=sub_gen_cfg.get("bulk_shift", 0.0),
    top_termination=sub_gen_cfg.get("top_termination"),    # <-- added
    bottom_termination=sub_gen_cfg.get("bottom_termination"),  # <-- added
    verbose=True,
)
```

---

## 3. Verified Working Config (`config.yaml`)

The key fields required for a correct SiO2 O-terminated passivated slab:

```yaml
surface_prep:
  slab_generation:
    enabled: true
    miller: [0, 0, 1]
    thickness_ang: 12.0
    vacuum_ang: 15.0
    target_area_ang2: 250.0
    top_termination: "O"      # Exposes O layer at top surface
    bottom_termination: "O"   # Exposes O layer at bottom surface

  passivation:
    enabled: true
    element: "H"
    side: "bottom"
    coverage: 1.0

  surface_analysis:
    ideal_coordination:
      Si: 4
      O: 2
```

**Result**: `O144Si63` bare slab → `H18O144Si63` passivated slab.
18 undercoordinated bottom-surface O atoms each receive one H via VSEPR bond placement.

The working full config is in `config_mod.yaml`.

---

## 4. Final Verification (`diag_passivation.py`)

```
STEP 1: Config values parsed from config.yaml
  slab_generation.enabled  : True
  slab_generation.top_term : O
  slab_generation.bot_term : O
  passivation.enabled      : True
  passivation.element      : H
  passivation.side         : bottom
  valence_map              : {'Si': 4, 'O': 2, ...}

STEP 2: Generating bare slab from bulk...
  Bare slab composition: O144Si63   (207 atoms)

STEP 3: Detecting dangling bonds (before passivation)...
  Dangling bonds found: 18

STEP 4: Running passivation...
  Passivated slab: H18O144Si63  (225 atoms)
  H atoms added  : 18
  H z-range      : [0.500, 1.272] Ang

  RESULT: [OK] PASSIVATION SUCCESSFUL
```
