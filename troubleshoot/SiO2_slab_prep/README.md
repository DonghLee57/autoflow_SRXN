# Troubleshooting: SiO2 Slab Prep Termination & Passivation

This directory contains the files used to diagnose and resolve issues with the slab termination and hydrogen passivation of silica ($\text{SiO}_2$) surfaces.

---

## 1. Issue Description

When attempting to generate a slab from `POSCAR_SiO2.vasp` with both top and bottom surfaces terminated by oxygen (`top_termination: "O"`, `bottom_termination: "O"`) and the bottom surface passivated with hydrogen (`element: "H"`), the passivation failed or led to structural collapse.

### A. Parameter Omission in Slab Generator
In the codebase, `main_workflow.py` called the slab creator `create_slab_from_bulk` but did not pass the `top_termination` and `bottom_termination` parameters from the configuration file.
```python
# Omitted parameters
slab = create_slab_from_bulk(
    bulk_atoms=read(paths["substrate_bulk"]),
    miller_indices=sub_gen_cfg.get("miller", [1, 0, 0]),
    thickness=sub_gen_cfg.get("thickness_ang", 10.0),
    ...
)
```
Consequently, the slab was cut exposing default bulk planes, resulting in a silicon-terminated bottom surface (`O144Si72`) instead of the requested oxygen-terminated one.

### B. Unphysical Passivation
Because the bottom layer consisted of Si atoms, the passivation algorithm attempted to satisfy Si valences by placing H downward, forming unstable Si-H species. During geometry optimization (Stage 0.5), these species deformed and detached due to the high strain of the incorrect termination.

---

## 2. Solutions

### A. Code Correction
We updated `main_workflow.py` to forward the termination configurations to `create_slab_from_bulk`:
```python
top_termination=sub_gen_cfg.get("top_termination"),
bottom_termination=sub_gen_cfg.get("bottom_termination"),
```

### B. Verification
With the fix applied:
1. The slab generator outputs an oxygen-terminated raw slab (`O144Si63`) with O atoms at both boundary planes.
2. The passivation algorithm correctly identifies bottom-layer O atoms (coordination 2) and attaches H to form stable O-H hydroxyl bonds.
3. The O-H bond length matches the covalent radius sum:
   $$d_{\text{O-H}} = R_{\text{cov},\text{O}} + R_{\text{cov},\text{H}} = 0.66 + 0.30 = 0.96\ \text{Å}$$
4. The relaxed slab configuration retains all passivated H atoms in a stable arrangement.

The working parameters are provided in `config_mod.yaml`.

---

## 3. Second-Stage Diagnosis: H Passivation & Constraint Collapse

Even after correcting the oxygen termination, the H-passivated bottom surface still exhibited physical and structural anomalies during relaxation. Two root causes were identified and corrected:

### A. Collinear (180-degree) Si-O-H Angle Bug
*   **Problem**: In `generate_vsepr_vectors`, for a 2-coordinated atom like Oxygen (`valence = 2`) that has only one neighbor (Silicon) at the surface, `num_missing` is 1. The default VSEPR logic returned `-sum_vec`, pointing directly opposite to the Si-O bond vector. This created a perfectly collinear 180-degree Si-O-H bond angle, which is chemically highly unstable compared to the bent tetrahedral-like silanol angle (~115°).
*   **Symmetry Lock**: Because the initial configuration was exactly collinear, the perpendicular force components on the H atoms were zero by symmetry, preventing standard optimizers from bending and relaxing the bonds.
*   **Solution**: We modified `generate_vsepr_vectors` to apply a 30-degree tilt to the dangling bond vector when `num_missing == 1` and `len(vectors) == 1`. This breaks the collinear symmetry and places the H atoms at a realistic initial bend angle (~150°).

### B. H-Atom Freezing by Slab Constraints
*   **Problem**: The Z-based FixAtoms constraint determines the bottom of the slab using the absolute minimum Z coordinate of all atoms:
    $$z_{\text{min}} = \min(z_{\text{atoms}})$$
    Since H passivation atoms are placed at the bottom ($z = 0.5\ \text{Å}$), the constraint engine calculated $z_{\text{min}} = 0.5\ \text{Å}$ and froze all atoms with $z < 0.5 + 5.5 = 6.0\ \text{Å}$. Consequently, the bottom H, O, and Si atoms were all frozen, locking them in the collinear state and preventing any relaxation.
*   **Solution**: We modified `_apply_constraints` in `potentials.py` to:
    1. Determine the substrate's $z_{\text{min}}$ by considering only non-H atoms.
    2. Explicitly exclude H atoms from the `FixAtoms` indices, ensuring passivation atoms are always free to relax and find their local energy minima.

With these two fixes, the 1-step MACE-relaxed slab preserves a stable, bent silanol passivation configuration with realistic Si-O-H bond angles (~149.4°) and correct $d_{\text{O-H}} \approx 0.96\ \text{Å}$, saved in `correct_SiO2_slab.vasp`.

