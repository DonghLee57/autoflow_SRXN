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
