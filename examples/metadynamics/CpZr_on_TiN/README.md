# CpZr(NMe₂)₃ on TiN(100) — 2D Metadynamics (SevenNet 7net-0)

A GPU-ready metadynamics example: the cyclopentadienyl-tris(dimethylamido)
zirconium precursor reacting on a TiN(100) surface, sampled with the
universal MLIP **7net-0**. Reconstructs a 2-D free-energy surface for the
chemisorption / ligand-exchange reaction.

## System
- **Slab**: TiN(100), rocksalt, 3×3×2 cells, 4 layers (144 atoms); bottom ~half frozen.
- **Precursor**: CpZr(NMe₂)₃ = `C11H23N3Zr` (38 atoms).
- **Total**: 182 atoms.
- Tags mark substrate (`tag<2`) vs adsorbate (`tag≥2`); stored in the `.extxyz`
  input so the `Element@region` CV selectors resolve automatically.

## Collective variables (`config.yaml`)
| CV | Type | Meaning |
|----|------|---------|
| `Zr_surfN`  | `distance`     | Zr ↔ nearest surface lattice N (**forming** bond) |
| `Zr_amidoN` | `coordination` | Zr ↔ its amido N ligands (**breaking** bond as a –NMe₂ leaves) |
| `proton_transfer` (optional) | `proton_transfer` | H transfer forming the leaving amine HNMe₂ |

The 2-D FES is plotted over (`Zr_surfN`, `Zr_amidoN`).

## Prerequisites (GPU server)
```bash
pip install sevenn                 # provides the 7net-0 potential
# PyTorch with CUDA (match your CUDA toolkit), e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu124
# optional extra GPU speed-up for SevenNet (then set enable_cueq: true in config):
pip install cuequivariance-torch
```
`config.yaml` sets `device: cuda` and `dtype: float32` (a single knob; TF32
fast-math is auto-enabled on CUDA). It falls back to CPU if no GPU is present.

## Run
```bash
python build_system.py            # generates inputs/ (already included)
python run_metad.py --steps 2000  # quick GPU smoke test (~1-2 min)
python run_metad.py               # full run: 200,000 steps (config.yaml)
```
`run_metad.py` freezes the bottom slab layers, does a short MLIP relaxation of
the physisorbed start, then runs well-tempered metadynamics.

## Outputs (`results/`)
- `fes_2d.png` — 2-D free-energy contour over the two CVs.
- `fes_2d.npz` — FES grid (`x`, `y`, `fes`) for re-plotting.
- `COLVAR`, `HILLS` — CV/bias time series and deposited Gaussians.
- `metad_traj.extxyz`, `metad_final.vasp` — trajectory and final structure.

## Expected wall-clock (200,000 steps)

| Hardware | 7net-0 throughput | full run |
|----------|------------------:|---------:|
| Modern CUDA GPU (e.g. RTX 4090) | ~25–50 steps/s | **~1.5–2 h** |
| Mid CUDA GPU (e.g. RTX 4070)    | ~15–30 steps/s | ~2–4 h |
| CPU only (6-core)               | ~1 step/s      | ~2–3 days |

Throughput is dominated by the MLIP force evaluation (182 atoms). On CPU this
example is impractical for a converged FES — use a CUDA GPU. To cut cost
further: shrink the slab, raise `freeze_below_z`, increase the timestep with
hydrogen-mass repartitioning, or restrict to 2 CVs.

## Validation note
The metadynamics engine itself is validated against known answers
(analytic double-well barrier; Cu(100) diffusion barrier vs NEB on the same
EMT potential) — see `DOCUMENTATION.md` §9.8 and
`unittests/test_metadynamics_validation.py`. For this system, cross-check the
metaD barrier against a NEB run (Stage 2.5) on the **same** 7net-0 potential.
