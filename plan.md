# Research Plan: Ni Precursor Adsorption on Si/SiO2 Substrates with Inhibitor

**System:** AllylCpNi, Ni(PF3)4 on Si and SiO2 substrates  
**Inhibitor:** secret_inhibitor  
**Potential:** SevenNet-0 (7net-0)  
**Framework:** autoflow_srxn

---

## Overview

목표: Si 및 SiO2 기판 위에서 ALD 전구체 AllylCpNi, Ni(PF3)4의 physisorption/chemisorption 거동을 계산하고, inhibitor 존재 시 선택적 흡착 억제 메커니즘을 규명한다.

### 연구 시스템 구성

| 역할 | 분자 | 비고 |
|---|---|---|
| 전구체 1 | AllylCpNi | η3-allyl + η5-Cp, haptic 결합 |
| 전구체 2 | Ni(PF3)4 | σ-donor PF3, haptic 없음 |
| Inhibitor | secret_inhibitor | C5NO2H13 계열 |

| 기판 | 면 | Termination |
|---|---|---|
| Si | (100) | 2×1 buckled dimer reconstruction |
| SiO2 | (001) | O-terminated (both faces) |
| SiO2 | (001) | Si-terminated (both faces) |

---

## Phase 0: Structure Preparation ✅ COMPLETED

### 목표
- 모든 분자 구조 최적화 (mode-following relaxation)
- 벌크 구조 최적화 (cell relax)
- AllylCpNi haptic 코드 검증

### 수행 내용
1. **Ni(PF3)4 구조 빌드** (`phase0/build_NiPF3_4.py`)
   - Td 대칭, Ni-P=2.05 Å, P-F=1.57 Å, Ni-P-F=117°
   - 17 atoms (F12P4Ni)

2. **분자 mode-following relaxation** (`phase0/relax_molecules.py`, SevenNet-0)
   - AllylCpNi: 6 cycle, min_freq=-0.337 THz (잔류 imaginary)
   - inhibitor: 1 cycle 수렴, min_freq=-0.075 THz
   - NiPF3_4: 2 cycle 수렴, min_freq=-0.080 THz

3. **Bulk cell relaxation** (`phase0/relax_bulk.py`, ExpCellFilter)
   - Si: a=5.463 Å (Fd-3m #227 cubic)
   - SiO2: a=b=5.043, c=7.383 Å (I-42d #122 tetragonal)

4. **Bulk 대칭성 검증** (`phase0/check_fix_symmetry.py`, symprec=0.01 Å)
   - Si: Fd-3m 유지, 셀 보정 0.000000 Å ✓
   - SiO2: I-42d 유지, a=b 평균화 보정 0.000116 Å ✓

5. **원자 Z-순서 정렬** (`phase0/sort_structures.py`)
   - AllylCpNi, inhibitor, NiPF3_4 (original+relaxed 각각)
   - ase.data.atomic_numbers 기준 오름차순 정렬

6. **AllylCpNi haptic 코드 검증** (`phase0/validate_allylcpni.py`)
   - 17/17 PASS
   - η5-Cp (hapticity=5), η3-allyl (hapticity=3) 정상 인식
   - 원자 정렬 후 Ni 인덱스 0→18 변경에도 코드 정상 작동

### 이슈 및 대응

| 이슈 | 대응 |
|---|---|
| AllylCpNi 잔류 imaginary mode 2개 (-0.337 THz) | η5-Cp ring 자유 회전 + η3-allyl sigmatropic 이동은 기체상 고립 분자의 본질적 soft mode. 표면 흡착 후 제약됨. 구조 그대로 사용. max_iter=6으로 부족, 필요 시 증가 |
| Windows cp949 콘솔 인코딩 오류 (em-dash, ✓) | ASCII 대체 문자 사용 |

---

## Phase 1: Substrate Preparation ✅ COMPLETED

### 목표
- Si(100) 슬랩: 2×1 buckled dimer 재건 + ML relax
- SiO2(001) O-terminated 슬랩: ML relax
- SiO2(001) Si-terminated 슬랩: ML relax
- 하부 5.5 Å 고정, H passivation (bottom)
- Site map 및 dangling bond 분포 확인

### 코드 아키텍처 결정

#### 문제
`surface_utils.py`에 Si(100)-specific 함수들이 혼재:
- `reconstruct_si100_2x1_buckled`
- `identify_surface_bonds` / `oxidize_si_surface`
- `insert_o_bridge_pure_geo`
- `build_si100_slab` / `generate_standard_surfaces`

#### 결정: `reconstruction_recipes.py` 분리

```
autoflow_srxn/surface/
├── surface_utils.py          # 범용 geometry util (변경됨)
│   └── auto_reconstruct_surface(atoms, side, miller=None)  ← miller 파라미터 추가
└── reconstruction_recipes.py  # 시스템별 하드코딩 구현체 (신규)
    ├── reconstruct_si100_2x1_buckled()
    ├── identify_surface_bonds() / oxidize_si_surface()
    └── build_si100_slab() / generate_standard_surfaces()
```

**원칙:**
- `surface_utils.py`: crystal-system-agnostic 범용 함수만 보유
- `reconstruction_recipes.py`: 명시적으로 system-specific임을 표시, 사용 시 직접 import
- `auto_reconstruct_surface`: `miller` 파라미터를 받아 Si(100)인 경우만 recipe 호출, 나머지는 random_noise + ML relax에 위임

### 슬랩 설정 및 결과 (최종, bulk-shift 수정 후)

| 기판 | Miller | Atoms | Cell (Å) | E/atom (eV) | Relax steps | Frozen |
|---|---|---|---|---|---|---|
| Si(100) | (1,0,0) | 112 | 10.93×10.93 | -5.002 | 67 | 5.5 Å |
| SiO2(001) O-term | (0,0,1) | 112 | 10.09×10.09 | -7.308 | 48 | 5.5 Å |
| SiO2(001) Si-term | (0,0,1) | 108 | 10.09×10.09 | -7.406 | 44 | 5.5 Å |

### 표면 구조 분석 (post-relax, 최종)

**Si(100):**
- bulk +a/4 shift 후: 전 층 Si 원자수 = 8 (이전: 표면층 4, 내부층 8 불균형 해결)
- **c(4×2) reconstruction** 자발 형성: 4개 dimers / 2×2 cell
  - 강하게 buckled dimer 2쌍: Δz=1.05 Å, bond=2.358 Å (DFT ref: ~2.35 Å ✓)
  - 약하게 buckled dimer 2쌍: Δz=0.04 Å, bond=2.534 Å
  - c(4×2)는 2×1보다 더 안정적인 Si(100) 저온 바닥 상태 (7net-0 올바르게 예측)
- Dangling bonds: 4 (각 dimer당 1개); 분석 코드 보고치(6)는 임계값 이슈

**SiO2 O-term:**
- Top layer: 8개 terminal O (2×2 확장), coord=1 (Si-O bond 1개, dangling bond 1개/O)
- 물리적으로 올바른 O-terminated surface (silanol 전구체 구조)
- 분석 코드 수정 완료: 원소쌍별 covalent cutoff로 O-O 근접쌍을 비결합으로 분리하여 dangling bond 8개 정상 감지

**SiO2 Si-term:**
- Top layer: under-coordinated Si (coord=2), dangling bonds=8 (분석 코드 정확)
- 4개 bridging O가 바로 아래에 위치; I-42d (001) 구조 특성

### 체크리스트
- [x] 각 슬랩의 표면 dangling bond 수 및 방향 확인 (분석 코드 이슈 있으나 구조는 정확)
- [x] Si(100) 층별 원자수 균형 확인 (전 층 8개 ✓)
- [x] Site map 생성 (top/bridge/hollow) - `structures/slabs/site_maps/`에 PNG/CSV 저장
- [x] H passivation 커버리지 확인 (bottom 1.0)
- [x] Slab relax 후 forces < 0.05 eV/Å 확인 (모두 수렴)
- [x] 표면 원자 좌표 합리성 검토 (모두 물리적으로 합리적)

### Site map 결과 (Phase 2 입력)

`phase1/generate_site_maps.py`로 workflow의 `generate_and_plot_site_map()`과 동일한 top/bridge/hollow 후보 생성 및 symmetry reduction을 수행했다. Candidate 수를 줄이기 위해 site 병합 기준은 `symprec=1.5 Å`로 설정했으며, symmetry operation 탐지는 더 보수적인 tolerance를 사용하고 top/bridge/hollow type별로 따로 병합해 site type 소실을 방지했다.

| 기판 | Unique sites | Top | Bridge | Hollow | 산출물 |
|---|---:|---:|---:|---:|---|
| Si(100) | 5 | 1 | 2 | 2 | `structures/slabs/site_maps/Si100_site_map.png`, `Si100_sites.csv` |
| SiO2 O-term | 8 | 1 | 4 | 3 | `structures/slabs/site_maps/SiO2_O_term_site_map.png`, `SiO2_O_term_sites.csv` |
| SiO2 Si-term | 7 | 1 | 3 | 3 | `structures/slabs/site_maps/SiO2_Si_term_site_map.png`, `SiO2_Si_term_sites.csv` |

---

## Phase 2: Inhibitor Adsorption ✅ COMPLETED

### 목표
- Si(100), SiO2(O-term), SiO2(Si-term) 위 inhibitor physisorption/chemisorption
- `branching_limit: 3` → 상위 3개 inhibitor-modified surface 전달

### 설정
```yaml
reaction_search:
  mechanisms:
    inhibitor:
      enabled: true
      center: 13  # inhibitor center atom (확인 필요)
      physisorption:
        placement_height: 3.0
        n_rot: 8
        gravity_pull: { enabled: true }
      chemisorption:
        enabled: true
        rot_steps: 8
      branching_limit: 3
```

### 체크리스트
- [x] Inhibitor center atom 종류 확인: 설정값 `center: 13`은 0-based ASE index 기준 C. 단, inhibitor physisorption 배치는 workflow에서 COM 기준 사용.
- [x] Physi candidate 수 확인 (`symprec=1.5 Å`, `n_rot=8`, physisorption only)
- [x] 흡착 에너지 순위 확인 (상위 8개 pre-screened 후보 relax, top 3 선정)
- [x] Inhibitor chemisorption on Si100 확인 (N-Si 공유결합 -1.80 eV 확인)
- [x] Inhibitor chemisorption on SiO2_Si_term 확인 (cov=0, 물리흡착만 존재)

### 2-A: Physisorption 결과 (SevenNet-0, relaxed) - Fibonacci 수정 후 최종

계산 설정: `placement_height=3.0 A`, `n_rot=8` (true Fibonacci sphere), `symprec=1.5 A`, 후보 생성 후 single-point pre-screen 상위 8개 relaxation.
흡착 에너지: `E_ads = E(slab+inhibitor) - E(slab) - E(inhibitor_gas)`, E_gas = -113.64 eV.

| 기판 | Generated | Relaxed | Rank 1 E_ads (eV) | Rank 2 | Rank 3 | CovBonds | 비고 |
|---|---:|---:|---:|---:|---:|---:|---|
| Si(100) | 40 | 8 | +0.009 | +0.015 | +0.015 | 0 | physisorption 비우호적. MinDist ~2.3-2.6 A (H-Si vdW) |
| SiO2 O-term | 64 | 8 | -1.095 | -1.084 | -1.045 | 0 | gravity_pull=False 채택. MinDist ~2.0-2.5 A (H-bond 지배) |
| SiO2 Si-term | 56 | 8 | -0.642 | -0.078 | -0.044 | 0 | rank 1만 안정, 나머지 급격 약화 |

**old (polar grid) vs new (Fibonacci) 비교:**

| 기판 | Old cand | New cand | Old rank1 | New rank1 | 주요 차이 |
|---|---:|---:|---:|---:|---|
| Si(100) | 28 | 40 | +0.014 | +0.009 | 개선 미미; Si(100) 자체가 physisorption 비우호적 |
| SiO2 O-term | 48 | 64 | -0.996 | -1.095 | +0.10 eV 추가 안정화; top-4 모두 -0.99 eV 이하로 향상 |
| SiO2_Si_term | 40 | 56 | -0.640 | -0.642 | rank1 동일; old의 rank2-5 (-0.63 eV 군)은 사라짐 |

**SiO2_Si_term 주목:** 구 polar grid에서는 rank1~5가 모두 -0.60~-0.64 eV로 비슷해 보였지만, 이는 동일한 2개 tilt각에서 4개 azimuthal spin이 유사한 in-plane 배향을 반복 샘플링한 artifact. 진정한 Fibonacci SO(3) 샘플링 결과 rank1(-0.642 eV) 1개만 안정하고 나머지는 -0.08 eV 이상으로 크게 다름 → 실제 안정 흡착 구조는 하나의 특정 배향뿐임을 확인.

주요 산출물:
- 전체 요약: `phase2/results/inhibitor_physisorption/physisorption_summary.txt`
- Si(100): `phase2/results/inhibitor_physisorption/Si100/Si100_inhibitor_physi_rank0{1-3}.vasp`
- SiO2 O-term: `phase2/results/inhibitor_physisorption/SiO2_O_term_nograv/SiO2_O_term_nograv_inhibitor_physi_rank0{1-3}.vasp`
- SiO2 Si-term: `phase2/results/inhibitor_physisorption/SiO2_Si_term/SiO2_Si_term_inhibitor_physi_rank0{1-3}.vasp`

### 2-B: Chemisorption 결과 (`phase2/run_chemisorption.py`)

계산 설정: Si100 = `build_chemisorption_structures(center=C/N)`, SiO2_Si_term = site-map 기반 (reactive atoms=O1,O2,N, heights=2.5-4.0 A, step=0.5 A). E_gas = -113.64 eV.

| 기판 | Method | Relaxed | Rank 1 E_ads (eV) | min_dist (pair) | cov | 비고 |
|---|---|---:|---:|---:|---:|---|
| Si100 | builder (C/N center) | 12 | **-1.800** | 1.750 A (N-Si) | 2 | N-Si 공유결합 확인 |
| Si100 | — | — | -1.775 | 1.679 A (O-Si) | 2 | rank2: O-Si 결합 |
| Si100 | — | — | -1.648 | 1.767 A (N-Si) | 2 | rank3: N-Si 결합 |
| SiO2_Si_term | site-map | 12 | -0.109 | 3.543 A (H-Si) | 0 | 공유결합 없음; 물리흡착만 |

**판단:**
- Si(100): inhibitor N 원자가 Si 댄글링 본드에 N-Si 공유결합 형성, E_ads = -1.80 eV (강한 chemisorption).
- SiO2(Si-term): chemisorption 없음. 가장 안정한 구조도 cov=0, E_ads = -0.11 eV (physi 수준).
- SiO2(O-term): 별도 계산 생략 — phase3 supercell 계산에서 O-H 비물리적 결합 확인 (아래 참조).
- **Si(100) inhibitor chemisorption E_ads (-1.80 eV) vs Si(100) inhibitor physisorption (+0.009 eV): physisorption 탐색에서는 chemisorption 포착 불가.** Phase 2 physi 결과는 local minimum에 머물며 N-Si 공유결합 basin을 발견하지 못한 것. 이 점이 Phase 3 supercell 재계산의 동기.

### 이슈 및 대응 (Phase 2)

| 이슈 | 대응 |
|---|---|
| inhibitor Si(100) physi 에너지 +0.009 eV (반발) | c(4×2) dimer surface 기하 때문에 flat placement가 불리. chemisorption 탐색에서 N-Si -1.80 eV 발견으로 해소 |
| SiO2_O_term chemi 계산 생략 | Phase 3에서 O-H 비물리 결합 확인 → ML potential 한계로 SiO2_O_term은 별도 DFT 검증 필요로 분류 |

---

## Phase 3: Precursor Adsorption ✅ COMPLETED

### 개요 및 셀 크기 결정

전구체 분자 footprint 분석:
- AllylCpNi: ~4.5 Å → 원본 셀 (10.9 Å Si100, 10.1 Å SiO2) 에 적합, 원본 셀 사용
- Ni(PF3)4: ~5.7 Å → 원본 셀은 PF3 그룹이 인접 이미지와 겹침; **[1,1]/[-1,1] 2x supercell 필요**

Supercell 생성 (`phase3/setup_supercells.py`, `P=[[1,1,0],[-1,1,0],[0,0,1]]`):

| 기판 | 원본 원자수 | 2x 원자수 | 셀 대각 (Å) |
|---|---:|---:|---:|
| Si100 | 112 | 224 | 15.45 |
| SiO2_Si_term | 108 | 216 | 14.26 |
| SiO2_O_term | 112 | 224 | 14.26 |

### 3.1 AllylCpNi 흡착 결과

계산 설정: `phase3/run_allylcpni.py` (원본 셀), HEIGHT_PHYSI=3.5 A, N_SPIN=4, PRESELECT=10(physi)/12(chemi). E_gas = -112.034 eV.

**Si100 - AllylCpNi:**
- Physi: `flat placement` → rank1 **-2.359 eV, dist=1.924 A (C-Si), cov=2 [CHEM]**
  - 주목: physi 탐색임에도 relax 중 C-Si 공유결합 형성 (allyl C가 Si 댄글링 본드에 결합)
- Chemi: `build_chemisorption_structures(center=Ni)` → rank1 -2.170 eV, dist=1.937 A (C-Si), cov=2
  - builder를 통한 chemi 탐색에서도 C-Si 결합 확인

**SiO2_Si_term - AllylCpNi:**
- Physi: rank1 -0.957 eV, dist=3.028 A (H-O), cov=0 [PHYSI]
- Chemi (Ni-down, site-map): rank1 -0.934 eV, dist=2.864 A (H-Si), cov=0 [PHYSI]
  - ⚠️ 공유결합 없음 — Ni를 아래로 향하게 했을 때 allyl C가 표면에서 멀어지는 방향 배치 문제
  - → `run_allylcpni_sio2_fix.py`로 재계산 (아래 참조)

**SiO2_O_term - AllylCpNi:**
- Physi: rank1 -8.911 eV, dist=0.970 A (H-O), cov=3 [CHEM]  ⚠️ H-O 결합 (비물리)
- Chemi: rank1 -9,732,167 eV ⚠️ **ML potential 발산** (C-O dist=0.246 A; atoms overlap)

**AllylCpNi SiO2 Chemi Fix (`phase3/run_allylcpni_sio2_fix.py`):**

문제 원인: `orient_atom_toward_surface(mol, NI_IDX=18)`로 Ni를 아래로 내리면 allyl C(idx=15~17)가 오히려 위로 올라감 (Ni 아래에 allyl C가 있는 기하).

해결: allyl C(idx=15)를 reactive atom으로 설정 → `orient_atom_toward_surface(mol, ALLYL_C_IDX=15)`.

| 기판 | Rank 1 E_ads (eV) | dist (pair) | cov | 판단 |
|---|---:|---:|---:|---|
| SiO2_Si_term | -0.732 | 3.420 A (H-Si) | 0 | 물리흡착만 (C-Si 결합 없음) — Si-O 친화도 > Si-C, 물리적으로 타당 |
| SiO2_O_term | -11.115 | 0.970 A (H-O) | 4 | allyl H의 O-H 결합 — ML potential 한계 |

### 3.2 Ni(PF3)4 흡착 결과

계산 설정: `phase3/run_nipf3_4.py` (2x supercell 전 기판), HEIGHT_PHYSI=4.5 A, N_SPIN=4, PRESELECT=10. E_gas = -87.927 eV.

**Si100 - Ni(PF3)4:**
- Physi (supercell flat): rank1 **-0.038 eV**, dist=3.213 A (F-Si), cov=0 [PHYSI]
- Chemi (P-down site-map): rank1 -0.036 eV, dist=3.429 A (F-Si), cov=0 [PHYSI]
  - ⚠️ chemi 탐색에서도 공유결합 없음 → Si 댄글링 본드와 P/Ni의 반응성 매우 낮음

**SiO2_Si_term - Ni(PF3)4:**
- Physi (supercell flat): rank1 **-1.540 eV**, dist=3.345 A (F-O), cov=0 [PHYSI]
  - F-O van der Waals/electrostatic 상호작용 지배적
- Chemi (P-down site-map): rank1 -0.068 eV, dist=3.375 A (F-Si), cov=0 [PHYSI]
  - → physi가 chemi보다 훨씬 안정; P-Si 결합 형성 없음

**SiO2_O_term - Ni(PF3)4:**
- Physi: rank1 -0.113 eV, dist=2.805 A (F-O), cov=0 [PHYSI]  (다른 기판 대비 매우 약함)
- Chemi: rank1 -0.083 eV, dist=3.001 A (F-O), cov=0 [PHYSI]

### 3.3 Inhibitor Supercell 재계산 결과

계산 설정: `phase3/run_inhibitor_supercell.py` (2x supercell), HEIGHT_PHYSI=2.5 A, E_gas = -113.640 eV.
목적: Ni(PF3)4와 동일한 supercell에서 inhibitor 기준값 확보.

| 기판 | Mode | Rank 1 E_ads (eV) | dist (pair) | cov | 비고 |
|---|---|---:|---:|---:|---|
| Si100 | physi | -0.260 | 2.178 A (N-Si) | 0 | N-Si 근접이나 공유결합 미형성 |
| Si100 | chemi (site-map) | -641,521,642 | 0.072 A (H-Si) | 2 | ⚠️ **ML 발산** (H-Si at 0.07 A) |
| SiO2_Si_term | physi | **-1.738** | 2.783 A (H-O) | 0 | H-bond 지배 physi |
| SiO2_Si_term | chemi (site-map) | -0.113 | 3.541 A (H-Si) | 0 | 공유결합 없음 |
| SiO2_O_term | physi | -11.390 | 0.971 A (H-O) | 4 | ⚠️ H-O 결합 (비물리) |
| SiO2_O_term | chemi | -2,045 | 0.215 A (H-O) | 4 | ⚠️ **ML 발산** |

**주요 발견:** Si100 inhibitor chemi on supercell (site-map 방식)이 ML 발산. Phase 2 builder 방식의 -1.800 eV (N-Si, cov=2) 결과가 유효. Site-map 방식은 Si 댄글링 본드의 정확한 위치를 포착하지 못해 원자 겹침 발생.

### Phase 3 흡착 에너지 총괄 매트릭스

E_ads = E(slab+mol) - E(slab) - E(mol_gas) [eV], SevenNet-0, FIRE relax.  
⚠️ = ML potential 발산 또는 비물리 결합; — = 계산 제외(발산 위험)

|  | Si100 | SiO2_Si_term | SiO2_O_term |
|---|---|---|---|
| **Inhibitor physi** | +0.009 (H-Si, cov=0) | -0.642 (H-O, cov=0) | -1.095 (H-O, cov=0) |
| **Inhibitor chemi** | **-1.800** ★ (N-Si, cov=2) †| -0.109 (H-Si, cov=0) | — |
| **AllylCpNi physi→** | **-2.359** (C-Si, cov=2) ★ | -0.957 (H-O, cov=0) | ⚠️ -8.91 H-O |
| **AllylCpNi chemi** | -2.170 (C-Si, cov=2) | -0.732 (H-Si, cov=0) ‡ | ⚠️ H-O |
| **Ni(PF3)4 physi** | -0.038 (F-Si, cov=0) | **-1.540** (F-O, cov=0) | -0.113 (F-O, cov=0) |
| **Ni(PF3)4 chemi** | -0.036 (F-Si, cov=0) | -0.068 (F-Si, cov=0) | -0.083 (F-O, cov=0) |

† Phase 2 builder 결과 (원본 셀). Supercell site-map 재계산은 ML 발산.  
‡ allyl C-down 재계산 결과 (`run_allylcpni_sio2_fix.py`).  
★ = 핵심 결과.

**물리적 해석:**
- AllylCpNi는 Si100에서 강한 chemisorption (-2.36 eV, allyl C-Si 공유결합 2개). SiO2에서는 공유결합 없음 → Si100 선택적 흡착.
- Ni(PF3)4는 어떤 기판에서도 공유결합 없음. SiO2_Si_term에서 F-O vdW (-1.54 eV)가 가장 강하나 가역적 physisorption. Si100에서는 사실상 비결합 (-0.04 eV).
- Inhibitor는 Si100에서만 N-Si 공유결합 (-1.80 eV) → Si100 댄글링 본드를 선점적으로 블로킹.

### 이슈 및 대응 (Phase 3)

| 이슈 | 원인 | 대응 | 상태 |
|---|---|---|---|
| Windows cp949 UnicodeEncodeError (em-dash, eta 기호) | chemisorption_builder.py 내부 출력 문자 | `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')` 모든 스크립트 상단에 추가 | closed |
| Ni(PF3)4 Si100 chemi: builder 후보 0개 | 4개 PF3 그룹이 Si dimer 표면에서 입체 장애 → 유효 후보 생성 불가 | site-map + P-down 방식으로 교체 | closed |
| ML potential 발산 (E_ads ~ -10^6 eV) | 원자 간 거리 < 0.3 A (atoms overlap) 시 ML interatomic potential 발산 | `E_ADS_MIN = -500.0 eV` 필터 도입 (`phase3/utils.py`), 발산 구조 'N(low-E)' 플래그 | closed |
| AllylCpNi SiO2 chemi: 모든 결과 cov=0 | `orient_atom_toward_surface(mol, NI_IDX=18)` → Ni 아래, allyl C 위로 올라가 표면 접촉 불가 | `run_allylcpni_sio2_fix.py`: `ALLYL_C_IDX=15`로 reactive atom 변경 | closed |
| Inhibitor Si100 chemi on supercell (site-map): ML 발산 | site-map이 Si 댄글링 본드 정확한 위치를 포착 못함 → 원자 겹침 | Phase 2 builder 결과(-1.800 eV) 유효. Supercell builder 재계산 필요 | open ⚠️ |
| SiO2_O_term: 전 분자에서 H-O 결합 형성 | 표면 terminal O가 ML potential에서 과반응 → physi도 H-O covalent bond 형성 | SiO2_O_term 결과 전체를 "DFT 검증 필요"로 분류, Phase 4 경쟁흡착에서 제외 | open ⚠️ |

---

## Phase 4: Competitive Adsorption (SCRIPTED — 미실행)

### 목표
inhibitor가 선점한 기판에서 AllylCpNi / Ni(PF3)4의 흡착 거동 비교.

```
E_ads_prec = E(slab+inh+prec) - E(slab+inh) - E(prec_gas)
```

Phase 3 clean-surface 결과 대비:
- E_ads_prec > E_ads_clean: inhibitor가 흡착을 차단
- E_ads_prec ≈ E_ads_clean: inhibitor가 흡착을 차단하지 않음

### 구현 (`phase4/run_competitive_adsorption.py`)

**기준 구조 (inhibitor-covered slab):**
- Si100[2x] + inhibitor physi rank01 (`phase3/results/inhibitor_supercell/Si100/physi/rank01.vasp`, 245원자)
- SiO2_Si_term[2x] + inhibitor physi rank01 (237원자)
- SiO2_O_term: 제외 (ML 발산 위험)

**전구체 배치:**
- AllylCpNi: Ni center (idx=18), HEIGHT=3.5 A
- Ni(PF3)4: Ni center (idx=16), HEIGHT=4.5 A

**탐색 그리드:**
- 6×6 분율좌표 grid (f1,f2 ∈ linspace(0.1, 0.9, 6)) × 4 spins = 144 후보/case
- Pre-screen → 상위 10개 relax
- 인히비터 centroid XY (C, N 원자 평균): Si100 (-5.35, 10.23 Å), SiO2_Si_term (-3.56, 12.84 Å)

**계산 케이스:**

| 기판 | 전구체 | 상태 |
|---|---|---|
| Si100+inh | AllylCpNi | 미실행 |
| Si100+inh | Ni(PF3)4 | 미실행 |
| SiO2_Si_term+inh | AllylCpNi | 미실행 |
| SiO2_Si_term+inh | Ni(PF3)4 | 미실행 |

### 예상 결과 및 가설

| 케이스 | 예상 E_ads_prec | 예상 차단 여부 | 근거 |
|---|---|---|---|
| Si100+inh / AllylCpNi | > -2.36 eV (약화) | 부분 차단 | inhibitor N-Si가 댄글링 본드 점유 → allyl C 결합 site 감소 |
| Si100+inh / Ni(PF3)4 | ≈ -0.04 eV (변화 없음) | 없음 | Ni(PF3)4가 Si100에서 원래 비결합 |
| SiO2_Si_term+inh / AllylCpNi | ≈ -1.0 eV (inhibitor physi 위에 physi) | 없음/약함 | inhibitor와 AllylCpNi 모두 physi → 적층 가능 |
| SiO2_Si_term+inh / Ni(PF3)4 | 불명 (F-O vdW vs inhibitor 표면 차폐) | 미지 | 실험 필요 |

### 체크리스트
- [x] `phase4/run_competitive_adsorption.py` 작성 완료
- [ ] Si100+inh / AllylCpNi 계산 실행
- [ ] Si100+inh / Ni(PF3)4 계산 실행
- [ ] SiO2_Si_term+inh / AllylCpNi 계산 실행
- [ ] SiO2_Si_term+inh / Ni(PF3)4 계산 실행
- [ ] Phase 3 clean-surface 결과와 비교 분석
- [ ] 선택성 지수 S 계산

### 선택성 지수 정의

```
S_inhibitor = [E_ads(inh, Si100) - E_ads(inh, SiO2_Si)] 
            = [-1.800] - [-0.642] = -1.158 eV  (inhibitor가 Si100을 크게 선호)

S_AllylCpNi = [E_ads(AllylCpNi, Si100) - E_ads(AllylCpNi, SiO2_Si)]
            = [-2.359] - [-0.957] = -1.402 eV  (AllylCpNi도 Si100을 선호)

S_NiPF3 = [E_ads(NiPF3, Si100) - E_ads(NiPF3, SiO2_Si)]
         = [-0.038] - [-1.540] = +1.502 eV   (Ni(PF3)4는 SiO2_Si를 선호)
```

→ Inhibitor와 AllylCpNi가 Si100을 같이 선호한다면, inhibitor는 AllylCpNi 흡착을 완전 차단하기 어려울 수 있음. Phase 4 실험이 핵심.

---

## 검증 필요 항목 (Verification Backlog)

| 항목 | 우선순위 | 방법 | 비고 |
|---|---|---|---|
| **Si100 inhibitor chemi on 2x supercell** | 高 | `build_chemisorption_structures` (builder) 방식으로 재계산 | site-map 방식은 ML 발산; builder가 Si 댄글링 본드 정확히 타깃팅 |
| **Phase 4 competitive adsorption 계산 실행** | 高 | `python phase4/run_competitive_adsorption.py` | 스크립트 완성됨, 아직 미실행 |
| **SiO2_O_term 결과 전체** | 高 | DFT (VASP/QE) 단일점 계산 또는 더 보수적인 ML potential | SevenNet-0가 O-terminated 표면에서 H-O 과결합 경향; terminal O 반응성 과대평가 의심 |
| **AllylCpNi Si100 chemi 구조 시각화** | 中 | VESTA로 rank01-03 VASP 파일 확인 | allyl C-Si 결합 2개의 기하 (hapticity 변화 유무) 확인 필요 |
| **Ni(PF3)4 Si100에서 Ni-Si 결합 가능성** | 中 | PF3 하나를 사전 해리한 상태의 구조로 계산 | 현재 intact Ni(PF3)4로는 P 그룹이 입체 장애. 해리 경로는 TS 계산 필요 |
| **AllylCpNi imaginary mode 잔류** | 低 | mode-following relaxation max_iter 증가 (>6) | eta-ring 회전/allyl sigmatropic은 soft mode이므로 구조 영향 미미 |
| **Inhibitor SiO2_Si_term chemi cov=0 물리성** | 低 | Si-O vs Si-C 결합 에너지 비교 (문헌 DFT) | 현재 결과는 Si-C < Si-O 해석과 일치; 정량 검증 필요 |

---

## Code Architecture Log

### reconstruction_recipes.py 분리 (Phase 1)

**배경:** `surface_utils.py`의 `reconstruct_si100_2x1_buckled` 및 관련 Si-specific 함수들이 범용 utility 모듈에 혼재.

**결정 근거:**
- 범용성 없는 하드코딩 구현체를 core 모듈에 포함하면 확장성 저하
- 신규 기판 추가 시 `surface_utils.py`가 비대해짐
- `auto_reconstruct_surface`의 is_iv 분기가 너무 광범위 (Si(110), Si(111)에도 잘못 적용)

**구현:**
- `reconstruction_recipes.py`: Si-specific 함수들 + 향후 다른 시스템별 recipe 추가 공간
- `auto_reconstruct_surface(miller=None)`: miller 파라미터로 Si(100) recipe 명시적 선택
- Miller 미지정 또는 Si(100) 이외: random_noise → ML relax에 위임

---

## Issues Log

| Phase | 이슈 | 대응 | 상태 |
|---|---|---|---|
| 0 | AllylCpNi 잔류 imaginary mode (eta-ring 자유 회전) | soft mode로 판단, 구조 그대로 사용 | closed |
| 0 | Windows cp949 인코딩 오류 (스크립트 출력) | em-dash/특수문자 ASCII로 대체 | closed |
| 0 | SiO2 a-b 비대칭 (0.000116 Å) | 대칭화 후 저장 | closed |
| 1 | Si-specific 함수 core 모듈 혼재 | reconstruction_recipes.py 분리 | closed |
| 1 | Si(100) dimer recipe: 0 dimers (소형 셀 문제) | FIRE가 자발적 buckled-dimer 형성으로 해결; dimer recipe는 target_area >= 120 Å² 필요 | closed |
| 1 | `get_all_dangling_bonds_general` O-term O 미감지 | 원인: 고정 2.6 Å neighbor cutoff가 terminal O 주변 O-O 근접쌍(~2.26 Å)을 결합으로 오인. 수정: 원소쌍별 covalent cutoff(+slack)로 filtering | fixed |
| 1 | SiO2 Si-term 상단 Si+O 혼재 | I-42d (001) 구조 특성 - 표면 Si가 coord=2로 언더코디네이션, 물리적으로 정상 | closed |
| 1 | **슬랩 면적 부족** (inhibitor+precursor 동시 계산 불가) | target_area: Si=60→120 Å², SiO2=60→100 Å²; x축 2배 확장 | fixed |
| 1 | **PBC 경계 원자 이탈** (Si bottom/top layer 원자 누락) | create_slab_from_bulk 후 slab.wrap() 추가; standardize_vasp_atoms에도 wrap 적용 | fixed |
| 1 | **Si(100) 층 원자수 불균형** (표면 4개 vs 내부 8개) | bulk를 +a/4 이동 후 슬랩 생성 → 전 층 8개 보장, c(4×2) reconstruction 형성 | fixed |
| 1 | **Si(100) Top dimer seed 과밀 매칭** (relax 후 adatom-like 원자 2개 상승) | top-layer Si perfect matching 후보 평가해 비결합 Si-Si 최소거리 최대인 matching 선택 | fixed |
| 2 | inhibitor Si(100) physi +0.009 eV (반발) | c(4×2) dimer 기하 때문에 flat placement 불리. chemi 탐색에서 N-Si -1.80 eV 발견 | closed |
| 3 | cp949 UnicodeEncodeError (library em-dash 출력) | `sys.stdout = io.TextIOWrapper(..., encoding='utf-8', errors='replace')` 전 스크립트에 추가 | closed |
| 3 | Ni(PF3)4 Si100 chemi builder 후보 0개 | 4개 PF3 그룹 입체 장애 → site-map + P-down 방식으로 교체 | closed |
| 3 | ML potential 발산 (E_ads ~ -10^6 eV) | 원자 겹침 시 발산. `E_ADS_MIN = -500.0 eV` 필터 + 'N(low-E)' 플래그 (`phase3/utils.py`) | closed |
| 3 | AllylCpNi SiO2 chemi 전부 cov=0 | Ni-down 배치 시 allyl C가 오히려 위로 올라감 → `ALLYL_C_IDX=15`로 reactive atom 변경 | closed |
| 3 | Inhibitor Si100 chemi (supercell, site-map): ML 발산 | site-map이 댄글링 본드 위치 미포착 → 원자 겹침. Phase 2 builder 결과(-1.80 eV) 유효 | **open** |
| 3 | SiO2_O_term: 전 분자에서 H-O covalent 결합 형성 | ML potential의 terminal O 과반응 경향. 결과 신뢰 불가 | **open (DFT 필요)** |
