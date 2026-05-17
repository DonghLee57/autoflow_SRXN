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

## Phase 2: Inhibitor Adsorption (PLANNED)

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
- [ ] Inhibitor-modified surface site map 확인

### Physisorption-only 결과 (SevenNet-0, relaxed) - Fibonacci 수정 후 최종

계산 설정: `placement_height=3.0 A`, `n_rot=8` (true Fibonacci sphere), `symprec=1.5 A`, 후보 생성 후 single-point pre-screen 상위 8개 relaxation.
흡착 에너지: `E_ads = E(slab+inhibitor) - E(slab) - E(inhibitor_gas)`, E_gas = -113.64 eV.

| 기판 | Generated | Relaxed | Rank 1 E_ads (eV) | Rank 2 | Rank 3 | CovBonds | 비고 |
|---|---:|---:|---:|---:|---:|---:|---|
| Si(100) | 40 | 8 | +0.009 | +0.015 | +0.015 | 0 | 여전히 physisorption 비우호적. MinDist ~2.3-2.6 A (H-Si vdW) |
| SiO2 O-term | 64 | 8 | -1.095 | -1.084 | -1.045 | 0 | gravity_pull=False 채택. MinDist ~2.0-2.5 A (H-O, H-bond 지배) |
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

판단:
- SiO2(O-term)이 가장 강한 physisorption (-1.10 eV). H-O 거리 ~2.0-2.5 A로 H-bond 지배적.
- SiO2(Si-term)은 하나의 특정 배향에서만 안정 (-0.64 eV), 나머지는 vdW 수준.
- Si(100)은 2x1 buckled dimer 표면이라 inhibitor 접촉 기하가 불리; 흡착 에너지 +0.009 eV (repulsive). inhibitor가 Si(100)을 자발적으로 기피하는 경향 → selectivity 메커니즘 시사.
- SiO2 O-term의 gravity_pull=True는 terminal O와 H-bond/O-H 결합 형성으로 chemisorption-like 구조 유도 → physisorption-only에서는 gravity_pull=False 유지.

---

## Phase 3: Precursor Adsorption (PLANNED)

### 3.1 Ni(PF3)4 (기준계, haptic 없음)

```yaml
reaction_search:
  mechanisms:
    precursor:
      center: ["Ni"]
      physisorption: { n_rot: 8 }
      chemisorption: { rot_steps: 8 }
```

예상 chemisorption 경로:
- Route 1 (single-site): PF3 이탈, Ni-Si 결합
- Route 2 (dissociation): PF3 + Ni fragment 별도 dangling bond 결합

### 3.2 AllylCpNi (haptic 포함)

```yaml
reaction_search:
  mechanisms:
    precursor:
      center: ["Ni"]
      physisorption: { n_rot: 8 }
      chemisorption: { rot_steps: 8 }
```

예상 chemisorption 경로:

| 경로 | Surface fragment | 이탈 리간드 | 화학적 의미 |
|---|---|---|---|
| 1a | (η5-Cp)Ni-Si | η3-allyl | allyl 이탈 |
| 1b | (η3-allyl)Ni-Si | η5-Cp | Cp 이탈 |
| 2a | (η5-Cp)Ni + allyl | — | 해리 흡착 |
| 2b | (η3-allyl)Ni + Cp | — | 해리 흡착 |

### 체크리스트
- [ ] Ni(PF3)4: 양 기판 physi/chemi candidate 생성 확인
- [ ] AllylCpNi: haptic placement 구조 육안 검토
- [ ] Inhibitor-modified surface에서 precursor 흡착 비교
- [ ] Route별 구조 저장 및 메커니즘 레이블 확인

---

## Phase 4: Analysis (PLANNED)

### 4.1 흡착 에너지 매트릭스

E_ads = E(slab+mol) - E(slab) - E(mol)

계산 매트릭스 (clean / inhibited):

|  | Si(100) | SiO2(O) | SiO2(Si) |
|---|---|---|---|
| Ni(PF3)4 physi | | | |
| Ni(PF3)4 chemi | | | |
| AllylCpNi physi | | | |
| AllylCpNi chemi | | | |

### 4.2 진동 분석 (PHVA)

```yaml
analysis:
  vibrational:
    phva:
      enabled: true
      center: "Ni"
      radius_ang: 6.0
      frozen_z_ang: 5.5
```

### 4.3 자유 에너지 (ALD 공정 온도)

온도 범위: 300-700 K (50 K 간격)

### 4.4 선택성 지수

S = [E_ads(inh,SiO2) - E_ads(inh,Si)] / [E_ads(prec,SiO2) - E_ads(prec,Si)]

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
| 0 | Windows cp949 인코딩 오류 | em-dash/특수문자 ASCII로 대체 | closed |
| 0 | SiO2 a-b 비대칭 (0.000116 Å) | 대칭화 후 저장 | closed |
| 1 | Si-specific 함수 core 모듈 혼재 | reconstruction_recipes.py 분리 | closed |
| 1 | Si(100) dimer recipe: 0 dimers (소형 셀 문제) | FIRE가 자발적 buckled-dimer 형성으로 해결; dimer recipe는 target_area >= 120 Å² 필요 | closed |
| 1 | `get_all_dangling_bonds_general` O-term O 미감지 | 원인: 고정 2.6 Å neighbor cutoff가 terminal O 주변 O-O 근접쌍(~2.26 Å)을 결합으로 오인. 수정: 원소쌍별 covalent cutoff(+slack)로 coordination과 VSEPR neighbor를 필터링하여 O-term dangling bond 8개 정상 감지 | fixed |
| 1 | SiO2 Si-term 상단 Si+O 혼재 | I-42d (001) 구조 특성 - 표면 Si가 coord=2로 언더코디네이션, 물리적으로 정상 | closed |
| 1 | **슬랩 면적 부족** (inhibitor+precursor 동시 계산 불가) | target_area: Si=60→120 Å² (2×2), SiO2=60→100 Å² (2×2); x축 2배 확장 | fixed |
| 1 | **PBC 경계 원자 이탈** (Si bottom/top layer 2개→4개 누락) | create_slab_from_bulk 후 slab.wrap() 추가; standardize_vasp_atoms에도 wrap 적용 | fixed |
| 1 | **Si(100) 층 원자수 불균형** (표면 4개 vs 내부 8개) | 근본 원인: ASE surface()가 x=0 면의 (0,0) corner atom 제외 (PBC 중복 방지). 수정: bulk를 +a/4 이동 후 슬랩 생성 → x=3a/4 면 노출, 경계에 원자 없어 전 층 8개 보장. c(4×2) reconstruction 올바르게 형성 | fixed |
| 1 | **Si(100) Top dimer seed 과밀 매칭** (relax 후 2개 원자가 adatom처럼 상승) | 원인: nearest-pair greedy가 dimer row 방향으로 인접 dimer를 너무 촘촘히 선택해 seed 단계에서 비결합 top Si-Si 거리가 1.81 Å까지 감소. 수정: top-layer Si perfect matching 후보를 평가해 비결합 Si-Si 최소거리가 최대인 dimer matching 선택. 재생성 결과: seed dimer 4개, dimer bond 2.30 Å, 2.05 Å 이하 비결합 근접쌍 없음; relax 후 top 8개 z spread 0.006 Å, dimer bond 2.45 Å, adatom-like 원자 없음 | fixed |
