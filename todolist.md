# AutoFlow-SRXN 개발 로드맵

---

## Part 1. Metadynamics 구현 계획

### 배경 및 목표

현재 워크플로우는 정적 구조(minima)만 탐색한다. Metadynamics(MetaD)를 도입하면:
- 흡착 → 전이 상태 → 표면반응/탈착 경로의 자유에너지 면(FES) 재구성
- 희귀 이벤트(반응, 탈착) 가속 샘플링
- CRN 구축에 필요한 전이 상태 구조 자동 수집

구현 전략: ASE + PLUMED Python API (plumed PyPI 패키지) 연동.  
ASE 경로(MACE, SevenNet)는 PLUMED의 Python 커널을 통해 온더플라이(on-the-fly) 바이어스 계산.

---

### Phase 1 — 인프라: MD 궤적 덤프 + PLUMED 연동 준비

**목표**: MetaD 없이 plain MD 궤적을 저장하고, PLUMED 입력 뼈대를 자동 생성하는 유틸리티 구축.

**작업 목록**

- [ ] `src/potentials.py` — ASE `Dynamics`와 PLUMED 연동 테스트
- [ ] `src/metadynamics.py` (신규) — `PlumedInputBuilder` 클래스
  - `build_distance_cv(i, j)` → PLUMED DISTANCE 블록 생성
  - `build_coordination_cv(group1, group2, r0, nn, mm)` → COORDINATION 블록
  - `build_metad_block(cv_names, sigma, height, biasfactor, pace, stride)` → METAD 블록
  - `write_plumed_dat(path)` → 완성된 `plumed.dat` 파일 저장
- [ ] `config_full.yaml` — `metadynamics` 섹션 초안 추가
  ```yaml
  metadynamics:
    enabled: false
    cvs:
      - type: "distance"   # or "coordination"
        atoms: [idx_A, idx_B]
    sigma: 0.1
    height: 1.0      # kJ/mol
    biasfactor: 10
    pace: 500
    temp_K: 500
    steps: 500000
  ```

---

### Phase 2 — Well-Tempered MetaD 워크플로우

**목표**: 흡착질-표면 z-거리 또는 배위수를 CV로 삼아 탈착 자유에너지 ΔG_des 를 자동으로 계산.

**작업 목록**

- [ ] `src/metadynamics.py` — `WellTemperedMetaDEngine` 클래스
  - `setup(atoms, cv_config)`: CV 원자 인덱스 결정 (흡착질 무게중심 ↔ 표면 상단 원자)
  - `run(engine, steps, temp_K)`: ASE MD + PLUMED 바이어스 적용 → HILLS 파일 수집
  - `reconstruct_fes(hills_path, cvs)`: `plumed sum_hills` 래퍼로 FES 재구성
  - `extract_barriers()` → `{forward: float, reverse: float}` 반환 (eV 단위)
- [ ] `src/ads_workflow_mgr.py` — `run_metad_desorption(candidate)` 추가
  - 최적화된 흡착 구조(minima)를 시작점으로 MetaD 실행
  - FES에서 ΔG_des, ΔG_act 추출 → `candidate.thermo` 딕셔너리에 기록
- [ ] 출력: `results/metad_fes.png` (FES 시각화), `results/metad_summary.yaml`

**검증 기준**:  
SiO2(001)-OH 위 DIPAS 탈착 ΔG가 DFT 문헌값(~0.6–1.0 eV) 범위에 들어오는지 확인.

---

### Phase 3 — Path CV + Funnel MetaD (고급)

**목표**: 반응 경로 전체(흡착 → 리간드 교환 → 화학결합 형성)에 걸친 FES 탐색.

**작업 목록**

- [ ] Path CV 구현: 기존 relaxed candidate 구조들을 경로 노드로 사용
  - `src/metadynamics.py` — `PathCVBuilder.from_extxyz(path_xyz)`: extxyz에서 정렬된 구조 시퀀스 로드
  - PLUMED `PATH` collective variable 블록 자동 생성
- [ ] Funnel MetaD 지원: 흡착 공간을 실린더로 제한하는 PLUMED 입력 생성
- [ ] NEB(Nudged Elastic Band) 연동 가능성 평가
  - ASE `NEB` + MACE calculator로 TS 추출 → Path CV 노드로 활용
  - `src/neb_runner.py` (신규) 후보

**의존성**: Phase 1, 2 완료 후 착수.

---

## Part 2. Chemical Reaction Network (CRN) 구현 계획

### CRN이란?

Chemical Reaction Network(화학 반응 네트워크)는 화학종(species)을 **노드**, 소단계 반응(elementary step)을 **엣지**로 표현한 방향 그래프다.

```
A(g) + * ──k_ads──► A*          # 흡착
A*        ──k_des──► A(g) + *    # 탈착
A* + B*   ──k_rxn──► AB* + *    # 표면 반응
A*_site1  ──k_diff──► A*_site2  # 표면 확산
```

각 엣지는 **전방/역방 속도 상수** `k = ν × exp(-ΔG_act / kT)`를 가진다.  
`ν` (전-지수 인자): 전이 상태 이론(TST)으로 진동수에서 유도.  
`ΔG_act`: FES에서 추출하거나 NEB으로 계산.

표면 반응 시뮬레이션의 최종 산물은 미분방정식(ODE) 또는 KMC(Kinetic Monte Carlo)로 풀어 종 농도/덮힘율의 시간 진화를 얻는 것이다.

---

### AutoFlow-SRXN과 CRN의 매핑

| CRN 요소 | AutoFlow-SRXN 데이터 소스 |
|---|---|
| 화학종 구조 | `all_relaxed_candidates.extxyz` 각 프레임 |
| 전위 에너지 | `candidate.energy` (MLIP relaxation 결과) |
| 진동 자유에너지 G(T) | `qpoint.yaml` + `ThermoCalculator` |
| 흡착 에너지 ΔE_ads | `E(A*) - E(slab) - E(A_gas)` |
| 탈착/반응 장벽 ΔG_act | MetaD FES (Phase 2) 또는 NEB |
| 전-지수 인자 ν | PHVA 주파수 → TST `kT/h × exp(ΔS/k)` |
| 속도 상수 k | Arrhenius: `ν × exp(-ΔG_act / kT)` |

---

### 구현 계획

#### Step 1 — 데이터 모델 정의

**`src/crn.py`** (신규)

```python
@dataclass
class Species:
    name: str           # 예: "DIPAS*_site_1", "DIPAS(g)", "slab"
    atoms: Atoms        # ASE Atoms 객체
    energy: float       # eV, MLIP 결과
    gibbs: dict         # {T: G(T)} — ThermoCalculator 출력
    is_gas: bool = False

@dataclass
class ReactionStep:
    reactants: list[Species]
    products: list[Species]
    step_type: str      # "adsorption" | "desorption" | "surface_rxn" | "diffusion"
    dG_act_forward: float    # eV
    dG_act_reverse: float    # eV
    prefactor: float         # s^-1, TST에서 계산
    temperature_K: float

    def rate_constant(self) -> float:
        return self.prefactor * np.exp(-self.dG_act_forward / (kB * self.temperature_K))
```

#### Step 2 — CRN 그래프 빌더

**`src/crn.py`** — `CRNBuilder` 클래스

- `add_adsorption_step(slab, gas_mol, adsorbed)`: E_ads = E(adsorbed) - E(slab) - E(gas) 계산
- `add_desorption_step(adsorbed, slab, gas_mol, dG_act)`: MetaD 결과 연결
- `add_surface_rxn_step(reactant_ads, product_ads, dG_act)`: 표면 내 반응
- `add_diffusion_step(site_a, site_b, dG_act)`: 사이트 간 이동
- `build_graph()` → `networkx.DiGraph` 반환

#### Step 3 — 속도 상수 계산기

**`src/crn.py`** — `TSTRateCalculator` 클래스

- 입력: `Species` (reactant, TS 또는 product), `temperature_K`
- PHVA 주파수 (`qpoint.yaml`)로 ZPE·엔트로피 보정
- Eyring-Evans-Polanyi: `k = (kT/h) × exp(-ΔG_act / kT)`
- 흡착의 경우 비충돌 이론(Hertz-Knudsen)으로 `k_ads` 계산

#### Step 4 — ODE/KMC 인터페이스

- `src/crn_solver.py` (신규)
  - `MeanFieldSolver`: scipy ODE 기반 평균장 근사 (덮힘율 θ_i 시간 진화)
  - `KMCSolver`: 향후 확장용 (KMC 알고리즘 뼈대)
- 출력: `crn_results.yaml` — 각 온도·압력에서 `{species: coverage_at_steady_state}`

#### Step 5 — 워크플로우 통합

**`src/ads_workflow_mgr.py`** 확장

```
run() 완료 후:
  → CRNBuilder.add_adsorption_step() 자동 호출
  → MetaD 장벽이 있으면 add_desorption_step() 추가
  → crn.build_graph() → 그래프 저장 (JSON 또는 graphml)
  → TSTRateCalculator 실행 → rate constants 기록
  → MeanFieldSolver 실행 → 덮힘율 출력
```

---

### 원자 구조 관점의 인풋 전략

| 반응 유형 | 초기 구조 소스 | 전이 상태(TS) 탐색 방법 |
|---|---|---|
| 흡착 | `physisorption` 후보 (run_autoflow) → `chemisorption` relaxed 구조 | MetaD z-CV 또는 Hertz-Knudsen (무장벽 가정) |
| 탈착 | 흡착 구조 + 진공 분리 구조 | MetaD FES 역전 |
| 리간드 교환 | 흡착 단계 중간체 구조 | NEB (ASE NEB + MACE) |
| 표면 확산 | 인접 흡착 사이트 2개 | NEB 또는 MetaD Path CV |
| 해리 흡착 | `chemisorption_builder` 분할 구조 | NEB |

**extxyz 활용**: `all_relaxed_candidates.extxyz`의 각 프레임이 CRN의 한 Species 노드.  
`info` 딕셔너리에 `energy`, `candidate_type` 등이 저장되어 있어 직접 파싱 가능.

---

### CRN 출력 예시

```yaml
# crn_output.yaml
species:
  - name: "DIPAS(g) + SiO2_slab"
    energy: -1234.56   # eV
    gibbs_298K: -1234.20
  - name: "DIPAS_physi*"
    energy: -1235.12
    gibbs_298K: -1234.68
  - name: "DIPAS_chemi* (Si-O bond)"
    energy: -1236.80
    gibbs_298K: -1236.10

reactions:
  - type: "adsorption"
    from: "DIPAS(g) + SiO2_slab"
    to: "DIPAS_physi*"
    dG_act_fwd: 0.0    # barrierless
    dG_act_rev: 0.72   # eV, from MetaD
    k_fwd_298K: 1.2e7  # s^-1
    k_rev_298K: 3.4e2

  - type: "surface_rxn"
    from: "DIPAS_physi*"
    to: "DIPAS_chemi*"
    dG_act_fwd: 0.45
    dG_act_rev: 1.13
    k_fwd_298K: 8.5e4
    k_rev_298K: 1.1e-1
```

---

## 우선순위 요약

| 단계 | 작업 | 난이도 | 선행 조건 |
|---|---|---|---|
| 1 | MetaD Phase 1: PLUMED 인프라 | 중 | PLUMED 설치 |
| 2 | CRN 데이터 모델 (`crn.py`) | 낮 | 없음 |
| 3 | MetaD Phase 2: Well-Tempered MetaD | 높 | Phase 1 |
| 4 | CRN 속도 상수 계산 (TSTRateCalculator) | 중 | MetaD Phase 2, PHVA |
| 5 | ODE 솔버 (`crn_solver.py`) | 중 | CRN 데이터 모델 |
| 6 | MetaD Phase 3: Path CV + NEB | 매우 높 | Phase 2 |
| 7 | CRN 워크플로우 통합 | 중 | 모든 이전 단계 |
