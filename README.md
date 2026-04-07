# AutoFlow-SRXN: Automated Surface Reaction Workflow

**AutoFlow-SRXN**은 다양한 전구체(Precursor)와 기판(Substrate) 사이의 흡착 및 반응 구조를 고속으로 탐색하고 생성하기 위한 고도화된 자동화 프레임워크입니다.

---

## 🚀 주요 기능 (Key Features)

### 1. 지능형 전구체 파편화 및 중복 제거
- **그래프 기반 파편화**: RDKit 및 ASE를 활용하여 전구체 내 리간드를 자동으로 탐색합니다.
- **화학식 기반 그룹화**: 동일한 화학식을 가진 리간드(예: 3개의 H 리간드)를 자동으로 식별하고, 불필요한 반복 계산을 방지하여 탐색 효율을 극대화합니다.

### 2. 알고리즘 기반 메커니즘 라우팅 (Algorithmic Chemisorption Builder)
- **표면 반응성 동적 분석**: 기호(Label)에 의존하지 않고, 전달된 기판(Substrate)의 기하학적 구조를 동적으로 분석하여 반응 가능한 노출 다이머(Dimer)와 상단(Top) 표면의 수소 패시베이션 사이트를 찾아냅니다. 
- **물리 흡착 (Physisorption)**: 전구체 전체에 대한 구형 회전 및 높이별 배치 최적화.
- **해리형 화학 흡착 (Dissociative Chemisorption)**: 전구체 내 리간드 해리 경로 도출 후 표면 다이머 양 끝단에 자동으로 정밀 맵핑.
- **수소 교환 반응 (H-Exchange)**: 패시베이션된 표면에서 발생하는 리간드-수소 교환 구조 및 발생하는 가스 부산물(`ex: byproduct=H2`)을 자동 계산하여 반영.

### 3. 실리콘 표면 엔지니어링 (Si-Specific Utilities)
- **Si(100) 2x1 Reconstructed**: 버클링(Buckling) 정렬이 포함된 표준 재구성 표면 생성.
- **산화 및 패시베이션**: Greedy Max-Min 거리 알고리즘을 통한 균일한 산화막(Oxygen Bridge) 및 수소 패시베이션 처리.

### 4. Gibbs 자유 에너지 및 열역학 분석 (Thermodynamics)
- **VDOS 기반 분석**: Phonopy의 진동 상태 밀도(VDOS) 데이터를 파싱하여 $ZPE, S_{vib}(T), G_{vib}(T)$ 산출.
- **Phase별 모드 분화**: 
    - **Gas Mode**: Sackur-Tetrode 식과 로테이션 보정, 엔탈피($PV$) 항을 포함한 기체 분자 자유 에너지 계산.
    - **Adsorbate/Substrate Mode**: 고정된 표면계에 최적화된 조화 고체(Harmonic Solid) 모델 기반 분석.
- **단위 변환 및 자동화**: $THz, cm^{-1}, eV, kJ/mol$ 간의 자동 변환 및 온도 의존성 리포트 생성.

### 5. 정밀 자가 진단 (Atomic-Level Diagnostics)
- **`verbose=True` 모드**: 라이브러리 내부에서 리간드 분석 및 스테릭 충돌 현황을 실시간 보고.
- **원자 수준 로그**: `[Overlap] Si(12) - C(45) clash (dist: 1.15 A)`와 같이 충돌 원자의 인덱스와 거리를 상세히 출력하여 디버깅 용이성 확보.

---

## 🏗️ 시스템 아키텍처 (Architecture)

### Core Libraries
- `example_dipas/ads_workflow_mgr.py`: 범용 흡착 워크플로우를 관장하며 리간드/반응 파편을 생성하는 매니저 클래스 (`AdsorptionWorkflowManager`).
- `example_dipas/chemisorption_builder.py`: 기하 구조 기반 동적 표면 분석 및 경로/메커니즘 스티어링, 결과 구조 조립을 담당하는 모듈.
- `example_dipas/surface_utils.py`: 기하 구조 기반 사이트 식별 및 범용 표면 패시베이션 기능을 지원합니다.

### Specialty Modules
- `example_dipas/si_surface_utils.py`: 실리콘(Si) 표면 특화 재구성 및 산화/수화 유틸리티.
- `free_energy/thermo_engine.py`: 통계 역학 기반 열역학 보정 엔진.
- `free_energy/analyze_thermo.py`: Phonopy YAML 결과 파싱 및 자유 에너지 분석 CLI 도구.

---

## 🛠️ 시작하기 (Quick Start)

DIPAS 전구체를 활용한 통합 연구 예제를 실행하려면 `example_dipas` 디렉토리에서 다음 명령을 수행하십시오.

```bash
cd example_dipas
python run_dipas_study.py
```

이 명령은 다음 4가지 표준 표면을 자동 생성하고 흡착 시뮬레이션을 수행합니다:
1. `Clean 2x1 Reconstructed`
2. `H-Passivated`
3. `Oxidized (50%)`
4. `Oxidized + H-Passivated`

### 🌡️ 열역학 분석 (Gibbs Free Energy)

Phonopy 계산 결과를 바탕으로 자유 에너지를 산출하려면 `free_energy` 디렉토리에서 다음 명령을 수행하십시오.

```bash
cd free_energy
# 기체 분자 분석 예시 (질량 18.015, 대칭수 2)
python analyze_thermo.py phonopy.yaml --energy -14.5 --mode gas --mass 18.0 --sigma 2

# 흡착 구조 분석 예시
python analyze_thermo.py phonopy.yaml --energy -500.4 --mode adsorbent
```

---

## 📝 라이선스 및 연락처
본 프로젝트는 고효율 표면 반응 탐색을 위해 설계되었습니다. 문의 사항이 있으시면 담당 연구원에게 연락해 주시기 바랍니다.
