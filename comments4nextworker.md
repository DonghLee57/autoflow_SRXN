# AutoFlow-SRXN: Transition State Workflow Development Status

## 1. 요청 사항 (User Request)
- **핵심 목표:** 물리흡착(Physisorption)과 화학흡착(Chemisorption) 구조를 자동으로 연결하여 전이 상태(Transition State, TS)를 찾는 워크플로우 구현.
- **방법론:** NEB(Nudged Elastic Band)로 경로를 찾고, ARTn(Activation Relaxation Technique)으로 Saddle Point를 정밀화(Refinement)하는 2단계 프로세스.
- **검증 시스템:** 문헌 데이터가 존재하는 TiN(111) 표면 + TiCl4 전구체 시스템 (MACE-MP 포텐셜 사용).

## 2. 현재까지 수행한 작업 (Completed Tasks)
### A. 인프라 및 핵심 엔진 구현
- **Deterministic Metadata Tracking:** `ChemisorptionBuilder`가 분자를 쪼갤 때의 인덱스 변화를 `index_mapping` 정보로 저장하도록 수정. 이를 통해 NEB 실행 전 초기/기말 상태의 원자 순서를 완벽하게 일치시킴.
- **TransitionStateWorkflow 클래스 제작:**
  - `align_states`: `index_mapping` 및 COM(Center of Mass) 기반 주기적 경계 조건(PBC) 대응 정렬 로직 구현.
  - `run_ts_search`: NEB 실행 후 최고 에너지 이미지를 ARTn으로 넘겨주는 파이프라인 구축.
- **ARTn 엔진 개선:** `ARTSearcher`가 고정 원자(Fixed layers)가 포함된 시스템에서도 차원 오류 없이 작동하도록 `VibrationalAnalyzer`의 패딩된 고유벡터를 사용하도록 수정.

### B. 예제 구축 및 실행 (TiN + TiCl4)
- **구조 생성:** 2x2 TiN(111) Slab(32 atoms) 및 TiCl4 분자 생성 스크립트 작성. 또한, `C(N(C)C)(OC)(OC)` SMILES로부터 `secret_inhibitor.vasp` 구조를 생성하여 `structures/`에 저장함.
- **워크플로우 통합:** `run_adsorption.py`에 `execute_ts_search_stage`를 추가하여 후보군 탐색 후 자동으로 TS Search가 이어지도록 통합.
- **물리적 타당성 검사:** TS Barrier가 10eV를 초과할 경우 PBC 문제로 간주하고 경고를 출력하는 로직 추가.

## 3. 직면한 문제 및 해결 시도 (Current Issues & Debugging)
- **PBC Displacement Issue:** NEB 실행 시 원자들이 주기적 경계를 가로질러 이동하면서 비정상적으로 높은 Barrier(20eV+)가 발생하는 현상 확인.
  - *해결 시도:* `align_states`에서 흡착제의 COM을 기말로 맞추는 Shift 로직을 적용했으나, 여전히 NEB 과정에서 이미지가 꼬일 가능성이 있음.
- **ARTn Shape Mismatch:** 고정 원자가 있을 때 `VibrationalAnalyzer`가 활성 원자에 대해서만 계산하면서 발생한 Array mismatch 해결 완료.

## 4. 남은 작업 (Pending Tasks)
1.  **TiN + TiCl4 결과 최종 분석:** 현재 실행 중인 계산(PID 27240 혹은 재실행 필요)이 끝나면 `ts_refined.vasp`와 Barrier(0.5~1.0eV 예상)를 문헌 값과 비교.
2.  **NEB 경로 안정화:** 만약 여전히 20eV 이상의 Barrier가 나온다면, `ase.mep.neb.IDPP` 보간 시 원자들이 "최단 경로"가 아닌 "PBC를 넘는 경로"를 선택하지 않도록 `initial.wrap()` 및 `final.wrap()` 처리를 더 정밀하게 다듬어야 함.
3.  **Protector Exchange 대응:** 현재는 원자 수가 변하지 않는 해리(Dissociation) 반응만 지원함. 원자 수가 변하는 Protector exchange 메커니즘을 위한 인덱스 더미 원자 추가 로직 필요.
4.  **Vibrational Verification 연동:** TS 탐색 완료 후, 자동으로 주파수 계산을 수행하여 허수 진동 모드가 정확히 1개인지 확인하는 로직 추가.

## 5. 다음 작업자를 위한 가이드 (Instructions for Next Session)
- `examples/TiN_TiCl4_MACE/run_tin_example.py`를 실행하여 워크플로우가 끝까지 에러 없이 도는지 확인하십시오.
- `results/clean_on_TiCl4/ts_search_0/` 폴더 내의 VASP 파일들을 시각화하여 NEB 경로가 물리적으로 타당한지(원자가 순간이동 하지 않는지) 체크하십시오.
- Barrier가 1.5eV 이하로 내려가지 않는다면 `ts_workflow.py`의 `align_states` 함수에서 PBC 이미지 매칭 부분을 개선해야 합니다.
