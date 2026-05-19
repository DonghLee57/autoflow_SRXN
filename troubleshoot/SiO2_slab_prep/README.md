# Troubleshooting: SiO2 Slab Prep — Termination & Passivation

---

## 1. 증상

`run_generic_adsorption_study(config.yaml)` 실행 시 passivation 설정이 올바름에도
불구하고 출력 `prepared_slab.extxyz`에 H 원자가 나타나지 않음.

---

## 2. 근본 원인

`main_workflow.py`의 `prepare_slab_stage()` 함수에서 `create_slab_from_bulk()`를
호출할 때 `top_termination` / `bottom_termination` 파라미터를 전달하지 않았음.

```python
# 수정 전 — config의 termination 설정이 무시됨
slab = create_slab_from_bulk(
    bulk_atoms=read(paths["substrate_bulk"]),
    miller_indices=sub_gen_cfg.get("miller", [1, 0, 0]),
    thickness=sub_gen_cfg.get("thickness_ang", 10.0),
    vacuum=sub_gen_cfg.get("vacuum_ang", 10.0),
    target_area=sub_gen_cfg.get("target_area_ang2"),
    supercell_matrix=sub_gen_cfg.get("supercell_matrix"),
    bulk_shift=sub_gen_cfg.get("bulk_shift", 0.0),
    verbose=True,
    # top_termination, bottom_termination 누락!
)
```

결과: SiO2(001) 슬랩이 O-terminated 대신 디폴트 컷으로 생성되어
바닥 면이 Si-terminated 됨. `passivate_surface_coverage_general`은
Si 원자의 dangling bond에 H를 배치하게 되고, 이 Si-H 결합은 MLIP
릴랙스(`slab_relax: true`) 중에 분리되어 최종 출력에 H가 없는 것처럼 보임.

### 왜 `passivate_surface_coverage_general` 자체는 정상인가

- O-terminated slab(올바른 입력)에 직접 호출하면 바닥면 O 원자 18개 각각의
  dangling bond를 정확히 감지하고 O–H(0.96 Å)를 배치함.
- 문제는 슬랩 생성 단계에서 잘못된 면을 노출하는 것이지,
  passivation 로직 자체가 아님.

---

## 3. 수정 내용

`autoflow_srxn/surface/main_workflow.py` — `prepare_slab_stage()`:

```python
# 수정 후 — termination 파라미터 전달
slab = create_slab_from_bulk(
    bulk_atoms=read(paths["substrate_bulk"]),
    miller_indices=sub_gen_cfg.get("miller", [1, 0, 0]),
    thickness=sub_gen_cfg.get("thickness_ang", 10.0),
    vacuum=sub_gen_cfg.get("vacuum_ang", 10.0),
    target_area=sub_gen_cfg.get("target_area_ang2"),
    supercell_matrix=sub_gen_cfg.get("supercell_matrix"),
    bulk_shift=sub_gen_cfg.get("bulk_shift", 0.0),
    top_termination=sub_gen_cfg.get("top_termination"),       # 추가
    bottom_termination=sub_gen_cfg.get("bottom_termination"), # 추가
    verbose=True,
)
```

---

## 4. 검증된 config (`config.yaml`)

```yaml
surface_prep:
  slab_generation:
    enabled: true
    miller: [0, 0, 1]
    thickness_ang: 12.0
    vacuum_ang: 15.0
    target_area_ang2: 250.0
    top_termination: "O"      # O-terminated top
    bottom_termination: "O"   # O-terminated bottom (passivation 대상)

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

수정 후 결과: 바닥면 O 원자 18개에 H 배치 → `H18O144Si63` (225 atoms).

---

## 5. 알려진 추가 사항

### output_prefix 경로

`config.yaml`의 `output_prefix: "results"` 는 상대 경로이며,
`run_generic_adsorption_study` 실행 시 **현재 작업 디렉토리(CWD)** 기준으로
해석됨. config 파일 위치 기준이 아님.

- SiO2_slab_prep 디렉토리 안에서 실행할 경우:
  `troubleshoot/SiO2_slab_prep/results/prepared_slab.extxyz` 생성
- 프로젝트 루트에서 실행할 경우:
  `<repo_root>/results/prepared_slab.extxyz` 생성

절대 경로를 사용하거나 config 파일과 같은 디렉토리에서 실행하는 것을 권장.

### slab_relax와 MLIP 포텐셜

`slab_relax: true` 설정 시 바닥 H 원자는 `frozen_z_ang: 5.5` 규칙에 따라
동결됨(z_min + 5.5 Å 이하의 모든 원자 고정). 따라서 패시베이션 구조는 릴랙스
후에도 유지되어야 함. 단, 사용하는 MLIP 포텐셜(SevenNet matpes_pbe 등)이
SiO2 + O-H 환경에서 큰 비물리적 힘을 발생시킬 경우 상위 레이어의 릴랙스
동작이 불안정할 수 있음. 이 경우 `slab_relax: false`로 설정하여 패시베이션
결과를 먼저 확인 후 릴랙스 활성화를 권장.
