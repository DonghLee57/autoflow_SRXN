# Interface Builder Task Log

Target: Build crystal interfaces between t-ZrO2 (substrate) and NbO, NbO2, Nb2O5, Ta2O5 (films)
Branch: `dev`

---

## Structures

| Material | File | Space Group | Source |
|---|---|---|---|
| t-ZrO2 | `structures/ZrO2_t_bulk.vasp` | P42/nmc (#137) | existing |
| NbO | `structures/NbO_bulk.vasp` | Pm-3m (#221) | created |
| NbO2 | `structures/NbO2_bulk.vasp` | P42/mnm (#136, rutile) | created |
| B-Nb2O5 | `structures/Nb2O5_B_bulk.vasp` | C2/m (#12) | created |
| B-Ta2O5 | `structures/Ta2O5_B_bulk.vasp` | C2/m (#12) | created |

구조 생성/검증 스크립트:
- `structures/create_oxide_structures.py` — pymatgen으로 벌크 구조 생성
- `structures/validate_oxide_structures.py` — 격자 파라미터, 최소 원자간 거리 검증
- `structures/fetch_mp_structures.py` — Materials Project에서 공식 구조 다운로드 (mp-2574, mp-2311, mp-821, mp-581967, mp-10390)

---

## 계면 매칭 파이프라인 개요

```
bulk structure
    |
    v  get_surface_lattice_2d(structure, miller)
2D canonical lattice A  [|v1|, 0]
                        [|v2|*cos_gamma, |v2|*sin_gamma]
    |
    v  find_coincidences(A_sub, A_film, max_det, strain_cutoff)
       HNF 열거 (Zur & McGill 1984): Na, Nb 쌍 탐색
       F = (Na @ A_sub) @ inv(Nb @ A_film)
       vm (von Mises strain) = sqrt(0.5*(eps1^2 + eps2^2 + (eps1-eps2)^2))
    |
    v  build_symmetric_slab(structure, miller, min_thickness, vacuum, HNF)
       pymatgen SlabGenerator(symmetrize=True) + HNF supercell
       rotate normal -> z
    |
    v  stack_interface(sub_slab, film_slab, gap, vacuum)
       v1-alignment rotation + film_frac @ sub_cell_2d (epitaxial strain 모델)
```

---

## 완료된 작업

### [DONE] 구조 파일 및 예제 설정
- NbO, NbO2, B-Nb2O5, B-Ta2O5 벌크 VASP 파일 생성 및 검증
- 4개 계면 쌍 예제 디렉터리 및 config 작성:
  - `examples/interface_match/ZrO2_t_NbO/`
  - `examples/interface_match/ZrO2_t_NbO2/`
  - `examples/interface_match/ZrO2_t_Nb2O5/`
  - `examples/interface_match/ZrO2_t_Ta2O5/`
- 배치 실행 스크립트 `examples/interface_match/run_all_ZrO2_interfaces.py`

### [DONE] 슬랩 생성 단계 점검 (troubleshoot/)
진단 스크립트를 통해 파이프라인 각 단계를 체계적으로 검증:

| 단계 | 결과 |
|---|---|
| `get_surface_lattice_2d` 정규화 변환 | ZrO2/NbO/NbO2: 정확. Nb2O5/Ta2O5: 버그 발견 → 수정 완료 |
| `find_coincidences` HNF 열거 및 vm 계산 | 정확 |
| `build_symmetric_slab` HNF 슈퍼셀 | 3D 셀 크기가 2D 예측과 정확히 일치 (모든 케이스) |
| 원자 수 = det(HNF) × 기본셀 원자 수 | 전부 통과 |
| 최소 원자간 거리 > 1.5 A | 전부 통과 (2.0~2.1 A) |

### [DONE] Bug Fix: `get_surface_lattice_2d`

**원인**: `min_vacuum_size=0`을 사용하면 pymatgen이 진공 없는 벌크 셀(14원자, C2/m의 경우 B, C 벡터)을 슬랩으로 반환해 표면 격자가 아닌 벌크 셀 벡터를 in-plane으로 사용했음.

**증상**: B-Nb2O5/Ta2O5 (001)에서
- 수정 전: `|v2|=5.561, gamma=90°` (잘못된 벌크 c축)
- 수정 후: `|v2|=6.815, gamma=69°` (올바른 C2/m 표면 원시셀 = b, (A+B)/2)

**수정 내용** (`autoflow_srxn/interface/builder.py`):
```python
# Before
gen = SlabGenerator(..., min_slab_size=1, min_vacuum_size=0, ...)
slabs = gen.get_slabs()
v1, v2 = slab.lattice.matrix[0], slab.lattice.matrix[1]  # 3D 벡터를 그대로 사용

# After
gen = SlabGenerator(..., min_slab_size=8.0, min_vacuum_size=1.0, ...)
slabs = gen.get_slabs(symmetrize=False)
# rotate normal -> z (build_symmetric_slab과 동일)
atoms.rotate(normal, [0,0,1], rotate_cell=True)
v1_xy = cell[0, :2]  # z-회전 후 XY 성분만 사용
v2_xy = cell[1, :2]
```

**수정 후 검증** (`troubleshoot/verify_fix.py`):
- ZrO2(101,001,111), NbO2(001), NbO(110): 전부 `get_surface_lattice_2d` vs `build_symmetric_slab` 일치 확인
- Nb2O5(001): `get_surface_lattice_2d` 정확, `build_symmetric_slab(min_t=10)` 실패 → config 수정 필요 (아래)

**관련 config 수정**:
- `ZrO2_t_Nb2O5/config.yaml`: `min_slab_thickness: 10 → 15` (C2/m symmetrize=True 최소 요구치)
- `ZrO2_t_Ta2O5/config.yaml`: `min_slab_thickness: 10 → 15`

---

## 미완료 / 알려진 문제

### [TODO-1] `stack_interface`: 비직교 격자에서 회전 오류

**문제**: `stack_interface`가 v1 방향만 정렬하는 단일 각도 회전을 적용.
직교 격자(gamma=90°)에서는 충분하지만, 비직교 격자에서 최적 회전과 최대 ~9.8° 차이 발생.

**영향받는 케이스**:
- ZrO2(111)/NbO(110): sub gamma=66°, v2 mismatch 0.92 A, 회전 오차 9.79°
- Nb2O5(001) 포함 계면: gamma=69° (C2/m 표면 원시셀)

**수정 방향**:
```python
# 현재: v1 방향만 정렬
angle_sub  = arctan2(v1_sub[1], v1_sub[0])
angle_film = arctan2(v1_film[1], v1_film[0])
rot = angle_sub - angle_film

# 수정: polar decomp F = R @ U 에서 최적 회전 R 사용
from scipy.linalg import polar
F = A_Na @ inv(A_Nb)   # 2D canonical 공간에서
R, U = polar(F)
theta_opt = arctan2(R[1,0], R[0,0])
rot = (angle_sub - angle_film) + theta_opt
```

**참고**: vm 값은 rotation-invariant이므로 `find_coincidences`의 vm 계산은 올바름.
회전 오류는 원자 배치에만 영향 (비물리적 전단변형 추가).

### [TODO-2] `wrap_interface_for_dft` 제거

`builder.py` lines 323~395에 `wrap_interface_for_dft()` 함수가 존재.
사전에 추가된 것으로 현재 단계에서 불필요. 제거 필요.

### [TODO-3] ZrO2/NbO 매칭 범위 확장

ZrO2(001)/NbO(001): max_det=12 이내 매칭 없음.
NbO a=4.21 A vs ZrO2 a=3.60 A는 17% 불일치. 공약격자에는 det=42 이상 필요.
현재 ZrO2(111)/NbO(110)에서만 매칭 (vm=4.35%).

옵션:
- max_det 확대 (성능 저하)
- NbO 전용 Miller 지수 목록 직접 지정

### [TODO-4] Materials Project 공식 구조 사용

현재 구조는 `create_oxide_structures.py`로 생성한 것.
`fetch_mp_structures.py`로 MP 공식 구조 (mp-2311, mp-821, mp-581967, mp-10390) 다운로드 후 교체 예정.

---

## 진단 스크립트 (troubleshoot/)

| 파일 | 내용 |
|---|---|
| `inspect_matching_logic.py` | 매칭 파이프라인 전체 수치 추적 |
| `check_supercell_consistency.py` | 슈퍼셀 vs 벌크 동등성 검증 |
| `check_monoclinic_slabs.py` | Nb2O5/Ta2O5 슬랩 생성 가능 Miller 면 탐색 |
| `check_lattice2d_consistency.py` | min_slab_size/vacuum 파라미터 영향 비교 |
| `check_oriented_unit_cell.py` | oriented_unit_cell vs slab 격자 비교 |
| `print_slab_cells.py` | 실제 pymatgen slab 셀 행렬 출력 |
| `verify_fix.py` | get_surface_lattice_2d 수정 검증 |

---

## 예제 실행

```bash
# 단일 계면 쌍
cd examples/interface_match/ZrO2_t_NbO2
python run_interface.py

# 전체 배치
cd examples/interface_match
python run_all_ZrO2_interfaces.py --strain 0.08 --top-k 5
```

---

## 검증된 계면 후보 (수정 전 예비 결과)

| 계면 | 최적 면 | Na | Nb | vm | 원자수(예상) |
|---|---|---|---|---|---|
| ZrO2/NbO2 | (101)/(001) | [[4,0],[0,3]] | [[3,0],[0,4]] | 1.33% | 264+288=552 |
| ZrO2/NbO | (111)/(110) | [[5,0],[0,2]] | [[6,0],[1,2]] | 4.35% | 290+264=554 |
