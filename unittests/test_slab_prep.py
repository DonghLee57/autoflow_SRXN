"""
test_slab_prep.py
=================
Geometry-only tests for the slab preparation stage.

두 케이스를 검증합니다:
  1. Si(100) — Si_mp149.vasp + 2×1 buckled-dimer 재구성
  2. SiO2(001) — SiO2_mp-546794.vasp + O-terminated top/bottom + bottom H passivation

모든 테스트는 MLIP 없이 순수 geometry 연산만 실행합니다
(slab_relax=False, candidate_relax=False).

테스트 레벨
-----------
Unit        : prepare_slab_stage() 직접 호출 → 세분화된 구조 검증
              결과 slab 을 unittests/output_slab_prep/ 에 저장
Integration : run_generic_adsorption_study() 엔드투엔드 검증
              output_prefix = unittests/output_slab_prep/{si100,sio2}/

결과 파일 (테스트 후 VESTA 등으로 확인 가능)
-------------------------------------------
  output_slab_prep/si100_unit_prepared_slab.extxyz   ← Unit Si(100)
  output_slab_prep/sio2_unit_prepared_slab.extxyz    ← Unit SiO2(001)
  output_slab_prep/si100/prepared_slab.extxyz        ← Integration Si(100)
  output_slab_prep/sio2/prepared_slab.extxyz         ← Integration SiO2(001)

Config 파일
-----------
  config_slab_prep_si100.yaml  — structures/Si_mp149.vasp 사용
  config_slab_prep_sio2.yaml   — structures/SiO2_mp-546794.vasp 사용

Si(100) bulk_shift PBC 버그 수정 (surface_utils.py)
----------------------------------------------------
ASE surface(vacuum=0) 는 cell_c = z_max (원자 범위)로 설정하므로
최상단 원자가 분수 좌표 1.0 == 0.0 (PBC). bulk_shift=0.25 적용 시
z=0 원자와 z=cell_c 원자가 동일 위치로 합쳐져 16-원자 이중층 생성 (버그).
수정: bulk_shift 적용 전 cell_c = num_layers * d_hkl 로 재설정.

Si(100) bottom H passivation
------------------------------
bulk_shift 버그 수정 후 bottom Si 는 coord=2 (위로 2개 결합, 아래로 dangling 2개).
passivation.enabled=true → 각 bottom Si 에 H 2개 배치 (Si-H 1.48 A).
총 8 Si × 2 H = 16 H 원자.

bond_slack 가이드 (get_all_dangling_bonds_general)
---------------------------------------------------
bond_slack=0.20 → cutoff 2.42 A : MLIP 릴랙스 후 dimer 결합(~2.46 A) 불인식 (원래 버그)
bond_slack=0.45 → cutoff 2.67 A : 구 slab 에서 up-dimer 백본드(2.705 A) 불인식
bond_slack=0.50 → cutoff 2.72 A : 모든 케이스 커버 -- 테스트 기준값

SiO2 termination 이슈
----------------------
top_termination/bottom_termination 이 create_slab_from_bulk 에 전달되어야
O-terminated slab 이 생성됨. 누락 시 Si-terminated 기본 컷 적용 → H 미생성.
"""

import logging
import os
import unittest

import numpy as np
from ase.io import read, write as ase_write
from ase.neighborlist import neighbor_list

from autoflow_srxn.surface.main_workflow import (
    prepare_slab_stage,
    run_generic_adsorption_study,
)
from autoflow_srxn.surface.surface_utils import (
    find_surface_indices,
    get_all_dangling_bonds_general,
)
from autoflow_srxn.utils.config_utils import load_yaml_config


# ─────────────────────────────────────────────────────────────────────────────
# 경로 상수
# ─────────────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.normpath(os.path.join(_HERE, ".."))

_CONFIG_SI100 = os.path.join(_HERE, "config_slab_prep_si100.yaml")
_CONFIG_SIO2  = os.path.join(_HERE, "config_slab_prep_sio2.yaml")

# 테스트 결과 구조 파일을 저장하는 고정 디렉토리 (tmpdir 가 아님)
_OUTPUT_DIR = os.path.join(_HERE, "output_slab_prep")


# ─────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _silent_logger(name: str = "test_slab_prep") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _load_config(yaml_path: str, output_prefix: str) -> dict:
    """YAML 을 load_yaml_config 로 파싱 후 output_prefix 를 지정 경로로 교체.

    load_yaml_config 는 substrate_bulk 를 config 파일 위치 기준 절대경로로 변환.
    output_prefix 는 미존재 디렉토리이므로 여기서 명시 설정.
    """
    cfg = load_yaml_config(yaml_path)
    cfg["paths"]["output_prefix"] = output_prefix
    return cfg


def _sio2_vasp_exists() -> bool:
    return os.path.exists(os.path.join(_REPO_ROOT, "structures", "SiO2_mp-546794.vasp"))


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 : Si(100) — 2×1 buckled-dimer 재구성
# ─────────────────────────────────────────────────────────────────────────────

class TestSi100SlabPrep(unittest.TestCase):
    """
    config_slab_prep_si100.yaml (structures/Si_mp149.vasp) 을 사용해
    prepare_slab_stage() 로 생성한 Si(100) slab 을 검증.

    검증 항목:
      - 원소: Si + H (bottom H passivation)
      - Bottom: 각 Si 에 H 2개 (Si-H 1.4-1.6 A), bottom Si coord=4
      - Top:  2x1 buckled-dimer 재구성 (dimer 결합 2.10-2.65 A, dimer Si dangling=1)
      - 층 구조: 모든 내부 layer n=8 (bulk_shift PBC 버그 수정 후 double-layer 없음)

    결과 slab: output_slab_prep/si100_unit_prepared_slab.extxyz
    """

    @classmethod
    def setUpClass(cls):
        cls.logger = _silent_logger("test_si100")
        out_prefix = os.path.join(_OUTPUT_DIR, "si100_unit")
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        config = _load_config(_CONFIG_SI100, out_prefix)
        cls.slab = prepare_slab_stage(config, cls.logger)
        cls.syms = np.array(cls.slab.get_chemical_symbols())

        # 결과 구조 저장 (VESTA 등으로 육안 확인용)
        out_path = os.path.join(_OUTPUT_DIR, "si100_unit_prepared_slab.extxyz")
        ase_write(out_path, cls.slab)

    # ── 원소 구성 ─────────────────────────────────────────────────────────────

    def test_contains_si_and_h_only(self):
        """bottom H passivation 으로 Si + H 만 존재해야 한다."""
        sym_set = set(self.syms.tolist())
        self.assertIn("Si", sym_set, "Si 없음")
        self.assertIn("H",  sym_set, "H 없음 — bottom passivation 실패")
        self.assertFalse(sym_set - {"Si", "H"},
                         f"예상 외 원소: {sym_set - {'Si', 'H'}}")

    # ── Bottom H passivation ──────────────────────────────────────────────────

    def test_bottom_h_count(self):
        """8 bottom Si × 2 dangling bond = H 16개가 생성되어야 한다."""
        n_h = int((self.syms == "H").sum())
        self.assertEqual(n_h, 16,
                         f"H 원자 수={n_h} (기대 16 = 8 Si * 2 H)")

    def test_bottom_h_each_bonded_to_one_si(self):
        """각 H 는 Si 와 1.3-1.6 A 이내 결합 정확히 1개 (Si-H bond)."""
        i_arr, j_arr, _ = neighbor_list("ijD", self.slab, 1.6)
        h_idx = np.where(self.syms == "H")[0]
        self.assertGreater(len(h_idx), 0, "H 없음")
        for h in h_idx:
            si_nb = [j for ii, j in zip(i_arr, j_arr)
                     if ii == h and self.syms[j] == "Si"]
            self.assertEqual(len(si_nb), 1,
                             f"H[{h}] Si 이웃={len(si_nb)} (기대 1)")

    def test_bottom_h_atoms_below_si(self):
        """H 원자들은 최하단 Si 아래(또는 같은 z)에 배치되어야 한다."""
        z_min_si = self.slab.positions[self.syms == "Si", 2].min()
        z_h      = self.slab.positions[self.syms == "H",  2]
        self.assertGreater(len(z_h), 0, "H 없음")
        self.assertLessEqual(z_h.max(), z_min_si + 0.1,
                             f"H z_max={z_h.max():.3f} > Si z_min={z_min_si:.3f} "
                             f"-- H 가 bottom 에 없음")

    def test_bottom_si_fully_passivated(self):
        """bottom H 추가 후 bottom Si 의 coordination 이 4 이어야 한다.

        cutoff=2.60 A: Si-H(1.48 A) 와 Si-Si(2.36 A) 결합은 포함하되
        인접 Si 의 H 원자 (2.78 A, 비결합) 는 제외.
        """
        # 2.60 A: covers Si-H (1.48) and Si-Si (2.36), excludes non-bonded H (~2.78)
        i_arr, j_arr = neighbor_list("ij", self.slab, 2.60)
        z_min_si = self.slab.positions[self.syms == "Si", 2].min()
        bot_si = np.where(
            (self.slab.positions[:, 2] < z_min_si + 0.1) & (self.syms == "Si")
        )[0]
        self.assertGreater(len(bot_si), 0, "Bottom Si 없음")
        for idx in bot_si:
            nb = [j for ii, j in zip(i_arr, j_arr) if ii == idx]
            self.assertEqual(len(nb), 4,
                             f"Si[{idx}] coord={len(nb)} (기대 4 = 2 Si + 2 H)")

    def test_no_double_density_layer(self):
        """bulk_shift PBC 버그 수정 후 Si 층 각각 8원자 (16-원자 double layer 없음)."""
        si_z = np.round(self.slab.positions[self.syms == "Si", 2], 2)
        from collections import Counter
        layer_counts = Counter(si_z.tolist())
        doubles = {z: n for z, n in layer_counts.items() if n > 8}
        self.assertEqual(len(doubles), 0,
                         f"16-원자 이중층 발견: {doubles} -- bulk_shift PBC 버그 재발")

    # ── 2x1 Dimer 재구성 ─────────────────────────────────────────────────────

    def test_dimer_bonds_present_at_top_surface(self):
        """
        재구성 후 표면 Si 원자들 사이에 dimer 결합(2.10-2.65 A)이 존재해야 한다.
        threshold=0.8: buckled dimer 상하 원자(최대 buckle ~0.7 A)를 모두 포함.
        """
        top_idx = find_surface_indices(self.slab, side="top", threshold=0.8, species="Si")
        self.assertGreater(len(top_idx), 0, "표면 Si 원자 없음")
        top_set = set(top_idx.tolist())
        i_arr, j_arr, D_arr = neighbor_list("ijD", self.slab, 2.65)
        dimer_bonds = [
            1 for i, j, d in zip(i_arr, j_arr, D_arr)
            if i in top_set and j in top_set
            and self.syms[i] == "Si" and self.syms[j] == "Si"
            and 2.10 <= np.linalg.norm(d) <= 2.65
        ]
        self.assertGreater(len(dimer_bonds), 0,
                           "dimer 결합(2.10-2.65 A) 없음 -- 재구성 실패")

    def test_dimer_si_has_exactly_one_dangling_bond(self):
        """
        bond_slack=0.50 기준, 각 dimer Si 는 dangling bond 가 정확히 1개여야 한다.

        z > z_max-0.8 로 실제 dimer 원자만 평가
        (get_all_dangling_bonds_general hardcoded threshold=2.0 은 sub-surface 포함).
        """
        z_max = self.slab.positions[:, 2].max()
        dimer_si = {i for i in np.where(self.slab.positions[:, 2] > z_max - 0.8)[0]
                    if self.syms[i] == "Si"}
        self.assertGreater(len(dimer_si), 0, "Dimer Si 원자 없음")

        valence_map = {"Si": 4, "H": 1}
        dbs = get_all_dangling_bonds_general(
            self.slab, valence_map, side="top", bond_slack=0.50
        )
        counts = {}
        for db in dbs:
            if db["parent"] in dimer_si:
                counts[db["parent"]] = counts.get(db["parent"], 0) + 1

        over = {i: n for i, n in counts.items() if n > 1}
        self.assertEqual(len(over), 0,
                         f"dangling > 1 인 dimer Si: {list(over.keys())}")
        missing = dimer_si - set(counts.keys())
        self.assertEqual(len(missing), 0,
                         f"dangling bond 없는 dimer Si: {missing}")

    def test_top_dangling_bonds_point_outward(self):
        """실제 dimer Si 의 dangling bond 벡터 z 성분 > 0 (진공 방향)."""
        z_max = self.slab.positions[:, 2].max()
        dimer_si = {i for i in np.where(self.slab.positions[:, 2] > z_max - 0.8)[0]
                    if self.syms[i] == "Si"}
        valence_map = {"Si": 4, "H": 1}
        dbs = get_all_dangling_bonds_general(
            self.slab, valence_map, side="top", bond_slack=0.50
        )
        surface_dbs = [db for db in dbs if db["parent"] in dimer_si]
        self.assertGreater(len(surface_dbs), 0, "Dimer dangling bond 없음")
        for db in surface_dbs:
            self.assertGreater(db["vector"][2], 0.0,
                               f"atom {db['parent']} vz={db['vector'][2]:.3f} (진공 방향 아님)")


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 : SiO2(001) — O-terminated top+bottom + bottom H passivation
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_sio2_vasp_exists(), "structures/SiO2_mp-546794.vasp 없음 — 건너뜀")
class TestSiO2SlabPrep(unittest.TestCase):
    """
    config_slab_prep_sio2.yaml (structures/SiO2_mp-546794.vasp) 을 사용해
    prepare_slab_stage() 로 생성한 SiO2(001) slab 을 검증.
    결과 slab 은 output_slab_prep/sio2_unit_prepared_slab.extxyz 에 저장.
    """

    @classmethod
    def setUpClass(cls):
        cls.logger = _silent_logger("test_sio2")
        out_prefix = os.path.join(_OUTPUT_DIR, "sio2_unit")
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        config = _load_config(_CONFIG_SIO2, out_prefix)
        cls.slab = prepare_slab_stage(config, cls.logger)
        cls.syms = np.array(cls.slab.get_chemical_symbols())

        out_path = os.path.join(_OUTPUT_DIR, "sio2_unit_prepared_slab.extxyz")
        ase_write(out_path, cls.slab)

    # ── 원소 구성 ─────────────────────────────────────────────────────────────

    def test_contains_si_o_h(self):
        """Si, O, H 가 모두 존재해야 한다."""
        s = set(self.syms.tolist())
        self.assertIn("Si", s)
        self.assertIn("O",  s)
        self.assertIn("H",  s, "H 없음 — bottom passivation 실패")

    def test_o_si_ratio_at_least_two(self):
        """O:Si ≥ 2.0 (O-terminated 확인 — 비율 < 2 이면 Si-terminated)."""
        n_o, n_si = int((self.syms=="O").sum()), int((self.syms=="Si").sum())
        self.assertGreaterEqual(n_o/n_si, 2.0,
                                f"O:Si={n_o/n_si:.3f} < 2.0 — termination 설정 누락?")

    # ── Bottom O-H 구조 ───────────────────────────────────────────────────────

    def test_h_atoms_at_bottom(self):
        """H 원자들이 최하단 무거운 원자(O/Si) 아래에 배치되어야 한다."""
        z_heavy = self.slab.positions[np.isin(self.syms, ["Si","O"]), 2]
        z_h     = self.slab.positions[self.syms=="H", 2]
        self.assertGreater(len(z_h), 0, "H 없음")
        self.assertLessEqual(z_h.max(), z_heavy.min()+1.2,
                             f"H z_max={z_h.max():.3f} — 하단 배치 아님")

    def test_each_h_bonded_to_exactly_one_o(self):
        """각 H 는 O 와 1.3 Å 이내 결합 정확히 1개 (O-H bond)."""
        i_arr, j_arr, _ = neighbor_list("ijD", self.slab, 1.3)
        h_idx = np.where(self.syms=="H")[0]
        self.assertGreater(len(h_idx), 0, "H 없음")
        for h in h_idx:
            o_nb = [j for i, j in zip(i_arr, j_arr) if i==h and self.syms[j]=="O"]
            self.assertEqual(len(o_nb), 1, f"H[{h}] O 이웃={len(o_nb)} (기대 1)")

    def test_bottom_surface_is_o_terminated(self):
        """
        최하단 layer (H 제외, z_min+0.5 Å 이하) 에 Si 없어야 한다.
        Si 존재 시 top_termination/bottom_termination 누락 의심.
        SiO2(001) 층간격 ~0.92 Å → threshold=0.5 으로 O 층만 선택.
        """
        heavy = np.isin(self.syms, ["Si","O"])
        z_min = self.slab.positions[heavy, 2].min()
        bot = set(self.syms[heavy][self.slab.positions[heavy,2] < z_min+0.5].tolist())
        self.assertIn("O", bot, "최하단 O 없음")
        self.assertNotIn("Si", bot,
                         "최하단 layer 에 Si 존재 — O-terminated 아님")

    # ── Top O dangling bonds ──────────────────────────────────────────────────

    def test_top_surface_has_o_dangling_bonds(self):
        """상단 O 원자에 dangling bond (chemisorption active site) 가 존재해야 한다."""
        valence_map = {"Si": 4, "O": 2, "H": 1}
        dbs = get_all_dangling_bonds_general(self.slab, valence_map, side="top")
        self.assertGreater(len(dbs), 0, "상단 dangling bond 없음")
        self.assertIn("O", {db["parent_sym"] for db in dbs},
                      "상단 dangling bond 가 O 위에 없음")

    def test_top_dangling_bonds_point_upward(self):
        """상단 dangling bond 벡터 z > -0.1 (진공 방향)."""
        valence_map = {"Si": 4, "O": 2, "H": 1}
        dbs = get_all_dangling_bonds_general(self.slab, valence_map, side="top")
        for db in dbs:
            self.assertGreater(db["vector"][2], -0.1,
                               f"atom {db['parent']} vz={db['vector'][2]:.3f}")

    def test_bottom_dangling_bonds_fully_passivated(self):
        """coverage=1.0 → bottom dangling bonds == 0 (모두 H 로 채워짐)."""
        valence_map = {"Si": 4, "O": 2, "H": 1}
        dbs = get_all_dangling_bonds_general(self.slab, valence_map, side="bottom")
        self.assertEqual(len(dbs), 0,
                         f"Bottom dangling {len(dbs)}개 남음 — passivation 불완전")


# ─────────────────────────────────────────────────────────────────────────────
# Integration: run_generic_adsorption_study end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestSlabPrepIntegration(unittest.TestCase):
    """
    YAML config → load_yaml_config → run_generic_adsorption_study() 로
    prepared_slab.extxyz 생성을 end-to-end 검증.

    output_prefix 를 output_slab_prep/{si100,sio2}/ 로 고정해 결과 파일을 영구 보존.
    프로덕션 실행과 동등:
        run_generic_adsorption_study("config_slab_prep_si100.yaml")
    """

    def _run_and_load(self, yaml_path: str, subdir: str):
        out_prefix = os.path.join(_OUTPUT_DIR, subdir)
        os.makedirs(out_prefix, exist_ok=True)
        config = _load_config(yaml_path, out_prefix)
        run_generic_adsorption_study(config)
        out = os.path.join(out_prefix, "prepared_slab.extxyz")
        self.assertTrue(os.path.exists(out),
                        f"prepared_slab.extxyz 미생성: {out}")
        return read(out)

    # ── Si(100) ──────────────────────────────────────────────────────────────

    def test_si100_prepared_slab_written(self):
        """Si(100): prepared_slab.extxyz 에 Si + H (bottom passivation) 가 존재해야 한다."""
        slab = self._run_and_load(_CONFIG_SI100, "si100")
        syms = set(slab.get_chemical_symbols())
        self.assertIn("Si", syms)
        self.assertIn("H",  syms, "H 없음 -- bottom passivation 실패")
        self.assertFalse(syms - {"Si", "H"}, f"예상 외 원소: {syms - {'Si', 'H'}}")

    def test_si100_dimer_bonds_in_output(self):
        """Si(100): prepared_slab.extxyz 에 dimer 결합(2.10–2.65 Å)이 존재해야 한다."""
        slab = self._run_and_load(_CONFIG_SI100, "si100")
        syms = np.array(slab.get_chemical_symbols())
        top_idx = find_surface_indices(slab, side="top", threshold=0.8, species="Si")
        top_set = set(top_idx.tolist())
        i_arr, j_arr, D_arr = neighbor_list("ijD", slab, 2.65)
        dimers = [1 for i, j, d in zip(i_arr, j_arr, D_arr)
                  if i in top_set and j in top_set
                  and syms[i]=="Si" and syms[j]=="Si"
                  and 2.10 <= np.linalg.norm(d) <= 2.65]
        self.assertGreater(len(dimers), 0, "prepared_slab 에 dimer 결합 없음")

    # ── SiO2(001) ────────────────────────────────────────────────────────────

    @unittest.skipUnless(_sio2_vasp_exists(), "SiO2_mp-546794.vasp 없음 — 건너뜀")
    def test_sio2_prepared_slab_written(self):
        """SiO2(001): prepared_slab.extxyz 에 Si+O+H 가 존재해야 한다."""
        slab = self._run_and_load(_CONFIG_SIO2, "sio2")
        syms = set(slab.get_chemical_symbols())
        self.assertIn("Si", syms)
        self.assertIn("O",  syms)
        self.assertIn("H",  syms, "H 없음 — passivation 실패")

    @unittest.skipUnless(_sio2_vasp_exists(), "SiO2_mp-546794.vasp 없음 — 건너뜀")
    def test_sio2_o_si_ratio(self):
        """SiO2(001): O:Si ≥ 2.0 (O-terminated 확인)."""
        slab = self._run_and_load(_CONFIG_SIO2, "sio2")
        lst = list(slab.get_chemical_symbols())
        n_o, n_si = lst.count("O"), lst.count("Si")
        self.assertGreaterEqual(n_o/n_si, 2.0,
                                f"O:Si={n_o/n_si:.2f} < 2 — termination 누락 확인")


if __name__ == "__main__":
    unittest.main(verbosity=2)
