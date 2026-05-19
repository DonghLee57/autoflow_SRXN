"""
test_slab_prep.py
=================
Geometry-only tests for the slab preparation stage.

두 케이스를 검증합니다:
  1. Si(100) — create_slab_from_bulk + 2×1 buckled-dimer 재구성
  2. SiO2(001) — O-terminated top/bottom + bottom H passivation → top O dangling bonds

모든 테스트는 MLIP 없이 순수 geometry 연산만 실행합니다
(slab_relax=False, candidate_relax=False).

테스트 레벨
-----------
Unit        : prepare_slab_stage() 직접 호출 → 세분화된 구조 검증
Integration : run_generic_adsorption_study() 엔드투엔드 → prepared_slab.extxyz 검증

Config 는 unittests/ 아래의 YAML 파일에서 load_yaml_config 로 읽어 파싱합니다.
output_prefix 만 임시 디렉토리로 교체 후 전달합니다.
  config_slab_prep_si100.yaml  — Si(100) 케이스
  config_slab_prep_sio2.yaml   — SiO2(001) 케이스

Si(100) 핵심 이슈 (troubleshoot/Si_100_reconstruction 참고)
-----------------------------------------------------------
* MLIP 릴랙스 후 dimer 결합 길이 ~2.46 Å → bond_slack=0.45 필수
  (default 0.20 → cutoff 2.42 Å < 2.46 Å → dimer 불인식 → dangling bond 2개 오류)
* get_all_dangling_bonds_general 의 hardcoded threshold=2.0 때문에
  실제 dimer 층(z_max, z_max-0.4) 외에 sub-surface 층(z_max-1.1 정도)도 포함됨.
  → dangling bond 검사 시 z > z_max-0.8 필터로 실제 dimer 원자만 대상으로 삼아야 함.
* bulk_shift=0.25 + 2×2 supercell 에서 bottom Si 의 과협조 PBC 아티팩트 때문에
  MLIP-free 환경에서는 bottom passivation 이 작동하지 않으므로 config 에서 비활성화.

SiO2(001) 핵심 이슈 (troubleshoot/SiO2_slab_prep 참고)
------------------------------------------------------
* top_termination/bottom_termination 이 create_slab_from_bulk 에 전달되어야
  O-terminated slab 이 생성됨 (누락 시 Si-terminated 기본 컷 적용 → H 미생성).
* passivation(side="bottom") 으로 bottom O-H 형성,
  top O dangling bonds 는 chemisorption active sites 로 유지.
"""

import logging
import os
import tempfile
import unittest

import numpy as np
from ase.io import read
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
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

# 이 파일 위치 (unittests/)
_HERE = os.path.dirname(os.path.abspath(__file__))

# YAML config 파일 경로
_CONFIG_SI100 = os.path.join(_HERE, "config_slab_prep_si100.yaml")
_CONFIG_SIO2  = os.path.join(_HERE, "config_slab_prep_sio2.yaml")


def _silent_logger(name: str = "test_slab_prep") -> logging.Logger:
    """테스트 실행 중 로그 출력을 억제하는 NullHandler 로거."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _load_config(yaml_path: str, tmpdir: str) -> dict:
    """YAML 을 load_yaml_config 로 파싱한 뒤 output_prefix 를 tmpdir 로 교체.

    load_yaml_config 는 substrate_bulk 경로를 config 파일 위치 기준 절대경로로
    변환합니다. output_prefix 는 존재하지 않는 디렉토리이므로 변환되지 않으므로
    여기서 tmpdir 로 명시 설정합니다.
    """
    cfg = load_yaml_config(yaml_path)
    cfg["paths"]["output_prefix"] = tmpdir
    return cfg


def _sio2_poscar_exists() -> bool:
    """SiO2 벌크 POSCAR fixture 가 존재하는지 확인."""
    poscar = os.path.normpath(
        os.path.join(_HERE, "..", "troubleshoot", "SiO2_slab_prep", "POSCAR_SiO2.vasp")
    )
    return os.path.exists(poscar)


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 : Si(100) — 2×1 buckled-dimer 재구성
# ─────────────────────────────────────────────────────────────────────────────

class TestSi100SlabPrep(unittest.TestCase):
    """
    config_slab_prep_si100.yaml 을 load_yaml_config 로 파싱한 뒤
    prepare_slab_stage() 를 호출해 Si(100) 2×1 buckled-dimer 재구성을 검증합니다.

    Bottom passivation 은 이 셀 기하(bulk_shift=0.25, 2×2 supercell)에서
    MLIP-free 환경의 PBC 아티팩트로 작동하지 않아 config 에서 비활성화됩니다.
    핵심 검증 항목은 2×1 dimer 재구성 geometry 입니다.
    """

    @classmethod
    def setUpClass(cls):
        cls.logger = _silent_logger("test_si100")
        cls.tmpdir = tempfile.mkdtemp()
        config = _load_config(_CONFIG_SI100, cls.tmpdir)
        cls.slab = prepare_slab_stage(config, cls.logger)
        cls.syms = np.array(cls.slab.get_chemical_symbols())

    # ── 원소 구성 ─────────────────────────────────────────────────────────────

    def test_contains_only_si(self):
        """Passivation 비활성화 상태에서 slab 은 Si 만 포함해야 한다."""
        sym_set = set(self.syms.tolist())
        self.assertIn("Si", sym_set, "Si 원자 없음")
        self.assertFalse(
            sym_set - {"Si"},
            f"예상 외 원소 발견: {sym_set - {'Si'}}"
        )

    # ── 2×1 Dimer 재구성 ─────────────────────────────────────────────────────

    def test_dimer_bonds_present_at_top_surface(self):
        """
        재구성 후 표면 Si 원자들 사이에 dimer 결합(2.10–2.65 Å)이 존재해야 한다.
        threshold=0.8 으로 실제 dimer 층만 선택 (buckle ≈0.4 Å 이므로 0.8 Å 이면 충분).
        seed 에서 dimer 결합 ≈2.30 Å, MLIP 릴랙스 후 ≈2.46 Å.
        """
        top_idx = find_surface_indices(self.slab, side="top", threshold=0.8, species="Si")
        self.assertGreater(len(top_idx), 0, "표면 Si 원자 없음")

        top_set = set(top_idx.tolist())
        i_arr, j_arr, D_arr = neighbor_list("ijD", self.slab, 2.65)

        dimer_bonds = [
            (i, j, np.linalg.norm(d))
            for i, j, d in zip(i_arr, j_arr, D_arr)
            if i in top_set and j in top_set
            and self.syms[i] == "Si" and self.syms[j] == "Si"
            and 2.10 <= np.linalg.norm(d) <= 2.65
        ]
        self.assertGreater(
            len(dimer_bonds), 0,
            "표면 Si-Si dimer 결합(2.10–2.65 Å) 없음 — 재구성 실패"
        )

    def test_dimer_si_has_exactly_one_dangling_bond(self):
        """
        bond_slack=0.45 로 dimer 결합 인식 시 각 dimer Si 는
        dangling bond 가 정확히 1개여야 한다 (coord=3).

        주의: get_all_dangling_bonds_general 의 hardcoded threshold=2.0 은
        sub-surface 원자(z_max 에서 ~1.1 Å 아래)도 포함하므로, z > z_max-0.8 Å
        필터로 실제 dimer 원자만 평가합니다.

        bond_slack=0.20 이면 릴랙스 후 2.46 Å dimer 를 인식 못해
        dangling bond 가 2개로 계산되는 버그 발생 (이 테스트가 해당 회귀를 감지).
        """
        z_max = self.slab.positions[:, 2].max()
        # buckled dimer: up-atom at z_max, down-atom at z_max-0.4 → 0.8 Å margin
        dimer_idx = set(
            np.where(self.slab.positions[:, 2] > z_max - 0.8)[0].tolist()
        )
        # 해당 범위 내 Si 원자만
        dimer_si_idx = {i for i in dimer_idx if self.syms[i] == "Si"}
        self.assertGreater(len(dimer_si_idx), 0, "Dimer Si 원자 없음")

        valence_map = {"Si": 4, "H": 1}
        dbs = get_all_dangling_bonds_general(
            self.slab, valence_map, side="top", bond_slack=0.45
        )
        counts: dict[int, int] = {}
        for db in dbs:
            if db["parent"] in dimer_si_idx:
                counts[db["parent"]] = counts.get(db["parent"], 0) + 1

        over = {idx: n for idx, n in counts.items() if n > 1}
        self.assertEqual(
            len(over), 0,
            f"dangling bond > 1 인 dimer Si: atom indices {list(over.keys())}\n"
            f"  → bond_slack < 0.45 이면 2.46 Å dimer 결합을 인식하지 못합니다."
        )
        # 각 dimer 원자가 최소 1개 dangling bond 를 가져야 한다
        missing_db = dimer_si_idx - set(counts.keys())
        self.assertEqual(
            len(missing_db), 0,
            f"dangling bond 가 없는 dimer Si: {missing_db} — 재구성 또는 dangling 감지 실패"
        )

    def test_top_dangling_bonds_point_outward(self):
        """실제 dimer Si 의 dangling bond 벡터 z 성분이 양수여야 한다 (진공 방향)."""
        z_max = self.slab.positions[:, 2].max()
        dimer_si_idx = set(
            i for i in np.where(self.slab.positions[:, 2] > z_max - 0.8)[0]
            if self.syms[i] == "Si"
        )
        valence_map = {"Si": 4, "H": 1}
        dbs = get_all_dangling_bonds_general(
            self.slab, valence_map, side="top", bond_slack=0.45
        )
        surface_dbs = [db for db in dbs if db["parent"] in dimer_si_idx]
        self.assertGreater(len(surface_dbs), 0, "Dimer Si dangling bond 없음")
        for db in surface_dbs:
            self.assertGreater(
                db["vector"][2], 0.0,
                f"atom {db['parent']} dangling bond 가 표면 안쪽을 향함 "
                f"(vz={db['vector'][2]:.3f})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 : SiO2(001) — O-terminated top+bottom + bottom H passivation
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_sio2_poscar_exists(), "SiO2 벌크 POSCAR 없음 — 건너뜀")
class TestSiO2SlabPrep(unittest.TestCase):
    """
    config_slab_prep_sio2.yaml 을 load_yaml_config 로 파싱한 뒤
    prepare_slab_stage() 를 호출해 SiO2(001) slab 구조를 검증합니다.
    top 면의 O dangling bonds 는 chemisorption active sites 로 보존되어야 합니다.
    """

    @classmethod
    def setUpClass(cls):
        cls.logger = _silent_logger("test_sio2")
        cls.tmpdir = tempfile.mkdtemp()
        config = _load_config(_CONFIG_SIO2, cls.tmpdir)
        cls.slab = prepare_slab_stage(config, cls.logger)
        cls.syms = np.array(cls.slab.get_chemical_symbols())

    # ── 원소 구성 ─────────────────────────────────────────────────────────────

    def test_contains_si_o_h(self):
        """Slab 에 Si, O, H 가 모두 존재해야 한다."""
        sym_set = set(self.syms.tolist())
        self.assertIn("Si", sym_set)
        self.assertIn("O",  sym_set)
        self.assertIn("H",  sym_set, "H 원자 없음 — bottom passivation 실패")

    def test_o_si_ratio_at_least_two(self):
        """
        벌크 SiO2 의 O:Si = 2. 표면 O 원자가 추가되므로 비율은 ≥2 이어야 한다.
        비율 < 2 이면 Si-terminated slab 이 생성된 것 (termination 설정 누락 의심).
        """
        n_o  = int((self.syms == "O").sum())
        n_si = int((self.syms == "Si").sum())
        ratio = n_o / n_si
        self.assertGreaterEqual(
            ratio, 2.0,
            f"O:Si = {ratio:.3f} < 2.0 — O-terminated 면이 아님 (termination 설정 누락?)"
        )

    # ── Bottom O-H 구조 ───────────────────────────────────────────────────────

    def test_h_atoms_at_bottom(self):
        """H 원자들이 최하단 무거운 원자(O/Si) 아래에 배치되어야 한다."""
        z_heavy = self.slab.positions[np.isin(self.syms, ["Si", "O"]), 2]
        z_h     = self.slab.positions[self.syms == "H", 2]
        self.assertGreater(len(z_h), 0, "H 원자 없음")
        self.assertLessEqual(
            z_h.max(), z_heavy.min() + 1.2,
            f"H 최대 z={z_h.max():.3f} Å — 하단에 위치하지 않음"
        )

    def test_each_h_bonded_to_exactly_one_o(self):
        """각 H 원자는 O 와 1.3 Å 이내 결합을 정확히 1개 가져야 한다 (O-H bond)."""
        i_arr, j_arr, _ = neighbor_list("ijD", self.slab, 1.3)
        h_indices = np.where(self.syms == "H")[0]
        self.assertGreater(len(h_indices), 0, "H 원자 없음")
        for h_idx in h_indices:
            o_neighbors = [j for i, j in zip(i_arr, j_arr)
                           if i == h_idx and self.syms[j] == "O"]
            self.assertEqual(
                len(o_neighbors), 1,
                f"H[{h_idx}] O 이웃 수={len(o_neighbors)} (기대값 1)"
            )

    def test_bottom_surface_is_o_terminated(self):
        """
        최하단 layer (H 제외)가 O 원자만 포함해야 한다.
        Si 원자가 포함되면 top_termination/bottom_termination 이
        create_slab_from_bulk 에 전달되지 않은 것.
        """
        heavy_mask = np.isin(self.syms, ["Si", "O"])
        z_heavy = self.slab.positions[heavy_mask, 2]
        z_min   = z_heavy.min()
        # SiO2(001) 층 간격 ~0.92 Å → threshold=0.5 로 최하단 O 층만 선택
        # threshold=1.5 를 쓰면 바로 위 Si 층(z_min+~0.92)까지 포함됨
        bottom_syms = set(
            self.syms[heavy_mask][
                self.slab.positions[heavy_mask, 2] < z_min + 0.5
            ].tolist()
        )
        self.assertIn("O",  bottom_syms, "최하단 layer 에 O 없음")
        self.assertNotIn(
            "Si", bottom_syms,
            "최하단 layer 에 Si 존재 — O-terminated 아님 "
            "(top_termination/bottom_termination 이 create_slab_from_bulk 에 전달됐는지 확인)"
        )

    # ── Top O dangling bonds (chemisorption active sites) ────────────────────

    def test_top_surface_has_o_dangling_bonds(self):
        """
        상단면 O 원자들은 dangling bond 를 가져야 한다 (coord < 2).
        이 dangling bonds 가 chemisorption active sites 역할을 한다.
        """
        valence_map = {"Si": 4, "O": 2, "H": 1}
        dbs = get_all_dangling_bonds_general(self.slab, valence_map, side="top")
        self.assertGreater(
            len(dbs), 0,
            "상단면에 dangling bond 없음 — chemisorption site 가 존재하지 않음"
        )
        top_syms = {db["parent_sym"] for db in dbs}
        self.assertIn("O", top_syms, "상단면 dangling bond 가 O 원자 위에 없음")

    def test_top_dangling_bonds_point_upward(self):
        """상단면 dangling bond 벡터의 z 성분 > -0.1 이어야 한다 (진공 방향)."""
        valence_map = {"Si": 4, "O": 2, "H": 1}
        dbs = get_all_dangling_bonds_general(self.slab, valence_map, side="top")
        for db in dbs:
            self.assertGreater(
                db["vector"][2], -0.1,
                f"atom {db['parent']} ({db['parent_sym']}) dangling bond 가 아래를 향함 "
                f"(vz={db['vector'][2]:.3f})"
            )

    def test_bottom_dangling_bonds_fully_passivated(self):
        """
        Bottom passivation 완료 후 bottom dangling bonds 수 == 0 이어야 한다
        (coverage=1.0 → 모든 site 가 H 로 채워짐).
        """
        valence_map = {"Si": 4, "O": 2, "H": 1}
        bottom_dbs = get_all_dangling_bonds_general(
            self.slab, valence_map, side="bottom"
        )
        self.assertEqual(
            len(bottom_dbs), 0,
            f"Bottom dangling bonds {len(bottom_dbs)}개 남아 있음 "
            f"— passivation 이 완전하지 않음 (coverage=1.0 인데 미채워진 site 존재)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration: run_generic_adsorption_study — slab prep 경로 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestSlabPrepIntegration(unittest.TestCase):
    """
    YAML config 를 load_yaml_config 로 파싱 후 run_generic_adsorption_study() 에
    전달해 prepared_slab.extxyz 가 올바르게 생성되는지 end-to-end 로 검증합니다.

    이 흐름은 프로덕션에서 사용자가 다음과 같이 실행하는 것과 동등합니다:
        run_generic_adsorption_study("config_slab_prep_si100.yaml")
    단, output_prefix 를 tmpdir 로 교체해 테스트 환경을 격리합니다.
    """

    def _run_and_load_slab(self, yaml_path: str):
        """YAML 로드 → output_prefix 교체 → run_generic_adsorption_study 실행 →
        생성된 prepared_slab.extxyz 반환."""
        tmpdir = tempfile.mkdtemp()
        config = _load_config(yaml_path, tmpdir)
        run_generic_adsorption_study(config)

        out_path = os.path.join(tmpdir, "prepared_slab.extxyz")
        self.assertTrue(
            os.path.exists(out_path),
            f"prepared_slab.extxyz 가 생성되지 않음: {out_path}"
        )
        return read(out_path)

    # ── Si(100) 통합 ─────────────────────────────────────────────────────────

    def test_si100_prepared_slab_written(self):
        """Si(100): prepared_slab.extxyz 가 Si 로만 구성되어야 한다 (passivation 비활성)."""
        slab = self._run_and_load_slab(_CONFIG_SI100)
        syms = set(slab.get_chemical_symbols())
        self.assertIn("Si", syms, "Si 없음")
        self.assertFalse(
            syms - {"Si"},
            f"예상 외 원소 발견: {syms - {'Si'}}"
        )

    def test_si100_dimer_bonds_in_output(self):
        """Si(100): prepared_slab.extxyz 에 dimer 결합(2.10–2.65 Å)이 존재해야 한다."""
        slab = self._run_and_load_slab(_CONFIG_SI100)
        syms = np.array(slab.get_chemical_symbols())
        top_idx = find_surface_indices(slab, side="top", threshold=0.8, species="Si")
        top_set = set(top_idx.tolist())
        i_arr, j_arr, D_arr = neighbor_list("ijD", slab, 2.65)
        dimer_bonds = [
            1 for i, j, d in zip(i_arr, j_arr, D_arr)
            if i in top_set and j in top_set
            and syms[i] == "Si" and syms[j] == "Si"
            and 2.10 <= np.linalg.norm(d) <= 2.65
        ]
        self.assertGreater(len(dimer_bonds), 0, "prepared_slab 에 dimer 결합 없음")

    # ── SiO2(001) 통합 ───────────────────────────────────────────────────────

    @unittest.skipUnless(_sio2_poscar_exists(), "SiO2 벌크 POSCAR 없음 — 건너뜀")
    def test_sio2_prepared_slab_written(self):
        """SiO2(001): prepared_slab.extxyz 가 Si + O + H 로 구성되어야 한다."""
        slab = self._run_and_load_slab(_CONFIG_SIO2)
        syms_list = list(slab.get_chemical_symbols())
        syms_set  = set(syms_list)
        self.assertIn("Si", syms_set)
        self.assertIn("O",  syms_set)
        self.assertIn("H",  syms_set, "H 없음 — passivation 실패")

    @unittest.skipUnless(_sio2_poscar_exists(), "SiO2 벌크 POSCAR 없음 — 건너뜀")
    def test_sio2_o_si_ratio(self):
        """SiO2(001): extxyz 의 O:Si 비율이 2.0 이상이어야 한다 (O-terminated 확인)."""
        slab = self._run_and_load_slab(_CONFIG_SIO2)
        syms_list = list(slab.get_chemical_symbols())
        n_o  = syms_list.count("O")
        n_si = syms_list.count("Si")
        self.assertGreaterEqual(
            n_o / n_si, 2.0,
            f"O:Si={n_o/n_si:.2f} < 2 — O-terminated 아님 "
            "(top/bottom_termination 누락 확인)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
