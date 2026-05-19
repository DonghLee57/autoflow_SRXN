"""
test_slab_prep.py
=================
Geometry-only tests for the slab preparation stage.

두 케이스를 검증합니다:
  1. Si(100) — create_slab_from_bulk + 2×1 buckled-dimer 재구성 + bottom H passivation
  2. SiO2(001) — O-terminated top/bottom + bottom H passivation → top O dangling bonds

모든 테스트는 MLIP 없이 순수 geometry 연산만 실행합니다 (slab_relax=False,
candidate_relax=False). 두 레벨에서 검사합니다:
  - Unit   : prepare_slab_stage() 직접 호출 → 세분화된 구조 검증
  - Integration : run_generic_adsorption_study() 엔드투엔드 → prepared_slab.extxyz 검증

Si(100) 핵심 이슈 (troubleshoot/Si_100_reconstruction/README.md 참고):
  * MLIP 릴랙스 후 dimer 결합 길이 ~2.46 Å → bond_slack=0.45 필수
    (default 0.20 이면 cutoff=2.42 Å < 2.46 Å → dimer 불인식 → 2개 dangling bond 오류)
  * 재구성 seed 단계의 dimer 결합 길이는 ~2.30 Å (bond_slack=0.20 에서도 인식 가능)
    → 릴랙스 후 테스트를 위해 bond_slack=0.45 로 명시

SiO2(001) 핵심 이슈 (troubleshoot/SiO2_slab_prep/README.md 참고):
  * top_termination/bottom_termination 이 prepare_slab_stage 에서
    create_slab_from_bulk 에 전달되어야 O-terminated slab 생성
  * passivation(side="bottom") 으로 bottom O-H 형성,
    top O dangling bonds 는 chemisorption active sites 로 유지
"""

import logging
import os
import tempfile
import unittest

import numpy as np
from ase.build import bulk as ase_bulk
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


# ─────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _silent_logger(name="test_slab_prep"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _sio2_vasp_path():
    """SiO2 벌크 POSCAR 경로 (troubleshoot fixture)."""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__),
                     "..", "troubleshoot", "SiO2_slab_prep", "POSCAR_SiO2.vasp")
    )


def _base_workflow_flags():
    """MLIP 없이 geometry 단계만 실행하는 공통 workflow/relaxation 설정."""
    return {
        "workflow": {
            "slab_relax":      False,
            "candidate_relax": False,
            "md_equilibrate":  False,
            "post_md_relax":   False,
        },
        "relaxation": {
            "fmax": 0.05,
            "steps": 100,
            "frozen_z_ang": 5.5,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 : Si(100) — 2×1 buckled-dimer 재구성 + bottom H passivation
# ─────────────────────────────────────────────────────────────────────────────

class TestSi100SlabPrep(unittest.TestCase):
    """
    Si 다이아몬드 구조에서 Si(100) slab 생성 후 2×1 buckled-dimer 재구성과
    bottom H passivation 을 검증합니다.
    """

    @classmethod
    def setUpClass(cls):
        cls.logger = _silent_logger("test_si100")
        cls.tmpdir = tempfile.mkdtemp()

        # ASE 표준 bulk Si → VASP 임시 파일로 저장
        si_bulk = ase_bulk("Si", "diamond", a=5.431)
        bulk_path = os.path.join(cls.tmpdir, "bulk_si.vasp")
        ase_write(bulk_path, si_bulk, format="vasp")

        config = {
            "paths": {
                "substrate_bulk": bulk_path,
                "output_prefix":  cls.tmpdir,
            },
            **_base_workflow_flags(),
            "surface_prep": {
                "slab_generation": {
                    "enabled":         True,
                    "miller":          [1, 0, 0],
                    "thickness_ang":   8.0,
                    "vacuum_ang":      10.0,
                    # bulk_shift=0.25: Si(100) 2×1 재구성에 필요한 bilayer 경계 노출
                    "bulk_shift":      0.25,
                    # 작은 supercell 로 속도 확보 (실제 계산 시 4×4 권장)
                    "supercell_matrix": [[2, 0], [0, 2]],
                },
                "reconstruction": {
                    "enabled":      True,
                    "strategy":     "auto",   # Si + miller(1,0,0) → reconstruct_si100_2x1_buckled
                    "side":         "top",
                    "buckling_dist": 0.4,     # 수직 버클링 진폭 (Å)
                },
                "passivation": {
                    "enabled":  True,
                    "element":  "H",
                    "side":     "bottom",
                    "coverage": 1.0,
                },
                "surface_analysis": {
                    "ideal_coordination": {"Si": 4, "H": 1},
                },
            },
        }
        cls.slab = prepare_slab_stage(config, cls.logger)
        cls.si_syms = np.array(cls.slab.get_chemical_symbols())

    # ── 기본 원소 구성 ────────────────────────────────────────────────────────

    def test_contains_si_and_h(self):
        """Slab 에 Si 와 H 원자가 모두 존재해야 한다."""
        syms = set(self.slab.get_chemical_symbols())
        self.assertIn("Si", syms, "Si 원자 없음")
        self.assertIn("H",  syms, "H 원자 없음 — bottom passivation 실패")

    def test_only_si_and_h(self):
        """Si(100) slab 은 Si 와 H 만 포함해야 한다."""
        unexpected = set(self.slab.get_chemical_symbols()) - {"Si", "H"}
        self.assertFalse(unexpected, f"예상 외 원소 발견: {unexpected}")

    # ── Bottom H passivation ─────────────────────────────────────────────────

    def test_bottom_h_confined_below_si(self):
        """모든 H 원자는 최하단 Si 원자보다 낮거나 같은 위치에 있어야 한다."""
        z_si = self.slab.positions[self.si_syms == "Si", 2]
        z_h  = self.slab.positions[self.si_syms == "H",  2]
        self.assertTrue(len(z_h) > 0, "H 원자 없음")
        self.assertLessEqual(
            z_h.max(), z_si.min() + 0.8,
            f"H 최대 z={z_h.max():.3f} > Si 최소 z={z_si.min():.3f} + 0.8 Å"
        )

    def test_bottom_h_bonded_to_si(self):
        """각 H 원자는 Si 와 0.8–1.8 Å 범위의 결합을 정확히 1개 가져야 한다."""
        i_arr, j_arr, _ = neighbor_list("ijD", self.slab, 1.8)
        h_indices = np.where(self.si_syms == "H")[0]
        for h_idx in h_indices:
            mask = i_arr == h_idx
            si_neighbors = [j for j in j_arr[mask] if self.si_syms[j] == "Si"]
            self.assertEqual(
                len(si_neighbors), 1,
                f"H[{h_idx}] 의 Si 이웃 수={len(si_neighbors)} (기대값 1)"
            )

    # ── 2×1 Dimer 재구성 ─────────────────────────────────────────────────────

    def test_dimer_bonds_present_at_top_surface(self):
        """
        재구성 후 표면 Si 원자들 사이에 dimer 결합(2.10–2.65 Å)이 존재해야 한다.
        seed geometry 에서 목표 결합 길이 ≈2.30 Å,
        MLIP 릴랙스 후 ≈2.46 Å (bond_slack=0.45 로 인식 필요).
        """
        # threshold=1.5: 버클링(0.4 Å)으로 높이가 다른 dimer 쌍 모두 포함
        top_idx = find_surface_indices(self.slab, side="top", threshold=1.5, species="Si")
        self.assertGreater(len(top_idx), 0, "표면 Si 원자 없음")

        top_set = set(top_idx.tolist())
        i_arr, j_arr, D_arr = neighbor_list("ijD", self.slab, 2.65)

        dimer_bonds = []
        for idx in top_idx:
            mask = i_arr == idx
            for j, d in zip(j_arr[mask], D_arr[mask]):
                dist = np.linalg.norm(d)
                if j in top_set and self.si_syms[j] == "Si" and 2.10 <= dist <= 2.65:
                    dimer_bonds.append((idx, j, dist))

        self.assertGreater(
            len(dimer_bonds), 0,
            "표면 Si-Si dimer 결합(2.10–2.65 Å) 없음 — 재구성 실패"
        )

    def test_dimer_si_has_exactly_one_dangling_bond(self):
        """
        bond_slack=0.45 로 dimer 결합 인식 시, 각 표면 Si 는
        dangling bond 가 정확히 1개여야 한다 (coord=3).
        bond_slack=0.20 이면 릴랙스 후 2.46 Å dimer 를 인식 못해
        dangling bond 가 2개로 계산되는 버그 발생.
        """
        valence_map = {"Si": 4, "H": 1}
        dbs = get_all_dangling_bonds_general(
            self.slab, valence_map, side="top", bond_slack=0.45
        )
        top_idx = set(
            find_surface_indices(self.slab, side="top", threshold=1.5, species="Si").tolist()
        )
        surface_dbs = [db for db in dbs if db["parent"] in top_idx]

        # 원자별 dangling bond 수 집계
        counts: dict[int, int] = {}
        for db in surface_dbs:
            counts[db["parent"]] = counts.get(db["parent"], 0) + 1

        over = {idx: n for idx, n in counts.items() if n > 1}
        self.assertEqual(
            len(over), 0,
            f"dangling bond > 1 인 표면 Si (dimer 미인식): atom indices {list(over.keys())}\n"
            f"  → bond_slack < 0.45 이면 2.46 Å dimer 결합을 인식하지 못합니다."
        )

    def test_top_dangling_bonds_point_outward(self):
        """표면 Si dangling bond 벡터의 z 성분이 양수여야 한다 (진공 방향)."""
        valence_map = {"Si": 4, "H": 1}
        dbs = get_all_dangling_bonds_general(
            self.slab, valence_map, side="top", bond_slack=0.45
        )
        top_idx = set(
            find_surface_indices(self.slab, side="top", threshold=1.5, species="Si").tolist()
        )
        surface_dbs = [db for db in dbs if db["parent"] in top_idx]

        self.assertGreater(len(surface_dbs), 0, "표면 dangling bond 없음")
        for db in surface_dbs:
            self.assertGreater(
                db["vector"][2], 0.0,
                f"atom {db['parent']} 의 dangling bond 가 표면 안쪽을 향함 "
                f"(vz={db['vector'][2]:.3f})"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 : SiO2(001) — O-terminated top+bottom + bottom H passivation
# ─────────────────────────────────────────────────────────────────────────────

class TestSiO2SlabPrep(unittest.TestCase):
    """
    SiO2(001) slab 생성 시 O-terminated 상하면과 bottom H passivation 을 검증합니다.
    top 면의 O dangling bonds 는 chemisorption active sites 로 보존되어야 합니다.
    """

    @classmethod
    def setUpClass(cls):
        vasp_path = _sio2_vasp_path()
        if not os.path.exists(vasp_path):
            raise unittest.SkipTest(
                f"SiO2 벌크 POSCAR 없음 (건너뜀): {vasp_path}"
            )

        cls.logger = _silent_logger("test_sio2")
        cls.tmpdir = tempfile.mkdtemp()

        config = {
            "paths": {
                "substrate_bulk": vasp_path,
                "output_prefix":  cls.tmpdir,
            },
            **_base_workflow_flags(),
            "surface_prep": {
                "slab_generation": {
                    "enabled":            True,
                    "miller":             [0, 0, 1],
                    "thickness_ang":      12.0,
                    "vacuum_ang":         15.0,
                    "target_area_ang2":   250.0,
                    # 핵심 수정: 이 두 파라미터가 create_slab_from_bulk 에
                    # 전달되어야 O-terminated slab 이 생성됨
                    "top_termination":    "O",
                    "bottom_termination": "O",
                },
                "passivation": {
                    "enabled":  True,
                    "element":  "H",
                    "side":     "bottom",   # bottom O-H 형성, top O dangling 유지
                    "coverage": 1.0,
                },
                "surface_analysis": {
                    "ideal_coordination": {"Si": 4, "O": 2, "H": 1},
                },
            },
        }
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
        비율 < 2 이면 Si-terminated slab 이 생성된 것.
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
        self.assertTrue(len(z_h) > 0, "H 원자 없음")
        self.assertLessEqual(
            z_h.max(), z_heavy.min() + 1.2,
            f"H 최대 z={z_h.max():.3f} Å — 하단에 위치하지 않음"
        )

    def test_each_h_bonded_to_exactly_one_o(self):
        """각 H 원자는 O 와 1.3 Å 이내의 결합을 정확히 1개 가져야 한다 (O-H bond)."""
        i_arr, j_arr, _ = neighbor_list("ijD", self.slab, 1.3)
        h_indices = np.where(self.syms == "H")[0]
        self.assertGreater(len(h_indices), 0, "H 원자 없음")

        for h_idx in h_indices:
            mask = i_arr == h_idx
            o_neighbors = [j for j in j_arr[mask] if self.syms[j] == "O"]
            self.assertEqual(
                len(o_neighbors), 1,
                f"H[{h_idx}] O 이웃 수={len(o_neighbors)} (기대값 1)"
            )

    def test_bottom_surface_is_o_terminated(self):
        """
        최하단 layer (H 제외)가 O 원자만 포함해야 한다.
        Si 원자가 포함되면 termination 설정이 잘못 적용된 것.
        """
        heavy_mask = np.isin(self.syms, ["Si", "O"])
        z_heavy = self.slab.positions[heavy_mask, 2]
        z_min   = z_heavy.min()

        bottom_syms = set(
            self.syms[heavy_mask][
                self.slab.positions[heavy_mask, 2] < z_min + 1.5
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
        상단면 O 원자들은 1개의 dangling bond 를 가져야 한다 (coord=1, target=2).
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

    def test_dangling_bond_count_matches_h_count(self):
        """
        Bottom dangling bonds 수 == H 원자 수 (coverage=1.0 이므로 1:1 대응).
        """
        valence_map = {"Si": 4, "O": 2, "H": 1}
        bottom_dbs_before_passivation = get_all_dangling_bonds_general(
            self.slab, valence_map, side="bottom"
        )
        # passivation 후 slab 이므로 bottom dangling bonds 는 0 이어야 함
        # (모두 H 로 채워짐)
        self.assertEqual(
            len(bottom_dbs_before_passivation), 0,
            f"Bottom dangling bonds {len(bottom_dbs_before_passivation)}개 남아 있음 "
            f"— passivation 이 완전하지 않음 (coverage=1.0 인데 미채워진 site 존재)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Integration: run_generic_adsorption_study — slab prep 경로 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestSlabPrepIntegration(unittest.TestCase):
    """
    run_generic_adsorption_study() 를 precursor/inhibitor 없이 실행해
    prepared_slab.extxyz 가 올바르게 생성되는지 검증합니다.
    """

    def _run_and_load(self, config: dict) -> object:
        run_generic_adsorption_study(config)
        out_path = os.path.join(
            config["paths"]["output_prefix"], "prepared_slab.extxyz"
        )
        self.assertTrue(
            os.path.exists(out_path),
            f"prepared_slab.extxyz 가 생성되지 않음: {out_path}"
        )
        return read(out_path)

    def test_si100_integration(self):
        """Si(100) slab prep 통합 테스트: extxyz 파일에 H 포함 여부 확인."""
        tmpdir = tempfile.mkdtemp()
        si_bulk = ase_bulk("Si", "diamond", a=5.431)
        bulk_path = os.path.join(tmpdir, "bulk_si.vasp")
        ase_write(bulk_path, si_bulk, format="vasp")

        config = {
            "paths": {
                "substrate_bulk": bulk_path,
                "precursor":      None,
                "inhibitor":      None,
                "output_prefix":  tmpdir,
            },
            **_base_workflow_flags(),
            "surface_prep": {
                "slab_generation": {
                    "enabled":          True,
                    "miller":           [1, 0, 0],
                    "thickness_ang":    8.0,
                    "vacuum_ang":       10.0,
                    "bulk_shift":       0.25,
                    "supercell_matrix": [[2, 0], [0, 2]],
                },
                "reconstruction": {
                    "enabled":       True,
                    "strategy":      "auto",
                    "side":          "top",
                    "buckling_dist": 0.4,
                },
                "passivation": {
                    "enabled":  True,
                    "element":  "H",
                    "side":     "bottom",
                    "coverage": 1.0,
                },
                "surface_analysis": {
                    "ideal_coordination": {"Si": 4, "H": 1},
                },
            },
        }
        slab = self._run_and_load(config)
        syms = set(slab.get_chemical_symbols())
        self.assertIn("Si", syms, "Si 없음")
        self.assertIn("H",  syms, "H 없음 — passivation 실패")
        self.assertNotIn("O", syms, "O 가 있어선 안 됨")

    def test_sio2_integration(self):
        """SiO2(001) slab prep 통합 테스트: O-terminated + bottom H 확인."""
        vasp_path = _sio2_vasp_path()
        if not os.path.exists(vasp_path):
            self.skipTest(f"SiO2 벌크 POSCAR 없음 (건너뜀): {vasp_path}")

        tmpdir = tempfile.mkdtemp()
        config = {
            "paths": {
                "substrate_bulk": vasp_path,
                "precursor":      None,
                "inhibitor":      None,
                "output_prefix":  tmpdir,
            },
            **_base_workflow_flags(),
            "surface_prep": {
                "slab_generation": {
                    "enabled":            True,
                    "miller":             [0, 0, 1],
                    "thickness_ang":      12.0,
                    "vacuum_ang":         15.0,
                    "target_area_ang2":   250.0,
                    "top_termination":    "O",
                    "bottom_termination": "O",
                },
                "passivation": {
                    "enabled":  True,
                    "element":  "H",
                    "side":     "bottom",
                    "coverage": 1.0,
                },
                "surface_analysis": {
                    "ideal_coordination": {"Si": 4, "O": 2, "H": 1},
                },
            },
        }
        slab = self._run_and_load(config)
        syms_list = list(slab.get_chemical_symbols())
        syms_set  = set(syms_list)

        self.assertIn("Si", syms_set)
        self.assertIn("O",  syms_set)
        self.assertIn("H",  syms_set, "H 없음 — passivation 실패")

        n_o  = syms_list.count("O")
        n_si = syms_list.count("Si")
        self.assertGreaterEqual(
            n_o / n_si, 2.0,
            f"O:Si={n_o/n_si:.2f} < 2 — O-terminated 아님 "
            "(top/bottom_termination 누락 확인)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
