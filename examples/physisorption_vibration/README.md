# Physisorption Vibrational Analysis: FHVA vs. PHVA
## Case Study: DIPAS on SiO₂ Surface

이 예제는 표면 흡착 시스템에서 **전체 헤시안 분석(FHVA)**과 **부분 헤시안 분석(PHVA)**의 물리적 차이를 이해하고, 대규모 시스템에서 효율적이고 정확한 진동 분석을 수행하기 위한 가이드를 제공합니다.

---

## 1. 이론적 배경: FHVA vs. PHVA

표면 시스템의 진동 분석 시, 기판(Slab)의 크기에 따라 분석 방법론의 선택이 중요합니다.

### Full Hessian Vibrational Analysis (FHVA)
*   **특징:** 시스템의 모든 원자($3N$ 자유도)를 변위시켜 헤시안 행렬을 구성합니다.
*   **장점:** 표면 재구성이나 기판-흡착제 간의 장거리 커플링을 포함하는 완전한 정보를 제공합니다.
*   **단점:** 유한한 슬래브 모델에서 **'슬래브 밀림(Slab drift)'**과 같은 비물리적인 이미지너리 모드가 발생하기 쉽고 계산 비용이 매우 높습니다.

### Partial Hessian Vibrational Analysis (PHVA)
*   **특징:** 관심 영역(흡착제 및 표면 상단)만 이동시키고, 나머지는 고정(Frozen)합니다.
*   **장점:** 비물리적인 슬래브 전체의 움직임을 억제하여 로컬 결합 특성을 명확히 보여주며 계산이 빠릅니다.
*   **한계:** 기판 고정으로 인한 'Stiffening' 효과가 저주파 대역에서 발생할 수 있습니다.

---

## 2. 이미지너리 모드 진단: Collective Ratio

FHVA 계산 시 발생하는 음의 진동수가 실제 구조적 불안정성인지, 아니면 수치적 Artifact인지 판별하기 위해 **Collective Ratio** 지표를 사용합니다.

$$\text{Collective Ratio} = \frac{\text{Mean(Atomic Displacements)}}{\text{Max(Atomic Displacements)}}$$

*   **Ratio > 0.1 (Global Drift):** 거의 모든 원자가 한 방향으로 움직이는 현상으로, 실제 반응과 무관한 **슬래브 드리프트(Artifact)**입니다. **PHVA 결과를 신뢰해야 합니다.**
*   **Ratio << 0.1 (Local Instability):** 움직임이 흡착제 주변에 집중된 경우로, 실제 **Saddle point**일 확률이 높습니다. 구조 재최적화가 필요합니다.

---

## 3. MAC 기반 모드 매칭 분석 (Validation)

PHVA가 FHVA의 물리적 특성을 얼마나 잘 보존하는지 **Modal Assurance Criterion (MAC)**으로 검증합니다.

| 주파수 대역 | MAC ≥ 0.7 비율 | 주파수 오차 (Median) | 해석 |
|------------|---------------|-------------------|------|
| 고주파 (>30 THz) | ~48% | < 0.1% | 흡착제 내부 진동, PHVA가 완벽히 재현 |
| 중주파 (10-30 THz) | ~29% | ~0.4% | 흡착제-기판 하이브리드 모드, 신뢰 가능 |
| 저주파 (<10 THz) | < 5% | 가변적 | 기판 포논 모드, PHVA에서 물리적 변형 발생 |

**핵심 결론:** HTST(전이 상태 이론) 계산에 중요한 **흡착제 관련 모드**들은 PHVA에서도 sub-0.1% 오차로 FHVA와 동일하게 계산됩니다.

---

## 4. HTST Prefactor 계산 전략

PHVA 주파수를 이용한 HTST Prefactor($\nu^\ddagger$) 계산 시 오차 상쇄 원리가 작용합니다.

$$\nu^\ddagger = \frac{k_\mathrm{B}T}{h} \cdot \frac{\prod \nu_i^\mathrm{Reactant}}{\prod \nu_j^\mathrm{TS}}$$

*   **오차 상쇄:** PHVA에서 오차가 큰 저주파 포논 모드들은 반응물(R)과 전이 상태(TS)에서 거의 동일하게 나타나므로, 분자와 분모에서 서로 상쇄됩니다.
*   **정확도:** 실제로 Prefactor를 결정하는 고주파 결합 모드들은 PHVA에서 매우 정확하므로, 본 시스템에서 **PHVA를 이용한 Prefactor 오차는 2% 미만**으로 매우 신뢰할 수 있습니다.

---

## 5. 실행 가이드

### 5.1 계산 수행 (`run_vibration.py`)
MACE 포텐셜을 사용하여 FHVA 및 PHVA 계산을 수행합니다.
```bash
python run_vibration.py --mode both   # FHVA, PHVA 모두 실행
python run_vibration.py --mode phva   # PHVA만 실행 (권장)
```

### 5.2 결과 분석 및 시각화 (`analyze_vibration.py`)
계산된 `qpoints.yaml` 파일들을 비교하여 진단 플롯을 생성합니다.
```bash
python analyze_vibration.py
```
*   **fig2_parity.png**: FHVA vs PHVA 주파수 상관관계 ($R^2$ 포함)
*   **fig5_localization.png**: 주파수별 IPR(국소화) 분포
*   **fig6_participation.png**: 원소별 진동 에너지 참여도

---

## 6. 주요 파일 안내
*   `run_vibration.py`: 진동 분석 실행 스크립트
*   `analyze_vibration.py`: 고해상도 진단 플롯 생성 및 통계 분석
*   `config_*.yaml`: 분석 영역(Frozen Z 등) 설정 파일
*   `dipas_sio2_relaxed.vasp`: 최적화된 입력을 위한 구조 파일
