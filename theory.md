전이 상태 이론(Transition State Theory, TST)의 조화 근사 버전인 **Harmonic Transition State Theory (HTST)**에서 진동 항과 엔트로피의 관계를 유도하는 과정에 대해 설명해 드리겠습니다.

이 유도는 통계역학적 분배 함수(Partition Function)와 열역학적 상태량 사이의 연결을 기반으로 합니다.

---

### 1. 기본 공식: Transition State Theory (TST)
에어링 식(Eyring equation)에 따르면, 반응 속도 상수 $k$는 다음과 같이 정의됩니다:

$$k = \kappa \frac{k_B T}{h} \frac{Q^{\ddagger}}{Q_{IS}} \exp\left(-\frac{\Delta E_0^\ddagger}{k_B T}\right)$$

여기서:
- $\kappa$: 투과 계수 (Transmission coefficient, 대개 1로 가정)
- $Q^{\ddagger}$: 전이 상태(TS)의 분배 함수 (반응 좌표 방향의 운동이 제외된 $3N-1$ 자유도)
- $Q_{IS}$: 초기 상태(Initial State, EQ)의 분배 함수 ($3N$ 자유도)
- $\Delta E_0^\ddagger$: 영점 에너지(ZPE)를 포함한 활성화 에너지

에어링 식(Eyring equation)은 1935년 헨리 에어링(Henry Eyring), 메러디스 에반스(Meredith Evans), 마이클 폴라니(Michael Polanyi)에 의해 독립적으로 제안된 **전이 상태 이론(Transition State Theory, TST)**의 핵심 결과물입니다.

이 식은 단순히 실험적 관찰에 기반한 아레니우스 식($k = A e^{-E_a/RT}$)에 이론적 기반을 부여하기 위해, **통계역학(Statistical Mechanics)**과 **화학 평형(Chemical Equilibrium)**의 원리를 결합하여 유도되었습니다. 유도 과정을 차근차근 설명해 드리겠습니다.

---

### 1. 기본 가정: 활성화 복합체(Activated Complex)의 존재
반응물 $A$와 $B$가 생성물 $P$로 변할 때, 에너지가 가장 높은 지점인 **전이 상태(Transition State)**에서 일시적인 구조체인 $X^{\ddagger}$가 형성된다고 가정합니다.

$$A + B \rightleftharpoons X^{\ddagger} \rightarrow P$$

여기에는 두 가지 핵심 가정이 포함됩니다:
1.  **준-평형(Quasi-equilibrium) 가정**: 반응물($A, B$)과 활성화 복합체($X^{\ddagger}$) 사이에는 열역학적 평형이 유지된다.
2.  **반응 경로의 분리**: 전이 상태에 도달한 복합체는 특정 진동 모드를 통해 매우 낮은 빈도로 장벽을 넘어가며, 이 과정이 전체 반응 속도를 결정한다.

### 2. 평형 상수와 분배 함수의 관계
반응물과 활성화 복합체 사이의 평형 상수 $K^{\ddagger}$는 통계역학적으로 다음과 같이 정의됩니다:

$$K^{\ddagger} = \frac{[X^{\ddagger}]}{[A][B]} = \frac{Q^{\ddagger}}{Q_A Q_B} \exp\left(-\frac{\Delta E_0^{\ddagger}}{k_B T}\right)$$

여기서:
- $Q$: 각 상태의 총 분배 함수(Total Partition Function)
- $\Delta E_0^{\ddagger}$: $0\text{ K}$에서의 활성화 에너지

### 3. 반응 좌표(Reaction Coordinate)의 분리
활성화 복합체 $X^{\ddagger}$는 $3N$개의 자유도를 가집니다. 이 중 하나는 반응 경로를 따라 움직이는 **반응 좌표**입니다. 에어링은 이 자유도를 나머지 진동 모드와 분리하였습니다.

$$Q^{\ddagger} = q_{RC} \cdot Q'_{rest}$$

이때, 반응 좌표 방향의 분배 함수 $q_{RC}$를 매우 낮은 주파수 $\nu$를 가진 조화 진동자로 취급하여 고온 극한 근사를 적용합니다:
$$q_{RC} = \frac{1}{1 - \exp(-h\nu / k_B T)} \approx \frac{k_B T}{h\nu}$$

### 4. 반응 속도의 결정
전체 반응 속도 $v$는 전이 상태에 있는 복합체의 농도 $[X^{\ddagger}]$와 그 복합체가 장벽을 넘어가는 빈도(frequency)의 곱으로 표현됩니다. 여기서 빈도는 앞서 가정한 진동수 $\nu$와 같습니다.

$$v = \nu [X^{\ddagger}]$$

이제 앞서 구한 평형 상수 식을 $[X^{\ddagger}]$에 대해 정리하여 대입합니다:
$$v = \nu \cdot \left( [A][B] \frac{q_{RC} Q'_{rest}}{Q_A Q_B} \exp\left(-\frac{\Delta E_0^{\ddagger}}{k_B T}\right) \right)$$

여기에 $q_{RC} \approx \frac{k_B T}{h\nu}$를 대입하면 놀랍게도 $\nu$가 상쇄됩니다:
$$v = \nu \cdot \frac{k_B T}{h\nu} \cdot \frac{Q'_{rest}}{Q_A Q_B} \exp\left(-\frac{\Delta E_0^{\ddagger}}{k_B T}\right) [A][B]$$
$$v = \frac{k_B T}{h} \frac{Q'_{rest}}{Q_A Q_B} \exp\left(-\frac{\Delta E_0^{\ddagger}}{k_B T}\right) [A][B]$$

### 5. 최종 에어링 식 (Eyring Equation)
반응 속도 법칙 $v = k [A][B]$와 비교하면, 속도 상수 $k$는 다음과 같이 도출됩니다.

$$k = \kappa \frac{k_B T}{h} K^{\ddagger}$$
또는 열역학적 표현으로:
$$k = \kappa \frac{k_B T}{h} \exp\left(\frac{\Delta S^{\ddagger}}{R}\right) \exp\left(-\frac{\Delta H^{\ddagger}}{RT}\right)$$

여기서 $\kappa$는 전이 상태에 도달한 분자가 실제로 생성물로 넘어가는 비율을 나타내는 **투과 계수(Transmission Coefficient)**입니다 (일반적으로 1로 가정).

---

### 물리적 의미와 요약
1.  **$\frac{k_B T}{h}$의 의미**: 이는 '자연스러운' 속도의 단위로, 온도 $T$에서 시스템이 전이 상태를 통과하는 기본적인 빈도를 의미합니다. (대략 $10^{12} \sim 10^{13} \text{ s}^{-1}$ 수준)
2.  **아레니우스 식과의 연결**: 에어링 식은 아레니우스 식의 빈도 인자 $A$가 단순한 상수가 아니라, 온도($T$)와 활성화 엔트로피($\Delta S^{\ddagger}$)에 의존하는 함수임을 보여줍니다.
3.  **성공적인 통합**: 이 유도는 거시적인 화학 반응 속도를 분자 수준의 미시적인 양(진동수, 관성 모멘트 등)을 포함하는 분배 함수로 설명해냈다는 점에서 화학 역사상 가장 중요한 성과 중 하나로 꼽힙니다.

---

### 2. 조화 근사 (Harmonic Approximation)
표면 반응에서 흡착된 분자는 고체 표면의 퍼텐셜 우물에 갇혀 있으므로, 모든 자유도를 진동(vibration)으로 취급할 수 있습니다. 조화 진동자의 분배 함수 $q_{vib, i}$는 다음과 같습니다:

$$q_{vib, i} = \frac{\exp(-h\nu_i / 2k_B T)}{1 - \exp(-h\nu_i / k_B T)}$$

고온 극한($k_B T \gg h\nu_i$) 또는 고전적 한계에서 이는 다음과 같이 근사됩니다:
$$q_{vib, i} \approx \frac{k_B T}{h\nu_i}$$

### 3. 유도 과정: 진동 항에서 엔트로피 항으로

#### A. 분배 함수와 엔트로피의 관계
통계역학에서 엔트로피 $S$와 분배 함수 $Q$의 관계는 다음과 같습니다:
$$S = R \left( \ln Q + T \left( \frac{\partial \ln Q}{\partial T} \right)_{V, N} \right)$$

조화 진동자 근사($q_{vib, i} \approx \frac{k_B T}{h\nu_i}$)를 적용하면:
$$\ln q_{vib, i} = \ln \left( \frac{k_B T}{h\nu_i} \right)$$
$$T \frac{\partial \ln q_{vib, i}}{\partial T} = T \cdot \frac{1}{T} = 1$$

따라서, 한 개의 진동 모드에 대한 엔트로피는 다음과 같습니다:
$$S_{vib, i} = R \left( \ln \left( \frac{k_B T}{h\nu_i} \right) + 1 \right)$$

#### B. 상태별 총 엔트로피
- **초기 상태 (EQ, $N$개의 모드):**
  $$S_{EQ} = \sum_{i=1}^{N} S_{vib, i} = R \left( \sum_{i=1}^{N} \ln \left( \frac{k_B T}{h\nu_{i, EQ}} \right) + N \right)$$
- **전이 상태 (TS, $N-1$개의 모드):**
  $$S_{TS} = R \left( \sum_{j=1}^{N-1} \ln \left( \frac{k_B T}{h\nu_{j, TS}} \right) + (N-1) \right)$$

#### C. 엔트로피 변화량 ($\Delta S^\ddagger$)
$$\frac{S_{TS} - S_{EQ}}{R} = \sum_{j=1}^{N-1} \ln \left( \frac{k_{B}T}{h\nu_{j,TS}} \right) - \sum_{i=1}^{N} \ln \left( \frac{k_{B}T}{h\nu_{i,EQ}} \right) - 1$$
$$\exp\left(\frac{\Delta S^\ddagger}{R}\right) = \exp\left( \ln \left[ \frac{\prod_{i=1}^{N} \nu_{i,EQ}}{\prod_{j=1}^{N-1} \nu_{j,TS}} \cdot \frac{h}{k_B T} \right] - 1 \right)$$
$$\exp\left(\frac{\Delta S^\ddagger}{R}\right) = \frac{\prod \nu_{EQ}}{\prod \nu_{TS}} \cdot \frac{h}{k_B T} \cdot e^{-1}$$

*(참고: 여기서 $e^{-1}$ 항은 열역학적 정의와 통계역학적 정의 사이의 컨벤션 차이에 따라 속도 식에 흡수되거나 무시될 수 있으나, 핵심은 빈도 인자(Pre-exponential factor)가 진동수의 비로 표현된다는 점입니다.)*

### 4. 물리적 배경 및 가정
1.  **조화 근사 (Harmonic Approximation):** 퍼텐셜 에너지가 평형점 근처에서 2차 함수 형태를 가짐을 가정합니다. 이는 낮은 온도나 물리 흡착/화학 흡착 초기 단계에서 잘 맞습니다.
2.  **보른-오펜하이머 근사:** 핵의 운동과 전자의 운동을 분리하여 PES(Potential Energy Surface)를 정의합니다.
3.  **열적 평형:** 반응물들이 전이 상태를 지나기 전에 볼츠만 분포에 도달할 정도로 충분히 평형을 유지하고 있다고 가정합니다.
4.  **Classical Limit:** $k_B T$가 진동 에너지($h\nu$)보다 훨씬 크다는 가정 하에 $q \approx k_B T / h\nu$ 형태가 도출됩니다. 만약 매우 낮은 온도라면 양자역학적 보정($1/(1-e^{-u})$)이 필요합니다.

### 요약
결론적으로 `vib_ts / vib_eq`가 엔트로피 항으로 표현되는 이유는, **엔트로피 자체가 시스템이 가질 수 있는 상태 수(분배 함수)의 로그 값**이기 때문입니다. HTST에서는 빈도 인자(Pre-factor) $A$가 다음과 같이 비니어드(Vineyard) 식으로 표현됩니다:

$$A_{Vineyard} = \frac{\prod_{i=1}^{N} \nu_{i, EQ}}{\prod_{j=1}^{N-1} \nu_{j, TS}}$$

이 값은 결과적으로 전이 상태 형성에 따른 **구조적 무질서도의 변화(엔트로피 변화)**를 물리적으로 대변하게 됩니다.

> [!NOTE]
> 계산 화학 코드(ASE 등)에서 Gibbs Free Energy를 구할 때, $G = H - TS$ 식을 통해 속도 상수를 계산하면 자연스럽게 이 진동수 비가 엔트로피 항에 포함되어 계산됩니다.
