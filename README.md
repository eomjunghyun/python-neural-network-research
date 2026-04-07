# python-neural-network-research

여러 개의 사인파 성분으로 이루어진 1차원 시계열을 신경망이 `next-step prediction`으로 학습할 때, 내부 병목 표현이 어떤 차원 구조와 주파수 subspace 정렬을 보이는지 연구하는 레포지토리다.

현재 코드는 기존 연속시간 샘플링 방식과, 새로 추가된 이산시간 직접 생성 방식을 모두 지원한다.

## 연구 목표

- 사인파 합성 시계열을 예측하도록 학습했을 때 hidden representation의 유효 차원이 어떻게 형성되는가
- 이론적 rank `2 * NUM_FREQS`와 실제 특징 행렬의 rank proxy가 얼마나 가까운가
- hidden subspace가 실제 생성 basis의 subspace와 얼마나 잘 정렬되는가
- amplitude, phase, lag, sequence length, noise, 모델 구조, `k=NUM_FREQS` 변화가 위 현상에 어떤 영향을 주는가

## 신호 생성 모드

### 1. Continuous mode

기본 모드다.

\[
x(t)=\sum_{j=1}^{k} a_j \sin(\omega_j t+\phi_j), \qquad t_n=n \cdot DT
\]

- 연속시간 신호를 먼저 정의한 뒤 `DT` 간격으로 샘플링한다
- 설정 키는 `time_mode="continuous"`
- 주파수는 `FREQ_MIN ~ FREQ_MAX` 범위의 정수 각주파수에서 중복 없이 뽑는다

### 2. Discrete mode

새로 추가된 모드다.

\[
y_n=\sum_{j=1}^{k} a_j \sin(\theta_j n+\phi_j), \qquad n=0,1,\dots,N-1
\]

- 처음부터 이산시간 시계열을 직접 생성한다
- 설정 키는 `time_mode="discrete"`
- `theta_min`, `theta_max` 사이에서 discrete angular frequency를 직접 샘플링한다
- 기본 범위는 `0.05 * pi`부터 `0.85 * pi`까지다

## 학습 과제

길이 `LAG`의 슬라이딩 윈도우로 다음 시점을 예측한다.

- 입력: `x[t : t + LAG]`
- 타깃: `x[t + LAG]`

노이즈 실험에서는 noisy signal을 입력으로 쓰고, 타깃은 아래 둘 중 하나다.

- `TRAIN_TARGET="noisy"`: noisy 신호 자체를 예측
- `TRAIN_TARGET="clean"`: clean 신호를 복원하도록 학습

## 모델

`MODEL_ID`로 아래 네 가지 모델을 선택할 수 있다.

- `AN001_BN_RELU`
- `AN002_NO_BN_TANH`
- `AN003_LINEAR`
- `AN004_DEEP_TANH`

공통적으로 bottleneck representation `h`를 함께 반환하며,

- `bottleneck_dim = BOTTLENECK_MULTIPLIER * NUM_FREQS`
- `theoretical_rank = 2 * NUM_FREQS`

를 기준으로 representation 지표를 계산한다.

## 주요 지표

### 예측 성능

- train: `mse`, `mae`, `acc`, `train_r2`
- test: `test_mse`, `test_mae`, `test_acc`, `test_r2`

### rank 관련

- train: `train_rank_threshold`, `train_rank_entropy`, `train_rank_gap`, `train_rel_rank_gap`
- test: `rank_threshold`, `rank_entropy`, `rank_gap`, `rel_rank_gap`
- 추가 스펙트럼 지표:
  - `train_spectral_gap_2k`, `spectral_gap_2k`
  - `train_energy_ratio_2k`, `energy_ratio_2k`

### subspace alignment

특징 행렬의 열공간과, 샘플링된 `sin/cos` basis 행렬의 열공간을 비교한다.

- train:
  - `train_align_coverage`
  - `train_align_purity`
  - `train_alignment_score_2k`
  - `train_align_mean_cosine`
  - `train_mean_principal_angle_deg`
- test:
  - `align_coverage`
  - `align_purity`
  - `alignment_score_2k`
  - `align_mean_cosine`
  - `mean_principal_angle_deg`

### basis numerical dimension

샘플링된 basis 행렬의 수치적 랭크를 계산해 `fourier_numerical_dim`으로 기록한다.

- continuous mode에서는 sampled Fourier basis의 수치적 랭크
- discrete mode에서는 sampled discrete-time basis의 수치적 랭크

### noise 관련

`USE_NOISE=True`일 때 아래 지표가 추가된다.

- train: `train_input_snr_db`, `train_output_snr_db`, `train_snr_gain_db`
- test: `input_snr_db`, `output_snr_db`, `snr_gain_db`

모든 seed 집계 지표는 `*_mean`, `*_std`, `*_ci95_low`, `*_ci95_high` 형태로 저장된다.

## 실험 파이프라인

`run_experiment(cfg)`는 아래 순서로 동작한다.

1. `ExperimentConfig` 준비
2. 설정 검증
3. seed 고정
4. 모델 생성
5. signal set 생성
6. 필요 시 noise 추가
7. lag-window dataset 구성
8. chronological train/test split
9. 같은 signal set에 대해 여러 training seed 반복 학습
10. train/test 예측 지표 계산
11. train/test representation 지표 계산
12. set별 집계
13. overall summary와 `summary_df` 생성
14. `MAKE_PLOTS=True`일 때 기본 플롯 출력

## 주요 설정

핵심 설정은 `src/config.py`의 `ExperimentConfig`에 모여 있다.

### 시간 모드 관련

- `time_mode`: `"continuous"` 또는 `"discrete"`
- continuous 전용:
  - `DT`
  - `FREQ_MIN`
  - `FREQ_MAX`
  - `NYQUIST_MARGIN`
- discrete 전용:
  - `theta_min`
  - `theta_max`

### 공통 데이터 생성 관련

- `SEQ_LEN`
- `NUM_FREQS`
- `RANDOM_AMPLITUDE`, `AMP_MIN`, `AMP_MAX`
- `RANDOM_PHASE`, `PHASE_MIN`, `PHASE_MAX`

### 모델/학습 관련

- `MODEL_ID`
- `LAG`
- `HIDDEN_DIM`
- `BOTTLENECK_MULTIPLIER`
- `LR`
- `EPOCHS`

### 반복 실험 관련

- `NUM_EXPERIMENTS`
- `SEEDS_PER_FREQ`
- `TEST_RATIO`

### 출력 관련

- `VERBOSE`
- `MAKE_PLOTS`

## 빠른 시작

### 기본 실행

```python
from src import ExperimentConfig, run_experiment

cfg = ExperimentConfig()
results = run_experiment(cfg)

print(results["summary_df"])
```

### Continuous mode 예시

```python
from src import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    time_mode="continuous",
    NUM_FREQS=4,
    RANDOM_AMPLITUDE=True,
    RANDOM_PHASE=True,
    LAG=32,
)
results = run_experiment(cfg)
```

### Discrete mode 예시

```python
import numpy as np
from src import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    time_mode="discrete",
    NUM_FREQS=4,
    SEQ_LEN=2000,
    LAG=32,
    theta_min=0.05 * np.pi,
    theta_max=0.85 * np.pi,
    RANDOM_AMPLITUDE=False,
    RANDOM_PHASE=True,
)
results = run_experiment(cfg)
```

### 노이즈 복원 예시

```python
from src import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    USE_NOISE=True,
    NOISE_TYPE="ar1",
    SNR_DB=5.0,
    TRAIN_TARGET="clean",
    RANDOM_AMPLITUDE=True,
    RANDOM_PHASE=True,
)
results = run_experiment(cfg)
```

## 노트북

실험용 노트북은 `experiments/` 아래에 정리한다.

- `experiments/hyperparameter_k_sweep/20260406_k_sweep_k01_initial.ipynb`
  - continuous mode에서 `k=1..16` sweep
- `experiments/hyperparameter_k_sweep/20260406_k_sweep_discrete_k01_k16.ipynb`
  - discrete mode에서 `k=1..16` sweep

두 노트북 모두 실험 셀 하나당 하나의 `k`만 바꾸도록 구성되어 있다.

## 레포 구조

```text
.
├─ README.md
├─ requirements.txt
├─ experiments/
│  └─ hyperparameter_k_sweep/
└─ src/
   ├─ README.md
   ├─ __init__.py
   ├─ common_utils.py
   ├─ config.py
   ├─ data_utils.py
   ├─ experiment_runner.py
   ├─ metrics.py
   └─ models.py
```

## 추가 문서

- `src` 내부 함수/모듈 설명: [src/README.md](src/README.md)
