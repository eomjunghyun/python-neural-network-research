# `src` Guide

`src/`는 실험 공통 로직을 모듈화한 코드 영역이다. 설정, 모델, 시계열 생성, 노이즈 주입, 지표 계산, 실험 실행이 모두 여기 모여 있다.

현재 구현은 연속시간 샘플링과 이산시간 직접 생성 둘 다 지원한다.

## 디렉터리 구성

```text
src/
  __init__.py
  common_utils.py
  config.py
  data_utils.py
  experiment_runner.py
  metrics.py
  models.py
```

## 빠른 사용법

```python
from src import ExperimentConfig, run_experiment

cfg = ExperimentConfig(
    MODEL_ID="AN001_BN_RELU",
    NUM_FREQS=4,
    RANDOM_AMPLITUDE=True,
    RANDOM_PHASE=True,
    NORMALIZE_H_COLUMNS=True,
)

results = run_experiment(cfg)
print(results["summary_df"])
```

Discrete mode 예시는 아래와 같다.

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

## 공개 API

`src/__init__.py`에서 아래 주요 객체와 함수를 export한다.

- 설정:
  - `ExperimentConfig`
- 모델:
  - `build_model`
  - `MODEL_REGISTRY`
  - `AnalyticNetAN001BnReLU`
  - `AnalyticNetAN002NoBnTanh`
  - `AnalyticNetAN003Linear`
  - `AnalyticNetAN004DeepTanh`
- 데이터:
  - `generate_sin_data`
  - `generate_continuous_sin_data`
  - `generate_discrete_sin_data`
  - `add_noise_to_signal`
  - `make_dataset`
  - `split_train_test_tensors`
- 지표:
  - `calculate_rank_metrics`
  - `calculate_subspace_alignment_metrics`
  - `calculate_sampled_basis_numerical_dim`
  - `build_sampled_basis_matrix`
  - `min_delta_f`
  - `min_delta_theta`
  - `regression_accuracy`
  - `regression_r2`
  - `snr_db_from_tensors`
  - `normalize_feature_columns`
  - `topk_energy_ratio`
- 실험 러너:
  - `train_one_seed`
  - `aggregate_seed_results`
  - `make_summary_dataframe`
  - `plot_results`
  - `print_overall_summary`
  - `print_overall_ci95_summary`
  - `run_experiment`

## 모듈별 설명

### `config.py`

`ExperimentConfig` dataclass를 정의한다. 모든 실험 하이퍼파라미터는 여기에 모인다.

핵심 필드는 다음과 같다.

- seed 관련:
  - `GLOBAL_SEED`
  - `DATA_SEED_BASE`
  - `NOISE_SEED_BASE`
  - `TRAIN_SEED_BASE`
- 시간 모드 관련:
  - `time_mode`
  - `DT`
  - `theta_min`
  - `theta_max`
- 데이터 생성 관련:
  - `SEQ_LEN`
  - `NUM_FREQS`
  - `NUM_FREQS_MIN`
  - `NUM_FREQS_MAX`
  - `FREQ_MIN`
  - `FREQ_MAX`
  - `NYQUIST_MARGIN`
- amplitude / phase 관련:
  - `RANDOM_AMPLITUDE`
  - `AMP_MIN`
  - `AMP_MAX`
  - `RANDOM_PHASE`
  - `PHASE_MIN`
  - `PHASE_MAX`
- noise 관련:
  - `USE_NOISE`
  - `NOISE_TYPE`
  - `SNR_DB`
  - `AR1_RHO`
  - `IMPULSE_PROB`
  - `IMPULSE_SCALE`
- 학습 타깃:
  - `TRAIN_TARGET`
- 모델 / 최적화:
  - `MODEL_ID`
  - `LAG`
  - `HIDDEN_DIM`
  - `BOTTLENECK_MULTIPLIER`
  - `LR`
  - `EPOCHS`
- 반복 실험:
  - `NUM_EXPERIMENTS`
  - `SEEDS_PER_FREQ`
  - `TEST_RATIO`
- 지표 / 출력:
  - `ACC_TOLERANCE`
  - `RANK_THRESHOLD`
  - `SCREE_TOPK`
  - `NORMALIZE_H_COLUMNS`
  - `VERBOSE`
  - `MAKE_PLOTS`

기본값은 backward compatibility를 위해 `time_mode="continuous"`다.

### `models.py`

식별자 기반으로 네 가지 모델을 제공한다.

- `AN001_BN_RELU`
  - `Linear -> BatchNorm1d -> ReLU -> Linear -> BatchNorm1d -> Linear`
- `AN002_NO_BN_TANH`
  - `Linear -> Tanh -> Linear -> Tanh -> Linear`
- `AN003_LINEAR`
  - `Linear -> Linear -> Linear`
- `AN004_DEEP_TANH`
  - `Linear -> Tanh -> Linear -> Tanh -> Linear -> Tanh -> Linear`

모든 모델은 `(y_hat, h)`를 반환한다.

- `y_hat`: next-step prediction
- `h`: bottleneck representation

### `common_utils.py`

실험 전역 공통 보조 함수들이다.

- `set_seed(seed)`
  - Python, NumPy, PyTorch seed를 동시에 고정한다
- `validate_sampling_constraints(dt, freq_max, margin=0.98)`
  - continuous mode에서 Nyquist margin 제약을 검사한다
- `validate_config(cfg)`
  - 설정 전체 유효성을 검사한다
  - `time_mode`는 `"continuous"` 또는 `"discrete"`만 허용
  - continuous mode에서는 정수 주파수 샘플링 가능성과 Nyquist 조건을 검사
  - discrete mode에서는 `0 < theta_min < theta_max < pi`를 검사
- `mean_std(values)`
  - 평균과 표본표준편차를 계산한다
- `mean_std_ci95(values)`
  - 평균, 표준편차, 정규근사 95% 신뢰구간을 계산한다

### `data_utils.py`

시계열 생성과 전처리를 담당한다.

#### `generate_continuous_sin_data(cfg, rng_py, rng_np)`

연속시간 신호를 먼저 정의하고 `DT`로 샘플링한다.

\[
x(t)=\sum_j a_j \sin(\omega_j t+\phi_j), \qquad t_n=n \cdot DT
\]

반환:

- `y`: shape `(SEQ_LEN,)`의 `float32` 배열
- `info`:
  - `time_mode="continuous"`
  - `freqs`
  - `thetas=None`
  - `amplitudes`
  - `phases`
  - `components`

#### `generate_discrete_sin_data(cfg, rng_py, rng_np)`

이산시간 시계열을 직접 생성한다.

\[
y_n=\sum_j a_j \sin(\theta_j n+\phi_j)
\]

반환:

- `y`: shape `(SEQ_LEN,)`의 `float32` 배열
- `info`:
  - `time_mode="discrete"`
  - `freqs=None`
  - `thetas`
  - `amplitudes`
  - `phases`
  - `components`

#### `generate_sin_data(cfg, rng_py, rng_np)`

공통 wrapper다.

- `cfg.time_mode == "continuous"`면 `generate_continuous_sin_data`
- `cfg.time_mode == "discrete"`면 `generate_discrete_sin_data`

기존 호출 방식은 유지되며, 새 실험에서만 `time_mode="discrete"`를 켜면 된다.

#### `add_noise_to_signal(clean_signal, snr_db, noise_type, rng, ...)`

지원 노이즈:

- `white`
- `ar1`
- `impulsive`

반환:

- `noisy`
- `noise`
- `actual_snr_db`

#### `make_dataset(data, lag)`

1차원 시계열을 lag-window 지도학습 텐서로 바꾼다.

- 입력: `data[i:i+lag]`
- 타깃: `data[i+lag]`

#### `split_train_test_tensors(*tensors, test_ratio)`

정렬된 tensor들을 시계열 순서를 유지한 채 train/test로 분할한다.

### `metrics.py`

표현 분석용 지표 모듈이다.

#### 기본 회귀 지표

- `regression_accuracy(y_true, y_pred, tol)`
- `regression_r2(y_true, y_pred)`

#### rank 관련

- `calculate_rank_metrics(S, threshold)`
  - `rank_threshold`
  - `rank_entropy`
- `topk_energy_ratio(singular_values, top_k)`

#### 주파수/각주파수 간격

- `min_delta_f(freqs)`
- `min_delta_theta(thetas)`

#### feature 전처리

- `normalize_feature_columns(H, eps=1e-12)`

#### basis 행렬 생성

- `build_sampled_fourier_matrix(freqs, dt, lag, seq_len)`
  - continuous mode용 sampled Fourier basis
- `build_sampled_discrete_basis_matrix(thetas, lag, seq_len)`
  - discrete mode용 sampled basis
- `build_sampled_basis_matrix(...)`
  - `time_mode`에 따라 위 둘 중 하나를 호출하는 통합 인터페이스

#### basis numerical dimension

- `calculate_sampled_fourier_numerical_dim(...)`
  - continuous 전용 helper
- `calculate_sampled_discrete_numerical_dim(...)`
  - discrete 전용 helper
- `calculate_sampled_basis_numerical_dim(...)`
  - mode-aware 통합 helper

현재 `fourier_numerical_dim`은 sampled basis 행렬의 `np.linalg.matrix_rank(...)` 결과다.

#### subspace alignment

- `calculate_subspace_alignment_metrics(H, freqs, dt, lag, top_k, *, thetas=None, time_mode="continuous")`

이 함수는 `H`의 열공간과, sampled basis 행렬의 열공간을 비교한다. 개별 열을 1:1로 매칭하는 방식은 아니다.

반환 지표:

- `align_coverage`
- `align_purity`
- `alignment_score_2k`
- `align_mean_cosine`
- `mean_principal_angle_deg`
- `principal_angles_deg`

`time_mode="continuous"`이면 `freqs` 기준, `time_mode="discrete"`이면 `thetas` 기준 basis를 만든다.

### `experiment_runner.py`

실험 전체 파이프라인을 관리한다.

#### `train_one_seed(...)`

한 training seed에 대해 모델 하나를 학습하고 지표를 계산한다.

출력에는 아래가 포함된다.

- train 예측 지표:
  - `mse`
  - `mae`
  - `acc`
  - `train_r2`
- test 예측 지표:
  - `test_mse`
  - `test_mae`
  - `test_acc`
  - `test_r2`
- train representation 지표:
  - `train_rank_threshold`
  - `train_rank_entropy`
  - `train_rank_gap`
  - `train_rel_rank_gap`
  - `train_spectral_gap_2k`
  - `train_energy_ratio_2k`
  - `train_align_coverage`
  - `train_align_purity`
  - `train_alignment_score_2k`
  - `train_align_mean_cosine`
  - `train_mean_principal_angle_deg`
- test representation 지표:
  - `rank_threshold`
  - `rank_entropy`
  - `rank_gap`
  - `rel_rank_gap`
  - `spectral_gap_2k`
  - `energy_ratio_2k`
  - `align_coverage`
  - `align_purity`
  - `alignment_score_2k`
  - `align_mean_cosine`
  - `mean_principal_angle_deg`

`USE_NOISE=True`일 때는 train/test SNR 지표도 포함된다.

#### `aggregate_seed_results(seed_results, cfg)`

같은 signal set에서 여러 training seed 결과를 집계한다.

- 모든 scalar metric에 대해
  - `*_mean`
  - `*_std`
  - `*_ci95_low`
  - `*_ci95_high`
  를 만든다
- `principal_angles_deg_all`
- `snorm_topk_all`
  도 함께 저장한다

#### `make_summary_dataframe(cfg, set_rows, overall_summary)`

set별 결과와 overall row를 하나의 `DataFrame`으로 정리한다.

주요 컬럼:

- 메타데이터:
  - `set_idx`
  - `freqs`
  - `thetas`
  - `amplitudes`
  - `phases`
  - `actual_snr_db`
  - `min_delta_f`
  - `min_delta_theta`
- basis numerical dim:
  - `fourier_numerical_dim_*`
- train/test 예측 및 representation 요약 지표들

참고로 set별 row에는 train/test 요약 지표가 폭넓게 들어가지만, `overall_summary`는 현재 주로 test 쪽 핵심 지표와 일부 공통 지표 중심으로 집계한다.

#### `plot_results(results, cfg)`

`cfg.MAKE_PLOTS=True`일 때 기본 플롯 두 개를 출력한다.

- `Train MSE by Set`
- `Subspace Coverage by Set`

#### `print_overall_summary(overall_summary, cfg)`

현재 overall 평균 요약을 콘솔 친화적으로 출력한다.

#### `print_overall_ci95_summary(overall_summary, cfg)`

현재 overall 95% 신뢰구간을 콘솔에 출력한다.

#### `run_experiment(cfg=None)`

메인 엔트리포인트다.

동작 순서:

1. `ExperimentConfig` 준비
2. `validate_config`
3. 전체 seed 고정
4. 모델 차원 계산
5. signal set 생성
6. noise 주입 여부 처리
7. lag-window dataset 생성
8. train/test split
9. 여러 training seed 반복 학습
10. set별 집계
11. overall summary 생성
12. `summary_df` 반환
13. 옵션에 따라 출력과 플롯 실행

반환 딕셔너리의 주요 키:

- `config_df`
- `config`
- `theoretical_rank`
- `bottleneck_dim`
- `set_results`
- `overall_summary`
- `summary_df`
- `all_principal_angles_deg`
- `all_snorm_topk`

## 추천 워크플로

### 1. 기본 실험

```python
from src import ExperimentConfig, run_experiment

cfg = ExperimentConfig()
results = run_experiment(cfg)
```

### 2. continuous/discrete 비교

```python
cfg_cont = ExperimentConfig(time_mode="continuous", NUM_FREQS=4)
cfg_disc = ExperimentConfig(time_mode="discrete", NUM_FREQS=4)
```

### 3. `k` sweep 노트북

`experiments/hyperparameter_k_sweep/` 아래 노트북은 다른 하이퍼파라미터는 고정하고 `NUM_FREQS`만 바꾸는 용도로 정리돼 있다.

## 유지 원칙

- 새 파라미터는 먼저 `ExperimentConfig`에 추가한다
- 재사용할 로직은 노트북 대신 `src/`로 올린다
- 기존 API는 가능한 유지하고, 새 기능은 mode-aware branch로 추가한다
- 문서 변경이 필요한 기능을 넣었으면 루트 `README.md`와 `src/README.md`를 함께 갱신한다
