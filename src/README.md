# `src` Guide

이 디렉터리는 `0326_실험5_정규화.ipynb`의 공통 로직을 재사용 가능한 파이썬 모듈로 분리한 연구 베이스 코드다.  
설정, 모델, 데이터 생성, 노이즈 주입, 지표 계산, 실험 실행 함수를 모두 `src` 아래로 옮겨서 이후 실험이 동일한 기준 코드를 사용하도록 정리했다.

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

모델 식별자를 바꾸면 같은 파이프라인에서 다른 모델 변형을 바로 비교할 수 있다.

```python
cfg = ExperimentConfig(MODEL_ID="AN004_DEEP_TANH")
results = run_experiment(cfg)
```

## 모듈별 설명

### `config.py`

`ExperimentConfig`를 정의한다. 노트북에서 흩어져 있던 실험 하이퍼파라미터를 하나의 dataclass로 고정해 재현성을 높인다.

중요 파라미터:

- `MODEL_ID`: 사용할 모델 식별자
- `NUM_FREQS`: 합성 주파수 개수
- `LAG`: 입력 윈도우 길이
- `TEST_RATIO`: chronological test split 비율
- `SEQ_LEN`, `DT`: 샘플 길이와 샘플링 간격
- `USE_NOISE`, `NOISE_TYPE`, `SNR_DB`: 노이즈 실험 설정
- `NORMALIZE_H_COLUMNS`: hidden column 정규화 여부

예시:

```python
from src import ExperimentConfig

cfg = ExperimentConfig(
    MODEL_ID="AN003_LINEAR",
    NUM_FREQS=5,
    LAG=80,
    SEQ_LEN=5000,
    DT=0.01,
    RANDOM_AMPLITUDE=False,
    NORMALIZE_H_COLUMNS=True,
)
```

### `models.py`

식별자 기반으로 4개 모델을 제공한다.

#### `AnalyticNetAN001BnReLU`

기존 베이스라인 모델이다.

1. `Linear(input_dim -> hidden_dim)`
2. `BatchNorm1d(hidden_dim)`
3. `ReLU()`
4. `Linear(hidden_dim -> bottleneck_dim)`
5. `BatchNorm1d(bottleneck_dim)`
6. `Linear(bottleneck_dim -> 1, bias=False)`

모델 식별자:

- `AN001_BN_RELU`

#### `AnalyticNetAN002NoBnTanh`

배치 정규화를 제거하고 활성화 함수를 `Tanh`로 교체한 모델이다.

1. `Linear(input_dim -> hidden_dim)`
2. `Tanh()`
3. `Linear(hidden_dim -> bottleneck_dim)`
4. `Tanh()`
5. `Linear(bottleneck_dim -> 1, bias=False)`

모델 식별자:

- `AN002_NO_BN_TANH`

#### `AnalyticNetAN003Linear`

활성화 함수를 제거한 선형 모델이다.

1. `Linear(input_dim -> hidden_dim)`
2. `Linear(hidden_dim -> bottleneck_dim)`
3. `Linear(bottleneck_dim -> 1, bias=False)`

모델 식별자:

- `AN003_LINEAR`

#### `AnalyticNetAN004DeepTanh`

`Tanh` 기반에 hidden layer를 하나 더 추가한 모델이다.

1. `Linear(input_dim -> hidden_dim)`
2. `Tanh()`
3. `Linear(hidden_dim -> hidden_dim)`
4. `Tanh()`
5. `Linear(hidden_dim -> bottleneck_dim)`
6. `Tanh()`
7. `Linear(bottleneck_dim -> 1, bias=False)`

모델 식별자:

- `AN004_DEEP_TANH`

공통 출력:

- `y_hat`: 다음 시점 예측값
- `h`: bottleneck representation

직접 생성 예시:

```python
from src.models import AnalyticNetAN002NoBnTanh
import torch

model = AnalyticNetAN002NoBnTanh(input_dim=32, hidden_dim=64, bottleneck_dim=16)
x = torch.randn(8, 32)
y_hat, h = model(x)
```

식별자 기반 생성 예시:

```python
from src import build_model

model = build_model(
    model_id="AN004_DEEP_TANH",
    input_dim=32,
    hidden_dim=64,
    bottleneck_dim=16,
)
```

### `common_utils.py`

실험 전역 공통 보조 함수를 둔다.

#### `set_seed(seed: int) -> None`

- Python, NumPy, PyTorch 시드를 동시에 고정한다.
- `PYTHONHASHSEED`, `torch.backends.cudnn.deterministic`, `benchmark` 설정까지 함께 다룬다.

#### `validate_sampling_constraints(dt: float, freq_max: float, margin: float = 0.98) -> None`

- 샘플링 간격 `DT`가 최대 각주파수 `FREQ_MAX`를 안전하게 표현할 수 있는지 검사한다.
- 기준은 `freq_max < margin * (pi / dt)` 이다.

#### `validate_config(cfg: ExperimentConfig) -> None`

다음을 한 번에 검사한다.

- 주파수 개수 범위
- 주파수 샘플링 가능 여부
- `TRAIN_TARGET` 유효성
- `MODEL_ID` 유효성
- `NOISE_TYPE` 유효성
- amplitude/phase 파라미터 정합성
- Nyquist 조건

#### `mean_std(values: List[float]) -> Tuple[float, float]`

- scalar metric 평균과 표본 표준편차를 계산한다.
- 값이 0개면 `nan`, 1개면 표준편차를 `0.0`으로 처리한다.

#### `mean_std_ci95(values: List[float]) -> Tuple[float, float, float, float]`

- 평균, 표준편차, 95% 신뢰구간 하한/상한을 반환한다.
- 현재 구현은 `1.96 * std / sqrt(n)`을 쓰는 정규 근사 방식이다.

### `data_utils.py`

데이터 생성과 전처리를 담당한다.

#### `generate_sin_data(cfg, rng_py, rng_np)`

- `NUM_FREQS`개의 정수 주파수를 `FREQ_MIN ~ FREQ_MAX` 범위에서 중복 없이 샘플링한다.
- 각 성분에 대해 `a_k * sin(w_k * t + phi_k)`를 만든 뒤 전부 합산한다.

반환값:

- `y`: shape `(SEQ_LEN,)`의 `float32` 시계열
- `info`: `freqs`, `amplitudes`, `phases`, `components`

#### `add_noise_to_signal(clean_signal, snr_db, noise_type, rng, ...)`

지원 노이즈:

- `white`
- `ar1`
- `impulsive`

기능:

- raw noise 생성
- 목표 `SNR_DB`에 맞도록 전력 재스케일링
- noisy signal, noise, 실제 달성 SNR 반환

#### `make_dataset(data, lag)`

1차원 시계열을 supervised next-step prediction 데이터셋으로 변환한다.

- 입력 `x[i] = data[i:i+lag]`
- 타깃 `y[i] = data[i+lag]`

반환 shape:

- `x_train`: `(N, lag)`
- `y_train`: `(N, 1)`

#### `split_train_test_tensors(*tensors, test_ratio)`

- 정렬된 tensor들을 시계열 순서를 유지한 채 train/test로 분할한다.
- 실험 코드는 이 함수를 사용해 `H_test` 기반 지표를 계산한다.

### `metrics.py`

표현 분석용 지표를 모은 모듈이다.

#### `regression_accuracy(y_true, y_pred, tol)`

- 절대 오차가 `tol` 이하인 비율을 계산한다.

#### `regression_r2(y_true, y_pred)`

- 테스트 결정계수 `R^2`를 계산한다.

#### `calculate_rank_metrics(S, threshold)`

- `rank_threshold`
- `rank_entropy`

를 계산한다.

`spectral_gap_2k`는 실험 러너에서 다음과 같이 계산한다.

```text
spectral_gap_2k = sigma_(2k) / sigma_(2k+1)
```

#### `min_delta_f(freqs)`

- 샘플링된 주파수들 사이 최소 간격을 계산한다.

#### `snr_db_from_tensors(clean_ref, observed)`

- clean 기준 대비 observed의 SNR을 dB 단위로 계산한다.

#### `normalize_feature_columns(H, eps=1e-12)`

- hidden feature matrix `H`의 각 column을 L2 normalize한다.

#### `topk_energy_ratio(singular_values, top_k)`

- 상위 `top_k` 특이값 에너지가 전체 에너지에서 차지하는 비율을 계산한다.

#### `calculate_subspace_alignment_metrics(H, freqs, dt, lag)`

핵심 분석 함수다.

1. hidden feature 행렬 `H`의 column space를 구한다.
2. 실제 신호를 구성한 주파수들에 대해 `sin`, `cos` basis를 만든다.
3. 테스트 특징 행렬 `H_test`의 좌측 특이벡터를 구한다.
4. 상위 `2k`개 좌측 특이벡터 `U_H^(2k)` 기준으로 alignment를 계산한다.

반환 지표:

- `align_coverage`
- `align_purity`
- `alignment_score_2k`
- `align_mean_cosine`
- `mean_principal_angle_deg`
- `principal_angles_deg`

### `experiment_runner.py`

전체 실험 파이프라인을 실행한다.

#### `train_one_seed(...)`

- `MODEL_ID`에 맞는 모델 1개를 한 seed로 학습한다.
- MSE, MAE, accuracy, rank, subspace alignment, SNR 관련 지표를 계산한다.

#### `aggregate_seed_results(seed_results, cfg)`

- 같은 frequency set에서 여러 seed 결과를 평균과 표준편차로 묶는다.

#### `make_summary_dataframe(cfg, set_rows, overall_summary)`

- set별 결과와 overall 결과를 하나의 `pandas.DataFrame`으로 정리한다.
- 각 scalar metric은 `mean`, `std`, `ci95_low`, `ci95_high` 컬럼을 가진다.

#### `plot_results(results, cfg)`

- 기본 요약 그래프를 출력한다.

#### `print_overall_summary(overall_summary, cfg)`

- 최종 평균 지표를 문자열로 출력한다.

#### `print_overall_ci95_summary(overall_summary, cfg)`

- 각 지표의 95% 신뢰구간을 별도로 출력한다.

#### `run_experiment(cfg=None)`

메인 엔트리포인트다.

실행 순서:

1. `ExperimentConfig` 준비
2. 설정 검증
3. 전체 seed 초기화
4. `MODEL_ID`에 맞는 모델 선택
5. bottleneck 차원 및 이론 rank 계산
6. `NUM_EXPERIMENTS`만큼 다른 frequency/amplitude/phase set 생성
7. 필요 시 noise 추가
8. lag-window dataset 구성
9. `TEST_RATIO` 기준 chronological train/test split 수행
10. `SEEDS_PER_FREQ`번 학습 반복
11. `H_test` 기준 핵심 지표 계산
12. set 단위 집계
13. 전체 요약 DataFrame과 dict 결과 생성
14. 옵션에 따라 plot 및 출력

주요 반환값:

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

### 1. 기본 실행

```python
from src import ExperimentConfig, run_experiment

cfg = ExperimentConfig()
results = run_experiment(cfg)
```

### 2. 모델 변형 비교

```python
cfg = ExperimentConfig(
    MODEL_ID="AN002_NO_BN_TANH",
    NUM_FREQS=5,
    RANDOM_AMPLITUDE=False,
    RANDOM_PHASE=True,
    NORMALIZE_H_COLUMNS=True,
    LAG=80,
    SEQ_LEN=5000,
    DT=0.01,
)
results = run_experiment(cfg)
```

### 3. 노이즈 제거/복원 실험

```python
cfg = ExperimentConfig(
    MODEL_ID="AN004_DEEP_TANH",
    USE_NOISE=True,
    NOISE_TYPE="ar1",
    SNR_DB=5.0,
    TRAIN_TARGET="clean",
)
results = run_experiment(cfg)
```

## 이후 실험을 위한 원칙

- 새 파라미터는 먼저 `ExperimentConfig`에 올릴 것
- 재사용 가능한 로직은 노트북에 직접 쓰지 말고 `src`로 이동할 것
- 실험 결과 표준 구조는 `run_experiment()` 반환 형태를 유지할 것
- 새 모델 변형은 모델 클래스와 `MODEL_ID`를 함께 추가할 것
