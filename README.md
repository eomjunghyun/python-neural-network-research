# python-neural-network-research

이 레포지토리는 `0326_실험5_정규화.ipynb`를 기준으로 정리한 연구 베이스 코드다.  
핵심 관심사는 여러 개의 사인파 성분으로 이루어진 시계열을 신경망이 어떻게 표현하는지, 그리고 그 내부 표현이 실제 생성 주파수 subspace와 얼마나 정렬되는지를 분석하는 것이다.

주요 연구 질문은 다음과 같다.

- 여러 개의 정수 주파수 성분을 합성한 시계열을 next-step prediction으로 학습할 때, 병목 표현의 차원 구조가 어떻게 형성되는가
- 이론적 rank `2 * NUM_FREQS`와 실제 hidden representation의 effective rank가 얼마나 가깝게 나타나는가
- hidden representation이 실제 생성 주파수의 `sin/cos` basis subspace를 얼마나 잘 포착하는가
- amplitude, phase, lag, sequence length, sampling interval, noise, 모델 구조가 위 현상에 어떤 영향을 주는가

## 문제 설정

길이 `SEQ_LEN`의 1차원 시계열을 생성한다. 각 시계열은 `NUM_FREQS`개의 정수 주파수 성분 합으로 구성된다.

```text
x_k(t) = a_k * sin(w_k * t + phi_k)
x(t) = sum_k x_k(t)
```

여기서

- `w_k`는 정수 각주파수
- `a_k`는 amplitude
- `phi_k`는 phase

이다. 실험에 따라 amplitude와 phase는 고정되거나 무작위 샘플링된다.

학습 과제는 길이 `LAG`짜리 슬라이딩 윈도우로 다음 시점을 예측하는 것이다.

- 입력: `x[t : t + LAG]`
- 타깃: `x[t + LAG]`

노이즈 실험에서는 noisy signal을 입력으로 사용하고, 타깃은 아래 둘 중 하나를 선택한다.

- `TRAIN_TARGET="noisy"`: noisy 신호 자체를 예측
- `TRAIN_TARGET="clean"`: clean 신호를 복원하도록 학습

## 모델 구성

현재 베이스 코드에는 식별자 기반으로 4개의 모델이 있다. 실험 시 `MODEL_ID`로 선택한다.

### `AN001_BN_RELU`

기존 베이스라인을 이름만 정리한 모델이다.

```text
Input(LAG)
 -> Linear(LAG, HIDDEN_DIM)
 -> BatchNorm1d
 -> ReLU
 -> Linear(HIDDEN_DIM, BOTTLENECK_MULTIPLIER * NUM_FREQS)
 -> BatchNorm1d
 -> Linear(bottleneck_dim, 1, bias=False)
```

### `AN002_NO_BN_TANH`

배치 정규화를 제거하고 활성화 함수를 `Tanh`로 바꾼 모델이다.

```text
Input(LAG)
 -> Linear(LAG, HIDDEN_DIM)
 -> Tanh
 -> Linear(HIDDEN_DIM, BOTTLENECK_MULTIPLIER * NUM_FREQS)
 -> Tanh
 -> Linear(bottleneck_dim, 1, bias=False)
```

### `AN003_LINEAR`

활성화 함수를 제거한 선형 모델이다.

```text
Input(LAG)
 -> Linear(LAG, HIDDEN_DIM)
 -> Linear(HIDDEN_DIM, BOTTLENECK_MULTIPLIER * NUM_FREQS)
 -> Linear(bottleneck_dim, 1, bias=False)
```

### `AN004_DEEP_TANH`

`Tanh` 기반 모델에 hidden layer를 하나 더 추가한 모델이다.

```text
Input(LAG)
 -> Linear(LAG, HIDDEN_DIM)
 -> Tanh
 -> Linear(HIDDEN_DIM, HIDDEN_DIM)
 -> Tanh
 -> Linear(HIDDEN_DIM, BOTTLENECK_MULTIPLIER * NUM_FREQS)
 -> Tanh
 -> Linear(bottleneck_dim, 1, bias=False)
```

공통 해석 포인트:

- `bottleneck_dim = BOTTLENECK_MULTIPLIER * NUM_FREQS`
- `theoretical_rank = 2 * NUM_FREQS`
- hidden representation의 column space와 실제 생성 주파수의 `sin/cos` basis subspace를 직접 비교한다

## 실험 파이프라인

한 번의 `run_experiment(cfg)`는 아래 순서로 진행된다.

1. `ExperimentConfig` 준비
2. 설정 검증
3. 전역 seed 고정
4. `MODEL_ID`에 맞는 모델 선택
5. bottleneck 차원과 이론 rank 계산
6. `NUM_EXPERIMENTS`개의 서로 다른 signal set 생성
7. 필요 시 noise 추가
8. lag-window dataset 생성
9. 시계열 순서를 유지한 채 `TEST_RATIO` 비율로 train/test 분할
10. 각 signal set마다 `SEEDS_PER_FREQ`개의 다른 학습 seed로 모델 반복 학습
11. `H_test` 기준 핵심 지표 계산
12. seed 평균 및 표준편차 집계
13. 같은 집계 단위에서 정규 근사 기반 95% 신뢰구간 계산
14. set별 결과와 overall 결과를 DataFrame으로 정리
15. 필요 시 plot 출력

## 주요 분석 지표

### 예측 성능

- `mse`: train mean squared error
- `mae`: train mean absolute error
- `acc`: `|error| <= ACC_TOLERANCE` 비율
- `test_mse`: 테스트 평균제곱오차
- `test_r2`: 테스트 결정계수

집계 출력은 각 지표에 대해 아래 세 가지를 함께 제공한다.

- 평균(`*_mean`)
- 표준편차(`*_std`)
- 95% 신뢰구간(`*_ci95_low`, `*_ci95_high`)

### rank 관련 지표

- `rank_threshold`: 정규화 singular value가 `RANK_THRESHOLD`보다 큰 개수
- `rank_entropy`: singular value entropy 기반 effective rank
- `rank_gap`: `|rank_threshold - theoretical_rank|`
- `rel_rank_gap`: `rank_gap / theoretical_rank`
- `spectral_gap_2k`: 이론 rank 위치의 singular value gap

```text
spectral_gap_2k = sigma_(2k) / sigma_(2k+1)
```

### subspace alignment 지표

실제 생성 주파수 `freqs`에 대해 `sin(freq * t)`와 `cos(freq * t)`를 basis로 구성하고, 테스트 특징 행렬 `H_test`의 SVD를 이용해 아래 지표를 계산한다.

- `alignment_score_2k`: 상위 `2k`차 정렬 점수

```text
A_2k = (1 / 2k) * || Q_F^T U_H^(2k) ||_F^2
```

- `mean_principal_angle_deg`: `Q_F`와 `U_H^(2k)` 사이 평균 주각
- `energy_ratio_2k`: 상위 `2k`개 특이값 에너지 비율

```text
E_2k = sum_{i=1}^{2k} sigma_i^2 / sum_i sigma_i^2
```

- `align_coverage`
- `align_purity`
- `alignment_score_2k`
- `align_mean_cosine`
- `mean_principal_angle_deg`
- `principal_angles_deg`

### noise 관련 지표

`USE_NOISE=True`일 때 아래 지표가 추가된다.

- `input_snr_db`
- `output_snr_db`
- `snr_gain_db`

## 하이퍼파라미터 전체 설명

아래는 현재 `ExperimentConfig`에 정의된 전체 파라미터다.

### 1. Seed 관련

| 이름 | 기본값 | 설명 |
| --- | ---: | --- |
| `GLOBAL_SEED` | `42` | 실험 전체 전역 재현성 seed |
| `DATA_SEED_BASE` | `1000` | signal set 생성 seed 시작값 |
| `NOISE_SEED_BASE` | `50000` | noise 생성 seed 시작값 |
| `TRAIN_SEED_BASE` | `100` | 학습 반복 seed 시작값 |

### 2. 데이터 생성 관련

| 이름 | 기본값 | 설명 |
| --- | ---: | --- |
| `SEQ_LEN` | `1000` | 시계열 길이 |
| `DT` | `0.05` | 샘플링 간격 |
| `NUM_FREQS` | `4` | 한 signal을 구성하는 주파수 개수 |
| `NUM_FREQS_MIN` | `1` | 허용 최소 주파수 개수 |
| `NUM_FREQS_MAX` | `30` | 허용 최대 주파수 개수 |
| `FREQ_MIN` | `1` | 샘플링 가능한 최소 정수 주파수 |
| `FREQ_MAX` | `60` | 샘플링 가능한 최대 정수 주파수 |
| `NYQUIST_MARGIN` | `0.98` | Nyquist 한계보다 얼마나 보수적으로 제약할지 나타내는 margin |

중요 제약:

- `NUM_FREQS <= FREQ_MAX - FREQ_MIN + 1`
- `FREQ_MAX < NYQUIST_MARGIN * (pi / DT)`

### 3. Amplitude / Phase 관련

| 이름 | 기본값 | 설명 |
| --- | ---: | --- |
| `RANDOM_AMPLITUDE` | `True` | amplitude를 랜덤 샘플링할지 여부 |
| `AMP_MIN` | `0.5` | amplitude 랜덤 샘플링 최소값 |
| `AMP_MAX` | `2.0` | amplitude 랜덤 샘플링 최대값 |
| `RANDOM_PHASE` | `False` | phase를 랜덤 샘플링할지 여부 |
| `PHASE_MIN` | `0.0` | phase 랜덤 샘플링 최소값 |
| `PHASE_MAX` | `2pi` | phase 랜덤 샘플링 최대값 |

동작:

- `RANDOM_AMPLITUDE=False`면 모든 amplitude는 `1`
- `RANDOM_PHASE=False`면 모든 phase는 `0`

### 4. Noise 관련

| 이름 | 기본값 | 설명 |
| --- | ---: | --- |
| `USE_NOISE` | `False` | noisy 입력 실험 여부 |
| `NOISE_TYPE` | `"white"` | noise 종류 (`white`, `ar1`, `impulsive`) |
| `SNR_DB` | `10.0` | 목표 SNR(dB) |
| `AR1_RHO` | `0.8` | AR(1) noise 자기회귀 계수 |
| `IMPULSE_PROB` | `0.01` | impulsive noise 발생 확률 |
| `IMPULSE_SCALE` | `8.0` | impulsive spike 크기 배수 |

### 5. 학습 타깃 관련

| 이름 | 기본값 | 설명 |
| --- | ---: | --- |
| `TRAIN_TARGET` | `"noisy"` | noisy를 맞출지 clean을 복원할지 결정 |

가능 값:

- `"noisy"`
- `"clean"`

### 6. 모델 / 최적화 관련

| 이름 | 기본값 | 설명 |
| --- | ---: | --- |
| `MODEL_ID` | `"AN001_BN_RELU"` | 사용할 모델 식별자 |
| `LAG` | `32` | 입력 윈도우 길이 |
| `HIDDEN_DIM` | `64` | 첫 hidden layer 차원 |
| `BOTTLENECK_MULTIPLIER` | `4` | bottleneck 차원 계산 계수 |
| `LR` | `0.01` | Adam 학습률 |
| `EPOCHS` | `1000` | seed별 학습 epoch 수 |

가능한 `MODEL_ID`:

- `AN001_BN_RELU`
- `AN002_NO_BN_TANH`
- `AN003_LINEAR`
- `AN004_DEEP_TANH`

유도되는 값:

- `theoretical_rank = 2 * NUM_FREQS`
- `bottleneck_dim = BOTTLENECK_MULTIPLIER * NUM_FREQS`

### 7. 반복 실험 관련

| 이름 | 기본값 | 설명 |
| --- | ---: | --- |
| `NUM_EXPERIMENTS` | `5` | 서로 다른 signal set 개수 |
| `SEEDS_PER_FREQ` | `10` | 각 signal set당 학습 반복 수 |
| `TEST_RATIO` | `0.2` | 시계열 순서를 유지한 test split 비율 |

### 8. 지표 계산 관련

| 이름 | 기본값 | 설명 |
| --- | ---: | --- |
| `ACC_TOLERANCE` | `0.1` | accuracy 계산 시 허용 오차 |
| `RANK_THRESHOLD` | `0.05` | threshold rank 기준값 |
| `SCREE_TOPK` | `20` | 상위 singular value 저장 개수 |
| `NORMALIZE_H_COLUMNS` | `False` | hidden matrix column 정규화 여부 |

### 9. 출력 제어 관련

| 이름 | 기본값 | 설명 |
| --- | ---: | --- |
| `VERBOSE` | `True` | 콘솔 상세 로그 출력 여부 |
| `MAKE_PLOTS` | `False` | 요약 그래프 생성 여부 |

## 현재 노트북에서 확인된 실험 케이스

제공된 `0326_실험5_정규화.ipynb` 기준으로 확인한 실행 케이스는 아래와 같다.

### 실험 1

- `NUM_FREQS=4`
- `RANDOM_AMPLITUDE=False`
- `NORMALIZE_H_COLUMNS=True`
- `RANDOM_PHASE=True`

### 실험 2

- `NUM_FREQS=4`
- `RANDOM_AMPLITUDE=True`
- `NORMALIZE_H_COLUMNS=True`
- `RANDOM_PHASE=True`

### 실험 3

- 실험 2와 동일
- `LAG=64`

### 실험 4

- 실험 2와 동일
- `LAG=3`

### 실험 5

- `NUM_FREQS=5`
- `RANDOM_AMPLITUDE=True`
- `NORMALIZE_H_COLUMNS=True`
- `RANDOM_PHASE=True`
- `LAG=3`

### 실험 6

- `NUM_FREQS=5`
- `RANDOM_AMPLITUDE=False`
- `NORMALIZE_H_COLUMNS=True`
- `LAG=80`

### 실험 7

- `NUM_FREQS=5`
- `RANDOM_AMPLITUDE=False`
- `NORMALIZE_H_COLUMNS=True`
- `LAG=80`
- `SEQ_LEN=5000`
- `DT=0.01`

### 실험 8

노트북상 실험 7과 설정이 완전히 동일하다. 중복 실행 또는 재현 확인용 셀로 해석할 수 있다.

### 실험 9

- 실험 7과 유사
- `LAG=32`

## 사용법

### 환경 준비

프로젝트 루트에서 실행:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 기본 실행

```python
from src import ExperimentConfig, run_experiment

cfg = ExperimentConfig()
results = run_experiment(cfg)

print(results["config_df"])
print(results["summary_df"])
```

### 모델 변형 비교 예시

```python
from src import ExperimentConfig, run_experiment

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

### 노이즈 복원 실험 예시

```python
cfg = ExperimentConfig(
    MODEL_ID="AN003_LINEAR",
    USE_NOISE=True,
    NOISE_TYPE="ar1",
    SNR_DB=5.0,
    TRAIN_TARGET="clean",
    RANDOM_AMPLITUDE=True,
    RANDOM_PHASE=True,
)
results = run_experiment(cfg)
```

```python
cfg = ExperimentConfig(
    MODEL_ID="AN004_DEEP_TANH",
    NUM_FREQS=5,
    RANDOM_PHASE=True,
)
results = run_experiment(cfg)
```

## 레포 구조

```text
.
├─ README.md
├─ requirements.txt
└─ src
   ├─ README.md
   ├─ __init__.py
   ├─ common_utils.py
   ├─ config.py
   ├─ data_utils.py
   ├─ experiment_runner.py
   ├─ metrics.py
   └─ models.py
```

## 참고

- 함수별 상세 설명과 사용 예시는 [`src/README.md`](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/src/README.md)에 정리했다.
- 이후 실험에서는 노트북에 공통 로직을 다시 쓰기보다 `src` 모듈을 확장하는 방식으로 유지하는 것을 기본 원칙으로 삼는다.
