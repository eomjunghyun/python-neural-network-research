# Experiments Guide

`experiments/`는 이 레포에서 사용하는 실험용 노트북들을 모아둔 디렉터리다.  
크게 두 갈래로 나뉜다.

- `hyperparameter_k_sweep/`
  - 빠르게 `k=NUM_FREQS`만 바꿔보는 탐색용 노트북
- `paper_protocol/`
  - 논문용 순서에 맞춰 정리한 본 실험 노트북

이 문서는 각 노트북이 무엇을 검증하려는 실험인지, 어떤 질문에 답하려고 하는지, 어떻게 해석해야 하는지를 정리한다.

## 공통 배경

현재 실험 파이프라인의 핵심은 다음과 같다.

- 기본 주 실험 모드는 `discrete` time mode
- 이론 차원은 기본적으로 `2k`
  - `k = NUM_FREQS`
  - continuous mode면 `2 * len(freqs)`
  - discrete mode면 `2 * len(thetas)`
- numerical rank는 진단용이다
  - `fourier_numerical_dim`
  - `fourier_min_singular_value`
  - `fourier_condition_number`
- alignment는 새 SVD 기반 지표를 쓴다
  - `align_coverage_full`
  - `align_purity_full`
  - `align_coverage_top`
  - `align_mean_angle_deg_full`
  - `recon_r2_qf_from_h`

즉, 이 실험들은 단순히 예측 MSE만 보는 것이 아니라,

- 이론 부분공간을 얼마나 잘 회복하는가
- `2k` 차원 구조가 실제 hidden representation에 어떻게 드러나는가
- gap, lag, bottleneck, architecture, noise가 그 구조 회복을 어떻게 망가뜨리거나 유지하는가

를 함께 보는 데 목적이 있다.

## 디렉터리 구조

```text
experiments/
├─ README.md
├─ hyperparameter_k_sweep/
│  ├─ 20260406_k_sweep_k01_initial.ipynb
│  └─ 20260406_k_sweep_discrete_k01_k16.ipynb
└─ paper_protocol/
   ├─ 20260408_experiment01_metric_sanity.ipynb
   ├─ 20260408_experiment02_linear_recovery.ipynb
   ├─ 20260408_experiment03_negative_controls.ipynb
   ├─ 20260408_experiment04_easy_condition_recovery.ipynb
   ├─ 20260408_experiment05_lag_boundary.ipynb
   ├─ 20260408_experiment06_k_scaling_normalized.ipynb
   ├─ 20260408_experiment07_k_scaling_fixed_L.ipynb
   ├─ 20260408_experiment08_min_gap_sweep.ipynb
   ├─ 20260408_experiment09_bottleneck_dim_sweep.ipynb
   ├─ 20260408_experiment10_architecture_comparison.ipynb
   ├─ 20260408_experiment11_seq_len_sweep.ipynb
   └─ 20260408_experiment12_robustness_phase_amplitude_noise.ipynb
```

## `hyperparameter_k_sweep`

이 폴더는 빠른 `k` sweep용이다.  
다른 하이퍼파라미터는 고정하고 `NUM_FREQS`만 바꾸는 실험을 돌릴 때 쓴다.

### [20260406_k_sweep_k01_initial.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/hyperparameter_k_sweep/20260406_k_sweep_k01_initial.ipynb)

연속시간(`time_mode="continuous"`) 기준 `k=1..16` sweep 노트북이다.

핵심 특징:

- 셀 하나당 하나의 `k`
- 나머지 하이퍼파라미터는 `FIXED_CONFIG`에 고정
- 각 실행 뒤 `summary_df` 핵심 컬럼과 basis diagnostic 표를 함께 보여줌
- sampled basis의
  - `fourier_theoretical_dim`
  - `fourier_numerical_dim`
  - `fourier_min_singular_value`
  - `fourier_condition_number`
  를 같이 확인할 수 있음

이 노트북은 연속시간 샘플링 조건에서 `k` 증가가 rank/정렬/예측 성능에 어떤 영향을 주는지 빠르게 보는 용도다.

### [20260406_k_sweep_discrete_k01_k16.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/hyperparameter_k_sweep/20260406_k_sweep_discrete_k01_k16.ipynb)

이산시간(`time_mode="discrete"`) 기준 `k=1..16` sweep 노트북이다.

핵심 특징:

- 셀 하나당 하나의 `k`
- 나머지 하이퍼파라미터는 고정
- 새 discrete-time 생성 방식에 맞춘 빠른 `k` sweep용
- theory dim과 numerical rank를 동시에 확인할 수 있음

이 노트북은 현재 주 실험 방향과 더 직접적으로 맞물린다.  
즉 “discrete basis 기준으로 `k`가 늘어날 때 실험이 얼마나 어려워지는가”를 빠르게 점검하는 탐색용 notebook이다.

## `paper_protocol`

이 폴더는 논문용 순서에 맞춘 실험 노트북 모음이다.  
번호 순서가 대체로 권장 실행 순서다.

### 공통 철학

대부분의 notebook은 아래 방향을 따른다.

- `time_mode="discrete"`
- raw series를 먼저 train/val/test로 자름
- 각 split 내부에서 lag-window dataset 생성
- absolute target index 기반으로 basis를 구성
- 이론 차원 `2k`는 고정 비교 기준
- numerical rank는 오직 diagnostic

### [20260408_experiment01_metric_sanity.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment01_metric_sanity.ipynb)

지표 sanity check 실험이다.

무엇을 검증하나:

- 새 full/top alignment 지표가 진짜 구조와 가짜 구조를 구분하는가

어떻게 하나:

- 정답 basis `F`를 만들고
- 인공 `H`를 네 종류로 구성한다
  - `same_subspace`
  - `same_plus_orth_noise`
  - `random`
  - `random_perp`

핵심 지표:

- `align_coverage_full`
- `align_purity_full`
- `align_coverage_top`
- `align_mean_angle_deg_top`
- `recon_r2_qf_from_h`

기대 해석:

- `same_subspace`: 거의 완전 회복
- `same_plus_orth_noise`: coverage는 높고 purity는 낮아짐
- `random`: 전반적으로 낮음
- `random_perp`: 가장 낮음

이 실험이 실패하면 이후 실험으로 넘어가면 안 된다.

### [20260408_experiment02_linear_recovery.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment02_linear_recovery.ipynb)

선형 기준선 회복 실험이다.

무엇을 검증하나:

- 다중 사인파 예측 문제가 쉬운 조건에서는 선형 구조만으로 잘 풀리는가

설정:

- 모델: `AN003_LINEAR`
- `k ∈ {2, 4, 8}`
- `L ∈ {2k, 4k, 8k}`

핵심 지표:

- `test_mse`
- `test_r2`
- `test_align_coverage_full`
- `test_recon_r2_qf_from_h`
- `test_align_mean_angle_deg_full`

의미:

이 단계에서 안 되면 비선형 모델 이전에 데이터 파이프라인이나 basis wiring 쪽부터 의심해야 한다.

### [20260408_experiment03_negative_controls.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment03_negative_controls.ipynb)

음성 대조군 실험이다.

무엇을 검증하나:

- 지표가 우연히 높아지는 것이 아닌가
- 정답 basis가 아닐 때도 높은 점수가 나오지 않는가

포함된 대조군:

- `wrong_basis`
- `random`
- `random_perp`

추가 확장 대상으로 적어둔 것:

- time-shuffled target
- random target
- mismatched basis evaluation

핵심 지표:

- `test_align_coverage_full`
- `test_recon_r2_qf_from_h`
- `test_align_mean_angle_deg_full`

### [20260408_experiment04_easy_condition_recovery.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment04_easy_condition_recovery.ipynb)

쉬운 조건에서의 본 회복 실험이다.

무엇을 검증하나:

- linear와 tanh 모델이 쉬운 조건에서 이론 부분공간을 잘 회복하는가

비교 모델:

- `AN003_LINEAR`
- `AN002_NO_BN_TANH`

핵심 설정:

- `NUM_FREQS=4`
- `LAG=32`
- `BOTTLENECK_MULTIPLIER=4`

핵심 지표:

- `train_mse`, `test_mse`
- `test_align_coverage_full`
- `test_recon_r2_qf_from_h`
- `test_align_mean_angle_deg_full`

### [20260408_experiment05_lag_boundary.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment05_lag_boundary.ipynb)

lag 경계 실험이다.

무엇을 검증하나:

- `L / (2k)`가 1 근처를 넘길 때 구조 회복이 급격히 좋아지는가

설정:

- `k ∈ {2, 4, 8, 16}`
- `L ∈ {k, 2k-1, 2k, 3k, 4k, 6k}`
- `BOTTLENECK_MULTIPLIER ∈ {2, 4}`
- 모델: linear, tanh

핵심 지표:

- `test_mse`
- `test_align_coverage_full`
- `test_recon_r2_qf_from_h`
- `test_align_mean_angle_deg_full`

이 실험은 본문 핵심 그림 후보가 되기 쉽다.

### [20260408_experiment06_k_scaling_normalized.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment06_k_scaling_normalized.ipynb)

`k` 증가 실험의 정규화 버전이다.

무엇을 검증하나:

- `k`가 커져도 관측 길이와 병목 차원을 같이 키우면 성능 저하가 완만해지는가

설정:

- `k ∈ {1, 2, 4, 8, 16}`
- `L = 4k`
- `b ∈ {2k, 4k}`
- 모델: linear, tanh

이 노트북은 “차원 증가 자체”의 영향보다, 조건을 맞춰줬을 때 얼마나 잘 유지되는지를 본다.

### [20260408_experiment07_k_scaling_fixed_L.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment07_k_scaling_fixed_L.ipynb)

`k` 증가 실험의 고정 `L` 버전이다.

무엇을 검증하나:

- `k`가 커질수록 왜 어려워지는가
- 관측 길이 고정과 용량 부족이 어떤 악화를 만드는가

설정:

- `L=32` 고정
- `BOTTLENECK_DIM_OVERRIDE=16`으로 고정 병목 차원 사용
- `k ∈ {1, 2, 4, 8, 16}`
- 모델: linear, tanh

Experiment 06과 짝으로 해석해야 한다.

### [20260408_experiment08_min_gap_sweep.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment08_min_gap_sweep.ipynb)

최소 주파수 간격 실험이다.

무엇을 검증하나:

- 주파수 분리도(`min_delta_theta`)가 작아질수록 basis conditioning과 회복 성능이 어떻게 무너지는가

설정:

- `NUM_FREQS ∈ {4, 8}`
- `MIN_DELTA_THETA ∈ {0.20π, 0.12π, 0.08π, 0.04π, 0.02π}`
- 모델: linear, tanh

핵심 지표:

- `min_delta_theta`
- `fourier_min_singular_value`
- `fourier_condition_number`
- `test_align_coverage_full`
- `test_recon_r2_qf_from_h`

### [20260408_experiment09_bottleneck_dim_sweep.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment09_bottleneck_dim_sweep.ipynb)

병목 차원 실험이다.

무엇을 검증하나:

- 정말 `b ≈ 2k`가 구조 회복의 경계인가

설정:

- `NUM_FREQS ∈ {4, 8}`
- `BOTTLENECK_MULTIPLIER ∈ {1, 2, 4, 8}`
- 모델: linear, tanh

핵심 지표:

- `test_align_coverage_full`
- `test_align_purity_full`
- `test_recon_r2_qf_from_h`
- `test_energy_top_theory_dim`

기대 해석:

- `b < 2k`: coverage 저하
- `b = 2k`: 회복 경계
- `b > 2k`: coverage 유지 가능, purity 저하 가능

### [20260408_experiment10_architecture_comparison.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment10_architecture_comparison.ipynb)

아키텍처 비교 실험이다.

무엇을 검증하나:

- 어떤 모델 구조가 이론 부분공간을 가장 안정적으로 회복하는가

비교 모델:

- `AN001_BN_RELU`
- `AN002_NO_BN_TANH`
- `AN003_LINEAR`
- `AN004_DEEP_TANH`

조건:

- `easy`
- `hard`

핵심 지표:

- `test_mse`
- `test_align_coverage_full`
- `test_recon_r2_qf_from_h`
- `test_align_mean_angle_deg_full`

### [20260408_experiment11_seq_len_sweep.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment11_seq_len_sweep.ipynb)

데이터 길이 실험이다.

무엇을 검증하나:

- 데이터가 많아질수록 추정치가 안정화되고 신뢰구간이 줄어드는가

설정:

- `SEQ_LEN ∈ {512, 1024, 2048, 4096, 8192}`

핵심 지표:

- `test_mse`
- `test_align_coverage_full`
- `test_recon_r2_qf_from_h`
- 각 지표의 95% CI

이 실험은 “이 결과가 소표본 우연은 아니다”를 보강하는 역할을 한다.

### [20260408_experiment12_robustness_phase_amplitude_noise.ipynb](C:/Users/WWindows10/Documents/github_project/python-neural-network-research/experiments/paper_protocol/20260408_experiment12_robustness_phase_amplitude_noise.ipynb)

강건성 실험이다.

무엇을 검증하나:

- 위상 랜덤화, 진폭 랜덤화, 잡음이 구조 회복을 얼마나 망가뜨리는가

조건 순서:

- `phase_only`
- `phase_amp`
- `noise_30db`
- `noise_20db`
- `noise_10db`
- `noise_5db`

핵심 지표:

- `test_mse`
- `test_align_coverage_full`
- `test_recon_r2_qf_from_h`
- `test_input_snr_db`
- `test_output_snr_db`
- `test_snr_gain_db`

이 실험은 “너무 이상적인 clean setting에서만 되는 것 아니냐”는 질문에 답하는 용도다.

## 권장 실행 순서

논리적으로는 아래 순서를 권장한다.

1. `experiment01_metric_sanity`
2. `experiment02_linear_recovery`
3. `experiment03_negative_controls`
4. `experiment04_easy_condition_recovery`
5. `experiment05_lag_boundary`
6. `experiment06` / `experiment07`
7. `experiment08`
8. `experiment09`
9. `experiment10`
10. `experiment11`
11. `experiment12`

## 시간이 없을 때 우선순위

최소 핵심 세트만 돌린다면 아래가 우선이다.

- `experiment01_metric_sanity`
- `experiment02_linear_recovery`
- `experiment04_easy_condition_recovery`
- `experiment05_lag_boundary`
- `experiment08_min_gap_sweep`
- `experiment09_bottleneck_dim_sweep`

## 주의사항

- 이 디렉터리의 notebook은 대부분 템플릿이다
- 기본적으로 실행 결과가 저장되어 있지 않을 수 있다
- `paper_protocol`은 설계된 실험 순서에 맞춰 만든 것이고, 반드시 전부 한 번에 돌려야 한다는 뜻은 아니다
- numerical rank는 어디까지나 diagnostic이다
- 주 비교 기준은 항상 이론 차원 `2k`다
