# src 폴더 가이드

이 문서는 `src` 내부 코드의 역할과 함수별 내부 동작을 정리한 문서입니다.

## 1) 폴더 목적

- `src`는 실험 스크립트(`experiments/...`)에서 재사용하는 핵심 로직을 모아두는 곳입니다.
- 원칙:
  - `src`: 재사용 가능한 함수/모델
  - `experiments`: 실험 루프, 하이퍼파라미터, 시각화, 로그 출력

현재 파일:

- `common_utils.py`: 공통 유틸 함수(재현성, 데이터 생성, 지표 계산)
- `models.py`: 모델 클래스(`AnalyticNet`)

---

## 2) common_utils.py 상세

### `set_seed(seed: int) -> None`

역할:

- Python/NumPy/PyTorch의 랜덤 시드를 동시에 고정해 실험 재현성을 높입니다.

내부 동작 순서:

1. `random.seed(seed)`로 Python 기본 RNG 고정
2. `np.random.seed(seed)`로 NumPy RNG 고정
3. `torch.manual_seed(seed)`로 PyTorch RNG 고정
4. `PYTHONHASHSEED` 지정으로 해시 기반 순서 변동 최소화
5. `torch.backends.cudnn.deterministic = True` 설정
6. `torch.backends.cudnn.benchmark = False` 설정

주의:

- GPU/라이브러리 버전에 따라 완전 동일 재현이 항상 보장되지는 않을 수 있습니다.

---

### `generate_sin_data(...) -> Tuple[np.ndarray, Tuple[int, ...]]`

시그니처:

```python
generate_sin_data(
    seq_len: int,
    dt: float,
    num_freqs: int,
    freq_min: int,
    freq_max: int,
    rng: random.Random,
    num_freqs_min: int = 1,
    num_freqs_max: int = 30,
) -> Tuple[np.ndarray, Tuple[int, ...]]
```

역할:

- 정수 주파수 집합에서 `num_freqs=k`개를 샘플링해, `y(t) = Σ sin(f_i t)` 형태의 합성 시계열을 생성합니다.

입력:

- `seq_len`: 시계열 길이
- `dt`: 샘플 간 시간 간격
- `num_freqs`: 사용할 주파수 개수
- `freq_min`, `freq_max`: 샘플링 가능한 주파수 구간(양끝 포함)
- `rng`: 실험별 독립 난수 객체
- `num_freqs_min`, `num_freqs_max`: `num_freqs` 유효 범위

출력:

- `y`: `float32` 1차원 배열, shape `(seq_len,)`
- `freqs`: 실제 뽑힌 주파수 튜플

내부 동작 순서:

1. `num_freqs`가 허용 범위에 있는지 검증
2. 샘플링 가능한 전체 개수(`population_size`) 계산
3. `num_freqs > population_size`면 오류 발생
4. `t = np.arange(seq_len) * dt` 시간축 생성
5. `rng.sample(...)`로 중복 없이 주파수 샘플링
6. 각 주파수에 대해 `sin(f*t)`를 합산
7. 결과를 `float32`로 변환해 반환

왜 `rng`를 외부에서 받는가:

- 세트별 데이터 시드를 독립적으로 통제하기 쉽고, 전역 RNG 오염을 줄일 수 있습니다.

---

### `make_dataset(data: np.ndarray, lag: int) -> Tuple[torch.Tensor, torch.Tensor]`

역할:

- 1차원 시계열을 슬라이딩 윈도우 방식의 지도학습 데이터로 변환합니다.

정의:

- `x[i] = data[i : i+lag]`
- `y[i] = data[i+lag]`

출력 shape:

- `x_tensor`: `(N, lag)` where `N = len(data) - lag`
- `y_tensor`: `(N, 1)` (`unsqueeze(1)` 적용)

내부 동작 순서:

1. 인덱스를 이동하며 입력 구간/다음 시점 타깃 생성
2. 리스트를 NumPy 배열로 묶음
3. `torch.float32` 텐서로 변환
4. 타깃은 `(N,)`에서 `(N,1)`로 확장

---

### `min_delta_f(freqs: Sequence[int]) -> float`

역할:

- 주파수 집합의 최소 간격 `min Δf`를 계산합니다.

내부 동작:

1. 길이가 2 미만이면 `NaN` 반환(간격 정의 불가)
2. 주파수 정렬
3. 인접 차이(`np.diff`) 계산
4. 최소값 반환

주요 사용처:

- 실험 세트에서 주파수 간격이 좁을수록 alignment가 어려운지 분석할 때 사용

---

### `mean_std(values: Sequence[float]) -> Tuple[float, float]`

역할:

- 평균과 표본 표준편차(`ddof=1`)를 계산합니다.

내부 동작:

1. `float64` 배열로 변환
2. 길이가 1 이하이면 표준편차를 `0.0`으로 반환
3. 일반 경우 평균/표본 표준편차 반환

주의:

- 실험 반복 횟수가 작을 때 표준편차 해석에 주의가 필요합니다.

---

### `regression_accuracy(y_true, y_pred, tol) -> float`

역할:

- 회귀 문제에서 오차 허용치 기반 정확도를 계산합니다.

정의:

- `Acc = mean( |y_true - y_pred| <= tol )`

내부 동작:

1. 절대 오차 계산
2. 임계치 이하 여부를 bool 마스크로 생성
3. float 변환 후 평균

---

### `calculate_rank_metrics(S, threshold) -> Dict[str, float]`

역할:

- 특이값 벡터 `S`에서 랭크 관련 요약 지표를 계산합니다.

반환 항목:

- `rank_threshold`: `S/S[0] > threshold`인 성분 수
- `rank_entropy`: 확률 분포화한 특이값의 entropy rank

내부 동작 순서:

1. `S`를 `float64`로 변환
2. 빈 벡터면 기본값 반환
3. `S[0]`이 0인 극단 상황은 `1e-12`로 보호
4. 정규화 특이값으로 임계 초과 개수 계산
5. `p_i = S_i / sum(S)` 구성
6. `exp(-sum(p_i log p_i))` 계산

해석:

- `rank_threshold`는 임계치 기준 유효 차원 추정
- `rank_entropy`는 에너지 분포의 퍼짐 정도 반영

---

### `calculate_subspace_alignment_metrics(H, freqs, dt, lag) -> Dict[...]`

역할:

- 특징 부분공간 `H`가 실제 Fourier 부분공간과 얼마나 정렬되는지 정량화합니다.

핵심 아이디어:

- 데이터 생성에 쓰인 주파수 `freqs`로 Fourier 기저 `F`를 만든 뒤,
- `H`와 `F`의 직교기저 간 projection을 통해 부분공간 유사도를 계산합니다.

입력:

- `H`: shape `(samples, bottleneck_dim)` 특징 행렬
- `freqs`: 데이터 생성 주파수 목록
- `dt`: 시간 간격
- `lag`: 타깃 정렬 보정

시간축 정렬:

- `H[i]`는 원 신호의 시점 `(i + lag) * dt`에 대응하므로
- Fourier 기저도 동일한 시점축을 사용해야 합니다.

내부 동작 순서:

1. `t = (arange(samples) + lag) * dt` 생성
2. `F = [sin(f t), cos(f t)]`를 주파수별로 쌓아 `(samples, 2k)` 구성
3. `QR` 분해로 `Q_f`, `Q_h` (각 부분공간의 직교기저) 획득
4. `projection = Q_f.T @ Q_h` 계산
5. `projection`의 제곱합으로 겹침 정도 계산
6. `align_coverage`, `align_purity` 산출
7. 차원 비율을 이용해 `purity_norm` 정규화
8. `projection`의 특이값으로 principal angle 계산
9. 평균 cosine/각도와 각도 배열 반환

반환 항목:

- `align_coverage`: Fourier 공간이 H에 얼마나 담기는지
- `align_purity`: H가 Fourier 공간에 얼마나 집중되는지
- `purity_norm`: 차원 차이 보정 purity
- `align_mean_cosine`: 주각 cosine 평균
- `align_mean_angle_deg`: 주각 평균(도)
- `principal_angles_deg`: 주각 벡터(도)

수치 안정화 포인트:

- `+1e-12`를 사용해 0 나눗셈 방지
- 특이값은 `[-1, 1]`로 clip 후 `arccos` 적용

---

## 3) models.py 상세

### `AnalyticNet(nn.Module)`

구조:

1. `Linear(input_dim -> hidden_dim)`
2. `Tanh()`
3. `Linear(hidden_dim -> bottleneck_dim)`
4. `Linear(bottleneck_dim -> 1, bias=False)` (readout)

`forward(x)` 반환:

- `y_hat`: 최종 예측값
- `h`: bottleneck 특징

왜 `(y_hat, h)`를 같이 반환하는가:

- 예측 성능과 함께 특징공간 분석(SVD/부분공간 정렬)을 동시에 하기 위함입니다.

---

## 4) 유지보수 규칙

- 실험 간 재사용 가능한 로직은 `src`로 이동
- 특정 실험에만 필요한 시각화/루프는 `experiments`에 유지
- 함수 시그니처를 바꾸면, 해당 함수를 import하는 실험 파일도 함께 수정

