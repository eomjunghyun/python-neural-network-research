import os
import random
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

"""실험 공통 유틸리티 모음.

이 파일은 다음 역할을 담당한다.
1) 재현성 고정(set_seed)
2) 합성 시계열 데이터 생성(generate_sin_data)
3) 슬라이딩 윈도우 데이터셋 구성(make_dataset)
4) 지표 계산(정확도/랭크/부분공간 정렬)
"""


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    """실험 재현성을 위해 Python/NumPy/PyTorch 시드를 동시에 고정한다.

    내부 동작:
    - random, numpy, torch RNG를 모두 같은 seed로 설정
    - PYTHONHASHSEED를 고정해 해시 기반 연산의 변동 최소화
    - cuDNN 결정론 모드 설정 (가능한 범위에서 동일 결과 보장)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Data utilities
# -----------------------------
def generate_sin_data(
    seq_len: int,
    dt: float,
    num_freqs: int,
    freq_min: int,
    freq_max: int,
    rng: random.Random,
    num_freqs_min: int = 1,
    num_freqs_max: int = 30,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """정수 주파수 성분의 sin 합으로 1차원 시계열을 생성한다.

    Args:
        seq_len: 시퀀스 길이
        dt: 샘플링 시간 간격
        num_freqs: 선택할 주파수 개수(k)
        freq_min, freq_max: 주파수 샘플링 범위(포함)
        rng: random.Random 인스턴스(실험별 독립 난수원)
        num_freqs_min, num_freqs_max: 허용되는 k 범위

    Returns:
        y: shape (seq_len,), dtype float32
        freqs: 실제로 샘플링된 주파수 튜플
    """
    if num_freqs < num_freqs_min or num_freqs > num_freqs_max:
        raise ValueError(f"num_freqs must be in [{num_freqs_min}, {num_freqs_max}].")

    # 샘플링 가능한 고유 주파수 개수
    population_size = freq_max - freq_min + 1
    if num_freqs > population_size:
        raise ValueError("num_freqs is larger than available frequency population.")

    # t = [0, dt, 2dt, ...] 시간축 (endpoint ambiguity 방지를 위해 arange 사용)
    t = np.arange(seq_len, dtype=np.float64) * dt
    # 중복 없는 정수 주파수 샘플링
    freqs = rng.sample(range(freq_min, freq_max + 1), num_freqs)

    # y(t) = Σ sin(f_i * t)
    y = np.zeros_like(t, dtype=np.float64)
    for f in freqs:
        y += np.sin(f * t)

    return y.astype(np.float32), tuple(freqs)


def make_dataset(data: np.ndarray, lag: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """시계열을 슬라이딩 윈도우 학습 데이터(x, y)로 변환한다.

    - x[i] = data[i : i+lag]
    - y[i] = data[i+lag]
    """
    x, y = [], []
    for i in range(len(data) - lag):
        x.append(data[i : i + lag])
        y.append(data[i + lag])

    x_tensor = torch.tensor(np.array(x), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    return x_tensor, y_tensor


def min_delta_f(freqs: Sequence[int]) -> float:
    """주파수 집합 내 최소 간격(min Δf)을 계산한다.

    주파수가 2개 미만이면 간격을 정의할 수 없어 NaN을 반환한다.
    """
    if len(freqs) < 2:
        return float("nan")
    sf = np.sort(np.asarray(freqs, dtype=np.float64))
    return float(np.min(np.diff(sf)))


# -----------------------------
# Metric utilities
# -----------------------------
def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """표본 평균/표본 표준편차(ddof=1)를 반환한다.

    값이 1개 이하이면 표준편차는 0으로 둔다.
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return float(np.mean(arr)), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def regression_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, tol: float) -> float:
    """회귀 정확도: |오차| <= tol 인 샘플 비율."""
    return float((torch.abs(y_true - y_pred) <= tol).float().mean().item())


def calculate_rank_metrics(S: np.ndarray, threshold: float) -> Dict[str, float]:
    """특이값 벡터 S로부터 두 가지 랭크 지표를 계산한다.

    1) rank_threshold:
       S_norm = S / S[0] 에서 threshold를 초과하는 성분 개수
    2) rank_entropy:
       p_i = S_i / ΣS 로 정규화한 뒤 exp(H(p)) 형태의 entropy rank
    """
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return {"rank_threshold": 0, "rank_entropy": 0.0}

    # 0으로 나누는 문제를 방지하기 위한 안전 epsilon
    s0 = S[0] if S[0] != 0 else 1e-12
    S_norm = S / s0
    rank_threshold = int(np.sum(S_norm > threshold))

    p = S / (np.sum(S) + 1e-12)
    rank_entropy = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

    return {
        "rank_threshold": rank_threshold,
        "rank_entropy": rank_entropy,
    }


def calculate_subspace_alignment_metrics(
    H: np.ndarray,
    freqs: Sequence[int],
    dt: float,
    lag: int,
) -> Dict[str, np.ndarray | float]:
    """특징공간 H와 Fourier 부분공간 F 사이의 정렬 정도를 측정한다.

    Args:
        H: 특징 행렬, shape (samples, bottleneck_dim)
        freqs: 데이터 생성에 사용된 실제 주파수 목록
        dt: 시간 간격
        lag: H의 첫 행이 대응하는 시점 보정값

    Returns:
        align_coverage:
            Fourier 부분공간이 H에 얼마나 커버되는지 (0~1에 가까울수록 좋음)
        align_purity:
            H가 Fourier 부분공간에 얼마나 집중되는지
        purity_norm:
            차원 차이(d_f, d_h) 보정한 purity
        align_mean_cosine:
            주각(principal angles)의 코사인 평균
        align_mean_angle_deg:
            주각 평균(도)
        principal_angles_deg:
            개별 주각 배열(도)
    """
    seq_len = H.shape[0]
    # H의 i번째 row는 원 시계열 시점 (i + lag) * dt에 대응
    t = (np.arange(seq_len, dtype=np.float64) + lag) * dt

    # F = [sin(f1 t), cos(f1 t), ..., sin(fk t), cos(fk t)]
    F = []
    for f in freqs:
        F.append(np.sin(f * t))
        F.append(np.cos(f * t))
    F = np.array(F).T  # (samples, 2k)

    # QR로 각 공간의 직교기저를 구성
    Q_f, _ = np.linalg.qr(F)
    Q_h, _ = np.linalg.qr(H)

    # 두 부분공간의 중첩 정보
    projection = Q_f.T @ Q_h
    proj_sq_sum = float(np.sum(projection**2))

    d_f = Q_f.shape[1]
    d_h = Q_h.shape[1]

    align_coverage = proj_sq_sum / (d_f + 1e-12)
    align_purity = proj_sq_sum / (d_h + 1e-12)

    # purity의 이론적 상한을 이용해 0~1 스케일에 가깝게 정규화
    purity_upper = d_f / (d_h + 1e-12)
    purity_norm = align_purity / (purity_upper + 1e-12)

    # projection 특이값 = principal angle cosine
    sv = np.linalg.svd(projection, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)

    principal_angles_deg = np.degrees(np.arccos(sv))
    align_mean_cosine = float(np.mean(sv))
    align_mean_angle_deg = float(np.mean(principal_angles_deg))

    return {
        "align_coverage": align_coverage,
        "align_purity": align_purity,
        "purity_norm": purity_norm,
        "align_mean_cosine": align_mean_cosine,
        "align_mean_angle_deg": align_mean_angle_deg,
        "principal_angles_deg": principal_angles_deg,
    }
