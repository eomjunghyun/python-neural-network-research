import os
import random
from typing import Dict, Sequence, Tuple

import numpy as np
import torch


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
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
    if num_freqs < num_freqs_min or num_freqs > num_freqs_max:
        raise ValueError(f"num_freqs must be in [{num_freqs_min}, {num_freqs_max}].")

    population_size = freq_max - freq_min + 1
    if num_freqs > population_size:
        raise ValueError("num_freqs is larger than available frequency population.")

    t = np.arange(seq_len, dtype=np.float64) * dt
    freqs = rng.sample(range(freq_min, freq_max + 1), num_freqs)

    y = np.zeros_like(t, dtype=np.float64)
    for f in freqs:
        y += np.sin(f * t)

    return y.astype(np.float32), tuple(freqs)


def make_dataset(data: np.ndarray, lag: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = [], []
    for i in range(len(data) - lag):
        x.append(data[i : i + lag])
        y.append(data[i + lag])

    x_tensor = torch.tensor(np.array(x), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    return x_tensor, y_tensor


def min_delta_f(freqs: Sequence[int]) -> float:
    if len(freqs) < 2:
        return float("nan")
    sf = np.sort(np.asarray(freqs, dtype=np.float64))
    return float(np.min(np.diff(sf)))


# -----------------------------
# Metric utilities
# -----------------------------
def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return float(np.mean(arr)), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def regression_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, tol: float) -> float:
    return float((torch.abs(y_true - y_pred) <= tol).float().mean().item())


def calculate_rank_metrics(S: np.ndarray, threshold: float) -> Dict[str, float]:
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return {"rank_threshold": 0, "rank_entropy": 0.0}

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
    """
    H: feature matrix with shape (samples, bottleneck_dim)
    freqs: true frequencies used for data generation
    """
    seq_len = H.shape[0]
    t = (np.arange(seq_len, dtype=np.float64) + lag) * dt

    F = []
    for f in freqs:
        F.append(np.sin(f * t))
        F.append(np.cos(f * t))
    F = np.array(F).T  # (samples, 2k)

    Q_f, _ = np.linalg.qr(F)
    Q_h, _ = np.linalg.qr(H)

    projection = Q_f.T @ Q_h
    proj_sq_sum = float(np.sum(projection**2))

    d_f = Q_f.shape[1]
    d_h = Q_h.shape[1]

    align_coverage = proj_sq_sum / (d_f + 1e-12)
    align_purity = proj_sq_sum / (d_h + 1e-12)

    purity_upper = d_f / (d_h + 1e-12)
    purity_norm = align_purity / (purity_upper + 1e-12)

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
