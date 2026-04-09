from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import torch

from .config import ExperimentConfig
from .models import MODEL_REGISTRY


def set_seed(seed: int) -> None:
    """Synchronize Python, NumPy, and PyTorch RNG state."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_sampling_constraints(dt: float, freq_max: float, margin: float = 0.98) -> None:
    """Check the angular-frequency Nyquist margin implied by dt."""

    nyquist_angular = np.pi / dt
    max_safe = margin * nyquist_angular
    if freq_max >= max_safe:
        recommended_dt = np.pi / (freq_max / margin)
        raise ValueError(
            f"FREQ_MAX={freq_max} is too high for DT={dt}. "
            f"Need FREQ_MAX < {max_safe:.4f} (margin={margin}, pi/dt={nyquist_angular:.4f}). "
            f"Either lower FREQ_MAX or use DT <= {recommended_dt:.6f}."
        )


def validate_config(cfg: ExperimentConfig) -> None:
    """Validate configuration consistency before running an experiment."""

    if cfg.time_mode not in ("continuous", "discrete"):
        raise ValueError("time_mode must be either 'continuous' or 'discrete'.")
    if cfg.NUM_FREQS < cfg.NUM_FREQS_MIN or cfg.NUM_FREQS > cfg.NUM_FREQS_MAX:
        raise ValueError(
            f"NUM_FREQS must be an integer between {cfg.NUM_FREQS_MIN} and {cfg.NUM_FREQS_MAX}."
        )
    if cfg.TRAIN_TARGET not in ("noisy", "clean"):
        raise ValueError("TRAIN_TARGET must be either 'noisy' or 'clean'.")
    if cfg.MODEL_ID not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"MODEL_ID must be one of: {supported}.")
    if cfg.DEVICE not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError("DEVICE must be one of: 'auto', 'cuda', 'mps', 'cpu'.")
    if not (0.0 < cfg.TEST_RATIO < 1.0):
        raise ValueError("TEST_RATIO must be strictly between 0 and 1.")
    if not (0.0 <= cfg.VAL_RATIO < 1.0):
        raise ValueError("VAL_RATIO must lie in [0, 1).")
    if cfg.VAL_RATIO + cfg.TEST_RATIO >= 1.0:
        raise ValueError("VAL_RATIO + TEST_RATIO must be strictly less than 1.")
    if cfg.NOISE_TYPE not in ("white", "ar1", "impulsive"):
        raise ValueError("NOISE_TYPE must be one of: 'white', 'ar1', 'impulsive'.")
    if cfg.AMP_MIN <= 0 or cfg.AMP_MAX <= 0 or cfg.AMP_MIN > cfg.AMP_MAX:
        raise ValueError("AMP_MIN and AMP_MAX must be positive and satisfy AMP_MIN <= AMP_MAX.")
    if cfg.PHASE_MIN > cfg.PHASE_MAX:
        raise ValueError("PHASE_MIN must be less than or equal to PHASE_MAX.")
    if cfg.BOTTLENECK_DIM_OVERRIDE is not None and cfg.BOTTLENECK_DIM_OVERRIDE < 1:
        raise ValueError("BOTTLENECK_DIM_OVERRIDE must be positive when provided.")
    if cfg.time_mode == "continuous":
        if cfg.NUM_FREQS > (cfg.FREQ_MAX - cfg.FREQ_MIN + 1):
            raise ValueError("NUM_FREQS exceeds the number of available integer frequencies.")
        validate_sampling_constraints(cfg.DT, cfg.FREQ_MAX, margin=cfg.NYQUIST_MARGIN)
    else:
        if not (0.0 < cfg.theta_min < cfg.theta_max < np.pi):
            raise ValueError("Discrete mode requires 0 < theta_min < theta_max < pi.")
        if cfg.MIN_DELTA_THETA < 0.0:
            raise ValueError("MIN_DELTA_THETA must be non-negative.")
        if cfg.THETA_SAMPLE_MAX_ATTEMPTS < 1:
            raise ValueError("THETA_SAMPLE_MAX_ATTEMPTS must be at least 1.")
        available_span = cfg.theta_max - cfg.theta_min
        required_span = cfg.MIN_DELTA_THETA * max(0, cfg.NUM_FREQS - 1)
        if required_span >= available_span:
            raise ValueError(
                "MIN_DELTA_THETA is too large for the requested NUM_FREQS and theta range."
            )


def resolve_torch_device(device_preference: str = "auto") -> torch.device:
    """Resolve a torch device from a stable config string.

    Preference order for `auto` is CUDA, then MPS, then CPU.
    """

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())

    if device_preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise ValueError("DEVICE='cuda' was requested but CUDA is not available.")

    if device_preference == "mps":
        if mps_available:
            return torch.device("mps")
        raise ValueError("DEVICE='mps' was requested but MPS is not available.")

    if device_preference == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def mean_std(values: List[float]) -> Tuple[float, float]:
    """Return sample mean and sample std, handling small lists safely."""

    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(np.mean(arr)), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def mean_std_ci95(values: List[float]) -> Tuple[float, float, float, float]:
    """Return mean, sample std, and a normal-approximation 95% CI."""

    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan

    mean_value = float(np.mean(arr))
    if len(arr) == 1:
        return mean_value, 0.0, mean_value, mean_value

    std_value = float(np.std(arr, ddof=1))
    margin = 1.96 * std_value / np.sqrt(len(arr))
    return mean_value, std_value, mean_value - margin, mean_value + margin
