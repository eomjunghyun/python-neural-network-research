from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .config import ExperimentConfig


def _sample_amplitudes(cfg: ExperimentConfig, rng_np: np.random.Generator) -> np.ndarray:
    if cfg.RANDOM_AMPLITUDE:
        return rng_np.uniform(cfg.AMP_MIN, cfg.AMP_MAX, size=cfg.NUM_FREQS)
    return np.ones(cfg.NUM_FREQS, dtype=np.float64)


def _sample_phases(cfg: ExperimentConfig, rng_np: np.random.Generator) -> np.ndarray:
    if cfg.RANDOM_PHASE:
        return rng_np.uniform(cfg.PHASE_MIN, cfg.PHASE_MAX, size=cfg.NUM_FREQS)
    return np.zeros(cfg.NUM_FREQS, dtype=np.float64)


def _sample_discrete_thetas(cfg: ExperimentConfig, rng_np: np.random.Generator) -> np.ndarray:
    for _ in range(cfg.THETA_SAMPLE_MAX_ATTEMPTS):
        thetas = np.sort(rng_np.uniform(cfg.theta_min, cfg.theta_max, size=cfg.NUM_FREQS))
        if cfg.MIN_DELTA_THETA <= 0.0:
            return thetas
        if cfg.NUM_FREQS < 2 or np.min(np.diff(thetas)) >= cfg.MIN_DELTA_THETA:
            return thetas

    raise ValueError(
        "Failed to sample discrete-time frequencies that satisfy MIN_DELTA_THETA. "
        "Reduce NUM_FREQS, lower MIN_DELTA_THETA, or widen the theta range."
    )


def generate_continuous_sin_data(
    cfg: ExperimentConfig,
    rng_py: random.Random,
    rng_np: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a continuous-time sinusoid mixture and sample it at dt."""

    population_size = cfg.FREQ_MAX - cfg.FREQ_MIN + 1
    if cfg.NUM_FREQS > population_size:
        raise ValueError("NUM_FREQS exceeds the number of available frequencies.")

    t = np.arange(cfg.SEQ_LEN, dtype=np.float64) * cfg.DT
    freqs = rng_py.sample(range(cfg.FREQ_MIN, cfg.FREQ_MAX + 1), cfg.NUM_FREQS)
    amplitudes = _sample_amplitudes(cfg, rng_np)
    phases = _sample_phases(cfg, rng_np)

    y = np.zeros_like(t, dtype=np.float64)
    components = []
    for amplitude, freq, phase in zip(amplitudes, freqs, phases):
        component = amplitude * np.sin(freq * t + phase)
        y += component
        components.append(
            {
                "freq": int(freq),
                "amplitude": float(amplitude),
                "phase": float(phase),
            }
        )

    info = {
        "time_mode": "continuous",
        "freqs": tuple(int(freq) for freq in freqs),
        "thetas": None,
        "amplitudes": tuple(float(amplitude) for amplitude in amplitudes),
        "phases": tuple(float(phase) for phase in phases),
        "components": components,
    }
    return y.astype(np.float32), info


def generate_discrete_sin_data(
    cfg: ExperimentConfig,
    rng_py: random.Random,
    rng_np: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a discrete-time sinusoid mixture directly on n = 0, 1, ..., N-1."""

    _ = rng_py
    n = np.arange(cfg.SEQ_LEN, dtype=np.float64)
    thetas = _sample_discrete_thetas(cfg, rng_np)
    amplitudes = _sample_amplitudes(cfg, rng_np)
    phases = _sample_phases(cfg, rng_np)

    y = np.zeros_like(n, dtype=np.float64)
    components = []
    for amplitude, theta, phase in zip(amplitudes, thetas, phases):
        component = amplitude * np.sin(theta * n + phase)
        y += component
        components.append(
            {
                "theta": float(theta),
                "amplitude": float(amplitude),
                "phase": float(phase),
            }
        )

    info = {
        "time_mode": "discrete",
        "freqs": None,
        "thetas": tuple(float(theta) for theta in thetas),
        "amplitudes": tuple(float(amplitude) for amplitude in amplitudes),
        "phases": tuple(float(phase) for phase in phases),
        "components": components,
    }
    return y.astype(np.float32), info


def generate_sin_data(
    cfg: ExperimentConfig,
    rng_py: random.Random,
    rng_np: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a sinusoid mixture for either continuous-time or discrete-time mode."""

    if cfg.time_mode == "continuous":
        return generate_continuous_sin_data(cfg, rng_py, rng_np)
    if cfg.time_mode == "discrete":
        return generate_discrete_sin_data(cfg, rng_py, rng_np)
    raise ValueError(f"Unsupported time_mode '{cfg.time_mode}'. Expected 'continuous' or 'discrete'.")


def add_noise_to_signal(
    clean_signal: np.ndarray,
    snr_db: float,
    noise_type: str,
    rng: np.random.Generator,
    ar1_rho: float = 0.8,
    impulse_prob: float = 0.01,
    impulse_scale: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Add scaled white, AR(1), or impulsive noise to a clean signal."""

    signal = np.asarray(clean_signal, dtype=np.float64)
    length = len(signal)

    if noise_type == "white":
        raw_noise = rng.standard_normal(length)
    elif noise_type == "ar1":
        eps = rng.standard_normal(length)
        raw_noise = np.zeros(length, dtype=np.float64)
        raw_noise[0] = eps[0]
        for idx in range(1, length):
            raw_noise[idx] = ar1_rho * raw_noise[idx - 1] + eps[idx]
    elif noise_type == "impulsive":
        raw_noise = rng.standard_normal(length)
        mask = rng.random(length) < impulse_prob
        if np.any(mask):
            raw_noise[mask] += impulse_scale * rng.standard_normal(np.sum(mask))
    else:
        raise ValueError("NOISE_TYPE must be one of: 'white', 'ar1', 'impulsive'.")

    signal_power = np.mean(signal**2)
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    raw_noise_power = np.mean(raw_noise**2) + 1e-12
    noise = raw_noise * np.sqrt(target_noise_power / raw_noise_power)

    noisy = signal + noise
    actual_snr_db = 10.0 * np.log10(
        (np.mean(signal**2) + 1e-12) / (np.mean(noise**2) + 1e-12)
    )
    return noisy.astype(np.float32), noise.astype(np.float32), float(actual_snr_db)


def make_dataset(
    data: np.ndarray,
    lag: int,
    *,
    start_idx: int = 0,
    return_target_indices: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Convert a 1D time series into lag-window regression tensors.

    When `return_target_indices=True`, absolute target indices are returned as
    a third output so downstream alignment metrics can be evaluated at the
    exact train/test target locations.
    """

    x, y, target_indices = [], [], []
    for idx in range(len(data) - lag):
        x.append(data[idx : idx + lag])
        y.append(data[idx + lag])
        target_indices.append(start_idx + idx + lag)

    if len(x) == 0:
        x_tensor = torch.empty((0, lag), dtype=torch.float32)
        y_tensor = torch.empty((0, 1), dtype=torch.float32)
        index_array = np.empty((0,), dtype=np.int64)
    else:
        x_tensor = torch.tensor(np.array(x), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
        index_array = np.asarray(target_indices, dtype=np.int64)

    if return_target_indices:
        return x_tensor, y_tensor, index_array
    return x_tensor, y_tensor


def split_raw_series_arrays(
    *arrays: np.ndarray,
    test_ratio: float,
    val_ratio: float = 0.0,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Dict[str, Tuple[int, int]]]:
    """Chronologically split aligned raw series into train/val/test segments."""

    if len(arrays) == 0:
        raise ValueError("At least one array is required for splitting.")

    n_samples = len(arrays[0])
    if any(len(array) != n_samples for array in arrays):
        raise ValueError("All arrays must have the same length.")

    n_test = max(1, int(np.floor(n_samples * test_ratio)))
    n_val = max(1, int(np.floor(n_samples * val_ratio))) if val_ratio > 0.0 else 0
    n_train = n_samples - n_val - n_test
    if n_train < 1:
        raise ValueError("The requested train/val/test split leaves no training samples.")

    train_end = n_train
    val_end = n_train + n_val

    train_arrays = tuple(np.asarray(array[:train_end]) for array in arrays)
    val_arrays = tuple(np.asarray(array[train_end:val_end]) for array in arrays)
    test_arrays = tuple(np.asarray(array[val_end:]) for array in arrays)
    split_bounds = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_samples),
    }
    return train_arrays, val_arrays, test_arrays, split_bounds


def split_train_test_tensors(
    *tensors: torch.Tensor,
    test_ratio: float,
) -> Tuple[torch.Tensor, ...]:
    """Chronologically split aligned tensors into train and test segments."""

    if len(tensors) == 0:
        raise ValueError("At least one tensor is required for splitting.")

    n_samples = tensors[0].shape[0]
    if any(tensor.shape[0] != n_samples for tensor in tensors):
        raise ValueError("All tensors must have the same first dimension.")

    n_test = max(1, int(np.floor(n_samples * test_ratio)))
    n_train = n_samples - n_test
    if n_train < 1:
        raise ValueError("TEST_RATIO leaves no training samples after the split.")

    outputs = []
    for tensor in tensors:
        outputs.append(tensor[:n_train])
        outputs.append(tensor[n_train:])
    return tuple(outputs)
