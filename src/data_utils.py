from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .config import ExperimentConfig


def generate_sin_data(
    cfg: ExperimentConfig,
    rng_py: random.Random,
    rng_np: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a sinusoid mixture with amplitude and phase controls."""

    population_size = cfg.FREQ_MAX - cfg.FREQ_MIN + 1
    if cfg.NUM_FREQS > population_size:
        raise ValueError("NUM_FREQS exceeds the number of available frequencies.")

    t = np.arange(cfg.SEQ_LEN, dtype=np.float64) * cfg.DT
    freqs = rng_py.sample(range(cfg.FREQ_MIN, cfg.FREQ_MAX + 1), cfg.NUM_FREQS)

    if cfg.RANDOM_AMPLITUDE:
        amplitudes = rng_np.uniform(cfg.AMP_MIN, cfg.AMP_MAX, size=cfg.NUM_FREQS)
    else:
        amplitudes = np.ones(cfg.NUM_FREQS, dtype=np.float64)

    if cfg.RANDOM_PHASE:
        phases = rng_np.uniform(cfg.PHASE_MIN, cfg.PHASE_MAX, size=cfg.NUM_FREQS)
    else:
        phases = np.zeros(cfg.NUM_FREQS, dtype=np.float64)

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
        "freqs": tuple(int(freq) for freq in freqs),
        "amplitudes": tuple(float(amplitude) for amplitude in amplitudes),
        "phases": tuple(float(phase) for phase in phases),
        "components": components,
    }
    return y.astype(np.float32), info


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


def make_dataset(data: np.ndarray, lag: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a 1D time series into lag-window regression tensors."""

    x, y = [], []
    for idx in range(len(data) - lag):
        x.append(data[idx : idx + lag])
        y.append(data[idx + lag])
    return (
        torch.tensor(np.array(x), dtype=torch.float32),
        torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1),
    )


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
