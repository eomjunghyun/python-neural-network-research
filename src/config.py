from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for the sinusoid-mixture representation experiment."""

    GLOBAL_SEED: int = 42
    DATA_SEED_BASE: int = 1000
    NOISE_SEED_BASE: int = 50000
    TRAIN_SEED_BASE: int = 100

    SEQ_LEN: int = 1000
    time_mode: str = "discrete"
    DT: float = 0.05
    theta_min: float = 0.05 * np.pi
    theta_max: float = 0.85 * np.pi
    MIN_DELTA_THETA: float = 0.0
    THETA_SAMPLE_MAX_ATTEMPTS: int = 1000
    NUM_FREQS: int = 4
    NUM_FREQS_MIN: int = 1
    NUM_FREQS_MAX: int = 30
    FREQ_MIN: int = 1
    FREQ_MAX: int = 60
    NYQUIST_MARGIN: float = 0.98

    RANDOM_AMPLITUDE: bool = True
    AMP_MIN: float = 0.5
    AMP_MAX: float = 2.0
    RANDOM_PHASE: bool = False
    PHASE_MIN: float = 0.0
    PHASE_MAX: float = 2.0 * np.pi

    USE_NOISE: bool = False
    NOISE_TYPE: str = "white"
    SNR_DB: float = 10.0
    AR1_RHO: float = 0.8
    IMPULSE_PROB: float = 0.01
    IMPULSE_SCALE: float = 8.0

    TRAIN_TARGET: str = "noisy"

    MODEL_ID: str = "AN001_BN_RELU"
    DEVICE: str = "auto"

    LAG: int = 32
    HIDDEN_DIM: int = 64
    BOTTLENECK_MULTIPLIER: int = 4
    BOTTLENECK_DIM_OVERRIDE: int | None = None
    LR: float = 0.01
    EPOCHS: int = 1000

    NUM_EXPERIMENTS: int = 5
    SEEDS_PER_FREQ: int = 10
    VAL_RATIO: float = 0.0
    TEST_RATIO: float = 0.2

    ACC_TOLERANCE: float = 0.1
    RANK_THRESHOLD: float = 0.05
    SCREE_TOPK: int = 20
    NORMALIZE_H_COLUMNS: bool = False

    VERBOSE: bool = True
    MAKE_PLOTS: bool = False
