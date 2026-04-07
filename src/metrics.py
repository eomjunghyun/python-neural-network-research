from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch


def regression_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, tol: float) -> float:
    """Fraction of predictions whose absolute error is below tol."""

    return float((torch.abs(y_true - y_pred) <= tol).float().mean().item())


def regression_r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Coefficient of determination for regression."""

    y_true_np = y_true.detach().cpu().numpy().reshape(-1)
    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    ss_res = float(np.sum((y_true_np - y_pred_np) ** 2))
    ss_tot = float(np.sum((y_true_np - np.mean(y_true_np)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan") if ss_res > 1e-12 else 1.0
    return float(1.0 - (ss_res / ss_tot))


def calculate_rank_metrics(S: np.ndarray, threshold: float) -> Dict[str, float]:
    """Rank proxies from singular values."""

    singular_values = np.asarray(S, dtype=np.float64)
    s_norm = singular_values / (singular_values[0] + 1e-12)
    rank_threshold = int(np.sum(s_norm > threshold))

    probs = singular_values / (np.sum(singular_values) + 1e-12)
    rank_entropy = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))

    return {
        "rank_threshold": rank_threshold,
        "rank_entropy": rank_entropy,
    }


def min_delta_f(freqs: Sequence[int] | None) -> float:
    """Smallest spacing among sampled frequencies."""

    if freqs is None:
        return np.nan
    if len(freqs) < 2:
        return np.nan
    sorted_freqs = np.sort(np.array(freqs, dtype=np.float64))
    return float(np.min(np.diff(sorted_freqs)))


def min_delta_theta(thetas: Sequence[float] | None) -> float:
    """Smallest spacing among sampled discrete-time angular frequencies."""

    if thetas is None:
        return np.nan
    if len(thetas) < 2:
        return np.nan
    sorted_thetas = np.sort(np.array(thetas, dtype=np.float64))
    return float(np.min(np.diff(sorted_thetas)))


def snr_db_from_tensors(clean_ref: torch.Tensor, observed: torch.Tensor) -> float:
    """Estimate SNR in dB between a clean reference and an observed signal."""

    clean = clean_ref.detach().cpu().numpy().reshape(-1)
    obs = observed.detach().cpu().numpy().reshape(-1)
    noise = obs - clean
    signal_power = np.mean(clean**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    return float(10.0 * np.log10(signal_power / noise_power))


def normalize_feature_columns(H: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize hidden-feature columns."""

    features = np.asarray(H, dtype=np.float64)
    col_norms = np.linalg.norm(features, axis=0, keepdims=True)
    col_norms = np.maximum(col_norms, eps)
    return features / col_norms


def build_sampled_fourier_matrix(
    freqs: Tuple[int, ...],
    dt: float,
    lag: int,
    seq_len: int,
) -> np.ndarray:
    """Build the sampled sin/cos basis matrix used for subspace comparison."""

    t = (np.arange(seq_len, dtype=np.float64) + lag) * dt
    basis = []
    for freq in freqs:
        basis.append(np.sin(freq * t))
        basis.append(np.cos(freq * t))
    return np.array(basis, dtype=np.float64).T


def build_sampled_discrete_basis_matrix(
    thetas: Tuple[float, ...],
    lag: int,
    seq_len: int,
) -> np.ndarray:
    """Build the sampled discrete-time sin/cos basis matrix."""

    n = np.arange(seq_len, dtype=np.float64) + lag
    basis = []
    for theta in thetas:
        basis.append(np.sin(theta * n))
        basis.append(np.cos(theta * n))
    return np.array(basis, dtype=np.float64).T


def build_sampled_basis_matrix(
    *,
    time_mode: str,
    seq_len: int,
    lag: int,
    freqs: Tuple[int, ...] | None = None,
    thetas: Tuple[float, ...] | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Build the sampled basis matrix for either continuous or discrete mode."""

    if time_mode == "continuous":
        if freqs is None:
            raise ValueError("Continuous mode requires 'freqs'.")
        return build_sampled_fourier_matrix(freqs=freqs, dt=dt, lag=lag, seq_len=seq_len)
    if time_mode == "discrete":
        if thetas is None:
            raise ValueError("Discrete mode requires 'thetas'.")
        return build_sampled_discrete_basis_matrix(thetas=thetas, lag=lag, seq_len=seq_len)
    raise ValueError(f"Unsupported time_mode '{time_mode}'. Expected 'continuous' or 'discrete'.")


def calculate_sampled_fourier_numerical_dim(
    freqs: Tuple[int, ...],
    dt: float,
    lag: int,
    seq_len: int,
) -> Dict[str, Any]:
    """Return numerical-dimension diagnostics for the sampled Fourier matrix."""

    basis_matrix = build_sampled_fourier_matrix(freqs=freqs, dt=dt, lag=lag, seq_len=seq_len)
    if basis_matrix.size == 0:
        return {
            "fourier_numerical_dim": 0,
            "fourier_singular_values": np.array([], dtype=np.float64),
        }

    _, singular_values, _ = np.linalg.svd(basis_matrix, full_matrices=False)
    return {
        "fourier_numerical_dim": int(np.linalg.matrix_rank(basis_matrix)),
        "fourier_singular_values": singular_values,
    }


def calculate_sampled_discrete_numerical_dim(
    thetas: Tuple[float, ...],
    lag: int,
    seq_len: int,
) -> Dict[str, Any]:
    """Return numerical-dimension diagnostics for the sampled discrete-time basis matrix."""

    basis_matrix = build_sampled_discrete_basis_matrix(thetas=thetas, lag=lag, seq_len=seq_len)
    if basis_matrix.size == 0:
        return {
            "fourier_numerical_dim": 0,
            "fourier_singular_values": np.array([], dtype=np.float64),
        }

    _, singular_values, _ = np.linalg.svd(basis_matrix, full_matrices=False)
    return {
        "fourier_numerical_dim": int(np.linalg.matrix_rank(basis_matrix)),
        "fourier_singular_values": singular_values,
    }


def calculate_sampled_basis_numerical_dim(
    *,
    time_mode: str,
    seq_len: int,
    lag: int,
    freqs: Tuple[int, ...] | None = None,
    thetas: Tuple[float, ...] | None = None,
    dt: float = 1.0,
) -> Dict[str, Any]:
    """Return numerical-dimension diagnostics for the sampled basis matrix."""

    basis_matrix = build_sampled_basis_matrix(
        time_mode=time_mode,
        seq_len=seq_len,
        lag=lag,
        freqs=freqs,
        thetas=thetas,
        dt=dt,
    )
    if basis_matrix.size == 0:
        return {
            "fourier_numerical_dim": 0,
            "fourier_singular_values": np.array([], dtype=np.float64),
        }

    _, singular_values, _ = np.linalg.svd(basis_matrix, full_matrices=False)
    return {
        "fourier_numerical_dim": int(np.linalg.matrix_rank(basis_matrix)),
        "fourier_singular_values": singular_values,
    }


def calculate_subspace_alignment_metrics(
    H: np.ndarray,
    freqs: Tuple[int, ...] | None,
    dt: float,
    lag: int,
    top_k: int,
    *,
    thetas: Tuple[float, ...] | None = None,
    time_mode: str = "continuous",
) -> Dict[str, Any]:
    """Compare hidden-feature subspace with the sinusoidal basis subspace."""

    seq_len = H.shape[0]
    basis_matrix = build_sampled_basis_matrix(
        time_mode=time_mode,
        seq_len=seq_len,
        lag=lag,
        freqs=freqs,
        thetas=thetas,
        dt=dt,
    )

    q_f, _ = np.linalg.qr(basis_matrix)
    q_h, _ = np.linalg.qr(H)
    u_h, _, _ = np.linalg.svd(H, full_matrices=False)

    projection = q_f.T @ q_h
    proj_sq_sum = float(np.sum(projection**2))

    d_f = q_f.shape[1]
    d_h = q_h.shape[1]

    align_coverage = proj_sq_sum / (d_f + 1e-12)
    align_purity = proj_sq_sum / (d_h + 1e-12)

    top_dim = min(top_k, u_h.shape[1])
    u_h_top = u_h[:, :top_dim]
    projection_top = q_f.T @ u_h_top
    alignment_score_2k = float(np.sum(projection_top**2) / (top_k + 1e-12))

    canonical_corr = np.linalg.svd(projection_top, compute_uv=False)
    canonical_corr = np.clip(canonical_corr, -1.0, 1.0)
    principal_angles_deg = np.degrees(np.arccos(canonical_corr))

    return {
        "align_coverage": float(align_coverage),
        "align_purity": float(align_purity),
        "alignment_score_2k": alignment_score_2k,
        "align_mean_cosine": float(np.mean(canonical_corr)),
        "mean_principal_angle_deg": float(np.mean(principal_angles_deg)),
        "principal_angles_deg": principal_angles_deg,
    }


def topk_energy_ratio(singular_values: np.ndarray, top_k: int) -> float:
    """Energy concentration in the top-k singular values."""

    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.size == 0:
        return np.nan
    top_k = min(top_k, singular_values.size)
    energy = singular_values**2
    return float(np.sum(energy[:top_k]) / (np.sum(energy) + 1e-12))
