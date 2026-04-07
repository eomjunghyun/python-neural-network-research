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
    """Rank-proxy diagnostics from singular values.

    These are auxiliary metrics only. Main experiments keep the comparison
    dimension fixed at the theoretical value `2k`; threshold-based or
    entropy-based ranks are diagnostic summaries and do not replace that
    theory dimension.
    """

    singular_values = np.asarray(S, dtype=np.float64)
    if singular_values.size == 0:
        return {
            "rank_threshold": 0,
            "rank_entropy": 0.0,
        }

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


def numerical_rank_from_singular_values(
    singular_values: np.ndarray,
    rel_tol: float = 1e-8,
) -> int:
    """Return a relative-tolerance numerical rank from singular values."""

    values = np.asarray(singular_values, dtype=np.float64)
    if values.size == 0:
        return 0

    sigma_max = float(values[0])
    if sigma_max <= 0.0:
        return 0
    return int(np.sum(values > sigma_max * rel_tol))


def orth_basis_from_matrix(
    X: np.ndarray,
    rel_tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return an orthonormal numerical column-space basis from SVD.

    Returns:
        Q: orthonormal basis of the numerical column space
        S: singular values of X
        r: numerical rank determined by `numerical_rank_from_singular_values`
    """

    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("orth_basis_from_matrix expects a 2D array.")
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64), np.array([], dtype=np.float64), 0

    U, S, _ = np.linalg.svd(matrix, full_matrices=False)
    r = numerical_rank_from_singular_values(S, rel_tol=rel_tol)
    return U[:, :r], S, r


def _coerce_target_indices(target_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(target_indices, dtype=np.float64).reshape(-1)
    return indices


def _basis_theoretical_dim(
    *,
    time_mode: str,
    freqs: Tuple[int, ...] | None = None,
    thetas: Tuple[float, ...] | None = None,
) -> int:
    if time_mode == "continuous":
        if freqs is None:
            raise ValueError("Continuous mode requires 'freqs'.")
        return 2 * len(freqs)
    if time_mode == "discrete":
        if thetas is None:
            raise ValueError("Discrete mode requires 'thetas'.")
        return 2 * len(thetas)
    raise ValueError(f"Unsupported time_mode '{time_mode}'. Expected 'continuous' or 'discrete'.")


def _condition_number_from_singular_values(singular_values: np.ndarray) -> float:
    values = np.asarray(singular_values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    sigma_min = float(values[-1])
    if sigma_min <= 1e-12:
        return np.inf
    return float(values[0] / sigma_min)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    return float(np.mean(values))


def _basis_diagnostics_dict(
    basis_matrix: np.ndarray,
    *,
    theory_dim: int,
    rel_tol: float = 1e-8,
) -> Dict[str, Any]:
    matrix = np.asarray(basis_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Basis matrix must be 2D.")
    if matrix.shape[1] == 0:
        singular_values = np.array([], dtype=np.float64)
        min_singular_value = np.nan
    else:
        _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        min_singular_value = float(singular_values[-1]) if singular_values.size > 0 else np.nan

    return {
        "fourier_theoretical_dim": int(theory_dim),
        "fourier_numerical_dim": int(
            numerical_rank_from_singular_values(singular_values, rel_tol=rel_tol)
        ),
        "fourier_singular_values": singular_values,
        "fourier_min_singular_value": min_singular_value,
        "fourier_condition_number": _condition_number_from_singular_values(singular_values),
    }


def build_sampled_fourier_matrix_from_indices(
    freqs: Tuple[int, ...],
    dt: float,
    target_indices: np.ndarray,
) -> np.ndarray:
    """Build a continuous-time sampled sin/cos basis at absolute target indices."""

    indices = _coerce_target_indices(target_indices)
    if len(freqs) == 0:
        return np.empty((indices.size, 0), dtype=np.float64)

    t = indices * dt
    basis_matrix = np.empty((indices.size, 2 * len(freqs)), dtype=np.float64)
    for idx, freq in enumerate(freqs):
        basis_matrix[:, 2 * idx] = np.sin(freq * t)
        basis_matrix[:, 2 * idx + 1] = np.cos(freq * t)
    return basis_matrix


def build_sampled_discrete_basis_matrix_from_indices(
    thetas: Tuple[float, ...],
    target_indices: np.ndarray,
) -> np.ndarray:
    """Build a discrete-time sampled sin/cos basis at absolute target indices."""

    indices = _coerce_target_indices(target_indices)
    if len(thetas) == 0:
        return np.empty((indices.size, 0), dtype=np.float64)

    n = indices
    basis_matrix = np.empty((indices.size, 2 * len(thetas)), dtype=np.float64)
    for idx, theta in enumerate(thetas):
        basis_matrix[:, 2 * idx] = np.sin(theta * n)
        basis_matrix[:, 2 * idx + 1] = np.cos(theta * n)
    return basis_matrix


def build_sampled_basis_matrix_from_indices(
    *,
    time_mode: str,
    target_indices: np.ndarray,
    freqs: Tuple[int, ...] | None = None,
    thetas: Tuple[float, ...] | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Build the sampled basis matrix at absolute target indices."""

    if time_mode == "continuous":
        if freqs is None:
            raise ValueError("Continuous mode requires 'freqs'.")
        return build_sampled_fourier_matrix_from_indices(
            freqs=freqs,
            dt=dt,
            target_indices=target_indices,
        )
    if time_mode == "discrete":
        if thetas is None:
            raise ValueError("Discrete mode requires 'thetas'.")
        return build_sampled_discrete_basis_matrix_from_indices(
            thetas=thetas,
            target_indices=target_indices,
        )
    raise ValueError(f"Unsupported time_mode '{time_mode}'. Expected 'continuous' or 'discrete'.")


def build_sampled_fourier_matrix(
    freqs: Tuple[int, ...],
    dt: float,
    lag: int,
    seq_len: int,
) -> np.ndarray:
    """Build the sampled continuous-time basis from `(lag, seq_len)`."""

    target_indices = np.arange(seq_len, dtype=np.float64) + lag
    return build_sampled_fourier_matrix_from_indices(
        freqs=freqs,
        dt=dt,
        target_indices=target_indices,
    )


def build_sampled_discrete_basis_matrix(
    thetas: Tuple[float, ...],
    lag: int,
    seq_len: int,
) -> np.ndarray:
    """Build the sampled discrete-time basis from `(lag, seq_len)`."""

    target_indices = np.arange(seq_len, dtype=np.float64) + lag
    return build_sampled_discrete_basis_matrix_from_indices(
        thetas=thetas,
        target_indices=target_indices,
    )


def build_sampled_basis_matrix(
    *,
    time_mode: str,
    seq_len: int,
    lag: int,
    freqs: Tuple[int, ...] | None = None,
    thetas: Tuple[float, ...] | None = None,
    dt: float = 1.0,
) -> np.ndarray:
    """Build the sampled basis matrix from `(lag, seq_len)`.

    This wrapper is kept for backward compatibility and dispatches to the new
    absolute-index implementation.
    """

    target_indices = np.arange(seq_len, dtype=np.float64) + lag
    return build_sampled_basis_matrix_from_indices(
        time_mode=time_mode,
        target_indices=target_indices,
        freqs=freqs,
        thetas=thetas,
        dt=dt,
    )


def calculate_sampled_fourier_numerical_dim(
    freqs: Tuple[int, ...],
    dt: float,
    lag: int,
    seq_len: int,
    rel_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Return sampled-basis diagnostics.

    Main experiments still use the theoretical dimension `2 * len(freqs)`.
    Numerical rank and conditioning are diagnostic only.
    """

    basis_matrix = build_sampled_fourier_matrix(freqs=freqs, dt=dt, lag=lag, seq_len=seq_len)
    return _basis_diagnostics_dict(
        basis_matrix,
        theory_dim=2 * len(freqs),
        rel_tol=rel_tol,
    )


def calculate_sampled_discrete_numerical_dim(
    thetas: Tuple[float, ...],
    lag: int,
    seq_len: int,
    rel_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Return sampled-basis diagnostics.

    Main experiments still use the theoretical dimension `2 * len(thetas)`.
    Numerical rank and conditioning are diagnostic only.
    """

    basis_matrix = build_sampled_discrete_basis_matrix(thetas=thetas, lag=lag, seq_len=seq_len)
    return _basis_diagnostics_dict(
        basis_matrix,
        theory_dim=2 * len(thetas),
        rel_tol=rel_tol,
    )


def calculate_sampled_basis_numerical_dim(
    *,
    time_mode: str,
    seq_len: int,
    lag: int,
    freqs: Tuple[int, ...] | None = None,
    thetas: Tuple[float, ...] | None = None,
    dt: float = 1.0,
    rel_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Return sampled-basis diagnostics.

    Main experiments still use the theoretical dimension `2k`. Numerical rank
    is reported only as a diagnostic and must not replace that theory
    dimension in downstream comparisons.
    """

    basis_matrix = build_sampled_basis_matrix(
        time_mode=time_mode,
        seq_len=seq_len,
        lag=lag,
        freqs=freqs,
        thetas=thetas,
        dt=dt,
    )
    theory_dim = _basis_theoretical_dim(
        time_mode=time_mode,
        freqs=freqs,
        thetas=thetas,
    )
    return _basis_diagnostics_dict(
        basis_matrix,
        theory_dim=theory_dim,
        rel_tol=rel_tol,
    )


def calculate_subspace_alignment_from_matrices(
    H: np.ndarray,
    F: np.ndarray,
    theory_dim: int,
    *,
    h_rel_tol: float = 1e-8,
    f_rel_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Return SVD-based full/top subspace diagnostics from explicit matrices.

    Main experiments use `theory_dim=2k`. Numerical ranks of `H` and `F` are
    diagnostic only and do not replace this theory dimension.
    """

    H_matrix = np.asarray(H, dtype=np.float64)
    F_matrix = np.asarray(F, dtype=np.float64)
    if H_matrix.ndim != 2 or F_matrix.ndim != 2:
        raise ValueError("H and F must both be 2D arrays.")
    if H_matrix.shape[0] != F_matrix.shape[0]:
        raise ValueError("H and F must have the same number of rows.")

    Q_f_full, S_f, r_f_num = orth_basis_from_matrix(F_matrix, rel_tol=f_rel_tol)
    Q_h_full, _, r_h_num = orth_basis_from_matrix(H_matrix, rel_tol=h_rel_tol)
    U_h, S_h_raw, _ = np.linalg.svd(H_matrix, full_matrices=False)

    if Q_f_full.shape[1] == 0 or Q_h_full.shape[1] == 0:
        sv_full = np.array([], dtype=np.float64)
    else:
        sv_full = np.linalg.svd(Q_f_full.T @ Q_h_full, compute_uv=False)
        sv_full = np.clip(sv_full, -1.0, 1.0)
    angles_full = np.degrees(np.arccos(sv_full))

    full_overlap_energy = float(np.sum(sv_full**2))
    align_coverage_full = full_overlap_energy / theory_dim if theory_dim > 0 else np.nan
    align_purity_full = full_overlap_energy / r_h_num if r_h_num > 0 else np.nan

    top_dim = min(theory_dim, U_h.shape[1])
    if Q_f_full.shape[1] == 0 or top_dim == 0:
        sv_top = np.array([], dtype=np.float64)
    else:
        U_h_top = U_h[:, :top_dim]
        sv_top = np.linalg.svd(Q_f_full.T @ U_h_top, compute_uv=False)
        sv_top = np.clip(sv_top, -1.0, 1.0)
    angles_top = np.degrees(np.arccos(sv_top))

    top_overlap_energy = float(np.sum(sv_top**2))
    align_coverage_top = top_overlap_energy / theory_dim if theory_dim > 0 else np.nan
    align_purity_top = top_overlap_energy / top_dim if top_dim > 0 else np.nan

    if Q_f_full.shape[1] == 0:
        recon_r2_qf_from_h = np.nan
    else:
        denom = float(np.sum(Q_f_full**2))
        if denom <= 1e-12:
            recon_r2_qf_from_h = np.nan
        else:
            if H_matrix.shape[1] == 0:
                Qf_hat = np.zeros_like(Q_f_full)
            else:
                W, *_ = np.linalg.lstsq(H_matrix, Q_f_full, rcond=None)
                Qf_hat = H_matrix @ W
            residual = float(np.sum((Q_f_full - Qf_hat) ** 2))
            recon_r2_qf_from_h = float(1.0 - (residual / denom))

    f_min_singular_value = float(S_f[-1]) if S_f.size > 0 else np.nan

    return {
        "theory_dim": int(theory_dim),
        "f_numerical_dim": int(r_f_num),
        "h_numerical_dim": int(r_h_num),
        "f_min_singular_value": f_min_singular_value,
        "f_condition_number": _condition_number_from_singular_values(S_f),
        "align_coverage_full": float(align_coverage_full),
        "align_purity_full": float(align_purity_full),
        "align_mean_cosine_full": _safe_mean(sv_full),
        "align_mean_angle_deg_full": _safe_mean(angles_full),
        "principal_angles_full_deg": angles_full,
        "align_coverage_top": float(align_coverage_top),
        "align_purity_top": float(align_purity_top),
        "align_mean_cosine_top": _safe_mean(sv_top),
        "align_mean_angle_deg_top": _safe_mean(angles_top),
        "principal_angles_top_deg": angles_top,
        "energy_top_theory_dim": topk_energy_ratio(S_h_raw, theory_dim),
        "recon_r2_qf_from_h": recon_r2_qf_from_h,
    }


def calculate_subspace_alignment_metrics_v2(
    H: np.ndarray,
    *,
    time_mode: str,
    target_indices: np.ndarray,
    freqs: Tuple[int, ...] | None = None,
    thetas: Tuple[float, ...] | None = None,
    dt: float = 1.0,
    theory_dim: int | None = None,
    h_rel_tol: float = 1e-8,
    f_rel_tol: float = 1e-8,
) -> Dict[str, Any]:
    """Return alignment metrics using absolute target indices and SVD bases."""

    basis_matrix = build_sampled_basis_matrix_from_indices(
        time_mode=time_mode,
        target_indices=target_indices,
        freqs=freqs,
        thetas=thetas,
        dt=dt,
    )
    if theory_dim is None:
        theory_dim = _basis_theoretical_dim(
            time_mode=time_mode,
            freqs=freqs,
            thetas=thetas,
        )

    return calculate_subspace_alignment_from_matrices(
        H=H,
        F=basis_matrix,
        theory_dim=theory_dim,
        h_rel_tol=h_rel_tol,
        f_rel_tol=f_rel_tol,
    )


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
    """Backward-compatible wrapper for subspace alignment metrics.

    The new implementation uses absolute target indices and SVD-based
    diagnostics internally. Legacy alias keys are preserved so existing
    notebooks can continue to read the old names.
    """

    seq_len = H.shape[0]
    target_indices = np.arange(seq_len, dtype=np.int64) + lag
    metrics = calculate_subspace_alignment_metrics_v2(
        H,
        time_mode=time_mode,
        target_indices=target_indices,
        freqs=freqs,
        thetas=thetas,
        dt=dt,
        theory_dim=top_k,
    )

    metrics["align_coverage"] = metrics["align_coverage_full"]
    metrics["align_purity"] = metrics["align_purity_full"]
    metrics["alignment_score_2k"] = metrics["align_coverage_top"]
    metrics["align_mean_cosine"] = metrics["align_mean_cosine_top"]
    metrics["mean_principal_angle_deg"] = metrics["align_mean_angle_deg_top"]
    metrics["principal_angles_deg"] = metrics["principal_angles_top_deg"]
    return metrics


def topk_energy_ratio(singular_values: np.ndarray, top_k: int) -> float:
    """Energy concentration in the top-k singular values.

    This is typically used as a theory-dimension diagnostic, for example with
    `top_k = 2k` in the main experiments.
    """

    singular_values = np.asarray(singular_values, dtype=np.float64)
    if singular_values.size == 0:
        return np.nan
    top_k = min(top_k, singular_values.size)
    energy = singular_values**2
    return float(np.sum(energy[:top_k]) / (np.sum(energy) + 1e-12))
