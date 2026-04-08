from __future__ import annotations

from dataclasses import asdict
import random
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .common_utils import mean_std_ci95, set_seed, validate_config
from .config import ExperimentConfig
from .data_utils import (
    add_noise_to_signal,
    generate_sin_data,
    make_dataset,
    split_raw_series_arrays,
)
from .metrics import (
    calculate_rank_metrics,
    calculate_subspace_alignment_metrics_v2,
    min_delta_f,
    min_delta_theta,
    normalize_feature_columns,
    regression_accuracy,
    regression_r2,
    snr_db_from_tensors,
    topk_energy_ratio,
)
from .models import build_model


LEGACY_ALIAS_SOURCES = {
    "mse": "train_mse",
    "mae": "train_mae",
    "acc": "train_acc",
    "rank_threshold": "test_rank_threshold",
    "rank_entropy": "test_rank_entropy",
    "rank_gap": "test_rank_gap",
    "rel_rank_gap": "test_rel_rank_gap",
    "spectral_gap_2k": "test_spectral_gap_2k",
    "energy_ratio_2k": "test_energy_ratio_2k",
    "align_coverage": "test_align_coverage_full",
    "align_purity": "test_align_purity_full",
    "alignment_score_2k": "test_align_coverage_top",
    "align_mean_cosine": "test_align_mean_cosine_top",
    "mean_principal_angle_deg": "test_align_mean_angle_deg_top",
    "fourier_theoretical_dim": "test_fourier_theoretical_dim",
    "fourier_numerical_dim": "test_fourier_numerical_dim",
    "fourier_min_singular_value": "test_fourier_min_singular_value",
    "fourier_condition_number": "test_fourier_condition_number",
    "recon_r2_qf_from_h": "test_recon_r2_qf_from_h",
    "energy_top_theory_dim": "test_energy_top_theory_dim",
}

ARRAY_ALIAS_SOURCES = {
    "principal_angles_deg": "test_principal_angles_top_deg",
    "snorm_topk": "test_snorm_topk",
}


def _copy_metric_aliases(metric_dict: Dict[str, Any]) -> Dict[str, Any]:
    for alias_key, source_key in LEGACY_ALIAS_SOURCES.items():
        if source_key in metric_dict:
            metric_dict[alias_key] = metric_dict[source_key]
    for alias_key, source_key in ARRAY_ALIAS_SOURCES.items():
        if source_key in metric_dict:
            metric_dict[alias_key] = metric_dict[source_key]
    return metric_dict


def _metric_mean_key(base_name: str) -> str:
    return f"{base_name}_mean"


def _metric_std_key(base_name: str) -> str:
    return f"{base_name}_std"


def _metric_ci95_low_key(base_name: str) -> str:
    return f"{base_name}_ci95_low"


def _metric_ci95_high_key(base_name: str) -> str:
    return f"{base_name}_ci95_high"


def _safe_spectral_gap(singular_values: np.ndarray, theory_dim: int) -> float:
    idx_2k = theory_dim - 1
    if idx_2k < 0 or idx_2k >= len(singular_values):
        return np.nan
    sigma_2k = singular_values[idx_2k]
    sigma_next = singular_values[idx_2k + 1] if (idx_2k + 1) < len(singular_values) else 0.0
    return float(sigma_2k / (sigma_next + 1e-12))


def _compute_representation_metrics(
    h_np: np.ndarray,
    cfg: ExperimentConfig,
    signal_info: Dict[str, Any],
    theoretical_rank: int,
    target_indices: np.ndarray,
) -> Dict[str, Any]:
    """Compute rank and SVD-based subspace diagnostics for one feature matrix."""

    _, singular_values, _ = np.linalg.svd(h_np, full_matrices=False)
    rank_metrics = calculate_rank_metrics(singular_values, threshold=cfg.RANK_THRESHOLD)
    rank_threshold = rank_metrics["rank_threshold"]
    rank_entropy = rank_metrics["rank_entropy"]

    rank_gap = abs(rank_threshold - theoretical_rank)
    rel_rank_gap = rank_gap / (theoretical_rank + 1e-12)

    energy_ratio_2k = topk_energy_ratio(singular_values, top_k=theoretical_rank)
    spectral_gap_2k = _safe_spectral_gap(singular_values, theoretical_rank)

    s_norm = singular_values / (singular_values[0] + 1e-12) if len(singular_values) > 0 else np.array([])
    topk = np.full(cfg.SCREE_TOPK, np.nan, dtype=np.float64)
    ncopy = min(cfg.SCREE_TOPK, len(s_norm))
    topk[:ncopy] = s_norm[:ncopy]

    align_metrics = calculate_subspace_alignment_metrics_v2(
        h_np,
        time_mode=cfg.time_mode,
        target_indices=target_indices,
        freqs=signal_info.get("freqs"),
        thetas=signal_info.get("thetas"),
        dt=cfg.DT,
        theory_dim=theoretical_rank,
    )

    metrics = {
        "rank_threshold": rank_threshold,
        "rank_entropy": rank_entropy,
        "rank_gap": float(rank_gap),
        "rel_rank_gap": float(rel_rank_gap),
        "spectral_gap_2k": spectral_gap_2k,
        "energy_ratio_2k": energy_ratio_2k,
        "energy_top_theory_dim": align_metrics["energy_top_theory_dim"],
        "align_coverage_full": align_metrics["align_coverage_full"],
        "align_purity_full": align_metrics["align_purity_full"],
        "align_mean_cosine_full": align_metrics["align_mean_cosine_full"],
        "align_mean_angle_deg_full": align_metrics["align_mean_angle_deg_full"],
        "principal_angles_full_deg": align_metrics["principal_angles_full_deg"],
        "align_coverage_top": align_metrics["align_coverage_top"],
        "align_purity_top": align_metrics["align_purity_top"],
        "align_mean_cosine_top": align_metrics["align_mean_cosine_top"],
        "align_mean_angle_deg_top": align_metrics["align_mean_angle_deg_top"],
        "principal_angles_top_deg": align_metrics["principal_angles_top_deg"],
        "recon_r2_qf_from_h": align_metrics["recon_r2_qf_from_h"],
        "fourier_theoretical_dim": float(align_metrics["theory_dim"]),
        "fourier_numerical_dim": float(align_metrics["f_numerical_dim"]),
        "fourier_min_singular_value": align_metrics["f_min_singular_value"],
        "fourier_condition_number": align_metrics["f_condition_number"],
        "h_numerical_dim": float(align_metrics["h_numerical_dim"]),
        "align_coverage": align_metrics["align_coverage_full"],
        "align_purity": align_metrics["align_purity_full"],
        "alignment_score_2k": align_metrics["align_coverage_top"],
        "align_mean_cosine": align_metrics["align_mean_cosine_top"],
        "mean_principal_angle_deg": align_metrics["align_mean_angle_deg_top"],
        "principal_angles_deg": align_metrics["principal_angles_top_deg"],
        "snorm_topk": topk,
    }
    return metrics


def _merge_prefixed_metrics(
    destination: Dict[str, Any],
    prefix: str,
    metrics: Dict[str, Any],
) -> None:
    for key, value in metrics.items():
        destination[f"{prefix}_{key}"] = value


def _collect_scalar_metric_keys(cfg: ExperimentConfig) -> List[str]:
    scalar_keys = [
        "train_mse",
        "train_mae",
        "train_acc",
        "train_r2",
        "test_mse",
        "test_mae",
        "test_acc",
        "test_r2",
    ]
    rep_suffixes = [
        "rank_threshold",
        "rank_entropy",
        "rank_gap",
        "rel_rank_gap",
        "spectral_gap_2k",
        "energy_ratio_2k",
        "energy_top_theory_dim",
        "align_coverage_full",
        "align_purity_full",
        "align_mean_cosine_full",
        "align_mean_angle_deg_full",
        "align_coverage_top",
        "align_purity_top",
        "align_mean_cosine_top",
        "align_mean_angle_deg_top",
        "recon_r2_qf_from_h",
        "fourier_theoretical_dim",
        "fourier_numerical_dim",
        "fourier_min_singular_value",
        "fourier_condition_number",
        "h_numerical_dim",
    ]
    for prefix in ("train", "test"):
        scalar_keys.extend(f"{prefix}_{suffix}" for suffix in rep_suffixes)

    if cfg.USE_NOISE:
        scalar_keys += [
            "train_input_snr_db",
            "train_output_snr_db",
            "train_snr_gain_db",
            "test_input_snr_db",
            "test_output_snr_db",
            "test_snr_gain_db",
        ]
    return scalar_keys


def _copy_summary_aliases(summary_dict: Dict[str, Any]) -> Dict[str, Any]:
    for alias_key, source_key in LEGACY_ALIAS_SOURCES.items():
        for suffix_key in ("mean", "std", "ci95_low", "ci95_high"):
            source_summary_key = f"{source_key}_{suffix_key}"
            alias_summary_key = f"{alias_key}_{suffix_key}"
            if source_summary_key in summary_dict:
                summary_dict[alias_summary_key] = summary_dict[source_summary_key]
    if "test_principal_angles_top_deg_all" in summary_dict:
        summary_dict["principal_angles_deg_all"] = summary_dict["test_principal_angles_top_deg_all"]
    return summary_dict


def train_one_seed(
    cfg: ExperimentConfig,
    x_train: torch.Tensor,
    y_train_target: torch.Tensor,
    train_target_indices: np.ndarray,
    x_test: torch.Tensor,
    y_test_target: torch.Tensor,
    test_target_indices: np.ndarray,
    y_clean_train: torch.Tensor,
    y_clean_test: torch.Tensor,
    y_noisy_train: torch.Tensor,
    y_noisy_test: torch.Tensor,
    signal_info: Dict[str, Any],
    theoretical_rank: int,
    bottleneck_dim: int,
    seed: int,
) -> Dict[str, Any]:
    """Train one model instance with a single training seed."""

    set_seed(seed)

    model = build_model(cfg.MODEL_ID, cfg.LAG, cfg.HIDDEN_DIM, bottleneck_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(cfg.EPOCHS):
        optimizer.zero_grad()
        y_hat, _ = model(x_train)
        loss = criterion(y_hat, y_train_target)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_train, h_train = model(x_train)
        y_pred_test, h_test = model(x_test)
        h_train_np = h_train.detach().cpu().numpy()
        h_test_np = h_test.detach().cpu().numpy()

        if cfg.NORMALIZE_H_COLUMNS:
            h_train_np = normalize_feature_columns(h_train_np)
            h_test_np = normalize_feature_columns(h_test_np)

        train_rep = _compute_representation_metrics(
            h_train_np,
            cfg=cfg,
            signal_info=signal_info,
            theoretical_rank=theoretical_rank,
            target_indices=train_target_indices,
        )
        test_rep = _compute_representation_metrics(
            h_test_np,
            cfg=cfg,
            signal_info=signal_info,
            theoretical_rank=theoretical_rank,
            target_indices=test_target_indices,
        )

        result: Dict[str, Any] = {
            "train_mse": float(criterion(y_pred_train, y_train_target).item()),
            "train_mae": float(nn.L1Loss()(y_pred_train, y_train_target).item()),
            "train_acc": regression_accuracy(y_train_target, y_pred_train, tol=cfg.ACC_TOLERANCE),
            "train_r2": regression_r2(y_train_target, y_pred_train),
            "test_mse": float(criterion(y_pred_test, y_test_target).item()),
            "test_mae": float(nn.L1Loss()(y_pred_test, y_test_target).item()),
            "test_acc": regression_accuracy(y_test_target, y_pred_test, tol=cfg.ACC_TOLERANCE),
            "test_r2": regression_r2(y_test_target, y_pred_test),
        }
        _merge_prefixed_metrics(result, "train", train_rep)
        _merge_prefixed_metrics(result, "test", test_rep)

        if cfg.USE_NOISE:
            train_input_snr = snr_db_from_tensors(y_clean_train, y_noisy_train)
            train_output_snr = snr_db_from_tensors(y_clean_train, y_pred_train)
            test_input_snr = snr_db_from_tensors(y_clean_test, y_noisy_test)
            test_output_snr = snr_db_from_tensors(y_clean_test, y_pred_test)
            result["train_input_snr_db"] = float(train_input_snr)
            result["train_output_snr_db"] = float(train_output_snr)
            result["train_snr_gain_db"] = float(train_output_snr - train_input_snr)
            result["test_input_snr_db"] = float(test_input_snr)
            result["test_output_snr_db"] = float(test_output_snr)
            result["test_snr_gain_db"] = float(test_output_snr - test_input_snr)

        _copy_metric_aliases(result)
    return result


def aggregate_seed_results(
    seed_results: List[Dict[str, Any]],
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    """Aggregate per-seed metrics into mean/std summaries."""

    scalar_keys = _collect_scalar_metric_keys(cfg)
    output: Dict[str, Any] = {}
    for key in scalar_keys:
        values = [float(result[key]) for result in seed_results]
        mean_value, std_value, ci95_low, ci95_high = mean_std_ci95(values)
        output[_metric_mean_key(key)] = mean_value
        output[_metric_std_key(key)] = std_value
        output[_metric_ci95_low_key(key)] = ci95_low
        output[_metric_ci95_high_key(key)] = ci95_high

    array_keys = [
        "train_principal_angles_full_deg",
        "train_principal_angles_top_deg",
        "test_principal_angles_full_deg",
        "test_principal_angles_top_deg",
        "snorm_topk",
    ]
    for key in array_keys:
        stacked = [np.asarray(result[key]) for result in seed_results]
        if key == "snorm_topk":
            output[f"{key}_all"] = np.vstack(stacked)
        else:
            output[f"{key}_all"] = np.concatenate(stacked)

    _copy_summary_aliases(output)
    return output


def make_summary_dataframe(
    cfg: ExperimentConfig,
    set_rows: List[Dict[str, Any]],
    overall_summary: Dict[str, Any],
) -> pd.DataFrame:
    """Create a tabular summary for per-set and overall experiment results."""

    if len(set_rows) == 0:
        return pd.DataFrame([{"set_idx": "overall", **overall_summary}])

    df = pd.DataFrame(set_rows)
    meta_cols = [
        "set_idx",
        "freqs",
        "thetas",
        "amplitudes",
        "phases",
        "actual_snr_db",
        "min_delta_f",
        "min_delta_theta",
        "n_train_targets",
        "n_val_targets",
        "n_test_targets",
    ]
    preferred_metric_bases = [
        "train_mse",
        "test_mse",
        "train_r2",
        "test_r2",
        "train_align_coverage_full",
        "test_align_coverage_full",
        "train_align_purity_full",
        "test_align_purity_full",
        "train_align_coverage_top",
        "test_align_coverage_top",
        "train_recon_r2_qf_from_h",
        "test_recon_r2_qf_from_h",
        "train_align_mean_angle_deg_full",
        "test_align_mean_angle_deg_full",
        "train_energy_top_theory_dim",
        "test_energy_top_theory_dim",
        "fourier_theoretical_dim",
        "fourier_numerical_dim",
        "fourier_min_singular_value",
        "fourier_condition_number",
        "rank_threshold",
        "rank_entropy",
        "mse",
        "test_mse",
        "align_coverage",
        "alignment_score_2k",
    ]
    suffixes = ["mean", "std", "ci95_low", "ci95_high"]
    ordered_cols = [col for col in meta_cols if col in df.columns]
    for base_name in preferred_metric_bases:
        for suffix_key in suffixes:
            column_name = f"{base_name}_{suffix_key}"
            if column_name in df.columns and column_name not in ordered_cols:
                ordered_cols.append(column_name)

    scalar_summary_cols = [
        column
        for column in df.columns
        if any(column.endswith(f"_{suffix_key}") for suffix_key in suffixes)
        and not column.endswith("_all")
    ]
    remaining_cols = [column for column in scalar_summary_cols if column not in ordered_cols]
    summary_df = df[ordered_cols + remaining_cols].copy()

    overall_row = {column: None for column in summary_df.columns}
    overall_row["set_idx"] = "overall"
    overall_row.update(overall_summary)
    return pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)


def plot_results(results: Dict[str, Any], cfg: ExperimentConfig) -> None:
    """Plot a compact test-focused summary for the current experiment."""

    summary_df = results["summary_df"]
    set_df = summary_df[summary_df["set_idx"] != "overall"].copy()
    if len(set_df) == 0:
        return

    if "test_mse_mean" in set_df.columns:
        plt.figure(figsize=(8, 4))
        plt.errorbar(set_df["set_idx"], set_df["test_mse_mean"], yerr=set_df.get("test_mse_std"), marker="o")
        plt.xlabel("set_idx")
        plt.ylabel("Test MSE")
        plt.title("Test MSE by Set")
        plt.grid(True, alpha=0.3)
        plt.show()

    if "test_align_coverage_full_mean" in set_df.columns:
        plt.figure(figsize=(8, 4))
        plt.errorbar(
            set_df["set_idx"],
            set_df["test_align_coverage_full_mean"],
            yerr=set_df.get("test_align_coverage_full_std"),
            marker="o",
        )
        plt.xlabel("set_idx")
        plt.ylabel("Test Coverage (Full)")
        plt.title("Test Subspace Coverage by Set")
        plt.grid(True, alpha=0.3)
        plt.show()

    if "test_recon_r2_qf_from_h_mean" in set_df.columns:
        plt.figure(figsize=(8, 4))
        plt.errorbar(
            set_df["set_idx"],
            set_df["test_recon_r2_qf_from_h_mean"],
            yerr=set_df.get("test_recon_r2_qf_from_h_std"),
            marker="o",
        )
        plt.xlabel("set_idx")
        plt.ylabel("Recon R^2")
        plt.title("Basis Reconstruction R^2 by Set")
        plt.grid(True, alpha=0.3)
        plt.show()


def print_overall_summary(overall_summary: Dict[str, Any], cfg: ExperimentConfig) -> None:
    """Print the aggregate metrics in a notebook-friendly format."""

    print("\n=== Overall summary (mean of set-level means plus set-level std) ===")
    metrics = [
        "train_mse",
        "test_mse",
        "train_r2",
        "test_r2",
        "test_align_coverage_full",
        "test_align_purity_full",
        "test_align_coverage_top",
        "test_align_mean_angle_deg_full",
        "test_recon_r2_qf_from_h",
        "fourier_theoretical_dim",
        "fourier_numerical_dim",
        "fourier_min_singular_value",
        "fourier_condition_number",
        "rank_threshold",
        "rank_entropy",
    ]
    if cfg.USE_NOISE:
        metrics += ["test_input_snr_db", "test_output_snr_db", "test_snr_gain_db"]

    for metric in metrics:
        mean_val = overall_summary.get(_metric_mean_key(metric), np.nan)
        std_val = overall_summary.get(_metric_std_key(metric), np.nan)
        print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")


def print_overall_ci95_summary(overall_summary: Dict[str, Any], cfg: ExperimentConfig) -> None:
    """Print 95% confidence intervals for the aggregate metrics."""

    print("\n=== Overall 95% Confidence Intervals ===")
    metrics = [
        "train_mse",
        "test_mse",
        "train_r2",
        "test_r2",
        "test_align_coverage_full",
        "test_align_purity_full",
        "test_align_coverage_top",
        "test_align_mean_angle_deg_full",
        "test_recon_r2_qf_from_h",
        "fourier_theoretical_dim",
        "fourier_numerical_dim",
        "fourier_min_singular_value",
        "fourier_condition_number",
        "rank_threshold",
        "rank_entropy",
    ]
    if cfg.USE_NOISE:
        metrics += ["test_input_snr_db", "test_output_snr_db", "test_snr_gain_db"]

    for metric in metrics:
        ci95_low_val = overall_summary.get(_metric_ci95_low_key(metric), np.nan)
        ci95_high_val = overall_summary.get(_metric_ci95_high_key(metric), np.nan)
        print(f"{metric}: [{ci95_low_val:.6f}, {ci95_high_val:.6f}]")


def run_experiment(cfg: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """Run the full experiment pipeline with raw-split and absolute-index diagnostics."""

    cfg = ExperimentConfig() if cfg is None else cfg
    cfg_dict = asdict(cfg)
    config_df = pd.DataFrame(list(cfg_dict.items()), columns=["hyperparameter", "value"])

    validate_config(cfg)
    set_seed(cfg.GLOBAL_SEED)

    if cfg.VERBOSE:
        print("\n=== Experiment hyperparameters ===")
        for key, value in cfg_dict.items():
            print(f"{key}: {value}")
        print("=" * 50)

    theoretical_rank = 2 * cfg.NUM_FREQS
    bottleneck_dim = (
        cfg.BOTTLENECK_DIM_OVERRIDE
        if cfg.BOTTLENECK_DIM_OVERRIDE is not None
        else cfg.BOTTLENECK_MULTIPLIER * cfg.NUM_FREQS
    )

    if cfg.VERBOSE:
        if cfg.time_mode == "continuous":
            print(
                f"--- Start experiment (mode=continuous, bottleneck={bottleneck_dim}, "
                f"num_freqs={cfg.NUM_FREQS}, theoretical_rank={theoretical_rank}, "
                f"freq_range=[{cfg.FREQ_MIN}, {cfg.FREQ_MAX}], dt={cfg.DT}) ---"
            )
        else:
            print(
                f"--- Start experiment (mode=discrete, bottleneck={bottleneck_dim}, "
                f"num_freqs={cfg.NUM_FREQS}, theoretical_rank={theoretical_rank}, "
                f"theta_range=[{cfg.theta_min:.4f}, {cfg.theta_max:.4f}], "
                f"min_delta_theta={cfg.MIN_DELTA_THETA:.4f}) ---"
            )

    set_rows: List[Dict[str, Any]] = []
    all_principal_angles_deg: List[np.ndarray] = []
    all_snorm_topk: List[np.ndarray] = []

    for experiment_idx in range(cfg.NUM_EXPERIMENTS):
        freqs_rng = random.Random(cfg.DATA_SEED_BASE + experiment_idx)
        rng_np = np.random.default_rng(cfg.DATA_SEED_BASE + experiment_idx)
        clean_data, signal_info = generate_sin_data(cfg, freqs_rng, rng_np)

        freqs = signal_info["freqs"]
        thetas = signal_info["thetas"]
        amplitudes = signal_info["amplitudes"]
        phases = signal_info["phases"]

        if cfg.USE_NOISE:
            noise_rng = np.random.default_rng(cfg.NOISE_SEED_BASE + experiment_idx)
            noisy_data, _, actual_snr = add_noise_to_signal(
                clean_signal=clean_data,
                snr_db=cfg.SNR_DB,
                noise_type=cfg.NOISE_TYPE,
                rng=noise_rng,
                ar1_rho=cfg.AR1_RHO,
                impulse_prob=cfg.IMPULSE_PROB,
                impulse_scale=cfg.IMPULSE_SCALE,
            )
        else:
            noisy_data = clean_data.copy()
            actual_snr = np.inf

        (clean_train_raw, noisy_train_raw), (clean_val_raw, noisy_val_raw), (clean_test_raw, noisy_test_raw), split_bounds = split_raw_series_arrays(
            clean_data,
            noisy_data,
            test_ratio=cfg.TEST_RATIO,
            val_ratio=cfg.VAL_RATIO,
        )

        train_start_idx = split_bounds["train"][0]
        val_start_idx = split_bounds["val"][0]
        test_start_idx = split_bounds["test"][0]

        x_train, y_noisy_train, train_target_indices = make_dataset(
            noisy_train_raw,
            lag=cfg.LAG,
            start_idx=train_start_idx,
            return_target_indices=True,
        )
        _, y_clean_train, _ = make_dataset(
            clean_train_raw,
            lag=cfg.LAG,
            start_idx=train_start_idx,
            return_target_indices=True,
        )
        x_test, y_noisy_test, test_target_indices = make_dataset(
            noisy_test_raw,
            lag=cfg.LAG,
            start_idx=test_start_idx,
            return_target_indices=True,
        )
        _, y_clean_test, _ = make_dataset(
            clean_test_raw,
            lag=cfg.LAG,
            start_idx=test_start_idx,
            return_target_indices=True,
        )

        if len(clean_val_raw) > cfg.LAG:
            _, _, val_target_indices = make_dataset(
                clean_val_raw,
                lag=cfg.LAG,
                start_idx=val_start_idx,
                return_target_indices=True,
            )
            n_val_targets = int(len(val_target_indices))
        else:
            n_val_targets = 0

        if x_train.shape[0] == 0:
            raise ValueError("Training split produced no lag windows. Increase SEQ_LEN or reduce LAG.")
        if x_test.shape[0] == 0:
            raise ValueError("Test split produced no lag windows. Increase SEQ_LEN or reduce LAG.")

        y_train_target = y_clean_train if (cfg.USE_NOISE and cfg.TRAIN_TARGET == "clean") else y_noisy_train
        y_test_target = y_clean_test if (cfg.USE_NOISE and cfg.TRAIN_TARGET == "clean") else y_noisy_test

        seed_results = []
        for seed_idx in range(cfg.SEEDS_PER_FREQ):
            seed_result = train_one_seed(
                cfg=cfg,
                x_train=x_train,
                y_train_target=y_train_target,
                train_target_indices=train_target_indices,
                x_test=x_test,
                y_test_target=y_test_target,
                test_target_indices=test_target_indices,
                y_clean_train=y_clean_train,
                y_clean_test=y_clean_test,
                y_noisy_train=y_noisy_train,
                y_noisy_test=y_noisy_test,
                signal_info=signal_info,
                theoretical_rank=theoretical_rank,
                bottleneck_dim=bottleneck_dim,
                seed=cfg.TRAIN_SEED_BASE + seed_idx,
            )
            seed_results.append(seed_result)

        aggregated = aggregate_seed_results(seed_results, cfg)
        if "test_principal_angles_top_deg_all" in aggregated:
            all_principal_angles_deg.append(aggregated["test_principal_angles_top_deg_all"])
        if "snorm_topk_all" in aggregated:
            all_snorm_topk.append(aggregated["snorm_topk_all"])

        row = {
            "set_idx": experiment_idx + 1,
            "freqs": freqs,
            "thetas": thetas,
            "amplitudes": amplitudes,
            "phases": phases,
            "actual_snr_db": float(actual_snr),
            "min_delta_f": min_delta_f(freqs),
            "min_delta_theta": min_delta_theta(thetas),
            "n_train_targets": int(len(train_target_indices)),
            "n_val_targets": int(n_val_targets),
            "n_test_targets": int(len(test_target_indices)),
            **aggregated,
        }
        set_rows.append(row)

        if cfg.VERBOSE:
            if cfg.time_mode == "continuous":
                print(f"set {experiment_idx + 1:2d}: freqs={freqs}")
            else:
                print(f"set {experiment_idx + 1:2d}: thetas={np.round(np.array(thetas), 6)}")
            print(f"  n_train_targets            = {len(train_target_indices)}")
            print(f"  n_val_targets              = {n_val_targets}")
            print(f"  n_test_targets             = {len(test_target_indices)}")
            print(f"  test_mse                   = {aggregated['test_mse_mean']:.6f} ± {aggregated['test_mse_std']:.6f}")
            print(f"  test_r2                    = {aggregated['test_r2_mean']:.6f} ± {aggregated['test_r2_std']:.6f}")
            print(
                f"  test_align_coverage_full   = "
                f"{aggregated['test_align_coverage_full_mean']:.4f} ± "
                f"{aggregated['test_align_coverage_full_std']:.4f}"
            )
            print(
                f"  test_align_coverage_top    = "
                f"{aggregated['test_align_coverage_top_mean']:.4f} ± "
                f"{aggregated['test_align_coverage_top_std']:.4f}"
            )
            print(
                f"  test_recon_r2_qf_from_h    = "
                f"{aggregated['test_recon_r2_qf_from_h_mean']:.4f} ± "
                f"{aggregated['test_recon_r2_qf_from_h_std']:.4f}"
            )
            print(
                f"  fourier_dim (theory / num) = "
                f"{aggregated['fourier_theoretical_dim_mean']:.2f} / "
                f"{aggregated['fourier_numerical_dim_mean']:.2f}"
            )
            print(
                f"  fourier_condition_number   = "
                f"{aggregated['fourier_condition_number_mean']:.4f} ± "
                f"{aggregated['fourier_condition_number_std']:.4f}"
            )

    overall_summary: Dict[str, Any] = {}
    if len(set_rows) > 0:
        mean_metric_bases = sorted(
            {
                key[:-5]
                for key, value in set_rows[0].items()
                if key.endswith("_mean") and np.isscalar(value)
            }
        )
        for metric_base in mean_metric_bases:
            values = [float(row[_metric_mean_key(metric_base)]) for row in set_rows]
            mean_value, std_value, ci95_low, ci95_high = mean_std_ci95(values)
            overall_summary[_metric_mean_key(metric_base)] = mean_value
            overall_summary[_metric_std_key(metric_base)] = std_value
            overall_summary[_metric_ci95_low_key(metric_base)] = ci95_low
            overall_summary[_metric_ci95_high_key(metric_base)] = ci95_high

    summary_df = make_summary_dataframe(cfg, set_rows, overall_summary)

    results = {
        "config_df": config_df,
        "config": asdict(cfg),
        "theoretical_rank": theoretical_rank,
        "bottleneck_dim": bottleneck_dim,
        "set_results": set_rows,
        "overall_summary": overall_summary,
        "summary_df": summary_df,
        "all_principal_angles_deg": (
            np.concatenate(all_principal_angles_deg) if all_principal_angles_deg else np.array([])
        ),
        "all_snorm_topk": (
            np.vstack(all_snorm_topk) if all_snorm_topk else np.empty((0, cfg.SCREE_TOPK))
        ),
    }

    if cfg.VERBOSE:
        print_overall_summary(overall_summary, cfg)
        print_overall_ci95_summary(overall_summary, cfg)
        if cfg.NORMALIZE_H_COLUMNS:
            print("--- feature matrix H column normalization: ON ---")
        else:
            print("--- feature matrix H column normalization: OFF ---")

    if cfg.MAKE_PLOTS:
        plot_results(results, cfg)

    return results
