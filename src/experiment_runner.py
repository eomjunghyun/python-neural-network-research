from __future__ import annotations

from dataclasses import asdict
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .common_utils import mean_std_ci95, set_seed, validate_config
from .config import ExperimentConfig
from .data_utils import add_noise_to_signal, generate_sin_data, make_dataset, split_train_test_tensors
from .metrics import (
    calculate_rank_metrics,
    calculate_subspace_alignment_metrics,
    min_delta_f,
    normalize_feature_columns,
    regression_accuracy,
    regression_r2,
    snr_db_from_tensors,
    topk_energy_ratio,
)
from .models import build_model


def train_one_seed(
    cfg: ExperimentConfig,
    x_train: torch.Tensor,
    y_train_target: torch.Tensor,
    x_test: torch.Tensor,
    y_test_target: torch.Tensor,
    y_clean_test: torch.Tensor,
    y_noisy_test: torch.Tensor,
    freqs: Tuple[int, ...],
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
        y_pred_train, _ = model(x_train)
        y_pred_test, h_test = model(x_test)
        h_np = h_test.detach().cpu().numpy()

        if cfg.NORMALIZE_H_COLUMNS:
            h_np = normalize_feature_columns(h_np)

        train_mse = float(criterion(y_pred_train, y_train_target).item())
        train_mae = float(nn.L1Loss()(y_pred_train, y_train_target).item())
        train_acc = regression_accuracy(y_train_target, y_pred_train, tol=cfg.ACC_TOLERANCE)

        test_mse = float(criterion(y_pred_test, y_test_target).item())
        test_mae = float(nn.L1Loss()(y_pred_test, y_test_target).item())
        test_acc = regression_accuracy(y_test_target, y_pred_test, tol=cfg.ACC_TOLERANCE)
        test_r2 = regression_r2(y_test_target, y_pred_test)

        _, singular_values, _ = np.linalg.svd(h_np, full_matrices=False)
        rank_metrics = calculate_rank_metrics(singular_values, threshold=cfg.RANK_THRESHOLD)
        rank_threshold = rank_metrics["rank_threshold"]
        rank_entropy = rank_metrics["rank_entropy"]

        rank_gap = abs(rank_threshold - theoretical_rank)
        rel_rank_gap = rank_gap / (theoretical_rank + 1e-12)

        idx_2k = theoretical_rank - 1
        if idx_2k < len(singular_values):
            sigma_2k = singular_values[idx_2k]
            sigma_next = singular_values[idx_2k + 1] if (idx_2k + 1) < len(singular_values) else 0.0
            spectral_gap_2k = float(sigma_2k / (sigma_next + 1e-12))
        else:
            spectral_gap_2k = np.nan

        energy_ratio_2k = topk_energy_ratio(singular_values, top_k=theoretical_rank)
        s_norm = singular_values / (singular_values[0] + 1e-12)
        topk = np.full(cfg.SCREE_TOPK, np.nan, dtype=np.float64)
        ncopy = min(cfg.SCREE_TOPK, len(s_norm))
        topk[:ncopy] = s_norm[:ncopy]

        align_metrics = calculate_subspace_alignment_metrics(
            h_np,
            freqs,
            dt=cfg.DT,
            lag=cfg.LAG,
            top_k=theoretical_rank,
        )

        result = {
            "mse": train_mse,
            "mae": train_mae,
            "acc": train_acc,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_acc": test_acc,
            "test_r2": test_r2,
            "rank_threshold": rank_threshold,
            "rank_entropy": rank_entropy,
            "rank_gap": float(rank_gap),
            "rel_rank_gap": float(rel_rank_gap),
            "spectral_gap_2k": spectral_gap_2k,
            "energy_ratio_2k": energy_ratio_2k,
            "align_coverage": align_metrics["align_coverage"],
            "align_purity": align_metrics["align_purity"],
            "alignment_score_2k": align_metrics["alignment_score_2k"],
            "align_mean_cosine": align_metrics["align_mean_cosine"],
            "mean_principal_angle_deg": align_metrics["mean_principal_angle_deg"],
            "principal_angles_deg": align_metrics["principal_angles_deg"],
            "snorm_topk": topk,
        }

        if cfg.USE_NOISE:
            input_snr = snr_db_from_tensors(y_clean_test, y_noisy_test)
            output_snr = snr_db_from_tensors(y_clean_test, y_pred_test)
            result["input_snr_db"] = float(input_snr)
            result["output_snr_db"] = float(output_snr)
            result["snr_gain_db"] = float(output_snr - input_snr)

    return result


def aggregate_seed_results(
    seed_results: List[Dict[str, Any]],
    cfg: ExperimentConfig,
) -> Dict[str, Any]:
    """Aggregate per-seed metrics into mean/std summaries."""

    scalar_keys = [
        "mse",
        "mae",
        "acc",
        "test_mse",
        "test_mae",
        "test_acc",
        "test_r2",
        "rank_threshold",
        "rank_entropy",
        "rank_gap",
        "rel_rank_gap",
        "spectral_gap_2k",
        "energy_ratio_2k",
        "align_coverage",
        "align_purity",
        "alignment_score_2k",
        "align_mean_cosine",
        "mean_principal_angle_deg",
    ]
    if cfg.USE_NOISE:
        scalar_keys += ["input_snr_db", "output_snr_db", "snr_gain_db"]

    output: Dict[str, Any] = {}
    for key in scalar_keys:
        values = [float(result[key]) for result in seed_results]
        mean_value, std_value, ci95_low, ci95_high = mean_std_ci95(values)
        output[f"{key}_mean"] = mean_value
        output[f"{key}_std"] = std_value
        output[f"{key}_ci95_low"] = ci95_low
        output[f"{key}_ci95_high"] = ci95_high

    output["principal_angles_deg_all"] = np.concatenate(
        [np.asarray(result["principal_angles_deg"]) for result in seed_results]
    )
    output["snorm_topk_all"] = np.vstack(
        [np.asarray(result["snorm_topk"]) for result in seed_results]
    )
    return output


def make_summary_dataframe(
    cfg: ExperimentConfig,
    set_rows: List[Dict[str, Any]],
    overall_summary: Dict[str, Any],
) -> pd.DataFrame:
    """Create a tabular summary for per-set and overall experiment results."""

    cols = [
        "set_idx",
        "freqs",
        "amplitudes",
        "phases",
        "actual_snr_db",
        "min_delta_f",
        "mse_mean",
        "mse_std",
        "mse_ci95_low",
        "mse_ci95_high",
        "mae_mean",
        "mae_std",
        "mae_ci95_low",
        "mae_ci95_high",
        "acc_mean",
        "acc_std",
        "acc_ci95_low",
        "acc_ci95_high",
        "test_mse_mean",
        "test_mse_std",
        "test_mse_ci95_low",
        "test_mse_ci95_high",
        "test_mae_mean",
        "test_mae_std",
        "test_mae_ci95_low",
        "test_mae_ci95_high",
        "test_acc_mean",
        "test_acc_std",
        "test_acc_ci95_low",
        "test_acc_ci95_high",
        "test_r2_mean",
        "test_r2_std",
        "test_r2_ci95_low",
        "test_r2_ci95_high",
        "rank_threshold_mean",
        "rank_threshold_std",
        "rank_threshold_ci95_low",
        "rank_threshold_ci95_high",
        "rank_entropy_mean",
        "rank_entropy_std",
        "rank_entropy_ci95_low",
        "rank_entropy_ci95_high",
        "rank_gap_mean",
        "rank_gap_std",
        "rank_gap_ci95_low",
        "rank_gap_ci95_high",
        "rel_rank_gap_mean",
        "rel_rank_gap_std",
        "rel_rank_gap_ci95_low",
        "rel_rank_gap_ci95_high",
        "spectral_gap_2k_mean",
        "spectral_gap_2k_std",
        "spectral_gap_2k_ci95_low",
        "spectral_gap_2k_ci95_high",
        "energy_ratio_2k_mean",
        "energy_ratio_2k_std",
        "energy_ratio_2k_ci95_low",
        "energy_ratio_2k_ci95_high",
        "align_coverage_mean",
        "align_coverage_std",
        "align_coverage_ci95_low",
        "align_coverage_ci95_high",
        "align_purity_mean",
        "align_purity_std",
        "align_purity_ci95_low",
        "align_purity_ci95_high",
        "alignment_score_2k_mean",
        "alignment_score_2k_std",
        "alignment_score_2k_ci95_low",
        "alignment_score_2k_ci95_high",
        "align_mean_cosine_mean",
        "align_mean_cosine_std",
        "align_mean_cosine_ci95_low",
        "align_mean_cosine_ci95_high",
        "mean_principal_angle_deg_mean",
        "mean_principal_angle_deg_std",
        "mean_principal_angle_deg_ci95_low",
        "mean_principal_angle_deg_ci95_high",
    ]
    if cfg.USE_NOISE:
        cols += [
            "input_snr_db_mean",
            "input_snr_db_std",
            "input_snr_db_ci95_low",
            "input_snr_db_ci95_high",
            "output_snr_db_mean",
            "output_snr_db_std",
            "output_snr_db_ci95_low",
            "output_snr_db_ci95_high",
            "snr_gain_db_mean",
            "snr_gain_db_std",
            "snr_gain_db_ci95_low",
            "snr_gain_db_ci95_high",
        ]

    df = pd.DataFrame(set_rows)
    available_cols = [col for col in cols if col in df.columns]
    summary_df = df[available_cols].copy()

    overall_row = {"set_idx": "overall", "freqs": None, "amplitudes": None, "phases": None}
    overall_row.update(overall_summary)
    return pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)


def plot_results(results: Dict[str, Any], cfg: ExperimentConfig) -> None:
    """Plot a minimal summary matching the notebook baseline."""

    summary_df = results["summary_df"]
    set_df = summary_df[summary_df["set_idx"] != "overall"].copy()
    if len(set_df) == 0:
        return

    plt.figure(figsize=(8, 4))
    plt.errorbar(set_df["set_idx"], set_df["mse_mean"], yerr=set_df["mse_std"], marker="o")
    plt.xlabel("set_idx")
    plt.ylabel("Train MSE")
    plt.title("Train MSE by Set")
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.errorbar(
        set_df["set_idx"],
        set_df["align_coverage_mean"],
        yerr=set_df["align_coverage_std"],
        marker="o",
    )
    plt.xlabel("set_idx")
    plt.ylabel("Coverage")
    plt.title("Subspace Coverage by Set")
    plt.grid(True, alpha=0.3)
    plt.show()


def print_overall_summary(overall_summary: Dict[str, Any], cfg: ExperimentConfig) -> None:
    """Print the aggregate metrics in a notebook-friendly format."""

    print("\n=== Overall summary (mean of set-level means plus set-level std) ===")
    metrics = [
        "mse",
        "mae",
        "acc",
        "test_mse",
        "test_mae",
        "test_acc",
        "test_r2",
        "rank_threshold",
        "rank_entropy",
        "rank_gap",
        "rel_rank_gap",
        "spectral_gap_2k",
        "energy_ratio_2k",
        "align_coverage",
        "align_purity",
        "alignment_score_2k",
        "align_mean_cosine",
        "mean_principal_angle_deg",
    ]
    if cfg.USE_NOISE:
        metrics += ["input_snr_db", "output_snr_db", "snr_gain_db"]

    for metric in metrics:
        mean_key = metric + "_mean"
        std_key = metric + "_std"
        mean_val = overall_summary.get(mean_key, np.nan)
        std_val = overall_summary.get(std_key, np.nan)
        print(f"{metric}: {mean_val:.6f} ± {std_val:.6f}")

def print_overall_ci95_summary(overall_summary: Dict[str, Any], cfg: ExperimentConfig) -> None:
    """Print 95% confidence intervals for the aggregate metrics."""

    print("\n=== Overall 95% Confidence Intervals ===")
    metrics = [
        "mse",
        "mae",
        "acc",
        "test_mse",
        "test_mae",
        "test_acc",
        "test_r2",
        "rank_threshold",
        "rank_entropy",
        "rank_gap",
        "rel_rank_gap",
        "spectral_gap_2k",
        "energy_ratio_2k",
        "align_coverage",
        "align_purity",
        "alignment_score_2k",
        "align_mean_cosine",
        "mean_principal_angle_deg",
    ]
    if cfg.USE_NOISE:
        metrics += ["input_snr_db", "output_snr_db", "snr_gain_db"]

    for metric in metrics:
        ci95_low_key = metric + "_ci95_low"
        ci95_high_key = metric + "_ci95_high"
        ci95_low_val = overall_summary.get(ci95_low_key, np.nan)
        ci95_high_val = overall_summary.get(ci95_high_key, np.nan)
        print(f"{metric}: [{ci95_low_val:.6f}, {ci95_high_val:.6f}]")


def run_experiment(cfg: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """Run the full notebook experiment pipeline."""

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
    bottleneck_dim = cfg.BOTTLENECK_MULTIPLIER * cfg.NUM_FREQS

    if cfg.VERBOSE:
        print(
            f"--- Start experiment (bottleneck={bottleneck_dim}, "
            f"num_freqs={cfg.NUM_FREQS}, theoretical_rank={theoretical_rank}, "
            f"freq_range=[{cfg.FREQ_MIN}, {cfg.FREQ_MAX}], dt={cfg.DT}) ---"
        )
        if cfg.RANDOM_AMPLITUDE:
            print(f"--- amplitude randomization: U[{cfg.AMP_MIN}, {cfg.AMP_MAX}] ---")
        else:
            print("--- amplitude randomization: OFF (all amplitudes = 1) ---")
        if cfg.RANDOM_PHASE:
            print(f"--- phase randomization: U[{cfg.PHASE_MIN}, {cfg.PHASE_MAX}] ---")
        else:
            print("--- phase randomization: OFF (all phases = 0) ---")
        if cfg.USE_NOISE:
            print(
                f"--- noise type={cfg.NOISE_TYPE}, target_snr_db={cfg.SNR_DB}, "
                f"train_target={cfg.TRAIN_TARGET} ---"
            )

    set_rows: List[Dict[str, Any]] = []
    all_principal_angles_deg: List[np.ndarray] = []
    all_snorm_topk: List[np.ndarray] = []

    for experiment_idx in range(cfg.NUM_EXPERIMENTS):
        freqs_rng = random.Random(cfg.DATA_SEED_BASE + experiment_idx)
        rng_np = np.random.default_rng(cfg.DATA_SEED_BASE + experiment_idx)
        clean_data, signal_info = generate_sin_data(cfg, freqs_rng, rng_np)

        freqs = signal_info["freqs"]
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

        x_all, y_noisy_all = make_dataset(noisy_data, lag=cfg.LAG)
        _, y_clean_all = make_dataset(clean_data, lag=cfg.LAG)
        y_target_all = y_clean_all if (cfg.USE_NOISE and cfg.TRAIN_TARGET == "clean") else y_noisy_all

        (
            x_train,
            x_test,
            y_train,
            y_test,
            _y_clean_train,
            y_clean_test,
            _y_noisy_train,
            y_noisy_test,
        ) = split_train_test_tensors(
            x_all,
            y_target_all,
            y_clean_all,
            y_noisy_all,
            test_ratio=cfg.TEST_RATIO,
        )

        seed_results = []
        for seed_idx in range(cfg.SEEDS_PER_FREQ):
            seed_result = train_one_seed(
                cfg=cfg,
                x_train=x_train,
                y_train_target=y_train,
                x_test=x_test,
                y_test_target=y_test,
                y_clean_test=y_clean_test,
                y_noisy_test=y_noisy_test,
                freqs=freqs,
                theoretical_rank=theoretical_rank,
                bottleneck_dim=bottleneck_dim,
                seed=cfg.TRAIN_SEED_BASE + seed_idx,
            )
            seed_results.append(seed_result)

        aggregated = aggregate_seed_results(seed_results, cfg)
        all_principal_angles_deg.append(aggregated["principal_angles_deg_all"])
        all_snorm_topk.append(aggregated["snorm_topk_all"])

        row = {
            "set_idx": experiment_idx + 1,
            "freqs": freqs,
            "amplitudes": amplitudes,
            "phases": phases,
            "actual_snr_db": float(actual_snr),
            "min_delta_f": min_delta_f(freqs),
            **aggregated,
        }
        set_rows.append(row)

        if cfg.VERBOSE:
            print(f"set {experiment_idx + 1:2d}: freqs={freqs}")
            print(f"  amplitudes      = {np.round(np.array(amplitudes), 4)}")
            if cfg.RANDOM_PHASE:
                print(f"  phases          = {np.round(np.array(phases), 4)}")
            print(f"  test_ratio      = {cfg.TEST_RATIO:.2f}")
            if cfg.USE_NOISE:
                print(f"  actual_snr_db   = {actual_snr:.2f}")
                print(
                    f"  input_snr_db    = {aggregated['input_snr_db_mean']:.3f} ± "
                    f"{aggregated['input_snr_db_std']:.3f}"
                )
                print(
                    f"  output_snr_db   = {aggregated['output_snr_db_mean']:.3f} ± "
                    f"{aggregated['output_snr_db_std']:.3f}"
                )
                print(
                    f"  snr_gain_db     = {aggregated['snr_gain_db_mean']:.3f} ± "
                    f"{aggregated['snr_gain_db_std']:.3f}"
                )
            print(f"  Train MSE       = {aggregated['mse_mean']:.6f} ± {aggregated['mse_std']:.6f}")
            print(f"  Train MAE       = {aggregated['mae_mean']:.6f} ± {aggregated['mae_std']:.6f}")
            print(f"  Acc(|err|<=0.1) = {aggregated['acc_mean']:.4f} ± {aggregated['acc_std']:.4f}")
            print(f"  Test MSE        = {aggregated['test_mse_mean']:.6f} ± {aggregated['test_mse_std']:.6f}")
            print(f"  Test MAE        = {aggregated['test_mae_mean']:.6f} ± {aggregated['test_mae_std']:.6f}")
            print(f"  Test Acc        = {aggregated['test_acc_mean']:.4f} ± {aggregated['test_acc_std']:.4f}")
            print(f"  Test R^2        = {aggregated['test_r2_mean']:.6f} ± {aggregated['test_r2_std']:.6f}")
            print(
                f"  Rank(th={cfg.RANK_THRESHOLD}) = "
                f"{aggregated['rank_threshold_mean']:.2f} ± {aggregated['rank_threshold_std']:.2f}"
            )
            print(
                f"  Rank(entropy)   = "
                f"{aggregated['rank_entropy_mean']:.2f} ± {aggregated['rank_entropy_std']:.2f}"
            )
            print(f"  rank_gap        = {aggregated['rank_gap_mean']:.2f} ± {aggregated['rank_gap_std']:.2f}")
            print(
                f"  rel_rank_gap    = "
                f"{aggregated['rel_rank_gap_mean']:.4f} ± {aggregated['rel_rank_gap_std']:.4f}"
            )
            print(
                f"  spectral_gap_2k = "
                f"{aggregated['spectral_gap_2k_mean']:.4f} ± {aggregated['spectral_gap_2k_std']:.4f}"
            )
            print(
                f"  energy_ratio_2k = "
                f"{aggregated['energy_ratio_2k_mean']:.4f} ± {aggregated['energy_ratio_2k_std']:.4f}"
            )
            print(
                f"  coverage        = "
                f"{aggregated['align_coverage_mean']:.4f} ± {aggregated['align_coverage_std']:.4f}"
            )
            print(
                f"  purity          = "
                f"{aggregated['align_purity_mean']:.4f} ± {aggregated['align_purity_std']:.4f}"
            )
            print(
                f"  alignment_2k    = "
                f"{aggregated['alignment_score_2k_mean']:.4f} ± {aggregated['alignment_score_2k_std']:.4f}"
            )
            print(
                f"  cosine          = "
                f"{aggregated['align_mean_cosine_mean']:.4f} ± {aggregated['align_mean_cosine_std']:.4f}"
            )
            print(
                f"  mean_angle_deg  = "
                f"{aggregated['mean_principal_angle_deg_mean']:.2f} ± "
                f"{aggregated['mean_principal_angle_deg_std']:.2f}"
            )

    overall_summary: Dict[str, Any] = {}
    scalar_summary_keys = [
        "mse_mean",
        "mae_mean",
        "acc_mean",
        "test_mse_mean",
        "test_mae_mean",
        "test_acc_mean",
        "test_r2_mean",
        "rank_threshold_mean",
        "rank_entropy_mean",
        "rank_gap_mean",
        "rel_rank_gap_mean",
        "spectral_gap_2k_mean",
        "energy_ratio_2k_mean",
        "align_coverage_mean",
        "align_purity_mean",
        "alignment_score_2k_mean",
        "align_mean_cosine_mean",
        "mean_principal_angle_deg_mean",
    ]
    if cfg.USE_NOISE:
        scalar_summary_keys += ["input_snr_db_mean", "output_snr_db_mean", "snr_gain_db_mean"]

    for key in scalar_summary_keys:
        values = [float(row[key]) for row in set_rows]
        mean_value, std_value, ci95_low, ci95_high = mean_std_ci95(values)
        overall_summary[key] = mean_value
        overall_summary[key.replace("_mean", "_std")] = std_value
        overall_summary[key.replace("_mean", "_ci95_low")] = ci95_low
        overall_summary[key.replace("_mean", "_ci95_high")] = ci95_high

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
