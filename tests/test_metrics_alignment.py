import unittest

import numpy as np

from src.metrics import (
    build_sampled_basis_matrix_from_indices,
    build_sampled_discrete_basis_matrix,
    build_sampled_discrete_basis_matrix_from_indices,
    build_sampled_fourier_matrix,
    build_sampled_fourier_matrix_from_indices,
    calculate_sampled_basis_numerical_dim,
    calculate_subspace_alignment_metrics,
    calculate_subspace_alignment_metrics_v2,
)


class MetricsAlignmentTests(unittest.TestCase):
    def test_continuous_basis_wrapper_matches_indices_version(self) -> None:
        freqs = (2, 5)
        dt = 0.1
        lag = 3
        seq_len = 8
        target_indices = np.arange(seq_len, dtype=np.int64) + lag

        old_matrix = build_sampled_fourier_matrix(freqs=freqs, dt=dt, lag=lag, seq_len=seq_len)
        new_matrix = build_sampled_fourier_matrix_from_indices(
            freqs=freqs,
            dt=dt,
            target_indices=target_indices,
        )

        np.testing.assert_allclose(old_matrix, new_matrix)

    def test_discrete_basis_wrapper_matches_indices_version(self) -> None:
        thetas = (0.2 * np.pi, 0.6 * np.pi)
        lag = 4
        seq_len = 9
        target_indices = np.arange(seq_len, dtype=np.int64) + lag

        old_matrix = build_sampled_discrete_basis_matrix(
            thetas=thetas,
            lag=lag,
            seq_len=seq_len,
        )
        new_matrix = build_sampled_discrete_basis_matrix_from_indices(
            thetas=thetas,
            target_indices=target_indices,
        )

        np.testing.assert_allclose(old_matrix, new_matrix)

    def test_same_subspace_alignment_is_high(self) -> None:
        rng = np.random.default_rng(0)
        target_indices = np.arange(80, dtype=np.int64) + 5
        freqs = (2, 7)
        F = build_sampled_basis_matrix_from_indices(
            time_mode="continuous",
            target_indices=target_indices,
            freqs=freqs,
            dt=0.05,
        )
        mixing = rng.standard_normal((F.shape[1], F.shape[1])) + np.eye(F.shape[1])
        H = F @ mixing

        metrics = calculate_subspace_alignment_metrics_v2(
            H,
            time_mode="continuous",
            target_indices=target_indices,
            freqs=freqs,
            dt=0.05,
        )

        self.assertGreater(metrics["align_coverage_full"], 0.999999)
        self.assertGreater(metrics["align_coverage_top"], 0.999999)
        self.assertGreater(metrics["recon_r2_qf_from_h"], 0.999999)
        self.assertEqual(metrics["theory_dim"], 4)
        self.assertEqual(metrics["f_numerical_dim"], 4)

    def test_random_low_dim_h_has_lower_reconstruction(self) -> None:
        rng = np.random.default_rng(1)
        target_indices = np.arange(80, dtype=np.int64) + 3
        thetas = (0.15 * np.pi, 0.45 * np.pi)
        H = rng.standard_normal((80, 2))

        metrics = calculate_subspace_alignment_metrics_v2(
            H,
            time_mode="discrete",
            target_indices=target_indices,
            thetas=thetas,
        )

        self.assertLess(metrics["recon_r2_qf_from_h"], 0.8)
        self.assertEqual(metrics["theory_dim"], 4)
        self.assertLessEqual(metrics["h_numerical_dim"], 2)

    def test_backward_compatibility_wrapper_returns_legacy_aliases(self) -> None:
        freqs = (1, 4)
        lag = 2
        seq_len = 40
        dt = 0.1
        H = build_sampled_fourier_matrix(freqs=freqs, dt=dt, lag=lag, seq_len=seq_len)

        metrics = calculate_subspace_alignment_metrics(
            H,
            freqs=freqs,
            dt=dt,
            lag=lag,
            top_k=4,
            time_mode="continuous",
        )

        self.assertIn("align_coverage", metrics)
        self.assertIn("align_purity", metrics)
        self.assertIn("alignment_score_2k", metrics)
        self.assertIn("align_mean_cosine", metrics)
        self.assertIn("mean_principal_angle_deg", metrics)
        self.assertIn("principal_angles_deg", metrics)
        self.assertAlmostEqual(metrics["align_coverage"], metrics["align_coverage_full"])
        self.assertAlmostEqual(metrics["align_purity"], metrics["align_purity_full"])
        self.assertAlmostEqual(metrics["alignment_score_2k"], metrics["align_coverage_top"])
        np.testing.assert_allclose(metrics["principal_angles_deg"], metrics["principal_angles_top_deg"])

    def test_basis_numerical_dim_reports_theory_and_diagnostics(self) -> None:
        metrics = calculate_sampled_basis_numerical_dim(
            time_mode="discrete",
            thetas=(0.2 * np.pi, 0.6 * np.pi),
            lag=1,
            seq_len=50,
        )

        self.assertEqual(metrics["fourier_theoretical_dim"], 4)
        self.assertEqual(metrics["fourier_numerical_dim"], 4)
        self.assertIn("fourier_singular_values", metrics)
        self.assertIn("fourier_min_singular_value", metrics)
        self.assertIn("fourier_condition_number", metrics)


if __name__ == "__main__":
    unittest.main()
