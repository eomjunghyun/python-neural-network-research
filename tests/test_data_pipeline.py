import unittest

import numpy as np

from src import ExperimentConfig, run_experiment
from src.data_utils import make_dataset, split_raw_series_arrays


class DataPipelineTests(unittest.TestCase):
    def test_make_dataset_can_return_absolute_target_indices(self) -> None:
        data = np.arange(10, dtype=np.float32)
        x, y, target_indices = make_dataset(
            data,
            lag=3,
            start_idx=20,
            return_target_indices=True,
        )

        self.assertEqual(x.shape, (7, 3))
        self.assertEqual(y.shape, (7, 1))
        np.testing.assert_array_equal(target_indices, np.arange(23, 30, dtype=np.int64))

    def test_split_raw_series_arrays_returns_chronological_bounds(self) -> None:
        a = np.arange(20, dtype=np.float32)
        b = -a

        train_arrays, val_arrays, test_arrays, bounds = split_raw_series_arrays(
            a,
            b,
            test_ratio=0.2,
            val_ratio=0.2,
        )

        self.assertEqual(bounds["train"], (0, 12))
        self.assertEqual(bounds["val"], (12, 16))
        self.assertEqual(bounds["test"], (16, 20))
        np.testing.assert_array_equal(train_arrays[0], a[:12])
        np.testing.assert_array_equal(val_arrays[1], b[12:16])
        np.testing.assert_array_equal(test_arrays[0], a[16:])

    def test_run_experiment_uses_new_alignment_keys_and_safe_overall_suffixes(self) -> None:
        cfg = ExperimentConfig(
            time_mode="discrete",
            MODEL_ID="AN003_LINEAR",
            NUM_FREQS=2,
            SEQ_LEN=64,
            LAG=8,
            HIDDEN_DIM=16,
            BOTTLENECK_MULTIPLIER=2,
            EPOCHS=2,
            NUM_EXPERIMENTS=1,
            SEEDS_PER_FREQ=1,
            VAL_RATIO=0.2,
            TEST_RATIO=0.2,
            VERBOSE=False,
            MAKE_PLOTS=False,
        )
        results = run_experiment(cfg)
        overall_summary = results["overall_summary"]
        summary_df = results["summary_df"]

        self.assertIn("test_align_coverage_full_mean", overall_summary)
        self.assertIn("test_recon_r2_qf_from_h_mean", overall_summary)
        self.assertIn("fourier_condition_number_mean", overall_summary)
        self.assertIn("align_mean_cosine_mean", overall_summary)
        self.assertIn("align_mean_cosine_std", overall_summary)
        self.assertNotIn("align_std_cosine_std", overall_summary)
        self.assertIn("test_align_coverage_full_mean", summary_df.columns)
        self.assertIn("test_recon_r2_qf_from_h_mean", summary_df.columns)


if __name__ == "__main__":
    unittest.main()
