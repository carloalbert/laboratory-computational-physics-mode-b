import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from autoregressive_residual_model import random_intercept_ar1_log_likelihood  # noqa: E402


class LineageOffsetARQTests(unittest.TestCase):
    def test_likelihood_recovers_profiled_mean(self):
        sequences = [np.array([-0.5, -0.4, -0.45]), np.array([-0.7, -0.6, -0.65])]
        generations = [np.arange(3), np.arange(3)]
        value, mean = random_intercept_ar1_log_likelihood(
            sequences, generations, 0.1, 0.3, 0.05
        )
        _, shifted_mean = random_intercept_ar1_log_likelihood(
            [values + 0.7 for values in sequences],
            generations,
            0.1,
            0.3,
            0.05,
        )
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(shifted_mean - mean, 0.7, places=12)

    def test_gap_resets_ar_but_not_offset(self):
        values = [np.array([-0.5, -0.4, -0.6])]
        contiguous = [np.array([0, 1, 2])]
        gapped = [np.array([0, 1, 8])]
        first, _ = random_intercept_ar1_log_likelihood(
            values, contiguous, 0.08, 0.7, 0.05
        )
        second, _ = random_intercept_ar1_log_likelihood(
            values, gapped, 0.08, 0.7, 0.05
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
