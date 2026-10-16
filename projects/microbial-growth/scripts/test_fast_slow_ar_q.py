import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from autoregressive_residual_model import two_ar1_log_likelihood  # noqa: E402


class FastSlowARQTests(unittest.TestCase):
    def test_finite_and_translation_equivariant(self):
        sequences = [np.array([-0.5, -0.4, -0.45]), np.array([-0.7, -0.6, -0.65])]
        generations = [np.arange(3), np.arange(3)]
        value, mean = two_ar1_log_likelihood(
            sequences, generations, 0.0, 0.07, 0.9, 0.01
        )
        _, shifted = two_ar1_log_likelihood(
            [sequence + 0.4 for sequence in sequences],
            generations,
            0.0,
            0.07,
            0.9,
            0.01,
        )
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(shifted - mean, 0.4, places=12)

    def test_gap_changes_covariance(self):
        values = [np.array([-0.5, -0.4, -0.6])]
        first, _ = two_ar1_log_likelihood(
            values, [np.array([0, 1, 2])], 0.1, 0.07, 0.9, 0.01
        )
        second, _ = two_ar1_log_likelihood(
            values, [np.array([0, 1, 8])], 0.1, 0.07, 0.9, 0.01
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
