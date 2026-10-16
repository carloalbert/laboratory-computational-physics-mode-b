import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from autoregressive_residual_model import (  # noqa: E402
    beta_regression_log_likelihood,
    lineage_acf,
    profile_hazard_log_likelihood,
    sample_growth_time,
)


class AutoregressiveResidualModelTests(unittest.TestCase):
    def test_hazard_sampler_matches_profile_density_scale(self):
        rng = np.random.default_rng(81)
        logb = np.full(4000, math.log(2.4))
        samples = sample_growth_time(logb, math.log(1.1), 8.0, 3.0, rng)
        self.assertTrue(np.all(samples > 0))
        self.assertGreater(samples.mean(), 0.05)
        value, omega = profile_hazard_log_likelihood(
            logb, np.ones(len(logb)), samples, math.log(1.1), math.log(3.0)
        )
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(omega, 8.0, delta=0.35)

    def test_beta_regression_prefers_correct_slope(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=4000)
        mean = 1.0 / (1.0 + np.exp(-(-0.3 + 0.7 * x)))
        f = rng.beta(mean * 80.0, (1.0 - mean) * 80.0)
        correct = beta_regression_log_likelihood(f, x, 0.0, -0.3, 0.7, math.log(80))
        wrong = beta_regression_log_likelihood(f, x, 0.0, -0.3, 0.0, math.log(80))
        self.assertGreater(correct, wrong)

    def test_short_lineage_acf_shape(self):
        values = np.array([[1.0, 2.0, 3.0, 0.0], [2.0, 1.0, 2.0, 1.0]])
        result = lineage_acf(values, np.array([3, 4]), max_lag=3)
        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(result[0, 0], 1.0)
        self.assertTrue(np.all(np.isfinite(result[0, :3])))


if __name__ == "__main__":
    unittest.main()
