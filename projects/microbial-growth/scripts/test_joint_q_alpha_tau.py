import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_joint_q_alpha_tau_glu import (  # noqa: E402
    cross_correlation_replicates,
    empirical_quantile,
    normal_score,
)


class JointQAlphaTauTests(unittest.TestCase):
    def test_empirical_transform_round_trip_at_observations(self):
        reference = np.array([1.0, 2.0, 3.0, 4.0])
        score = normal_score(reference, reference)
        recovered = empirical_quantile(
            __import__("scipy").stats.norm.cdf(score), reference
        )
        np.testing.assert_allclose(recovered, reference, atol=0.4)

    def test_cross_correlation_lag_convention(self):
        left = np.array([[1.0, 2.0, 3.0, 4.0]])
        right = np.array([[0.0, 1.0, 2.0, 3.0]])
        values = cross_correlation_replicates(
            left, right, np.array([4]), np.array([-1, 0, 1])
        )
        np.testing.assert_allclose(values, 1.0)


if __name__ == "__main__":
    unittest.main()
