import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_joint_alpha_q_glu import cross_correlation_replicates  # noqa: E402


class JointAlphaQTests(unittest.TestCase):
    def test_cross_correlation_lag_convention(self):
        alpha = np.array([[1.0, 2.0, 3.0, 4.0]])
        q = np.array([[0.0, 1.0, 2.0, 3.0]])
        values = cross_correlation_replicates(
            alpha, q, np.array([4]), np.array([-1, 0, 1])
        )
        self.assertEqual(values.shape, (1, 3))
        np.testing.assert_allclose(values, 1.0)


if __name__ == "__main__":
    unittest.main()
