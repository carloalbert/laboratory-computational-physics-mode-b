#!/usr/bin/env python3
"""Checks for the shared-E plus lineage-offset AR(1) q likelihood."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_environment_lineage_offset_ar_q_glu import (  # noqa: E402
    mode_and_log_density,
    q_covariance,
)


class EnvironmentLineageOffsetLikelihoodTests(unittest.TestCase):
    def test_profiled_density_matches_direct_gaussian_evaluation(self):
        sequences = [
            np.asarray([-0.50, -0.60, -0.45]),
            np.asarray([-0.55, -0.52]),
        ]
        theta = np.asarray(
            [-0.52, 0.15, math.log(0.02), math.log(0.08), math.log(0.01)]
        )
        fitted, structured = mode_and_log_density(theta, sequences, 3)
        effects = fitted["posterior_mean_E"]
        direct = multivariate_normal.logpdf(
            effects, mean=np.zeros(3), cov=np.eye(3) * math.exp(2.0 * theta[4])
        )
        for sequence in sequences:
            direct += multivariate_normal.logpdf(
                sequence,
                mean=theta[0] + effects[: len(sequence)],
                cov=q_covariance(
                    len(sequence), theta[1], math.exp(theta[2]), math.exp(theta[3])
                ),
            )
        self.assertAlmostEqual(structured, direct, places=10)


if __name__ == "__main__":
    unittest.main()
