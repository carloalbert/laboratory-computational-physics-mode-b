#!/usr/bin/env python3
"""Checks for the shared-environment plus two-AR q likelihood."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_environment_fast_slow_ar_q_glu import ar_covariance, log_likelihood  # noqa: E402


class EnvironmentFastSlowLikelihoodTests(unittest.TestCase):
    def test_structured_likelihood_matches_full_gaussian(self):
        rng = np.random.default_rng(17)
        lineages, horizon = 4, 5
        theta = np.array(
            [-0.52, -0.10, 0.90, np.log(0.07), np.log(0.012), np.log(0.015)]
        )
        lineage_covariance = ar_covariance(
            horizon, theta[1], np.exp(theta[3]), theta[2], np.exp(theta[4])
        )
        full_covariance = np.kron(np.eye(lineages), lineage_covariance) + np.kron(
            np.ones((lineages, lineages)),
            np.exp(2.0 * theta[5]) * np.eye(horizon),
        )
        sample = rng.multivariate_normal(
            np.full(lineages * horizon, theta[0]), full_covariance
        ).reshape(lineages, horizon)
        structured = log_likelihood(sample, theta)
        brute_force = multivariate_normal.logpdf(
            sample.ravel(),
            mean=np.full(lineages * horizon, theta[0]),
            cov=full_covariance,
        )
        self.assertAlmostEqual(structured, brute_force, places=10)


if __name__ == "__main__":
    unittest.main()
