#!/usr/bin/env python3
"""Checks for the individually inferred shared alpha environment."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_individual_q_alpha_environment_effects_glu import (  # noqa: E402
    alpha_mode_and_log_density,
)


class AlphaEnvironmentTests(unittest.TestCase):
    def test_posterior_mean_and_covariance_match_direct_gaussian_result(self):
        sequences = [
            np.asarray([-0.2, 0.1, 0.3]),
            np.asarray([0.0, 0.2]),
        ]
        rho, sigma_g = 0.35, 0.2
        fitted, _ = alpha_mode_and_log_density(
            np.asarray([rho, math.log(sigma_g)]), sequences, 3
        )
        residual_sd = math.sqrt(1.0 - sigma_g**2)
        precision = np.eye(3) / sigma_g**2
        score = np.zeros(3)
        for values in sequences:
            n = len(values)
            distance = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
            inverse = np.linalg.inv(residual_sd**2 * rho**distance)
            precision[:n, :n] += inverse
            score[:n] += inverse @ values
        covariance = np.linalg.inv(precision)
        mean = covariance @ score
        np.testing.assert_allclose(fitted["posterior_mean_G"], mean, atol=1e-12)
        np.testing.assert_allclose(fitted["posterior_covariance_G"], covariance, atol=1e-12)

    def test_latent_variance_is_preserved(self):
        fitted, _ = alpha_mode_and_log_density(
            np.asarray([0.4, math.log(0.3)]), [np.asarray([0.0, 0.1])], 2
        )
        total = (
            fitted["environment_prior_sd"] ** 2
            + fitted["stationary_residual_sd"] ** 2
        )
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_marginal_likelihood_matches_full_gaussian(self):
        sequences = [np.asarray([-0.2, 0.1]), np.asarray([0.0, 0.2])]
        rho, sigma_g = 0.35, 0.2
        fitted, structured = alpha_mode_and_log_density(
            np.asarray([rho, math.log(sigma_g)]), sequences, 2
        )
        residual_sd = fitted["stationary_residual_sd"]
        residual = residual_sd**2 * rho ** np.abs(
            np.subtract.outer(np.arange(2), np.arange(2))
        )
        covariance = np.block(
            [
                [residual + sigma_g**2 * np.eye(2), sigma_g**2 * np.eye(2)],
                [sigma_g**2 * np.eye(2), residual + sigma_g**2 * np.eye(2)],
            ]
        )
        direct = multivariate_normal.logpdf(
            np.concatenate(sequences), mean=np.zeros(4), cov=covariance
        )
        self.assertAlmostEqual(structured, direct, places=10)


if __name__ == "__main__":
    unittest.main()
