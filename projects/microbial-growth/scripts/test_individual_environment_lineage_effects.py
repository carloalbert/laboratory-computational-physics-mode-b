#!/usr/bin/env python3
"""Checks for the explicit A_l/E_i posterior used by the Glu q model."""

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_individual_environment_lineage_effects_glu import (  # noqa: E402
    apply_environment_gauge,
    ar_covariance,
    joint_effect_posterior,
)


class IndividualEffectPosteriorTests(unittest.TestCase):
    def test_structured_posterior_matches_direct_linear_gaussian_result(self):
        q = np.asarray([[-0.50, -0.48, 0.0], [-0.55, -0.57, -0.52]])
        lengths = np.asarray([2, 3])
        fit = {
            "mean": -0.53,
            "rho": 0.2,
            "sigma_lineage_offset": 0.03,
            "stationary_ar_sd": 0.08,
            "environment_prior_sd": 0.015,
        }
        result = joint_effect_posterior(q, lengths, fit)

        rows, values, covariance = [], [], []
        for lineage, length in enumerate(lengths):
            for generation in range(length):
                row = np.zeros(5)
                row[lineage] = 1.0
                row[2 + generation] = 1.0
                rows.append(row)
                values.append(q[lineage, generation])
            covariance.append(ar_covariance(length, fit["rho"], fit["stationary_ar_sd"]))
        x = np.asarray(rows)
        noise = np.zeros((sum(lengths), sum(lengths)))
        start = 0
        for block in covariance:
            stop = start + len(block)
            noise[start:stop, start:stop] = block
            start = stop
        prior_mean = np.asarray([fit["mean"], fit["mean"], 0.0, 0.0, 0.0])
        prior_cov = np.diag(
            [fit["sigma_lineage_offset"] ** 2] * 2
            + [fit["environment_prior_sd"] ** 2] * 3
        )
        direct_cov = np.linalg.inv(np.linalg.inv(prior_cov) + x.T @ np.linalg.inv(noise) @ x)
        direct_mean = direct_cov @ (
            np.linalg.inv(prior_cov) @ prior_mean
            + x.T @ np.linalg.inv(noise) @ np.asarray(values)
        )
        np.testing.assert_allclose(result["mean"], direct_mean, atol=1e-12)
        np.testing.assert_allclose(result["covariance"], direct_cov, atol=1e-12)

    def test_gauge_preserves_all_additive_predictors(self):
        a = np.asarray([[-0.5, -0.6], [-0.4, -0.7]])
        e = np.asarray([[0.02, -0.01, 0.03], [-0.01, 0.04, 0.02]])
        weights = np.asarray([3.0, 2.0, 1.0])
        gauged_a, gauged_e = apply_environment_gauge(a, e, weights)
        np.testing.assert_allclose(
            a[:, :, None] + e[:, None, :],
            gauged_a[:, :, None] + gauged_e[:, None, :],
        )
        np.testing.assert_allclose(gauged_e @ weights, 0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
