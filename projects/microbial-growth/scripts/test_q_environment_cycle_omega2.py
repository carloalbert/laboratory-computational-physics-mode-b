#!/usr/bin/env python3
"""Checks for the hierarchical cycle-specific omega2 model."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_q_environment_cycle_omega2_glu import (  # noqa: E402
    CycleRateLikelihood,
    draw_cycle_log_omega2,
)


def group(alpha, time, birth=2.0):
    return pd.DataFrame({
        "generation": np.arange(len(alpha)), "alpha": alpha,
        "time": time, "length_birth": np.full(len(alpha), birth),
    })


class CycleRateTests(unittest.TestCase):
    def setUp(self):
        self.prefixes = [
            group([0.5, 0.5, 0.5], [1.0, 1.1, 1.0]),
            group([0.6, 0.6, 0.6], [0.8, 0.9, 0.8]),
        ]

    def test_nearly_zero_random_effect_matches_shared_likelihood(self):
        likelihood = CycleRateLikelihood(
            self.prefixes, integration_order=48, random_effect_order=32
        )
        shared = np.log([1.5, 10.0, 7.0])
        marginal = likelihood.marginal_log_likelihood(
            np.r_[shared, math.log(1e-6)]
        )
        self.assertAlmostEqual(
            marginal, likelihood.shared_log_likelihood(shared), places=8
        )

    def test_posterior_probabilities_and_draws_are_valid(self):
        likelihood = CycleRateLikelihood(self.prefixes)
        theta = np.asarray([math.log(1.5), math.log(10.0), math.log(7.0), math.log(0.2)])
        nodes, probabilities = likelihood.posterior(theta)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
        fit = {
            "posterior_nodes_log_omega2": nodes,
            "posterior_probabilities": probabilities,
        }
        draws = draw_cycle_log_omega2(fit, 50, np.random.default_rng(21))
        self.assertEqual(draws.shape, (50, 2))
        self.assertTrue(np.all(np.isin(draws, nodes)))


if __name__ == "__main__":
    unittest.main()
