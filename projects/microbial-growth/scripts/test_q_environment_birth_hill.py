#!/usr/bin/env python3
"""Checks for the explicit-birth Hill extension used with the q environment."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from autoregressive_residual_model import (  # noqa: E402
    profile_birth_dependent_hazard_log_likelihood,
    profile_hazard_log_likelihood,
    sample_birth_dependent_growth_time,
)


class BirthHillTests(unittest.TestCase):
    def test_kappa_zero_is_exactly_the_three_parameter_hazard(self):
        birth = np.asarray([1.2, 2.0, 3.5])
        alpha = np.asarray([0.4, 0.6, 0.8])
        time = np.asarray([1.5, 0.9, 0.7])
        old, old_omega = profile_hazard_log_likelihood(
            np.log(birth), alpha, time, math.log(1.4), math.log(7.0), order=64
        )
        new, new_omega = profile_birth_dependent_hazard_log_likelihood(
            np.log(birth),
            alpha,
            time,
            math.log(1.4),
            math.log(7.0),
            0.0,
            math.log(2.0),
            order=64,
        )
        self.assertAlmostEqual(old, new, places=11)
        self.assertAlmostEqual(old_omega, new_omega, places=11)

    def test_positive_kappa_makes_large_births_divide_after_less_growth(self):
        rng = np.random.default_rng(9182)
        small = sample_birth_dependent_growth_time(
            np.full(4000, math.log(1.0)),
            math.log(1.5),
            10.0,
            7.0,
            3.0,
            math.log(2.0),
            rng,
        )
        large = sample_birth_dependent_growth_time(
            np.full(4000, math.log(4.0)),
            math.log(1.5),
            10.0,
            7.0,
            3.0,
            math.log(2.0),
            rng,
        )
        self.assertLess(float(large.mean()), float(small.mean()))


if __name__ == "__main__":
    unittest.main()
