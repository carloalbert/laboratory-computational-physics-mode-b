"""Measured-input replay with a slowly evolving latent division-size scale.

The current physical parameters remain unchanged, except that the scale used
for the division event in generation n is

    u_n = u * exp(z_n - sigma_z**2/2),
    z_(n+1) = phi*z_n + sigma_z*sqrt(1-phi**2)*xi_n,

where xi_n is standard normal.  Thus z is stationary N(0, sigma_z**2) and the
mean multiplicative scale factor is one.  The measured alpha and f sequences
are replayed exactly.
"""

from __future__ import annotations

import math

import numpy as np

from loglogistic_endpoint_residual import (
    EPS,
    sample_added_size,
    sample_valid_residual,
)


def simulate_measured_input_replay(
    physical_params: np.ndarray,
    phi: float,
    sigma_z: float,
    lineage_inputs: list[tuple[float, np.ndarray, np.ndarray]],
    seed: int,
) -> list[dict[str, float | int]]:
    """Simulate lineages while replaying their measured alpha and f histories."""
    if not (0.0 <= phi < 1.0 and sigma_z >= 0.0):
        raise ValueError("Require 0 <= phi < 1 and sigma_z >= 0")
    (
        u,
        omega2,
        eta,
        _,
        _,
        _,
        _,
        mu_epsilon,
        sigma_epsilon,
    ) = map(float, physical_params)
    rng = np.random.default_rng(seed)
    innovation_scale = sigma_z * math.sqrt(max(1.0 - phi**2, 0.0))
    mean_correction = 0.5 * sigma_z**2
    rows: list[dict[str, float | int]] = []
    for lineage, (initial_birth, alphas, fractions) in enumerate(lineage_inputs):
        birth = float(initial_birth)
        z = float(rng.normal(0.0, sigma_z))
        for generation, alpha in enumerate(alphas):
            effective_u = u * math.exp(np.clip(z - mean_correction, -20.0, 20.0))
            added = sample_added_size(
                birth, effective_u, omega2, eta, rng
            )
            final = birth + added
            endpoint_log_growth = math.log1p(added / birth)
            epsilon = sample_valid_residual(
                endpoint_log_growth,
                mu_epsilon,
                sigma_epsilon,
                rng,
            )
            division_time = (endpoint_log_growth - epsilon) / float(alpha)
            rows.append(
                {
                    "lineage": lineage,
                    "generation": generation,
                    "time": division_time,
                    "alpha": float(alpha),
                    "f": float(fractions[generation]),
                    "length_birth": birth,
                    "length_final": final,
                    "added_size": added,
                    "epsilon": epsilon,
                    "latent_log_scale": z,
                    "effective_u": effective_u,
                }
            )
            if generation + 1 < len(alphas):
                birth = max(float(fractions[generation + 1]) * final, EPS)
                z = phi * z + innovation_scale * float(rng.normal())
    return rows
