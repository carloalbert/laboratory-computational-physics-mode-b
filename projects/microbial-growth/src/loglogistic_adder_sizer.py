"""Normalized three-parameter log-logistic mixed adder-sizer model.

The conditional survival law is

    S(p | b) = (1 + (eta*b/u)**omega2)
               / (1 + ((p + eta*b)/u)**omega2).

It is the original modified-Burr survival shifted by ``eta*b``, conditioned on
survival at birth, and restricted to outer Burr exponent one.  The fitted
division parameters are therefore ``u``, ``omega2`` and ``eta``.
"""

from __future__ import annotations

import math

import numpy as np

from weibull_adder_sizer import (
    EPS,
    added_size,
    beta_ab,
    gamma_shape_scale,
    log_beta_pdf_mean_sd,
    log_gamma_pdf_mean_sd,
)


def _log1p_power(x: np.ndarray, exponent: float) -> np.ndarray:
    """Return log(1 + x**exponent) without overflowing."""
    x = np.asarray(x, dtype=float)
    log_x = np.log(np.maximum(x, EPS))
    return np.logaddexp(0.0, exponent * log_x)


def cumulative_hazard(
    p: np.ndarray,
    length_birth: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
) -> np.ndarray:
    """Conditional cumulative hazard, normalized to zero at p=0."""
    p = np.asarray(p, dtype=float)
    length_birth = np.asarray(length_birth, dtype=float)
    if not (u > 0.0 and omega2 > 0.0 and 0.0 <= eta <= 1.0):
        return np.full(np.broadcast(p, length_birth).shape, np.inf)
    x = np.maximum(p + eta * length_birth, EPS) / u
    x0 = np.maximum(eta * length_birth, 0.0) / u
    value = _log1p_power(x, omega2) - _log1p_power(x0, omega2)
    return np.maximum(value, 0.0)


def log_survival(
    p: np.ndarray,
    length_birth: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
) -> np.ndarray:
    return -cumulative_hazard(p, length_birth, u, omega2, eta)


def survival(
    p: np.ndarray,
    length_birth: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
) -> np.ndarray:
    return np.exp(np.clip(log_survival(p, length_birth, u, omega2, eta), -745.0, 0.0))


def log_division_time_density(
    time: np.ndarray,
    length_birth: np.ndarray,
    alpha: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
) -> np.ndarray:
    """Log division-time density including the p-to-time Jacobian."""
    time = np.asarray(time, dtype=float)
    length_birth = np.asarray(length_birth, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if not (u > 0.0 and omega2 > 0.0 and 0.0 <= eta <= 1.0):
        return np.full_like(time, -np.inf)

    p = added_size(time, length_birth, alpha)
    x = np.maximum(p + eta * length_birth, EPS) / u
    log_denominator = _log1p_power(x, omega2)
    log_hazard_p = (
        math.log(omega2)
        - math.log(u)
        + (omega2 - 1.0) * np.log(x)
        - log_denominator
    )
    log_dpdt = (
        np.log(np.maximum(length_birth, EPS))
        + np.log(np.maximum(alpha, EPS))
        + alpha * time
    )
    return log_hazard_p + log_dpdt + log_survival(p, length_birth, u, omega2, eta)


def log_likelihood(params: np.ndarray, data: dict[str, np.ndarray]) -> float:
    u, omega2, eta, mu_alpha, sigma_alpha, mu_f, sigma_f = params
    terms = (
        log_division_time_density(
            data["time"], data["length_birth"], data["alpha"], u, omega2, eta
        )
        + log_gamma_pdf_mean_sd(data["alpha"], mu_alpha, sigma_alpha)
        + log_beta_pdf_mean_sd(data["f"], mu_f, sigma_f)
    )
    if not np.all(np.isfinite(terms)):
        return -np.inf
    value = float(np.sum(terms))
    return value if np.isfinite(value) else -np.inf


def sample_added_size(
    length_birth: float,
    u: float,
    omega2: float,
    eta: float,
    rng: np.random.Generator,
) -> float:
    """Inverse-survival sample of added size."""
    uniform = max(float(rng.uniform()), np.finfo(float).tiny)
    x0_power = (eta * length_birth / u) ** omega2
    x_power = (1.0 + x0_power) / uniform - 1.0
    q = u * x_power ** (1.0 / omega2)
    return max(q - eta * length_birth, 0.0)


def simulate_lineages(
    params: np.ndarray,
    initial_lengths: np.ndarray,
    sim_lineages: int,
    sim_generations: int,
    seed: int,
) -> list[dict[str, float | int]]:
    u, omega2, eta, mu_alpha, sigma_alpha, mu_f, sigma_f = map(float, params)
    gamma_shape, gamma_scale = gamma_shape_scale(mu_alpha, sigma_alpha)
    beta_a, beta_b = beta_ab(mu_f, sigma_f)
    if gamma_shape is None or beta_a is None:
        raise ValueError("Invalid Gamma or Beta parameters")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for lineage in range(sim_lineages):
        birth = float(rng.choice(initial_lengths))
        for generation in range(sim_generations):
            alpha = float(rng.gamma(gamma_shape, gamma_scale))
            fraction = float(rng.beta(beta_a, beta_b))
            p = sample_added_size(birth, u, omega2, eta, rng)
            final = birth + p
            division_time = math.log1p(p / birth) / alpha
            rows.append(
                {
                    "lineage": lineage,
                    "generation": generation,
                    "time": division_time,
                    "alpha": alpha,
                    "f": fraction,
                    "length_birth": birth,
                    "length_final": final,
                    "added_size": p,
                }
            )
            birth = max(fraction * final, EPS)
    return rows

