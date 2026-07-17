"""Alpha-scaled saturating division hazard with an endpoint residual.

The division hazard is defined in physical time by

    h(t | b, alpha) = alpha * omega2 * x(t)**eta / (1 + x(t)**eta),
    x(t) = b * (exp(alpha*t) - 1) / u.

Equivalently, in growth time s = alpha*t the hazard is bounded by omega2.
At division, an endpoint residual maps modeled growth time to measured final
size,

    L_final = b * exp(s + epsilon),

with epsilon following a lower-truncated Student-t distribution so that final
size remains larger than birth size.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, gammaln, stdtr

from weibull_adder_sizer import EPS, beta_ab, gamma_shape_scale


RESIDUAL_DF = 4.0
QUADRATURE_ORDER = 24
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(
    QUADRATURE_ORDER
)


def _log_hill_fraction(
    modeled_added_size: np.ndarray, u: float, eta: float
) -> np.ndarray:
    log_ratio = np.log(np.maximum(modeled_added_size, EPS) / u)
    return -np.logaddexp(0.0, -eta * log_ratio)


def cumulative_hazard_growth_time(
    growth_time: np.ndarray,
    length_birth: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
) -> np.ndarray:
    """Cumulative hazard in growth time, evaluated by Gauss-Legendre quadrature."""
    growth_time = np.asarray(growth_time, dtype=float)
    length_birth = np.asarray(length_birth, dtype=float)
    if not (u > 0.0 and omega2 > 0.0 and eta > 0.0):
        return np.full(np.broadcast(growth_time, length_birth).shape, np.inf)
    growth_time, length_birth = np.broadcast_arrays(growth_time, length_birth)
    integration_time = 0.5 * growth_time[..., None] * (
        _QUADRATURE_NODES + 1.0
    )
    modeled_added = length_birth[..., None] * np.expm1(
        np.clip(integration_time, 0.0, 700.0)
    )
    hill_fraction = expit(
        eta * np.log(np.maximum(modeled_added, EPS) / u)
    )
    integral = 0.5 * growth_time * np.sum(
        _QUADRATURE_WEIGHTS * hill_fraction, axis=-1
    )
    return omega2 * integral


def log_time_density(
    time: np.ndarray,
    length_birth: np.ndarray,
    alpha: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
) -> np.ndarray:
    """Log event-time density conditional on observed birth size and alpha."""
    time = np.asarray(time, dtype=float)
    length_birth = np.asarray(length_birth, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    shape = np.broadcast(time, length_birth, alpha).shape
    if not (u > 0.0 and omega2 > 0.0 and eta > 0.0):
        return np.full(shape, -np.inf)
    growth_time = alpha * time
    modeled_added = length_birth * np.expm1(
        np.clip(growth_time, 0.0, 700.0)
    )
    log_fraction = _log_hill_fraction(modeled_added, u, eta)
    cumulative = cumulative_hazard_growth_time(
        growth_time, length_birth, u, omega2, eta
    )
    values = np.log(alpha) + math.log(omega2) + log_fraction - cumulative
    valid = (
        (time > 0.0)
        & (length_birth > 0.0)
        & (alpha > 0.0)
        & np.isfinite(values)
    )
    return np.where(valid, values, -np.inf)


def log_student_t_location_scale(
    values: np.ndarray,
    location: float,
    scale: float,
    df: float = RESIDUAL_DF,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not (scale > 0.0 and df > 0.0):
        return np.full_like(values, -np.inf)
    standardized = (values - location) / scale
    constant = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * math.log(df * math.pi)
        - math.log(scale)
    )
    return constant - 0.5 * (df + 1.0) * np.log1p(
        standardized**2 / df
    )


def log_lower_truncated_residual_density(
    values: np.ndarray,
    lower: np.ndarray,
    location: float,
    scale: float,
    df: float = RESIDUAL_DF,
) -> np.ndarray:
    """Student-t density conditional on epsilon > lower."""
    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    base = log_student_t_location_scale(values, location, scale, df)
    if not (scale > 0.0 and df > 0.0):
        return np.full(np.broadcast(values, lower).shape, -np.inf)
    standardized_lower = (lower - location) / scale
    survival_lower = stdtr(df, -standardized_lower)
    normalization = np.log(
        np.maximum(survival_lower, np.finfo(float).tiny)
    )
    return np.where(values > lower, base - normalization, -np.inf)


def sample_growth_time(
    length_birth: float,
    u: float,
    omega2: float,
    eta: float,
    rng: np.random.Generator,
) -> float:
    """Exact thinning sample in growth time s=alpha*t."""
    growth_time = 0.0
    for _ in range(100_000):
        growth_time += float(rng.exponential(1.0 / omega2))
        modeled_added = length_birth * math.expm1(min(growth_time, 700.0))
        acceptance = float(
            expit(eta * math.log(max(modeled_added, EPS) / u))
        )
        if rng.uniform() < acceptance:
            return growth_time
    raise RuntimeError("Unable to sample division growth time")


def sample_lower_truncated_residual(
    lower: float,
    location: float,
    scale: float,
    rng: np.random.Generator,
    df: float = RESIDUAL_DF,
) -> float:
    for _ in range(100_000):
        value = float(location + scale * rng.standard_t(df))
        if np.isfinite(value) and value > lower:
            return value
    raise RuntimeError("Unable to sample a positive-endpoint residual")


def simulate_lineages(
    params: np.ndarray,
    initial_lengths: np.ndarray,
    sim_lineages: int,
    sim_generations: int,
    seed: int,
    fixed_initial_lengths: bool = False,
) -> list[dict[str, float | int]]:
    (
        u,
        omega2,
        eta,
        mu_alpha,
        sigma_alpha,
        mu_f,
        sigma_f,
        mu_epsilon,
        sigma_epsilon,
    ) = map(float, params)
    gamma_shape, gamma_scale = gamma_shape_scale(mu_alpha, sigma_alpha)
    beta_a, beta_b = beta_ab(mu_f, sigma_f)
    if gamma_shape is None or beta_a is None:
        raise ValueError("Invalid Gamma or Beta parameters")
    initial_lengths = np.asarray(initial_lengths, dtype=float)
    if fixed_initial_lengths and len(initial_lengths) != sim_lineages:
        raise ValueError(
            "fixed_initial_lengths requires one initial size per lineage"
        )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for lineage in range(sim_lineages):
        birth = float(
            initial_lengths[lineage]
            if fixed_initial_lengths
            else rng.choice(initial_lengths)
        )
        current_fraction = float(rng.beta(beta_a, beta_b))
        for generation in range(sim_generations):
            alpha = float(rng.gamma(gamma_shape, gamma_scale))
            growth_time = sample_growth_time(
                birth, u, omega2, eta, rng
            )
            division_time = growth_time / alpha
            epsilon = sample_lower_truncated_residual(
                -growth_time, mu_epsilon, sigma_epsilon, rng
            )
            endpoint_log_growth = growth_time + epsilon
            final = birth * math.exp(min(endpoint_log_growth, 700.0))
            added = final - birth
            hazard_added = birth * math.expm1(min(growth_time, 700.0))
            next_fraction = float(rng.beta(beta_a, beta_b))
            rows.append(
                {
                    "lineage": lineage,
                    "generation": generation,
                    "time": division_time,
                    "growth_time": growth_time,
                    "alpha": alpha,
                    "f": current_fraction,
                    "next_f": next_fraction,
                    "length_birth": birth,
                    "length_final": final,
                    "added_size": added,
                    "hazard_added_size": hazard_added,
                    "epsilon": epsilon,
                }
            )
            birth = max(next_fraction * final, EPS)
            current_fraction = next_fraction
    return rows
