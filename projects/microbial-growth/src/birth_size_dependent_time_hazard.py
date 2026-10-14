"""Alpha-scaled Hill hazard with independent birth-size feedback.

The hazard log odds depend on both accumulated licensing material ``p`` and
birth size ``b``:

    h(a | b) = alpha * omega2 * sigmoid(log_odds(a)),
    log_odds(a) = eta*log(p(a)/u) + kappa*log(b/b_star),
    p(a) = b*(exp(alpha*a) - 1).

``b_star`` is fixed from the data.  Setting ``kappa=0`` recovers the previous
three-parameter alpha-scaled Hill hazard.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


EPS = np.finfo(float).tiny
DEFAULT_QUADRATURE_ORDER = 24


@lru_cache(maxsize=None)
def quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(order)


def hill_fraction(log_odds: np.ndarray) -> np.ndarray:
    log_odds = np.asarray(log_odds, dtype=float)
    positive = log_odds >= 0.0
    result = np.empty_like(log_odds)
    result[positive] = 1.0 / (1.0 + np.exp(-log_odds[positive]))
    exponential = np.exp(log_odds[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def cumulative_hazard_growth_time(
    growth_time: np.ndarray | float,
    length_birth: np.ndarray | float,
    u: float,
    omega2: float,
    eta: float,
    kappa: float,
    b_star: float,
    order: int = DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    growth_time, length_birth = np.broadcast_arrays(
        np.asarray(growth_time, dtype=float),
        np.asarray(length_birth, dtype=float),
    )
    if not (
        u > 0.0
        and omega2 > 0.0
        and eta > 0.0
        and b_star > 0.0
        and np.all(length_birth > 0.0)
    ):
        return np.full(growth_time.shape, np.inf)
    nodes, weights = quadrature(order)
    integration_time = 0.5 * growth_time[..., None] * (nodes + 1.0)
    modeled_added = length_birth[..., None] * np.expm1(
        np.clip(integration_time, 0.0, 700.0)
    )
    log_odds = (
        eta * (np.log(np.maximum(modeled_added, EPS)) - math.log(u))
        + kappa * np.log(length_birth[..., None] / b_star)
    )
    integral = 0.5 * growth_time * np.sum(
        weights * hill_fraction(log_odds), axis=-1
    )
    return omega2 * integral


def log_time_density(
    time: np.ndarray | float,
    length_birth: np.ndarray | float,
    alpha: np.ndarray | float,
    u: float,
    omega2: float,
    eta: float,
    kappa: float,
    b_star: float,
    order: int = DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    time, length_birth, alpha = np.broadcast_arrays(
        np.asarray(time, dtype=float),
        np.asarray(length_birth, dtype=float),
        np.asarray(alpha, dtype=float),
    )
    growth_time = alpha * time
    modeled_added = length_birth * np.expm1(np.clip(growth_time, 0.0, 700.0))
    log_odds = (
        eta * (np.log(np.maximum(modeled_added, EPS)) - math.log(u))
        + kappa * np.log(length_birth / b_star)
    )
    cumulative = cumulative_hazard_growth_time(
        growth_time,
        length_birth,
        u,
        omega2,
        eta,
        kappa,
        b_star,
        order=order,
    )
    values = (
        np.log(alpha)
        + math.log(omega2)
        - np.logaddexp(0.0, -log_odds)
        - cumulative
    )
    valid = (
        (time > 0.0)
        & (length_birth > 0.0)
        & (alpha > 0.0)
        & np.isfinite(values)
    )
    return np.where(valid, values, -np.inf)


def sample_growth_time(
    length_birth: float,
    u: float,
    omega2: float,
    eta: float,
    kappa: float,
    b_star: float,
    rng: np.random.Generator,
) -> float:
    """Sample growth time exactly by thinning against rate ``omega2``."""
    birth_term = kappa * math.log(length_birth / b_star)
    growth_time = 0.0
    for _ in range(100_000):
        growth_time += float(rng.exponential(1.0 / omega2))
        modeled_added = length_birth * math.expm1(min(growth_time, 700.0))
        log_odds = eta * (
            math.log(max(modeled_added, EPS)) - math.log(u)
        ) + birth_term
        acceptance = float(hill_fraction(np.asarray([log_odds]))[0])
        if rng.uniform() < acceptance:
            return growth_time
    raise RuntimeError("Unable to sample division growth time")
