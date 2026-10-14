"""Alpha-scaled Hill division hazard with deterministic lineage-age memory.

For absolute lineage time ``T`` the memory state and effective division scale
are

    m(T) = exp(-T / tau_m)
    u_eff(T) = u * exp(beta * m(T)).

Within a cell cycle, growth time is ``s = alpha * a`` and accumulated licensing
material is ``p(s) = b * (exp(s) - 1)``.  The physical-time hazard is

    h(a) = alpha * omega2 * (p/u_eff)**eta / (1 + (p/u_eff)**eta).

The memory state is continuous at division; its clock starts at zero at the
first observed birth in each lineage.
"""

from __future__ import annotations

import math

import numpy as np


EPS = np.finfo(float).tiny
DEFAULT_QUADRATURE_ORDER = 32


def quadrature(order: int = DEFAULT_QUADRATURE_ORDER) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(order)


def log_hill_fraction(log_ratio: np.ndarray) -> np.ndarray:
    """Return log(sigmoid(log_ratio)) stably."""
    return -np.logaddexp(0.0, -np.asarray(log_ratio, dtype=float))


def cumulative_hazard_growth_time(
    growth_time: np.ndarray,
    length_birth: np.ndarray,
    alpha: np.ndarray,
    lineage_time_start: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
    tau_m: float,
    beta: float,
    order: int = DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    """Cumulative hazard up to growth time ``s = alpha * a``."""
    growth_time, length_birth, alpha, lineage_time_start = np.broadcast_arrays(
        np.asarray(growth_time, dtype=float),
        np.asarray(length_birth, dtype=float),
        np.asarray(alpha, dtype=float),
        np.asarray(lineage_time_start, dtype=float),
    )
    shape = growth_time.shape
    if not (
        u > 0.0
        and omega2 > 0.0
        and eta > 0.0
        and tau_m > 0.0
        and np.all(alpha > 0.0)
    ):
        return np.full(shape, np.inf)

    nodes, weights = quadrature(order)
    integration_growth_time = 0.5 * growth_time[..., None] * (nodes + 1.0)
    physical_age = integration_growth_time / alpha[..., None]
    absolute_time = lineage_time_start[..., None] + physical_age
    memory = np.exp(-absolute_time / tau_m)
    log_u_effective = math.log(u) + beta * memory
    modeled_added = length_birth[..., None] * np.expm1(
        np.clip(integration_growth_time, 0.0, 700.0)
    )
    log_ratio = eta * (
        np.log(np.maximum(modeled_added, EPS)) - log_u_effective
    )
    fraction = np.exp(log_hill_fraction(log_ratio))
    integral = 0.5 * growth_time * np.sum(weights * fraction, axis=-1)
    return omega2 * integral


def log_time_density(
    time: np.ndarray,
    length_birth: np.ndarray,
    alpha: np.ndarray,
    lineage_time_start: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
    tau_m: float,
    beta: float,
    order: int = DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    """Log event-time density conditional on birth size and measured alpha."""
    time, length_birth, alpha, lineage_time_start = np.broadcast_arrays(
        np.asarray(time, dtype=float),
        np.asarray(length_birth, dtype=float),
        np.asarray(alpha, dtype=float),
        np.asarray(lineage_time_start, dtype=float),
    )
    shape = time.shape
    if not (u > 0.0 and omega2 > 0.0 and eta > 0.0 and tau_m > 0.0):
        return np.full(shape, -np.inf)

    growth_time = alpha * time
    modeled_added = length_birth * np.expm1(np.clip(growth_time, 0.0, 700.0))
    memory_at_event = np.exp(-(lineage_time_start + time) / tau_m)
    log_u_effective = math.log(u) + beta * memory_at_event
    event_log_ratio = eta * (
        np.log(np.maximum(modeled_added, EPS)) - log_u_effective
    )
    cumulative = cumulative_hazard_growth_time(
        growth_time,
        length_birth,
        alpha,
        lineage_time_start,
        u,
        omega2,
        eta,
        tau_m,
        beta,
        order=order,
    )
    values = (
        np.log(alpha)
        + math.log(omega2)
        + log_hill_fraction(event_log_ratio)
        - cumulative
    )
    valid = (
        (time > 0.0)
        & (length_birth > 0.0)
        & (alpha > 0.0)
        & (lineage_time_start >= 0.0)
        & np.isfinite(values)
    )
    return np.where(valid, values, -np.inf)


def sample_growth_time(
    length_birth: float,
    alpha: float,
    lineage_time_start: float,
    u: float,
    omega2: float,
    eta: float,
    tau_m: float,
    beta: float,
    rng: np.random.Generator,
) -> float:
    """Exact thinning sample in growth time for the time-dependent hazard."""
    growth_time = 0.0
    for _ in range(100_000):
        growth_time += float(rng.exponential(1.0 / omega2))
        physical_age = growth_time / alpha
        memory = math.exp(-(lineage_time_start + physical_age) / tau_m)
        log_u_effective = math.log(u) + beta * memory
        modeled_added = length_birth * math.expm1(min(growth_time, 700.0))
        log_ratio = eta * (
            math.log(max(modeled_added, EPS)) - log_u_effective
        )
        acceptance = math.exp(float(log_hill_fraction(log_ratio)))
        if rng.uniform() < acceptance:
            return growth_time
    raise RuntimeError("Unable to sample division growth time")
