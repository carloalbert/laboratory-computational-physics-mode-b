"""Alpha-scaled Hill division hazard with stationary leaky memory.

At the birth of generation ``n`` the memory state is ``m_n``.  During the
cycle it decays deterministically,

    m_n(a) = m_n * exp(-a / tau_m),

and modulates the division scale through

    u_eff(a) = u * exp(beta * m_n(a)).

The physical-time division hazard is

    h_n(a) = alpha_n * omega2 * z_n(a)**eta / (1 + z_n(a)**eta),
    z_n(a) = b_n * (exp(alpha_n*a) - 1) / u_eff(a).

Let ``E_n`` be the unit-exponential threshold that generates the division
event, so that the cumulative hazard at division equals ``E_n``.  The same
random number is transformed to a standard-normal innovation,

    xi_n = Phi^-1(1 - exp(-E_n)),

and the division update is

    m_(n+1) = rho_n*m_n + sqrt(1-rho_n**2)*xi_n,
    rho_n = exp(-d_n/tau_m).

Thus the added memory mode has only two fitted parameters, ``tau_m`` and
``beta``; it does not introduce another noise-scale parameter or random
source.  Conditional on the observed history, ``E_n`` is simply the
cumulative hazard evaluated at the observed division time, so the sequential
event-time likelihood remains directly evaluable.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


EPS = np.finfo(float).tiny
DEFAULT_QUADRATURE_ORDER = 16


@lru_cache(maxsize=None)
def _quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(order)


def _normal_ppf(probability: np.ndarray | float) -> np.ndarray:
    """Vectorized inverse standard-normal CDF (Acklam approximation)."""
    probability = np.asarray(probability, dtype=float)
    p = np.clip(probability, 1e-14, 1.0 - 1e-14)
    result = np.empty_like(p)

    a = np.asarray(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.asarray(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.asarray(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.asarray(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    lower = p < 0.02425
    upper = p > 1.0 - 0.02425
    central = ~(lower | upper)
    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(p[lower]))
        result[lower] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(upper):
        q = np.sqrt(-2.0 * np.log1p(-p[upper]))
        result[upper] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(central):
        q = p[central] - 0.5
        r = q * q
        result[central] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    return result


def innovation_from_threshold(threshold: np.ndarray | float) -> np.ndarray:
    """Map an Exp(1) division threshold to an exactly N(0,1) innovation."""
    threshold = np.asarray(threshold, dtype=float)
    probability = -np.expm1(-np.maximum(threshold, 0.0))
    return _normal_ppf(probability)


def _hill_fraction(log_ratio: np.ndarray) -> np.ndarray:
    positive = log_ratio >= 0.0
    result = np.empty_like(log_ratio)
    result[positive] = 1.0 / (1.0 + np.exp(-log_ratio[positive]))
    exponential = np.exp(log_ratio[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def cumulative_hazard_growth_time(
    growth_time: np.ndarray | float,
    length_birth: np.ndarray | float,
    alpha: np.ndarray | float,
    memory_birth: np.ndarray | float,
    u: float,
    omega2: float,
    eta: float,
    tau_m: float,
    beta: float,
    order: int = DEFAULT_QUADRATURE_ORDER,
) -> np.ndarray:
    """Cumulative hazard up to growth time ``s=alpha*a``."""
    growth_time, length_birth, alpha, memory_birth = np.broadcast_arrays(
        np.asarray(growth_time, dtype=float),
        np.asarray(length_birth, dtype=float),
        np.asarray(alpha, dtype=float),
        np.asarray(memory_birth, dtype=float),
    )
    if not (
        u > 0.0
        and omega2 > 0.0
        and eta > 0.0
        and tau_m > 0.0
        and np.all(alpha > 0.0)
    ):
        return np.full(growth_time.shape, np.inf)
    nodes, weights = _quadrature(order)
    integration_growth_time = 0.5 * growth_time[..., None] * (nodes + 1.0)
    physical_age = integration_growth_time / alpha[..., None]
    memory = memory_birth[..., None] * np.exp(-physical_age / tau_m)
    log_u_effective = math.log(u) + beta * memory
    modeled_added = length_birth[..., None] * np.expm1(
        np.clip(integration_growth_time, 0.0, 700.0)
    )
    log_ratio = eta * (
        np.log(np.maximum(modeled_added, EPS)) - log_u_effective
    )
    integral = 0.5 * growth_time * np.sum(
        weights * _hill_fraction(log_ratio), axis=-1
    )
    return omega2 * integral


def log_time_density_and_threshold(
    time: np.ndarray | float,
    length_birth: np.ndarray | float,
    alpha: np.ndarray | float,
    memory_birth: np.ndarray | float,
    u: float,
    omega2: float,
    eta: float,
    tau_m: float,
    beta: float,
    order: int = DEFAULT_QUADRATURE_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    """Return event log density and inferred exponential threshold."""
    time, length_birth, alpha, memory_birth = np.broadcast_arrays(
        np.asarray(time, dtype=float),
        np.asarray(length_birth, dtype=float),
        np.asarray(alpha, dtype=float),
        np.asarray(memory_birth, dtype=float),
    )
    growth_time = alpha * time
    threshold = cumulative_hazard_growth_time(
        growth_time,
        length_birth,
        alpha,
        memory_birth,
        u,
        omega2,
        eta,
        tau_m,
        beta,
        order=order,
    )
    memory_event = memory_birth * np.exp(-time / tau_m)
    log_u_effective = math.log(u) + beta * memory_event
    modeled_added = length_birth * np.expm1(np.clip(growth_time, 0.0, 700.0))
    log_ratio = eta * (
        np.log(np.maximum(modeled_added, EPS)) - log_u_effective
    )
    log_fraction = -np.logaddexp(0.0, -log_ratio)
    values = np.log(alpha) + math.log(omega2) + log_fraction - threshold
    valid = (
        (time > 0.0)
        & (length_birth > 0.0)
        & (alpha > 0.0)
        & np.isfinite(values)
    )
    return np.where(valid, values, -np.inf), threshold


def update_memory(
    memory_birth: np.ndarray | float,
    division_time: np.ndarray | float,
    threshold: np.ndarray | float,
    tau_m: float,
) -> np.ndarray:
    """Apply decay through a cycle and the parameter-free division jump."""
    memory_birth, division_time, threshold = np.broadcast_arrays(
        np.asarray(memory_birth, dtype=float),
        np.asarray(division_time, dtype=float),
        np.asarray(threshold, dtype=float),
    )
    rho = np.exp(-division_time / tau_m)
    innovation = innovation_from_threshold(threshold)
    return rho * memory_birth + np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) * innovation


def sample_cycle(
    length_birth: float,
    alpha: float,
    memory_birth: float,
    u: float,
    omega2: float,
    eta: float,
    tau_m: float,
    beta: float,
    rng: np.random.Generator,
    order: int = DEFAULT_QUADRATURE_ORDER,
) -> tuple[float, float, float]:
    """Sample division by exact thinning of the bounded hazard.

    Returns ``(growth_time, threshold, memory_next)``.
    """
    growth_time = 0.0
    for _ in range(100_000):
        growth_time += float(rng.exponential(1.0 / omega2))
        physical_age = growth_time / alpha
        memory = memory_birth * math.exp(-physical_age / tau_m)
        log_u_effective = math.log(u) + beta * memory
        modeled_added = length_birth * math.expm1(min(growth_time, 700.0))
        log_ratio = eta * (math.log(max(modeled_added, EPS)) - log_u_effective)
        acceptance = float(_hill_fraction(np.asarray([log_ratio]))[0])
        if rng.uniform() < acceptance:
            break
    else:
        raise RuntimeError("Unable to sample division time")
    threshold = float(
        cumulative_hazard_growth_time(
            growth_time,
            length_birth,
            alpha,
            memory_birth,
            u,
            omega2,
            eta,
            tau_m,
            beta,
            order=order,
        )
    )
    division_time = growth_time / alpha
    memory_next = float(
        update_memory(memory_birth, division_time, threshold, tau_m)
    )
    return growth_time, threshold, memory_next
