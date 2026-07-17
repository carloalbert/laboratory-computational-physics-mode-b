"""Weibull mixed adder-sizer model for microbial division.

The division-control variable is

    q = p + eta * length_birth,

where p is added size.  The cumulative hazard is a Weibull cumulative
hazard in q, conditioned on the cell having survived to its birth size:

    H(p | b) = ((p + eta*b) / u)**omega2 - ((eta*b) / u)**omega2.

The parameter ``omega2`` is the Weibull shape exponent.  It replaces the
``v`` parameter from the original modified-Burr model, leaving seven fitted
parameters in total when the Gamma growth-rate and Beta division-fraction
components are included.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import betaln, gammaln


EPS = 1e-300
MAX_LOG = 700.0


def gamma_shape_scale(mu: float, sigma: float) -> tuple[float | None, float | None]:
    """Convert Gamma mean/SD parameters to shape/scale."""
    if not (mu > 0.0 and sigma > 0.0):
        return None, None
    shape = (mu / sigma) ** 2
    scale = sigma**2 / mu
    if not (np.isfinite(shape) and np.isfinite(scale) and shape > 0 and scale > 0):
        return None, None
    return float(shape), float(scale)


def beta_ab(mu: float, sigma: float) -> tuple[float | None, float | None]:
    """Convert Beta mean/SD parameters to alpha/beta."""
    if not (0.0 < mu < 1.0 and sigma > 0.0):
        return None, None
    variance = sigma**2
    max_variance = mu * (1.0 - mu)
    if not (variance < max_variance):
        return None, None
    common = max_variance / variance - 1.0
    a = mu * common
    b = (1.0 - mu) * common
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return None, None
    return float(a), float(b)


def log_gamma_pdf_mean_sd(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    shape, scale = gamma_shape_scale(mu, sigma)
    if shape is None:
        return np.full_like(np.asarray(x, dtype=float), -np.inf)
    x = np.asarray(x, dtype=float)
    return (
        (shape - 1.0) * np.log(np.maximum(x, EPS))
        - x / scale
        - shape * np.log(scale)
        - gammaln(shape)
    )


def log_beta_pdf_mean_sd(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    a, b = beta_ab(mu, sigma)
    if a is None:
        return np.full_like(np.asarray(x, dtype=float), -np.inf)
    x = np.clip(np.asarray(x, dtype=float), EPS, 1.0 - EPS)
    return (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) - betaln(a, b)


def added_size(time: np.ndarray, length_birth: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Added size p(t) = b * (exp(alpha*t) - 1)."""
    time = np.asarray(time, dtype=float)
    length_birth = np.asarray(length_birth, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    return length_birth * np.expm1(np.clip(alpha * time, 0.0, MAX_LOG))


def cumulative_hazard(
    p: np.ndarray,
    length_birth: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
) -> np.ndarray:
    """Stable evaluation of the conditional Weibull cumulative hazard."""
    p = np.asarray(p, dtype=float)
    length_birth = np.asarray(length_birth, dtype=float)
    if not (u > 0.0 and omega2 > 0.0 and 0.0 <= eta <= 1.0):
        return np.full(np.broadcast(p, length_birth).shape, np.inf)

    q = np.maximum(p + eta * length_birth, EPS)
    log_first = omega2 * (np.log(q) - math.log(u))

    if eta == 0.0:
        factor = np.ones_like(log_first)
    else:
        q0 = np.maximum(eta * length_birth, EPS)
        log_second = omega2 * (np.log(q0) - math.log(u))
        factor = -np.expm1(np.minimum(log_second - log_first, 0.0))

    with np.errstate(over="ignore", invalid="ignore"):
        first = np.exp(np.minimum(log_first, MAX_LOG))
        result = first * factor
    result = np.where(log_first > MAX_LOG, np.inf, result)
    return np.maximum(result, 0.0)


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
    """Log density with respect to time, including dp/dt."""
    time = np.asarray(time, dtype=float)
    length_birth = np.asarray(length_birth, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if not (u > 0.0 and omega2 > 0.0 and 0.0 <= eta <= 1.0):
        return np.full_like(time, -np.inf)

    p = added_size(time, length_birth, alpha)
    q = np.maximum(p + eta * length_birth, EPS)
    log_q_over_u = np.log(q) - math.log(u)
    log_hazard_p = math.log(omega2) - math.log(u) + (omega2 - 1.0) * log_q_over_u
    log_dpdt = np.log(np.maximum(length_birth, EPS)) + np.log(np.maximum(alpha, EPS)) + alpha * time
    cumulative = cumulative_hazard(p, length_birth, u, omega2, eta)
    return log_hazard_p + log_dpdt - cumulative


def log_likelihood(params: np.ndarray, data: dict[str, np.ndarray]) -> float:
    """Full one-environment likelihood used by inference."""
    u, omega2, eta, mu_alpha, sigma_alpha, mu_f, sigma_f = params
    log_time = log_division_time_density(
        data["time"], data["length_birth"], data["alpha"], u, omega2, eta
    )
    log_alpha = log_gamma_pdf_mean_sd(data["alpha"], mu_alpha, sigma_alpha)
    log_fraction = log_beta_pdf_mean_sd(data["f"], mu_f, sigma_f)
    terms = log_time + log_alpha + log_fraction
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
    """Inverse-CDF sample of added size using an Exp(1) variate."""
    exponential = float(rng.exponential())
    base = (eta * length_birth / u) ** omega2
    q = u * (exponential + base) ** (1.0 / omega2)
    return max(q - eta * length_birth, 0.0)


def simulate_lineages(
    params: np.ndarray,
    initial_lengths: np.ndarray,
    sim_lineages: int,
    sim_generations: int,
    seed: int,
) -> list[dict[str, float | int]]:
    """Simulate lineages from a fitted posterior parameter vector."""
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
            time = math.log1p(p / birth) / alpha
            rows.append(
                {
                    "lineage": lineage,
                    "generation": generation,
                    "time": time,
                    "alpha": alpha,
                    "f": fraction,
                    "length_birth": birth,
                    "length_final": final,
                    "added_size": p,
                }
            )
            birth = max(fraction * final, EPS)
    return rows

