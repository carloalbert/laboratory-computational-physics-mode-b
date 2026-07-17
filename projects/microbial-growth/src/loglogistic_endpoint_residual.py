"""Log-logistic mixed adder-sizer with an endpoint-growth residual.

For consecutive observations in one lineage the measured endpoint is recovered
from the *next* row's division fraction,

    L_final,n = b_(n+1) / f_(n+1),
    p_n       = L_final,n - b_n,

and the discrepancy between the endpoint growth and the fitted within-cycle
growth rate is

    epsilon_n = log(L_final,n / b_n) - alpha_n * tau_n.

The division hazard is fitted directly to measured ``p_n`` using

    S(p | b) = (1 + (eta*b/u)**omega2)
               / (1 + ((p + eta*b)/u)**omega2),

For inference the physical scale is pivoted at a fixed reference birth length,

    p_star = u - eta * b_star,
    u      = p_star + eta * b_star,

which removes the near-degeneracy between ``u`` and ``eta`` without changing
the physical model.  The transformation has unit Jacobian.  Meanwhile epsilon
has a robust Student-t location-scale distribution with fixed
degrees of freedom.  Simulated lineages use the exact size recursion

    b_(n+1) = f_(n+1) * (b_n + p_n)

and reconstruct division time from

    tau_n = (log(1 + p_n/b_n) - epsilon_n) / alpha_n.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, stdtr

from weibull_adder_sizer import (
    EPS,
    beta_ab,
    gamma_shape_scale,
    log_beta_pdf_mean_sd,
    log_gamma_pdf_mean_sd,
)


RESIDUAL_DF = 4.0


def pivoted_to_physical_params(
    params: np.ndarray, birth_pivot: float
) -> np.ndarray:
    """Convert sampled (p_star, omega2, eta, ...) to physical (u, ...)."""
    physical = np.asarray(params, dtype=float).copy()
    if physical.shape[-1] != 9:
        raise ValueError("Expected nine endpoint-residual model parameters")
    physical[..., 0] = physical[..., 0] + physical[..., 2] * birth_pivot
    return physical


def log_likelihood_pivoted(
    params: np.ndarray,
    data: dict[str, np.ndarray],
    birth_pivot: float,
) -> float:
    return log_likelihood(pivoted_to_physical_params(params, birth_pivot), data)


def _log1p_power(x: np.ndarray, exponent: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.logaddexp(0.0, exponent * np.log(np.maximum(x, EPS)))


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
    shape = np.broadcast(p, length_birth).shape
    if not (u > 0.0 and omega2 > 0.0 and 0.0 <= eta <= 1.0):
        return np.full(shape, np.inf)
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
    return np.exp(
        np.clip(log_survival(p, length_birth, u, omega2, eta), -745.0, 0.0)
    )


def log_added_size_density(
    p: np.ndarray,
    length_birth: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
) -> np.ndarray:
    """Log density of measured added size; no time Jacobian is required."""
    p = np.asarray(p, dtype=float)
    length_birth = np.asarray(length_birth, dtype=float)
    if not (u > 0.0 and omega2 > 0.0 and 0.0 <= eta <= 1.0):
        return np.full(np.broadcast(p, length_birth).shape, -np.inf)
    x = np.maximum(p + eta * length_birth, EPS) / u
    log_hazard = (
        math.log(omega2)
        - math.log(u)
        + (omega2 - 1.0) * np.log(x)
        - _log1p_power(x, omega2)
    )
    values = log_hazard + log_survival(p, length_birth, u, omega2, eta)
    return np.where((p > 0.0) & (length_birth > 0.0), values, -np.inf)


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
    return constant - 0.5 * (df + 1.0) * np.log1p(standardized**2 / df)


def log_endpoint_residual_density(
    values: np.ndarray,
    upper: np.ndarray,
    location: float,
    scale: float,
    df: float = RESIDUAL_DF,
) -> np.ndarray:
    """Student-t residual density truncated so reconstructed time is positive."""
    values = np.asarray(values, dtype=float)
    upper = np.asarray(upper, dtype=float)
    base = log_student_t_location_scale(values, location, scale, df)
    if not (scale > 0.0 and df > 0.0):
        return np.full(np.broadcast(values, upper).shape, -np.inf)
    upper_cdf = stdtr(df, (upper - location) / scale)
    normalization = np.log(np.maximum(upper_cdf, np.finfo(float).tiny))
    return np.where(values < upper, base - normalization, -np.inf)


def log_likelihood(params: np.ndarray, data: dict[str, np.ndarray]) -> float:
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
    ) = params
    pieces = (
        log_added_size_density(
            data["added_size"], data["transition_birth"], u, omega2, eta
        ),
        log_gamma_pdf_mean_sd(data["alpha"], mu_alpha, sigma_alpha),
        log_beta_pdf_mean_sd(data["f"], mu_f, sigma_f),
        log_endpoint_residual_density(
            data["epsilon"],
            np.log1p(data["added_size"] / data["transition_birth"]),
            mu_epsilon,
            sigma_epsilon,
        ),
    )
    if any(not np.all(np.isfinite(piece)) for piece in pieces):
        return -np.inf
    # For observed (b_(n+1), tau_n), the transformation to (p_n, epsilon_n)
    # contributes |d(p, epsilon)/d(b_(n+1), tau)| = alpha_n/f_(n+1).
    # It is constant in the fitted parameters but retained for a coherent
    # likelihood value and future model comparisons.
    log_jacobian = float(data.get("transition_log_jacobian", 0.0))
    value = float(sum(np.sum(piece) for piece in pieces) + log_jacobian)
    return value if np.isfinite(value) else -np.inf


def sample_added_size(
    length_birth: float,
    u: float,
    omega2: float,
    eta: float,
    rng: np.random.Generator,
) -> float:
    """Analytic inverse-survival sample of measured added size."""
    uniform = max(float(rng.uniform()), np.finfo(float).tiny)
    x0_power = (eta * length_birth / u) ** omega2
    x_power = (1.0 + x0_power) / uniform - 1.0
    endpoint = u * x_power ** (1.0 / omega2)
    return max(endpoint - eta * length_birth, EPS)


def sample_valid_residual(
    max_residual: float,
    location: float,
    scale: float,
    rng: np.random.Generator,
    df: float = RESIDUAL_DF,
) -> float:
    """Sample epsilon conditional on the reconstructed division time being positive."""
    for _ in range(10_000):
        epsilon = float(location + scale * rng.standard_t(df))
        if np.isfinite(epsilon) and epsilon < max_residual:
            return epsilon
    raise RuntimeError("Unable to sample a positive-time endpoint residual")


def simulate_lineages(
    params: np.ndarray,
    initial_lengths: np.ndarray,
    sim_lineages: int,
    sim_generations: int,
    seed: int,
    fixed_initial_lengths: bool = False,
) -> list[dict[str, float | int]]:
    """Simulate lineages with sampled or explicitly paired initial sizes.

    With ``fixed_initial_lengths=False`` each simulated lineage samples its
    initial size from the empirical pool.  With ``True``, lineage ``i`` starts
    from ``initial_lengths[i]``; this is used by observation-matched ACF checks
    to preserve the measured (b0, lineage-length) pairing.
    """
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
            p = sample_added_size(birth, u, omega2, eta, rng)
            final = birth + p
            endpoint_log_growth = math.log1p(p / birth)
            epsilon = sample_valid_residual(
                endpoint_log_growth, mu_epsilon, sigma_epsilon, rng
            )
            division_time = (endpoint_log_growth - epsilon) / alpha
            next_fraction = float(rng.beta(beta_a, beta_b))
            rows.append(
                {
                    "lineage": lineage,
                    "generation": generation,
                    "time": division_time,
                    "alpha": alpha,
                    "f": current_fraction,
                    "next_f": next_fraction,
                    "length_birth": birth,
                    "length_final": final,
                    "added_size": p,
                    "epsilon": epsilon,
                }
            )
            birth = max(next_fraction * final, EPS)
            current_fraction = next_fraction
    return rows
