"""Autonomous Glu growth/division model with AR alpha and endpoint residuals.

For transition i in a lineage,

    b[i+1] = b[i] * f[i] * exp(alpha[i] * tau[i] + epsilon[i]).

The event time follows the three-parameter protein Hill hazard.  Growth rates
and endpoint residuals have Gamma and Student-t stationary marginals,
respectively, coupled in time by Gaussian-copula AR(1) processes.  The actual
division fraction is beta distributed with a birth-size-dependent mean.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, gammaln, ndtr
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm, t as t_dist


EPS = np.finfo(float).tiny
RESIDUAL_DF = 4.0


def log_hill_fraction(log_birth, growth_time, log_u, eta):
    """Log of sigmoid(eta log[b(exp(s)-1)/u]), evaluated stably."""
    s = np.asarray(growth_time, dtype=float)
    log_added = np.asarray(log_birth, dtype=float) + np.log(np.expm1(np.minimum(s, 700.0)))
    return -np.logaddexp(0.0, -eta * (log_added - log_u))


def profile_hazard_components(log_birth, growth_time, log_u, eta, order=48):
    """Return log Hill fractions and integrals excluding omega2."""
    log_birth = np.asarray(log_birth, dtype=float).ravel()
    growth_time = np.asarray(growth_time, dtype=float).ravel()
    nodes, weights = np.polynomial.legendre.leggauss(order)
    integration_time = 0.5 * growth_time[:, None] * (nodes + 1.0)
    log_added = log_birth[:, None] + np.log(
        np.expm1(np.minimum(integration_time, 700.0))
    )
    fraction = expit(eta * (log_added - log_u))
    integral = 0.5 * growth_time * np.sum(weights * fraction, axis=1)
    log_fraction = log_hill_fraction(log_birth, growth_time, log_u, eta)
    return log_fraction, integral


def profile_hazard_log_likelihood(log_birth, alpha, time, log_u, log_eta, order=48):
    """Profile omega2 analytically and return (log likelihood, omega2)."""
    eta = math.exp(log_eta)
    growth_time = np.asarray(alpha, dtype=float) * np.asarray(time, dtype=float)
    log_fraction, integral = profile_hazard_components(
        log_birth, growth_time, log_u, eta, order=order
    )
    if np.any(integral <= 0.0) or not np.all(np.isfinite(integral)):
        return -np.inf, np.nan
    omega2 = len(growth_time) / float(integral.sum())
    value = np.sum(
        np.log(alpha) + math.log(omega2) + log_fraction - omega2 * integral
    )
    return float(value), float(omega2)


def profile_birth_dependent_hazard_components(
    log_birth, growth_time, log_u, eta, kappa, log_b_star, order=48
):
    """Event log-Hill factors and integrals with explicit birth-size feedback."""
    log_birth = np.asarray(log_birth, dtype=float).ravel()
    growth_time = np.asarray(growth_time, dtype=float).ravel()
    nodes, weights = np.polynomial.legendre.leggauss(order)
    integration_time = 0.5 * growth_time[:, None] * (nodes + 1.0)
    log_added = log_birth[:, None] + np.log(
        np.expm1(np.minimum(integration_time, 700.0))
    )
    birth_term = kappa * (log_birth - log_b_star)
    integration_odds = eta * (log_added - log_u) + birth_term[:, None]
    integral = 0.5 * growth_time * np.sum(weights * expit(integration_odds), axis=1)
    event_log_added = log_birth + np.log(
        np.expm1(np.minimum(growth_time, 700.0))
    )
    event_log_fraction = -np.logaddexp(
        0.0,
        -(eta * (event_log_added - log_u) + birth_term),
    )
    return event_log_fraction, integral


def profile_birth_dependent_hazard_log_likelihood(
    log_birth,
    alpha,
    time,
    log_u,
    log_eta,
    kappa,
    log_b_star,
    order=48,
):
    """Profile omega2 for the four-parameter p-and-b Hill timing hazard."""
    eta = math.exp(log_eta)
    growth_time = np.asarray(alpha, dtype=float) * np.asarray(time, dtype=float)
    log_fraction, integral = profile_birth_dependent_hazard_components(
        log_birth,
        growth_time,
        log_u,
        eta,
        kappa,
        log_b_star,
        order=order,
    )
    if np.any(integral <= 0.0) or not np.all(np.isfinite(integral)):
        return -np.inf, np.nan
    omega2 = len(growth_time) / float(integral.sum())
    value = np.sum(
        np.log(alpha) + math.log(omega2) + log_fraction - omega2 * integral
    )
    return float(value), float(omega2)


def beta_regression_log_likelihood(fraction, log_birth, log_birth_pivot, beta0, beta_b, log_phi):
    fraction = np.asarray(fraction, dtype=float)
    x = np.asarray(log_birth, dtype=float) - log_birth_pivot
    mean = expit(beta0 + beta_b * x)
    phi = math.exp(log_phi)
    a = np.maximum(mean * phi, 1e-12)
    b = np.maximum((1.0 - mean) * phi, 1e-12)
    values = (
        gammaln(a + b)
        - gammaln(a)
        - gammaln(b)
        + (a - 1.0) * np.log(fraction)
        + (b - 1.0) * np.log1p(-fraction)
    )
    return float(values.sum())


def gaussian_copula_ar_log_likelihood(values, predecessor, marginal, params, rho):
    """Marginal log density plus AR(1) Gaussian-copula corrections.

    predecessor[j] is the index preceding j, or -1 at a segment start.
    """
    values = np.asarray(values, dtype=float)
    predecessor = np.asarray(predecessor, dtype=int)
    if not (-0.995 < rho < 0.995):
        return -np.inf
    if marginal == "gamma":
        shape, scale = params
        if shape <= 0.0 or scale <= 0.0:
            return -np.inf
        logpdf = gamma_dist.logpdf(values, a=shape, scale=scale)
        cdf = gamma_dist.cdf(values, a=shape, scale=scale)
    elif marginal == "student_t":
        location, scale = params
        if scale <= 0.0:
            return -np.inf
        zt = (values - location) / scale
        logpdf = t_dist.logpdf(zt, df=RESIDUAL_DF) - math.log(scale)
        cdf = t_dist.cdf(zt, df=RESIDUAL_DF)
    else:
        raise ValueError(marginal)
    z = norm.ppf(np.clip(cdf, 1e-10, 1.0 - 1e-10))
    linked = predecessor >= 0
    innovation_sd = math.sqrt(1.0 - rho * rho)
    correction = norm.logpdf(
        z[linked], loc=rho * z[predecessor[linked]], scale=innovation_sd
    ) - norm.logpdf(z[linked])
    total = float(np.sum(logpdf) + np.sum(correction))
    return total if np.isfinite(total) else -np.inf


def transform_gamma(z, shape, scale):
    return gamma_dist.ppf(np.clip(ndtr(z), 1e-10, 1.0 - 1e-10), a=shape, scale=scale)


def transform_student_t(z, location, scale):
    return location + scale * t_dist.ppf(
        np.clip(ndtr(z), 1e-10, 1.0 - 1e-10), df=RESIDUAL_DF
    )


def sample_growth_time(log_birth, log_u, omega2, eta, rng):
    """Vectorized exact thinning for growth time s=alpha*tau."""
    log_birth = np.asarray(log_birth, dtype=float)
    log_u = np.asarray(log_u, dtype=float)
    omega2 = np.asarray(omega2, dtype=float)
    if log_u.ndim and log_u.shape != log_birth.shape:
        raise ValueError("Array-valued log_u must have the same shape as log_birth")
    if omega2.ndim and omega2.shape != log_birth.shape:
        raise ValueError("Array-valued omega2 must have the same shape as log_birth")
    growth = np.zeros_like(log_birth)
    pending = np.ones(log_birth.shape, dtype=bool)
    for _ in range(100_000):
        if not pending.any():
            return growth
        indices = np.flatnonzero(pending)
        pending_omega2 = omega2 if omega2.ndim == 0 else omega2[indices]
        growth[indices] += rng.exponential(1.0 / pending_omega2, len(indices))
        log_added = log_birth[indices] + np.log(
            np.expm1(np.minimum(growth[indices], 700.0))
        )
        pending_log_u = log_u if log_u.ndim == 0 else log_u[indices]
        accepted = rng.random(len(indices)) < expit(
            eta * (log_added - pending_log_u)
        )
        pending[indices[accepted]] = False
    raise RuntimeError("Exact thinning did not converge")


def sample_birth_dependent_growth_time(
    log_birth, log_u, omega2, eta, kappa, log_b_star, rng
):
    """Vectorized exact thinning for the p-and-b Hill hazard in growth time."""
    log_birth = np.asarray(log_birth, dtype=float)
    growth = np.zeros_like(log_birth)
    pending = np.ones(log_birth.shape, dtype=bool)
    birth_term = kappa * (log_birth - log_b_star)
    for _ in range(100_000):
        if not pending.any():
            return growth
        indices = np.flatnonzero(pending)
        growth[indices] += rng.exponential(1.0 / omega2, len(indices))
        log_added = log_birth[indices] + np.log(
            np.expm1(np.minimum(growth[indices], 700.0))
        )
        log_odds = eta * (log_added - log_u) + birth_term[indices]
        accepted = rng.random(len(indices)) < expit(log_odds)
        pending[indices[accepted]] = False
    raise RuntimeError("Exact thinning did not converge")


def lineage_acf(series, lengths, max_lag=15):
    """Mean short-lineage ACF, preserving one equal vote per lineage."""
    series = np.asarray(series, dtype=float)
    lengths = np.asarray(lengths, dtype=int)
    if series.ndim == 2:
        series = series[None, ...]
    result = np.full((series.shape[0], max_lag + 1), np.nan)
    result[:, 0] = 1.0
    for replicate in range(series.shape[0]):
        centered = []
        denominators = []
        for lineage, length in enumerate(lengths):
            values = series[replicate, lineage, :length]
            if length:
                values = values - values.mean()
            centered.append(values)
            denominators.append(float(np.sum(values * values)))
        for lag in range(1, max_lag + 1):
            values = [
                float(np.sum(x[:-lag] * x[lag:]) / den)
                for x, den in zip(centered, denominators)
                if len(x) > lag and len(x) >= 3 and den > 0.0
            ]
            if values:
                result[replicate, lag] = float(np.mean(values))
    return result


def random_intercept_ar1_log_likelihood(
    sequences,
    generations,
    sigma_offset,
    rho,
    sigma_innovation,
):
    """Gaussian marginal likelihood for a random intercept plus AR(1).

    AR residuals restart after generation gaps, while the random intercept is
    shared by all observations from the same biological lineage.  The common
    mean is profiled analytically.
    """
    if not (
        sigma_offset > 0.0
        and sigma_innovation > 0.0
        and -0.98 < rho < 0.98
    ):
        return -np.inf, np.nan
    stationary_variance = sigma_innovation**2 / (1.0 - rho**2)
    cached = []
    numerator = 0.0
    denominator = 0.0
    for values, generation in zip(sequences, generations):
        values = np.asarray(values, dtype=float)
        generation = np.asarray(generation, dtype=int)
        if not len(values):
            continue
        segment = np.r_[0, np.cumsum(np.diff(generation) != 1)]
        distance = np.abs(generation[:, None] - generation[None, :])
        same_segment = segment[:, None] == segment[None, :]
        ar_covariance = stationary_variance * np.where(
            same_segment, rho**distance, 0.0
        )
        covariance = ar_covariance + sigma_offset**2
        covariance.flat[:: len(values) + 1] += 1e-12
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return -np.inf, np.nan
        ones = np.ones(len(values))
        solved_ones = np.linalg.solve(
            cholesky.T, np.linalg.solve(cholesky, ones)
        )
        solved_values = np.linalg.solve(
            cholesky.T, np.linalg.solve(cholesky, values)
        )
        numerator += float(ones @ solved_values)
        denominator += float(ones @ solved_ones)
        cached.append((values, cholesky))
    if not cached or denominator <= 0.0:
        return -np.inf, np.nan
    mean = numerator / denominator
    log_likelihood = 0.0
    for values, cholesky in cached:
        residual = values - mean
        whitened = np.linalg.solve(cholesky, residual)
        log_likelihood += -0.5 * (
            len(values) * math.log(2.0 * math.pi)
            + 2.0 * np.log(np.diag(cholesky)).sum()
            + whitened @ whitened
        )
    return float(log_likelihood), float(mean)


def two_ar1_log_likelihood(
    sequences,
    generations,
    rho_fast,
    sigma_fast_innovation,
    rho_slow,
    sigma_slow_innovation,
):
    """Gaussian likelihood for the sum of independent fast and slow AR(1)s."""
    if not (
        -0.98 < rho_fast < 0.98
        and -0.98 < rho_slow < 0.9995
        and sigma_fast_innovation > 0.0
        and sigma_slow_innovation > 0.0
    ):
        return -np.inf, np.nan
    fast_variance = sigma_fast_innovation**2 / (1.0 - rho_fast**2)
    slow_variance = sigma_slow_innovation**2 / (1.0 - rho_slow**2)
    cached = []
    numerator = 0.0
    denominator = 0.0
    for values, generation in zip(sequences, generations):
        values = np.asarray(values, dtype=float)
        generation = np.asarray(generation, dtype=int)
        if not len(values):
            continue
        segment = np.r_[0, np.cumsum(np.diff(generation) != 1)]
        distance = np.abs(generation[:, None] - generation[None, :])
        same_segment = segment[:, None] == segment[None, :]
        covariance = np.where(
            same_segment,
            fast_variance * rho_fast**distance
            + slow_variance * rho_slow**distance,
            0.0,
        )
        covariance.flat[:: len(values) + 1] += 1e-12
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return -np.inf, np.nan
        ones = np.ones(len(values))
        solved_ones = np.linalg.solve(
            cholesky.T, np.linalg.solve(cholesky, ones)
        )
        solved_values = np.linalg.solve(
            cholesky.T, np.linalg.solve(cholesky, values)
        )
        numerator += float(ones @ solved_values)
        denominator += float(ones @ solved_ones)
        cached.append((values, cholesky))
    if not cached or denominator <= 0.0:
        return -np.inf, np.nan
    mean = numerator / denominator
    log_likelihood = 0.0
    for values, cholesky in cached:
        residual = values - mean
        whitened = np.linalg.solve(cholesky, residual)
        log_likelihood += -0.5 * (
            len(values) * math.log(2.0 * math.pi)
            + 2.0 * np.log(np.diag(cholesky)).sum()
            + whitened @ whitened
        )
    return float(log_likelihood), float(mean)
