"""Endpoint-residual model with a persistent division scale per lineage.

The lineage scale is log-normal,

    log(u_l) ~ Normal(log(u_bar), sigma_log_u**2),

and is shared by every transition in lineage ``l``.  The division likelihood
for each lineage is marginalized over ``u_l`` with Gauss-Hermite quadrature,
so Bayesian inference only needs the four population-level division
parameters.  Conditional quadrature weights recover posterior draws of the
individual lineage scales for posterior-predictive checks.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

from loglogistic_endpoint_residual import EPS, log_added_size_density


HIERARCHICAL_PARAMETER_NAMES = ["p_star", "omega2", "eta", "sigma_log_u"]


def population_scale(
    params: np.ndarray, birth_pivot: float
) -> float:
    """Return the median population scale ``u_bar`` from pivoted parameters."""
    values = np.asarray(params, dtype=float)
    return float(values[0] + values[2] * birth_pivot)


def hermite_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and normalized log weights for expectation under N(0, 1)."""
    if order < 4:
        raise ValueError("Gauss-Hermite order must be at least four")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    z_nodes = math.sqrt(2.0) * nodes
    log_weights = np.log(weights) - 0.5 * math.log(math.pi)
    return z_nodes, log_weights


def prepare_lineage_transitions(
    transition_birth: np.ndarray,
    added_size: np.ndarray,
    lineage: np.ndarray,
) -> dict[str, np.ndarray | int]:
    """Sort transition arrays and encode lineage blocks for fast reduction."""
    birth = np.asarray(transition_birth, dtype=float)
    added = np.asarray(added_size, dtype=float)
    labels = np.asarray(lineage)
    if not (len(birth) == len(added) == len(labels)):
        raise ValueError("Transition arrays must have equal length")
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    unique, starts, counts = np.unique(
        labels, return_index=True, return_counts=True
    )
    return {
        "transition_birth": birth[order],
        "added_size": added[order],
        "lineage_labels": unique,
        "lineage_starts": starts,
        "lineage_counts": counts,
        "n_lineages": int(len(unique)),
    }


def lineage_node_log_likelihood(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Division log likelihood per lineage and quadrature node."""
    p_star, omega2, eta, sigma_log_u = map(float, params)
    u_bar = p_star + eta * birth_pivot
    n_lineages = int(prepared["n_lineages"])
    if not (
        u_bar > 0.0
        and omega2 > 0.0
        and 0.0 <= eta <= 1.0
        and sigma_log_u >= 0.0
    ):
        return (
            np.full((n_lineages, len(z_nodes)), -np.inf),
            np.full(len(z_nodes), np.nan),
        )

    # u_bar is the median lineage scale.  No mean correction is applied:
    # sigma_log_u describes biological dispersion on the log scale directly.
    u_nodes = u_bar * np.exp(sigma_log_u * np.asarray(z_nodes, dtype=float))
    birth = np.asarray(prepared["transition_birth"], dtype=float)[:, None]
    added = np.asarray(prepared["added_size"], dtype=float)[:, None]
    scale = u_nodes[None, :]
    x = np.maximum(added + eta * birth, EPS) / scale
    x0 = np.maximum(eta * birth, 0.0) / scale
    log_one_plus_x_power = np.logaddexp(0.0, omega2 * np.log(x))
    log_one_plus_x0_power = np.logaddexp(
        0.0, omega2 * np.log(np.maximum(x0, EPS))
    )
    log_density = (
        math.log(omega2)
        - np.log(scale)
        + (omega2 - 1.0) * np.log(x)
        + log_one_plus_x0_power
        - 2.0 * log_one_plus_x_power
    )
    if not np.all(np.isfinite(log_density)):
        return np.full((n_lineages, len(z_nodes)), -np.inf), u_nodes
    starts = np.asarray(prepared["lineage_starts"], dtype=int)
    return np.add.reduceat(log_density, starts, axis=0), u_nodes


def log_marginal_division_likelihood(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
) -> float:
    """Log likelihood after integrating one persistent ``u_l`` per lineage."""
    lineage_log_likelihood, _ = lineage_node_log_likelihood(
        params, prepared, birth_pivot, z_nodes
    )
    if not np.all(np.isfinite(lineage_log_likelihood)):
        return -np.inf
    value = np.sum(
        logsumexp(lineage_log_likelihood + log_weights[None, :], axis=1)
    )
    return float(value) if np.isfinite(value) else -np.inf


def conditional_lineage_scale_weights(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return quadrature scales and P(u_l | lineage transitions, params)."""
    lineage_log_likelihood, u_nodes = lineage_node_log_likelihood(
        params, prepared, birth_pivot, z_nodes
    )
    log_posterior = lineage_log_likelihood + log_weights[None, :]
    log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
    return u_nodes, np.exp(log_posterior)


def sample_conditional_lineage_scales(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one posterior ``u_l`` for every transition-bearing lineage."""
    u_nodes, weights = conditional_lineage_scale_weights(
        params, prepared, birth_pivot, z_nodes, log_weights
    )
    uniforms = rng.uniform(size=weights.shape[0])
    cumulative = np.cumsum(weights, axis=1)
    indices = np.sum(cumulative < uniforms[:, None], axis=1)
    indices = np.minimum(indices, len(u_nodes) - 1)
    return u_nodes[indices]


def common_u_log_likelihood(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
) -> float:
    """Reference likelihood at sigma_log_u=0 for numerical tests."""
    u_bar = population_scale(params, birth_pivot)
    values = log_added_size_density(
        np.asarray(prepared["added_size"], dtype=float),
        np.asarray(prepared["transition_birth"], dtype=float),
        u_bar,
        float(params[1]),
        float(params[2]),
    )
    return float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf


def sample_population_lineage_scales(
    u_bar: float,
    sigma_log_u: float,
    n_lineages: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw scales for new lineages from the fitted population distribution."""
    return u_bar * np.exp(sigma_log_u * rng.normal(size=n_lineages))
