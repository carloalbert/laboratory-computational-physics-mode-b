"""Endpoint-residual model with a persistent hazard shape per lineage.

The lineage-specific log-logistic shape obeys

    log(omega2_l) ~ Normal(log(omega2_bar), sigma_log_omega2**2),

while the physical size scale ``u`` and feedback ``eta`` remain global.  Each
lineage effect is integrated with Gauss-Hermite quadrature during inference and
recovered from conditional quadrature weights for posterior prediction.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

from hierarchical_lineage_u_endpoint_residual import population_scale
from loglogistic_endpoint_residual import EPS, log_added_size_density


HIERARCHICAL_OMEGA_PARAMETER_NAMES = [
    "p_star",
    "omega2",
    "eta",
    "sigma_log_omega2",
]


def lineage_node_log_likelihood(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Division log likelihood per lineage and log-normal shape node."""
    p_star, omega2_bar, eta, sigma_log_omega2 = map(float, params)
    u = p_star + eta * birth_pivot
    n_lineages = int(prepared["n_lineages"])
    if not (
        u > 0.0
        and omega2_bar > 0.0
        and 0.0 <= eta <= 1.0
        and sigma_log_omega2 >= 0.0
    ):
        return (
            np.full((n_lineages, len(z_nodes)), -np.inf),
            np.full(len(z_nodes), np.nan),
        )

    omega_nodes = omega2_bar * np.exp(
        sigma_log_omega2 * np.asarray(z_nodes, dtype=float)
    )
    birth = np.asarray(prepared["transition_birth"], dtype=float)[:, None]
    added = np.asarray(prepared["added_size"], dtype=float)[:, None]
    x = np.maximum(added + eta * birth, EPS) / u
    x0 = np.maximum(eta * birth, 0.0) / u
    omega = omega_nodes[None, :]
    log_density = (
        np.log(omega)
        - math.log(u)
        + (omega - 1.0) * np.log(x)
        + np.logaddexp(
            0.0, omega * np.log(np.maximum(x0, EPS))
        )
        - 2.0 * np.logaddexp(0.0, omega * np.log(x))
    )
    if not np.all(np.isfinite(log_density)):
        return np.full((n_lineages, len(z_nodes)), -np.inf), omega_nodes
    starts = np.asarray(prepared["lineage_starts"], dtype=int)
    return np.add.reduceat(log_density, starts, axis=0), omega_nodes


def log_marginal_division_likelihood(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
) -> float:
    """Log likelihood after integrating one ``omega2_l`` per lineage."""
    lineage_log_likelihood, _ = lineage_node_log_likelihood(
        params, prepared, birth_pivot, z_nodes
    )
    if not np.all(np.isfinite(lineage_log_likelihood)):
        return -np.inf
    value = np.sum(
        logsumexp(lineage_log_likelihood + log_weights[None, :], axis=1)
    )
    return float(value) if np.isfinite(value) else -np.inf


def conditional_lineage_omega2_weights(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shape nodes and P(omega2_l | lineage transitions, params)."""
    lineage_log_likelihood, omega_nodes = lineage_node_log_likelihood(
        params, prepared, birth_pivot, z_nodes
    )
    log_posterior = lineage_log_likelihood + log_weights[None, :]
    log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
    return omega_nodes, np.exp(log_posterior)


def sample_conditional_lineage_omega2(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one posterior shape for each transition-bearing lineage."""
    omega_nodes, weights = conditional_lineage_omega2_weights(
        params, prepared, birth_pivot, z_nodes, log_weights
    )
    uniforms = rng.uniform(size=weights.shape[0])
    cumulative = np.cumsum(weights, axis=1)
    indices = np.sum(cumulative < uniforms[:, None], axis=1)
    indices = np.minimum(indices, len(omega_nodes) - 1)
    return omega_nodes[indices]


def draw_observed_lineage_omega2(
    hierarchy_params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    groups: list,
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw aligned lineage shapes, using the population law if no transition exists."""
    transition_values = sample_conditional_lineage_omega2(
        hierarchy_params,
        prepared,
        birth_pivot,
        z_nodes,
        log_weights,
        rng,
    )
    labels = np.asarray(prepared["lineage_labels"])
    lookup = {
        label: value for label, value in zip(labels, transition_values)
    }
    omega2_bar = float(hierarchy_params[1])
    sigma = float(hierarchy_params[3])
    result = np.empty(len(groups), dtype=float)
    for index, group in enumerate(groups):
        label = group.lineage.iloc[0]
        result[index] = lookup.get(
            label,
            omega2_bar * math.exp(sigma * float(rng.normal())),
        )
    return result


def common_omega2_log_likelihood(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
) -> float:
    """Reference common-shape likelihood at sigma_log_omega2=0."""
    values = log_added_size_density(
        np.asarray(prepared["added_size"], dtype=float),
        np.asarray(prepared["transition_birth"], dtype=float),
        population_scale(params, birth_pivot),
        float(params[1]),
        float(params[2]),
    )
    return float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf

