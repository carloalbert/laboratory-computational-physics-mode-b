"""Endpoint-residual division model with lineage-specific scale and shape.

The two lineage effects have independent population distributions,

    log(u_l)      ~ Normal(log(u_bar), sigma_log_u**2)
    log(omega2_l) ~ Normal(log(omega2_bar), sigma_log_omega2**2),

and are integrated jointly with product Gauss-Hermite quadrature.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from loglogistic_endpoint_residual import EPS


PARAMETER_NAMES = [
    "p_star", "omega2", "eta", "sigma_log_u", "sigma_log_omega2"
]


def joint_nodes(
    params: np.ndarray, birth_pivot: float, z_nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    p_star, omega_bar, eta, sigma_u, sigma_omega = map(float, params)
    u_bar = p_star + eta * birth_pivot
    u = u_bar * np.exp(sigma_u * z_nodes[:, None])
    omega = omega_bar * np.exp(sigma_omega * z_nodes[None, :])
    shape = (len(z_nodes), len(z_nodes))
    return np.broadcast_to(u, shape).ravel(), np.broadcast_to(omega, shape).ravel()


def lineage_node_log_likelihood(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_star, omega_bar, eta, sigma_u, sigma_omega = map(float, params)
    u_bar = p_star + eta * birth_pivot
    n_lineages = int(prepared["n_lineages"])
    n_joint = len(z_nodes) ** 2
    if not (
        u_bar > 0 and omega_bar > 0 and 0 <= eta <= 1
        and sigma_u >= 0 and sigma_omega >= 0
    ):
        return (
            np.full((n_lineages, n_joint), -np.inf),
            np.full(n_joint, np.nan),
            np.full(n_joint, np.nan),
        )
    u, omega = joint_nodes(params, birth_pivot, z_nodes)
    birth = np.asarray(prepared["transition_birth"], float)[:, None]
    added = np.asarray(prepared["added_size"], float)[:, None]
    x = np.maximum(added + eta * birth, EPS) / u[None, :]
    x0 = np.maximum(eta * birth, EPS) / u[None, :]
    shape = omega[None, :]
    log_density = (
        np.log(shape) - np.log(u)[None, :]
        + (shape - 1) * np.log(x)
        + np.logaddexp(0, shape * np.log(x0))
        - 2 * np.logaddexp(0, shape * np.log(x))
    )
    starts = np.asarray(prepared["lineage_starts"], int)
    return np.add.reduceat(log_density, starts, axis=0), u, omega


def log_marginal_division_likelihood(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
) -> float:
    values, _, _ = lineage_node_log_likelihood(
        params, prepared, birth_pivot, z_nodes
    )
    if not np.all(np.isfinite(values)):
        return -np.inf
    joint_log_weights = (log_weights[:, None] + log_weights[None, :]).ravel()
    result = np.sum(logsumexp(values + joint_log_weights[None, :], axis=1))
    return float(result) if np.isfinite(result) else -np.inf


def sample_conditional_lineage_effects(
    params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    values, u, omega = lineage_node_log_likelihood(
        params, prepared, birth_pivot, z_nodes
    )
    joint_log_weights = (log_weights[:, None] + log_weights[None, :]).ravel()
    log_prob = values + joint_log_weights[None, :]
    log_prob -= logsumexp(log_prob, axis=1, keepdims=True)
    cumulative = np.cumsum(np.exp(log_prob), axis=1)
    indices = np.sum(cumulative < rng.uniform(size=(len(values), 1)), axis=1)
    indices = np.minimum(indices, len(u) - 1)
    return u[indices], omega[indices]
