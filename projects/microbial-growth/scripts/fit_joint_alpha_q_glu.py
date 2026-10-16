#!/usr/bin/env python3
"""Fit a directional joint marginal model for Glu alpha and two-mode q."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import norm


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from autoregressive_residual_model import sample_growth_time  # noqa: E402
from fit_autoregressive_residual_glu import (  # noqa: E402
    build_transitions,
    contiguous_prefixes,
    fit_hazard,
    fit_initial_birth,
    observed_matrix,
    summarize_acf,
)
from fit_fast_slow_ar_q_glu import fit_q  # noqa: E402
from fit_lineage_offset_ar_q_glu import pooled_acf_replicates, q_matrix  # noqa: E402
from glu_workflow import load_data  # noqa: E402


OUT = (
    PROJECT_DIR
    / "outputs/model_comparison/Glu/joint_alpha_q_protein_hazard"
)


def split_segments(group):
    boundaries = np.r_[
        0,
        1 + np.flatnonzero(np.diff(group.generation.to_numpy(int)) != 1),
        len(group),
    ]
    return [
        group.iloc[left:right].reset_index(drop=True)
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ]


def fit_alpha_marginal(groups):
    values = np.sort(
        np.concatenate([group.alpha.to_numpy(float) for group in groups])
    )
    return values, {
        "type": "empirical quantile marginal with rank-Gaussian copula",
        "n": len(values),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "minimum": float(values[0]),
        "maximum": float(values[-1]),
    }


def normal_score(alpha, sorted_reference):
    alpha = np.asarray(alpha, dtype=float)
    left = np.searchsorted(sorted_reference, alpha, side="left")
    right = np.searchsorted(sorted_reference, alpha, side="right")
    probability = (0.5 * (left + right) + 0.5) / len(sorted_reference)
    return norm.ppf(np.clip(probability, 1e-10, 1.0 - 1e-10))


def empirical_quantile(probability, sorted_reference):
    position = np.clip(probability, 0.0, 1.0) * (len(sorted_reference) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(sorted_reference) - 1)
    weight = position - lower
    return (1.0 - weight) * sorted_reference[lower] + weight * sorted_reference[upper]


def fit_joint_alpha(groups, alpha_reference, q_mean):
    design, target = [], []
    baseline_design = []
    for group in groups:
        for segment in split_segments(group):
            if len(segment) < 4:
                continue
            birth = segment.length_birth.to_numpy(float)
            alpha = segment.alpha.to_numpy(float)
            time = segment.time.to_numpy(float)
            q = np.diff(np.log(birth)) - alpha[:-1] * time[:-1]
            z = normal_score(alpha, alpha_reference)
            centered_q = q - q_mean
            for index in range(2, len(q)):
                design.append(
                    [
                        1.0,
                        z[index - 1],
                        z[index - 2],
                        centered_q[index],
                        centered_q[index - 1],
                        centered_q[index - 2],
                    ]
                )
                baseline_design.append([1.0, z[index - 1], z[index - 2]])
                target.append(z[index])
    design = np.asarray(design)
    baseline_design = np.asarray(baseline_design)
    target = np.asarray(target)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    sigma = float(np.sqrt(np.mean(residual**2)))
    log_likelihood = float(
        np.sum(-0.5 * (residual / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2.0 * math.pi))
    )
    baseline_coefficients = np.linalg.lstsq(baseline_design, target, rcond=None)[0]
    baseline_residual = target - baseline_design @ baseline_coefficients
    baseline_sigma = float(np.sqrt(np.mean(baseline_residual**2)))
    baseline_log_likelihood = float(
        np.sum(
            -0.5 * (baseline_residual / baseline_sigma) ** 2
            - math.log(baseline_sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )
    )
    parameter_gain = 3
    return {
        "intercept": float(coefficients[0]),
        "alpha_ar1": float(coefficients[1]),
        "alpha_ar2": float(coefficients[2]),
        "beta_q_current": float(coefficients[3]),
        "beta_q_lag1": float(coefficients[4]),
        "beta_q_lag2": float(coefficients[5]),
        "sigma_innovation": sigma,
        "n_conditional_observations": len(target),
        "conditional_log_likelihood": log_likelihood,
        "baseline_ar_log_likelihood": baseline_log_likelihood,
        "twice_log_likelihood_gain": 2.0 * (log_likelihood - baseline_log_likelihood),
        "aic_gain": 2.0 * (log_likelihood - baseline_log_likelihood) - 2.0 * parameter_gain,
        "baseline_coefficients": baseline_coefficients.tolist(),
        "baseline_sigma": baseline_sigma,
    }


def simulate(params, lengths, replicates, seed=84531, burn_in=500):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages, horizon = len(lengths), int(lengths.max())
    shape = (replicates, lineages)
    b0, alpha_reference, hazard, qfit, joint = (
        params["initial_birth"],
        params["alpha_reference"],
        params["hazard"],
        params["q"],
        params["joint_alpha_given_q"],
    )
    current = b0["log_mean"] + b0["log_sd"] * rng.normal(size=shape)
    fast = np.zeros(shape)
    slow = np.zeros(shape)
    z_alpha = np.zeros(shape)
    z_alpha_lag1 = np.zeros(shape)
    q_lag1 = np.zeros(shape)
    q_lag2 = np.zeros(shape)

    def advance():
        nonlocal fast, slow, z_alpha, z_alpha_lag1, q_lag1, q_lag2
        fast = (
            qfit["rho_fast"] * fast
            + qfit["sigma_fast_innovation"] * rng.normal(size=shape)
        )
        slow = (
            qfit["rho_slow"] * slow
            + qfit["sigma_slow_innovation"] * rng.normal(size=shape)
        )
        centered_q = fast + slow
        new_z_alpha = (
            joint["intercept"]
            + joint["alpha_ar1"] * z_alpha
            + joint["alpha_ar2"] * z_alpha_lag1
            + joint["beta_q_current"] * centered_q
            + joint["beta_q_lag1"] * q_lag1
            + joint["beta_q_lag2"] * q_lag2
            + joint["sigma_innovation"] * rng.normal(size=shape)
        )
        z_alpha_lag1, z_alpha = z_alpha, new_z_alpha
        q_lag2, q_lag1 = q_lag1, centered_q
        return centered_q

    for _ in range(burn_in):
        advance()
    phi1 = joint["alpha_ar1"]
    phi2 = joint["alpha_ar2"]
    beta0 = joint["beta_q_current"]
    beta1 = joint["beta_q_lag1"]
    beta2 = joint["beta_q_lag2"]
    transition_matrix = np.array(
        [
            [phi1, phi2, beta0 * qfit["rho_fast"], beta0 * qfit["rho_slow"], beta1, beta2],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, qfit["rho_fast"], 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, qfit["rho_slow"], 0.0, 0.0],
            [0.0, 0.0, qfit["rho_fast"], qfit["rho_slow"], 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    noise_map = np.array(
        [
            [joint["sigma_innovation"], beta0 * qfit["sigma_fast_innovation"], beta0 * qfit["sigma_slow_innovation"]],
            [0.0, 0.0, 0.0],
            [0.0, qfit["sigma_fast_innovation"], 0.0],
            [0.0, 0.0, qfit["sigma_slow_innovation"]],
            [0.0, qfit["sigma_fast_innovation"], qfit["sigma_slow_innovation"]],
            [0.0, 0.0, 0.0],
        ]
    )
    stationary_mean = np.linalg.solve(
        np.eye(6) - transition_matrix,
        np.array([joint["intercept"], 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    stationary_covariance = solve_discrete_lyapunov(
        transition_matrix, noise_map @ noise_map.T
    )
    alpha_z_mean = float(stationary_mean[0])
    alpha_z_sd = float(math.sqrt(stationary_covariance[0, 0]))
    log_birth = np.zeros((replicates, lineages, horizon))
    alpha = np.zeros_like(log_birth)
    q = np.full_like(log_birth, np.nan)
    q_fast = np.full_like(log_birth, np.nan)
    q_slow = np.full_like(log_birth, np.nan)
    growth = np.full_like(log_birth, np.nan)
    time = np.full_like(log_birth, np.nan)
    for generation in range(horizon):
        centered_q = advance()
        active = lengths > generation
        log_birth[:, active, generation] = current[:, active]
        alpha[:, active, generation] = empirical_quantile(
            norm.cdf((z_alpha[:, active] - alpha_z_mean) / alpha_z_sd),
            alpha_reference,
        )
        continuing = lengths > generation + 1
        if not continuing.any():
            continue
        q_value = qfit["mean"] + centered_q[:, continuing]
        growth_time = sample_growth_time(
            current[:, continuing].ravel(),
            math.log(hazard["u"]),
            hazard["omega2"],
            hazard["eta"],
            rng,
        ).reshape(replicates, int(continuing.sum()))
        q[:, continuing, generation] = q_value
        q_fast[:, continuing, generation] = fast[:, continuing]
        q_slow[:, continuing, generation] = slow[:, continuing]
        growth[:, continuing, generation] = growth_time
        time[:, continuing, generation] = growth_time / alpha[:, continuing, generation]
        current[:, continuing] = current[:, continuing] + growth_time + q_value
    return {
        "log_birth": log_birth,
        "birth": np.exp(np.clip(log_birth, -700.0, 700.0)),
        "alpha": alpha,
        "q": q,
        "q_fast": q_fast,
        "q_slow": q_slow,
        "growth": growth,
        "time": time,
    }


def observed_alpha_matrix(prefixes):
    lengths = np.asarray([max(len(group) - 1, 0) for group in prefixes], dtype=int)
    matrix = np.zeros((len(prefixes), int(lengths.max())))
    for index, group in enumerate(prefixes):
        matrix[index, : lengths[index]] = group.alpha.to_numpy(float)[:-1]
    return matrix, lengths


def cross_correlation_replicates(alpha, q, lengths, lags):
    if alpha.ndim == 2:
        alpha = alpha[None, ...]
    if q.ndim == 2:
        q = q[None, ...]
    result = np.full((len(alpha), len(lags)), np.nan)
    for replicate in range(len(alpha)):
        for column, lag in enumerate(lags):
            left, right = [], []
            for lineage, length in enumerate(lengths):
                if length <= abs(lag):
                    continue
                a = alpha[replicate, lineage, :length]
                values = q[replicate, lineage, :length]
                if lag >= 0:
                    left.append(a[: length - lag] if lag else a)
                    right.append(values[lag:])
                else:
                    offset = -lag
                    left.append(a[offset:])
                    right.append(values[:-offset])
            if left:
                result[replicate, column] = np.corrcoef(
                    np.concatenate(left), np.concatenate(right)
                )[0, 1]
    return result


def diagnostic_table(observed, simulated, lags):
    low, high = np.nanquantile(simulated, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "lag": lags,
            "observed": observed,
            "simulated_mean": np.nanmean(simulated, axis=0),
            "simulated_lo95": low,
            "simulated_hi95": high,
        }
    )


def plot_diagnostics(alpha_acf, q_acf, cross, birth_acf):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True)
    specs = [
        (axes[0, 0], alpha_acf, r"Growth-rate ACF ($\alpha$)"),
        (axes[0, 1], q_acf, r"Effective-jump ACF ($q$)"),
        (axes[1, 0], cross, r"Cross-correlation $\mathrm{Corr}(\alpha_i,q_{i+k})$"),
        (axes[1, 1], birth_acf, "Birth-size ACF"),
    ]
    for axis, table, title in specs:
        axis.fill_between(table.lag, table.simulated_lo95, table.simulated_hi95, color="#c4512c", alpha=0.20, label="Joint model, 95%")
        axis.plot(table.lag, table.simulated_mean, color="#c4512c", marker="s", ms=3.3, label="Joint model")
        axis.plot(table.lag, table.observed, color="#23364d", marker="o", ms=3.3, label="Observed")
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set(title=title, xlabel="Lag (generations)", ylabel="Correlation")
        axis.grid(alpha=0.16)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Glu: joint marginal model for growth rate and effective jump", fontsize=15)
    fig.savefig(OUT / "joint_alpha_q_diagnostics.png", dpi=190)
    fig.savefig(OUT / "joint_alpha_q_diagnostics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--max-lag", type=int, default=15)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    transition = build_transitions(groups)
    alpha_reference, alpha_marginal = fit_alpha_marginal(groups)
    qfit = fit_q(groups)
    joint = fit_joint_alpha(groups, alpha_reference, qfit["mean"])
    params = {
        "initial_birth": fit_initial_birth(groups),
        "alpha_marginal": alpha_marginal,
        "alpha_reference": alpha_reference,
        "hazard": fit_hazard(transition),
        "q": qfit,
        "joint_alpha_given_q": joint,
    }
    for name, values in params.items():
        if name != "alpha_reference":
            print(name, values, flush=True)
    observed_birth, birth_lengths = observed_matrix(prefixes)
    generated = simulate(params, birth_lengths, args.replicates)
    birth_acf = summarize_acf(
        observed_birth, generated["birth"], birth_lengths, args.max_lag
    )
    observed_q, lengths = q_matrix(prefixes)
    observed_alpha, _ = observed_alpha_matrix(prefixes)
    q_sim_acf = pooled_acf_replicates(generated["q"], lengths, args.max_lag)
    alpha_sim_acf = pooled_acf_replicates(
        generated["alpha"][:, :, : observed_alpha.shape[1]], lengths, args.max_lag
    )
    q_acf = diagnostic_table(
        pooled_acf_replicates(observed_q, lengths, args.max_lag)[0],
        q_sim_acf,
        np.arange(args.max_lag + 1),
    )
    alpha_acf = diagnostic_table(
        pooled_acf_replicates(observed_alpha, lengths, args.max_lag)[0],
        alpha_sim_acf,
        np.arange(args.max_lag + 1),
    )
    lags = np.arange(-8, 9)
    observed_cross = cross_correlation_replicates(
        observed_alpha, observed_q, lengths, lags
    )[0]
    simulated_cross = cross_correlation_replicates(
        generated["alpha"][:, :, : observed_alpha.shape[1]],
        generated["q"],
        lengths,
        lags,
    )
    cross = diagnostic_table(observed_cross, simulated_cross, lags)
    for filename, table in [
        ("alpha_acf.csv", alpha_acf),
        ("q_acf.csv", q_acf),
        ("alpha_q_cross_correlation.csv", cross),
        ("birth_size_acf.csv", birth_acf),
    ]:
        table.to_csv(OUT / filename, index=False)
    plot_diagnostics(alpha_acf, q_acf, cross, birth_acf)
    observed_alpha_flat = np.concatenate(
        [observed_alpha[index, :length] for index, length in enumerate(lengths)]
    )
    simulated_alpha_flat = np.concatenate(
        [generated["alpha"][:, index, :length].ravel() for index, length in enumerate(lengths)]
    )
    observed_q_flat = np.concatenate(
        [observed_q[index, :length] for index, length in enumerate(lengths)]
    )
    simulated_q_flat = np.concatenate(
        [generated["q"][:, index, :length].ravel() for index, length in enumerate(lengths)]
    )
    observed_birth_flat = np.concatenate(
        [observed_birth[index, :length] for index, length in enumerate(birth_lengths)]
    )
    simulated_birth_flat = np.concatenate(
        [generated["birth"][:, index, :length].ravel() for index, length in enumerate(birth_lengths)]
    )
    result = {
        "model": {
            "alpha_marginal": "Empirical quantile marginal with rank-Gaussian score z_alpha",
            "q": "sum of independent fast and slow Gaussian AR(1) modes",
            "joint": "z_alpha[i]=c+phi1*z_alpha[i-1]+phi2*z_alpha[i-2]+beta0*qtilde[i]+beta1*qtilde[i-1]+beta2*qtilde[i-2]+sigma*z[i]",
            "factorization": "p(q path) p(alpha path | q path)",
        },
        "fit": {key: value for key, value in params.items() if key != "alpha_reference"},
        "simulation": {
            "replicates": args.replicates,
            "replayed_quantities": "none; only contiguous-prefix lineage lengths are reused",
            "observed_alpha_mean_sd": [float(observed_alpha_flat.mean()), float(observed_alpha_flat.std())],
            "simulated_alpha_mean_sd": [float(simulated_alpha_flat.mean()), float(simulated_alpha_flat.std())],
            "observed_q_mean_sd": [float(observed_q_flat.mean()), float(observed_q_flat.std())],
            "simulated_q_mean_sd": [float(simulated_q_flat.mean()), float(simulated_q_flat.std())],
            "observed_birth_mean_sd": [float(observed_birth_flat.mean()), float(observed_birth_flat.std())],
            "simulated_birth_mean_sd": [float(simulated_birth_flat.mean()), float(simulated_birth_flat.std())],
            "alpha_acf_rmse_lags_1_to_max": float(np.sqrt(np.mean((alpha_acf.observed[1:] - alpha_acf.simulated_mean[1:]) ** 2))),
            "q_acf_rmse_lags_1_to_max": float(np.sqrt(np.mean((q_acf.observed[1:] - q_acf.simulated_mean[1:]) ** 2))),
            "cross_correlation_rmse": float(np.sqrt(np.mean((cross.observed - cross.simulated_mean) ** 2))),
            "birth_acf_rmse_lags_1_to_max": float(np.sqrt(np.mean((birth_acf.observed[1:] - birth_acf.simulated_mean[1:]) ** 2))),
        },
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2))
    print(cross.to_string(index=False), flush=True)
    print(json.dumps(result["simulation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
