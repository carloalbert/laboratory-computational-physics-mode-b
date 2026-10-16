#!/usr/bin/env python3
"""Replace the slow lineage q mode by a generation-shared random walk."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from analyze_tau_q_epsilon_correlations import balanced_arrays  # noqa: E402
from autoregressive_residual_model import sample_growth_time, transform_gamma  # noqa: E402
from fit_autoregressive_residual_glu import contiguous_prefixes, observed_matrix, summarize_acf  # noqa: E402
from fit_lineage_offset_ar_q_glu import pooled_acf_replicates, q_matrix  # noqa: E402
from glu_workflow import load_data  # noqa: E402


BASE = PROJECT_DIR / "outputs/model_comparison/Glu"
SOURCE = BASE / "fast_slow_ar_q_protein_hazard"
OUT = BASE / "shared_random_walk_q_protein_hazard"


def kalman_log_likelihood(mean_q, observation_variance, mu, drift, sigma):
    state_mean = 0.0
    state_variance = 0.0
    value = 0.0
    for generation, observed in enumerate(mean_q):
        if generation:
            state_mean += drift
            state_variance += sigma**2
        variance = state_variance + observation_variance[generation]
        residual = observed - mu - state_mean
        value += -0.5 * (
            math.log(2.0 * math.pi * variance) + residual**2 / variance
        )
        gain = state_variance / variance
        state_mean += gain * residual
        state_variance *= 1.0 - gain
    return value


def fit_shared_q(q):
    q = np.asarray(q, dtype=float)
    mean_q = q.mean(axis=0)
    observation_variance = q.var(axis=0, ddof=1) / len(q)
    generation = np.arange(q.shape[1])
    slope, intercept = np.polyfit(generation, mean_q, 1)

    def objective(theta):
        return -kalman_log_likelihood(
            mean_q,
            observation_variance,
            theta[0],
            theta[1],
            math.exp(theta[2]),
        )

    fit = minimize(
        objective,
        [intercept, slope, math.log(1e-4)],
        method="L-BFGS-B",
        bounds=[(-1.0, -0.1), (-0.02, 0.02), (math.log(1e-8), math.log(0.05))],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    mu, drift, log_sigma = map(float, fit.x)
    sigma = math.exp(log_sigma)

    # Estimate the fast lineage-specific component after removing the observed
    # generation mean. This prevents the common trend from appearing as
    # lineage memory.
    residual = q - mean_q
    rho_fast = float(
        np.sum(residual[:, :-1] * residual[:, 1:])
        / np.sum(residual[:, :-1] ** 2)
    )
    innovation = residual[:, 1:] - rho_fast * residual[:, :-1]
    sigma_fast = float(np.sqrt(np.mean(innovation**2)))
    stationary_fast_sd = sigma_fast / math.sqrt(1.0 - rho_fast**2)

    # A zero-drift nested fit tests whether the upward common trend is needed.
    def zero_drift_objective(theta):
        return -kalman_log_likelihood(
            mean_q,
            observation_variance,
            theta[0],
            0.0,
            math.exp(theta[1]),
        )

    zero = minimize(
        zero_drift_objective,
        [float(mean_q.mean()), math.log(1e-3)],
        method="L-BFGS-B",
        bounds=[(-1.0, -0.1), (math.log(1e-8), math.log(0.05))],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    log_likelihood = -float(fit.fun)
    zero_log_likelihood = -float(zero.fun)
    return {
        "mu": mu,
        "drift_per_generation": drift,
        "shared_innovation_sd": sigma,
        "rho_fast": rho_fast,
        "sigma_fast_innovation": sigma_fast,
        "stationary_fast_sd": stationary_fast_sd,
        "log_likelihood_generation_means": log_likelihood,
        "zero_drift_log_likelihood_generation_means": zero_log_likelihood,
        "aic_gain_over_zero_drift": 2.0 * (log_likelihood - zero_log_likelihood) - 2.0,
        "success": bool(fit.success),
        "message": str(fit.message),
        "observed_generation_mean_q": mean_q.tolist(),
        "observation_sd_generation_mean_q": np.sqrt(observation_variance).tolist(),
    }


def simulate(params, qfit, lengths, replicates=500, seed=716231):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages, horizon = len(lengths), int(lengths.max())
    shape = (replicates, lineages)
    log_birth = np.zeros((replicates, lineages, horizon))
    alpha = np.zeros_like(log_birth)
    growth = np.full_like(log_birth, np.nan)
    time = np.full_like(log_birth, np.nan)
    q = np.full_like(log_birth, np.nan)

    current = (
        params["initial_birth"]["log_mean"]
        + params["initial_birth"]["log_sd"] * rng.normal(size=shape)
    )
    alpha_latent = rng.normal(size=shape)
    fast = qfit["stationary_fast_sd"] * rng.normal(size=shape)
    shared = np.zeros((replicates, 1))
    afit, hazard = params["alpha"], params["hazard"]

    for generation in range(horizon):
        active = lengths > generation
        if generation:
            alpha_latent[:, active] = (
                afit["rho"] * alpha_latent[:, active]
                + math.sqrt(1.0 - afit["rho"] ** 2)
                * rng.normal(size=(replicates, int(active.sum())))
            )
        log_birth[:, active, generation] = current[:, active]
        alpha[:, active, generation] = transform_gamma(
            alpha_latent[:, active], afit["shape"], afit["scale"]
        )
        continuing = lengths > generation + 1
        if not continuing.any():
            continue
        if generation:
            shared += qfit["drift_per_generation"] + qfit[
                "shared_innovation_sd"
            ] * rng.normal(size=(replicates, 1))
            fast[:, continuing] = (
                qfit["rho_fast"] * fast[:, continuing]
                + qfit["sigma_fast_innovation"]
                * rng.normal(size=(replicates, int(continuing.sum())))
            )
        q_value = qfit["mu"] + shared + fast[:, continuing]
        growth_time = sample_growth_time(
            current[:, continuing].ravel(),
            math.log(hazard["u"]),
            hazard["omega2"],
            hazard["eta"],
            rng,
        ).reshape(replicates, int(continuing.sum()))
        q[:, continuing, generation] = q_value
        growth[:, continuing, generation] = growth_time
        time[:, continuing, generation] = growth_time / alpha[:, continuing, generation]
        current[:, continuing] = current[:, continuing] + growth_time + q_value
    return {
        "birth": np.exp(np.clip(log_birth, -700.0, 700.0)),
        "log_birth": log_birth,
        "alpha": alpha,
        "time": time,
        "growth": growth,
        "q": q,
    }


def interval(values):
    return (
        values.mean(axis=0),
        np.quantile(values, 0.025, axis=0),
        np.quantile(values, 0.975, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    source = json.loads((SOURCE / "fit_results.json").read_text())
    params = source["fit"]
    qfit = fit_shared_q(balanced_arrays()["q"])
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    generated = simulate(params, qfit, lengths, args.replicates)

    horizon = min(args.horizon, int(lengths.max()) - 1)
    generation = np.arange(horizon + 1)
    balanced = lengths > horizon
    observed_mean = observed_birth[balanced, : horizon + 1].mean(axis=0)
    simulated_means = generated["birth"][:, balanced, : horizon + 1].mean(axis=1)
    simulated_mean, simulated_low, simulated_high = interval(simulated_means)
    original = pd.read_csv(
        SOURCE / "mean_size_drift/balanced_generation_means.csv"
    ).query("generation <= @horizon")

    drift = pd.DataFrame(
        {
            "generation": generation,
            "n_balanced_lineages": int(balanced.sum()),
            "observed_mean_birth": observed_mean,
            "shared_rw_mean_birth": simulated_mean,
            "shared_rw_lo95_birth": simulated_low,
            "shared_rw_hi95_birth": simulated_high,
            "original_factorized_mean_birth": original.simulated_mean_birth.to_numpy(),
            "original_factorized_lo95_birth": original.simulated_lo95_birth.to_numpy(),
            "original_factorized_hi95_birth": original.simulated_hi95_birth.to_numpy(),
        }
    )
    drift.to_csv(OUT / "balanced_generation_means.csv", index=False)

    birth_acf = summarize_acf(
        observed_birth, generated["birth"], lengths, args.max_lag
    )
    birth_acf.to_csv(OUT / "birth_size_acf.csv", index=False)
    observed_q, q_lengths = q_matrix(prefixes)
    observed_q_acf = pooled_acf_replicates(observed_q, q_lengths, args.max_lag)[0]
    simulated_q_acf = pooled_acf_replicates(generated["q"], q_lengths, args.max_lag)
    q_low, q_high = np.nanquantile(simulated_q_acf, [0.025, 0.975], axis=0)
    q_acf = pd.DataFrame(
        {
            "lag": np.arange(args.max_lag + 1),
            "observed": observed_q_acf,
            "simulated_mean": np.nanmean(simulated_q_acf, axis=0),
            "simulated_lo95": q_low,
            "simulated_hi95": q_high,
        }
    )
    q_acf.to_csv(OUT / "q_acf.csv", index=False)

    # The common q mean itself is directly observed with high precision.
    observed_mean_q = np.asarray(qfit["observed_generation_mean_q"])
    q_generation = np.arange(len(observed_mean_q))
    shared_paths = np.zeros((args.replicates, len(observed_mean_q)))
    rng = np.random.default_rng(93517)
    for index in range(1, len(observed_mean_q)):
        shared_paths[:, index] = (
            shared_paths[:, index - 1]
            + qfit["drift_per_generation"]
            + qfit["shared_innovation_sd"] * rng.normal(size=args.replicates)
        )
    predicted_mean_q = qfit["mu"] + shared_paths
    mean_q, low_q, high_q = interval(predicted_mean_q)
    pd.DataFrame(
        {
            "generation": q_generation,
            "observed_mean_q": observed_mean_q,
            "simulated_mean_q": mean_q,
            "simulated_lo95_q": low_q,
            "simulated_hi95_q": high_q,
        }
    ).to_csv(OUT / "generation_mean_q.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axis = axes[0]
    axis.fill_between(generation, simulated_low, simulated_high, color="#c4512c", alpha=0.18)
    axis.plot(generation, simulated_mean, "s-", color="#c4512c", ms=3.5, label="Shared RW q")
    axis.plot(generation, original.simulated_mean_birth, "--", color="#8290a1", label="Original factorized")
    axis.plot(generation, observed_mean, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title="Balanced-cohort mean birth size", xlabel="Generation", ylabel="Mean birth size")
    axis.legend(frameon=False, fontsize=8.5)

    axis = axes[1]
    axis.fill_between(q_generation, low_q, high_q, color="#c4512c", alpha=0.18)
    axis.plot(q_generation, mean_q, "s-", color="#c4512c", ms=3.5, label="Shared RW q")
    axis.plot(q_generation, observed_mean_q, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title=r"Generation mean $q$", xlabel="Generation", ylabel=r"Mean $q$")

    axis = axes[2]
    axis.fill_between(birth_acf.lag, birth_acf.simulated_lo95, birth_acf.simulated_hi95,
                      color="#c4512c", alpha=0.18)
    axis.plot(birth_acf.lag, birth_acf.simulated_mean, "s-", color="#c4512c", ms=3.5, label="Shared RW q")
    axis.plot(birth_acf.lag, birth_acf.observed, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title="Birth-size ACF", xlabel="Lag (generations)", ylabel="Correlation")
    for axis in axes:
        axis.grid(alpha=0.16)
    fig.suptitle("Glu: generation-shared random-walk q in the original hazard model")
    fig.savefig(OUT / "shared_random_walk_q_diagnostics.png", dpi=190)
    fig.savefig(OUT / "shared_random_walk_q_diagnostics.pdf")
    plt.close(fig)

    observed_slope = float(np.polyfit(generation, observed_mean, 1)[0])
    simulated_slopes = np.polyfit(generation, simulated_means.T, 1)[0]
    summary = {
        "model": {
            "q": "q[l,i]=mu+m[i]+r[l,i]; m[i+1]=m[i]+delta+sigma_m*z[i]; r is lineage-specific stationary AR(1)",
            "alpha": source["model"]["alpha"],
            "hazard": source["model"]["hazard"],
            "shared_q_fit": qfit,
        },
        "simulation": {
            "replicates": args.replicates,
            "balanced_lineages": int(balanced.sum()),
            "observed_generation_20_change": float(observed_mean[-1] - observed_mean[0]),
            "shared_rw_generation_20_change": float(simulated_mean[-1] - simulated_mean[0]),
            "original_generation_20_change": float(
                original.simulated_mean_birth.iloc[-1]
                - original.simulated_mean_birth.iloc[0]
            ),
            "observed_linear_slope": observed_slope,
            "shared_rw_linear_slope_mean": float(simulated_slopes.mean()),
            "shared_rw_linear_slope_lo95": float(np.quantile(simulated_slopes, 0.025)),
            "shared_rw_linear_slope_hi95": float(np.quantile(simulated_slopes, 0.975)),
            "mean_birth_rmse": float(np.sqrt(np.mean((observed_mean - simulated_mean) ** 2))),
            "birth_acf_rmse": float(np.sqrt(np.mean((birth_acf.observed[1:] - birth_acf.simulated_mean[1:]) ** 2))),
            "q_acf_rmse": float(np.sqrt(np.mean((q_acf.observed[1:] - q_acf.simulated_mean[1:]) ** 2))),
        },
    }
    (OUT / "fit_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
