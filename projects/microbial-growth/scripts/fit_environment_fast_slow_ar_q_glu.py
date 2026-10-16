#!/usr/bin/env python3
"""Fit Glu with shared cycle effects plus fast and slow lineage AR modes in q."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
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
ENVIRONMENT_ONLY = BASE / "shared_cycle_environment_q_protein_hazard"
OUT = BASE / "environment_fast_slow_ar_q_protein_hazard"


def ar_covariance(horizon, rho_fast, sigma_fast, rho_slow, sigma_slow):
    """Stationary covariance of the sum of two independent AR(1) modes."""
    distance = np.abs(np.subtract.outer(np.arange(horizon), np.arange(horizon)))
    fast_variance = sigma_fast**2 / (1.0 - rho_fast**2)
    slow_variance = sigma_slow**2 / (1.0 - rho_slow**2)
    return fast_variance * rho_fast**distance + slow_variance * rho_slow**distance


def log_likelihood(q, theta, environment=True):
    """Exact balanced-panel likelihood after integrating the shared E vector."""
    mu, rho_fast, rho_slow, log_sigma_fast, log_sigma_slow = theta[:5]
    sigma_fast, sigma_slow = math.exp(log_sigma_fast), math.exp(log_sigma_slow)
    sigma_environment = math.exp(theta[5]) if environment else 1e-10
    lineages, horizon = q.shape
    covariance = ar_covariance(
        horizon, rho_fast, sigma_fast, rho_slow, sigma_slow
    )
    shared_covariance = covariance + lineages * sigma_environment**2 * np.eye(horizon)
    try:
        factor = cho_factor(covariance, lower=True, check_finite=False)
        shared_factor = cho_factor(shared_covariance, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf
    mean = q.mean(axis=0)
    centered = q - mean
    difference = mean - mu
    within_quadratic = float(
        np.sum(centered * cho_solve(factor, centered.T, check_finite=False).T)
    )
    mean_quadratic = lineages * float(
        difference @ cho_solve(shared_factor, difference, check_finite=False)
    )
    logdet = 2.0 * float(np.log(np.diag(factor[0])).sum())
    shared_logdet = 2.0 * float(np.log(np.diag(shared_factor[0])).sum())
    return -0.5 * (
        lineages * horizon * math.log(2.0 * math.pi)
        + (lineages - 1) * logdet
        + shared_logdet
        + within_quadratic
        + mean_quadratic
    )


def fit_q(q, source_q):
    starts = [
        [
            source_q["mean"],
            source_q["rho_fast"],
            source_q["rho_slow"],
            math.log(source_q["sigma_fast_innovation"]),
            math.log(source_q["sigma_slow_innovation"]),
            math.log(0.014),
        ],
        [-0.52, 0.0, 0.90, math.log(0.073), math.log(0.014), math.log(0.018)],
        [-0.52, -0.15, 0.97, math.log(0.075), math.log(0.009), math.log(0.012)],
        [-0.52, 0.15, 0.75, math.log(0.070), math.log(0.025), math.log(0.020)],
    ]
    bounds = [
        (-0.7, -0.3),
        (-0.7, 0.7),
        (0.3, 0.995),
        (math.log(1e-4), math.log(0.3)),
        (math.log(1e-4), math.log(0.3)),
        (math.log(1e-5), math.log(0.2)),
    ]
    fits = [
        minimize(
            lambda theta: -log_likelihood(q, theta),
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2500, "ftol": 1e-13},
        )
        for start in starts
    ]
    fitted = min(fits, key=lambda result: result.fun)
    theta = fitted.x

    null_starts = [start[:5] for start in starts]
    null_bounds = bounds[:5]
    null_fits = [
        minimize(
            lambda value: -log_likelihood(q, value, environment=False),
            start,
            method="L-BFGS-B",
            bounds=null_bounds,
            options={"maxiter": 2500, "ftol": 1e-13},
        )
        for start in null_starts
    ]
    null = min(null_fits, key=lambda result: result.fun)

    mu, rho_fast, rho_slow = map(float, theta[:3])
    sigma_fast, sigma_slow, sigma_environment = map(math.exp, theta[3:])
    horizon = q.shape[1]
    covariance = ar_covariance(
        horizon, rho_fast, sigma_fast, rho_slow, sigma_slow
    )
    cell_mean_covariance = covariance / q.shape[0]
    posterior_data_covariance = (
        cell_mean_covariance + sigma_environment**2 * np.eye(horizon)
    )
    solved = np.linalg.solve(posterior_data_covariance, np.eye(horizon))
    posterior_mean = sigma_environment**2 * np.linalg.solve(
        posterior_data_covariance, q.mean(axis=0) - mu
    )
    posterior_covariance = (
        sigma_environment**2 * np.eye(horizon)
        - sigma_environment**4 * solved
    )
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(posterior_covariance)
    posterior_covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T

    fast_sd = sigma_fast / math.sqrt(1.0 - rho_fast**2)
    slow_sd = sigma_slow / math.sqrt(1.0 - rho_slow**2)
    loglik = -float(fitted.fun)
    null_loglik = -float(null.fun)
    return {
        "mean": mu,
        "rho_fast": rho_fast,
        "sigma_fast_innovation": sigma_fast,
        "stationary_fast_sd": fast_sd,
        "rho_slow": rho_slow,
        "sigma_slow_innovation": sigma_slow,
        "stationary_slow_sd": slow_sd,
        "environment_prior_sd": sigma_environment,
        "posterior_mean_E": posterior_mean,
        "posterior_covariance_E": posterior_covariance,
        "posterior_sd_E": np.sqrt(np.diag(posterior_covariance)),
        "raw_generation_mean_q": q.mean(axis=0),
        "log_likelihood": loglik,
        "no_environment_log_likelihood": null_loglik,
        "twice_log_likelihood_gain": 2.0 * (loglik - null_loglik),
        "aic_gain": 2.0 * (loglik - null_loglik) - 2.0,
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "multistart_log_likelihoods": [-float(result.fun) for result in fits],
    }


def draw_environment(qfit, replicates, rng):
    return rng.multivariate_normal(
        qfit["posterior_mean_E"],
        qfit["posterior_covariance_E"],
        size=replicates,
        check_valid="raise",
    )


def simulate(params, qfit, lengths, replicates=500, seed=42611):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages, horizon = len(lengths), int(lengths.max())
    shape = (replicates, lineages)
    log_birth = np.zeros((replicates, lineages, horizon))
    alpha = np.zeros_like(log_birth)
    growth = np.full_like(log_birth, np.nan)
    time = np.full_like(log_birth, np.nan)
    q = np.full_like(log_birth, np.nan)
    q_fast = np.full_like(log_birth, np.nan)
    q_slow = np.full_like(log_birth, np.nan)
    current = (
        params["initial_birth"]["log_mean"]
        + params["initial_birth"]["log_sd"] * rng.normal(size=shape)
    )
    alpha_latent = rng.normal(size=shape)
    fast = qfit["stationary_fast_sd"] * rng.normal(size=shape)
    slow = qfit["stationary_slow_sd"] * rng.normal(size=shape)
    environment = draw_environment(qfit, replicates, rng)
    afit, hazard = params["alpha"], params["hazard"]

    for generation in range(horizon):
        active = lengths > generation
        if generation:
            alpha_latent[:, active] = (
                afit["rho"] * alpha_latent[:, active]
                + math.sqrt(1.0 - afit["rho"] ** 2)
                * rng.normal(size=(replicates, int(active.sum())))
            )
            fast[:, active] = (
                qfit["rho_fast"] * fast[:, active]
                + qfit["sigma_fast_innovation"]
                * rng.normal(size=(replicates, int(active.sum())))
            )
            slow[:, active] = (
                qfit["rho_slow"] * slow[:, active]
                + qfit["sigma_slow_innovation"]
                * rng.normal(size=(replicates, int(active.sum())))
            )
        log_birth[:, active, generation] = current[:, active]
        alpha[:, active, generation] = transform_gamma(
            alpha_latent[:, active], afit["shape"], afit["scale"]
        )
        continuing = lengths > generation + 1
        if not continuing.any():
            continue
        if generation < environment.shape[1]:
            effect = environment[:, generation, None]
        else:
            effect = qfit["environment_prior_sd"] * rng.normal(
                size=(replicates, 1)
            )
        q_value = (
            qfit["mean"]
            + effect
            + fast[:, continuing]
            + slow[:, continuing]
        )
        growth_value = sample_growth_time(
            current[:, continuing].ravel(),
            math.log(hazard["u"]),
            hazard["omega2"],
            hazard["eta"],
            rng,
        ).reshape(replicates, int(continuing.sum()))
        q[:, continuing, generation] = q_value
        q_fast[:, continuing, generation] = fast[:, continuing]
        q_slow[:, continuing, generation] = slow[:, continuing]
        growth[:, continuing, generation] = growth_value
        time[:, continuing, generation] = (
            growth_value / alpha[:, continuing, generation]
        )
        current[:, continuing] = current[:, continuing] + growth_value + q_value
    return {
        "birth": np.exp(np.clip(log_birth, -700.0, 700.0)),
        "log_birth": log_birth,
        "alpha": alpha,
        "growth": growth,
        "time": time,
        "q": q,
        "q_fast": q_fast,
        "q_slow": q_slow,
        "environment": environment,
    }


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    source = json.loads((SOURCE / "fit_results.json").read_text())
    params = source["fit"]
    balanced = balanced_arrays()
    qfit = fit_q(balanced["q"], params["q"])
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    generated = simulate(params, qfit, lengths, args.replicates)

    horizon = min(args.horizon, int(lengths.max()) - 1)
    generation = np.arange(horizon + 1)
    complete = lengths > horizon
    observed_mean = observed_birth[complete, : horizon + 1].mean(axis=0)
    simulated_means = generated["birth"][:, complete, : horizon + 1].mean(axis=1)
    simulated_mean = simulated_means.mean(axis=0)
    simulated_low, simulated_high = np.quantile(simulated_means, [0.025, 0.975], axis=0)
    original = pd.read_csv(
        SOURCE / "mean_size_drift/balanced_generation_means.csv"
    ).query("generation <= @horizon")
    environment_only = pd.read_csv(
        ENVIRONMENT_ONLY / "balanced_generation_means.csv"
    ).query("generation <= @horizon")
    pd.DataFrame(
        {
            "generation": generation,
            "observed_mean_birth": observed_mean,
            "simulated_mean_birth": simulated_mean,
            "simulated_lo95_birth": simulated_low,
            "simulated_hi95_birth": simulated_high,
            "environment_only_mean_birth": environment_only.simulated_mean_birth.to_numpy(),
            "original_factorized_mean_birth": original.simulated_mean_birth.to_numpy(),
        }
    ).to_csv(OUT / "balanced_generation_means.csv", index=False)

    birth_acf = summarize_acf(observed_birth, generated["birth"], lengths, args.max_lag)
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

    effect = pd.DataFrame(
        {
            "generation": np.arange(len(qfit["posterior_mean_E"])),
            "observed_mean_q": qfit["raw_generation_mean_q"],
            "posterior_mean_E": qfit["posterior_mean_E"],
            "posterior_sd_E": qfit["posterior_sd_E"],
        }
    )
    effect["posterior_mean_q"] = qfit["mean"] + effect.posterior_mean_E
    effect["posterior_lo95_E"] = effect.posterior_mean_E - 1.96 * effect.posterior_sd_E
    effect["posterior_hi95_E"] = effect.posterior_mean_E + 1.96 * effect.posterior_sd_E
    effect.to_csv(OUT / "environment_posterior.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    axis = axes[0, 0]
    axis.fill_between(generation, simulated_low, simulated_high, color="#c4512c", alpha=0.18)
    axis.plot(generation, simulated_mean, "s-", color="#c4512c", ms=3.5, label="Environment + AR modes")
    axis.plot(generation, environment_only.simulated_mean_birth, ":", color="#718096", label="Environment only")
    axis.plot(generation, original.simulated_mean_birth, "--", color="#a0aec0", label="AR modes only")
    axis.plot(generation, observed_mean, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title="Balanced-cohort mean birth size", xlabel="Generation", ylabel="Mean birth size")
    axis.legend(frameon=False, fontsize=8.1)

    axis = axes[0, 1]
    axis.fill_between(effect.generation, effect.posterior_lo95_E, effect.posterior_hi95_E,
                      color="#c4512c", alpha=0.18)
    axis.plot(effect.generation, effect.posterior_mean_E, "s-", color="#c4512c", ms=3.5)
    axis.axhline(0.0, color="0.6", lw=0.8)
    axis.set(title=r"Posterior environmental effect $E_i$", xlabel="Generation", ylabel=r"$E_i$")

    axis = axes[1, 0]
    axis.fill_between(birth_acf.lag, birth_acf.simulated_lo95, birth_acf.simulated_hi95,
                      color="#c4512c", alpha=0.18)
    axis.plot(birth_acf.lag, birth_acf.simulated_mean, "s-", color="#c4512c", ms=3.5, label="Environment + AR modes")
    axis.plot(birth_acf.lag, birth_acf.observed, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title="Birth-size ACF", xlabel="Lag (generations)", ylabel="Correlation")

    axis = axes[1, 1]
    axis.fill_between(q_acf.lag, q_acf.simulated_lo95, q_acf.simulated_hi95,
                      color="#c4512c", alpha=0.18)
    axis.plot(q_acf.lag, q_acf.simulated_mean, "s-", color="#c4512c", ms=3.5, label="Environment + AR modes")
    axis.plot(q_acf.lag, q_acf.observed, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title=r"Effective-jump ACF ($q$)", xlabel="Lag (generations)", ylabel="Correlation")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    fig.suptitle("Glu: shared cycle environment plus fast and slow lineage q modes")
    fig.savefig(OUT / "environment_fast_slow_q_diagnostics.png", dpi=190)
    fig.savefig(OUT / "environment_fast_slow_q_diagnostics.pdf")
    plt.close(fig)

    simulated_slopes = np.polyfit(generation, simulated_means.T, 1)[0]
    result = {
        "model": {
            "recursion": "log b[l,i+1]=log b[l,i]+alpha[l,i]*tau[l,i]+q[l,i]",
            "q": "q[l,i]=mu_q+E[i]+r_fast[l,i]+r_slow[l,i]",
            "environment": "E[i]~iid Normal(0,sigma_E^2); integrated in the likelihood and inferred jointly with both AR modes",
            "lineage_modes": "Independent stationary Gaussian AR(1) fast and slow modes",
            "hazard": source["model"]["hazard"],
            "alpha": source["model"]["alpha"],
        },
        "fit": {
            "q": serializable(qfit),
            "initial_birth": params["initial_birth"],
            "alpha": params["alpha"],
            "hazard": params["hazard"],
        },
        "simulation": {
            "replicates": args.replicates,
            "replayed_quantities": "none; only contiguous-prefix lineage lengths are reused",
            "observed_generation_20_change": float(observed_mean[-1] - observed_mean[0]),
            "simulated_generation_20_change": float(simulated_mean[-1] - simulated_mean[0]),
            "environment_only_generation_20_change": float(environment_only.simulated_mean_birth.iloc[-1] - environment_only.simulated_mean_birth.iloc[0]),
            "original_generation_20_change": float(original.simulated_mean_birth.iloc[-1] - original.simulated_mean_birth.iloc[0]),
            "observed_linear_slope": float(np.polyfit(generation, observed_mean, 1)[0]),
            "simulated_linear_slope_mean": float(simulated_slopes.mean()),
            "simulated_linear_slope_lo95": float(np.quantile(simulated_slopes, 0.025)),
            "simulated_linear_slope_hi95": float(np.quantile(simulated_slopes, 0.975)),
            "mean_birth_rmse": float(np.sqrt(np.mean((observed_mean - simulated_mean) ** 2))),
            "birth_acf_rmse": float(np.sqrt(np.mean((birth_acf.observed[1:] - birth_acf.simulated_mean[1:]) ** 2))),
            "q_acf_rmse": float(np.sqrt(np.mean((q_acf.observed[1:] - q_acf.simulated_mean[1:]) ** 2))),
        },
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
