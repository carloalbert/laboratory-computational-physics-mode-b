#!/usr/bin/env python3
"""Test iid actual fractions plus posterior-inferred shared cycle epsilon effects."""

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
from scipy.stats import beta as beta_distribution


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
OUT = BASE / "shared_cycle_epsilon_protein_hazard"


def fit_epsilon_random_effect(epsilon, mean_bounds=(-0.2, 0.2)):
    """Empirical-Bayes normal-normal fit for one shared effect per column."""
    epsilon = np.asarray(epsilon, dtype=float)
    n = epsilon.shape[0]
    means = epsilon.mean(axis=0)
    within_sums = np.sum((epsilon - means) ** 2, axis=0)

    def objective(theta):
        mu, log_cell_sd, log_environment_sd = theta
        cell_variance = math.exp(2.0 * log_cell_sd)
        environment_variance = math.exp(2.0 * log_environment_sd)
        # Exact marginal likelihood after integrating the shared epsilon_i.
        determinant = (n - 1) * math.log(cell_variance) + math.log(
            cell_variance + n * environment_variance
        )
        quadratic = within_sums / cell_variance + n * (means - mu) ** 2 / (
            cell_variance + n * environment_variance
        )
        return 0.5 * float(
            np.sum(n * math.log(2.0 * math.pi) + determinant + quadratic)
        )

    fitted = minimize(
        objective,
        [float(epsilon.mean()), math.log(0.05), math.log(0.008)],
        method="L-BFGS-B",
        bounds=[mean_bounds, (math.log(1e-4), math.log(0.5)), (math.log(1e-5), math.log(0.2))],
        options={"ftol": 1e-14, "maxiter": 2000},
    )
    mu, log_cell_sd, log_environment_sd = map(float, fitted.x)
    cell_variance = math.exp(2.0 * log_cell_sd)
    environment_variance = math.exp(2.0 * log_environment_sd)
    posterior_variance = 1.0 / (1.0 / environment_variance + n / cell_variance)
    posterior_mean = posterior_variance * (
        mu / environment_variance + n * means / cell_variance
    )
    return {
        "prior_mean": mu,
        "prior_sd": math.sqrt(environment_variance),
        "within_cycle_observation_sd": math.sqrt(cell_variance),
        "posterior_mean": posterior_mean,
        "posterior_sd": np.full_like(posterior_mean, math.sqrt(posterior_variance)),
        "raw_generation_mean": means,
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "log_likelihood": -float(fitted.fun),
    }


def fit_fraction(fraction):
    shape1, shape2, _, _ = beta_distribution.fit(
        np.asarray(fraction, dtype=float).ravel(), floc=0.0, fscale=1.0
    )
    return {
        "shape1": float(shape1),
        "shape2": float(shape2),
        "mean": float(shape1 / (shape1 + shape2)),
        "mean_log": float(
            __import__("scipy").special.digamma(shape1)
            - __import__("scipy").special.digamma(shape1 + shape2)
        ),
    }


def simulate(params, epsilon_fit, fraction_fit, lengths, replicates=500, seed=81512):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages, horizon = len(lengths), int(lengths.max())
    shape = (replicates, lineages)
    log_birth = np.zeros((replicates, lineages, horizon))
    alpha = np.zeros_like(log_birth)
    time = np.full_like(log_birth, np.nan)
    q = np.full_like(log_birth, np.nan)
    fraction = np.full_like(log_birth, np.nan)

    current = (
        params["initial_birth"]["log_mean"]
        + params["initial_birth"]["log_sd"] * rng.normal(size=shape)
    )
    alpha_latent = rng.normal(size=shape)
    afit, hazard = params["alpha"], params["hazard"]
    cycle_count = len(epsilon_fit["posterior_mean"])
    shared_epsilon = (
        epsilon_fit["posterior_mean"][None, :]
        + epsilon_fit["posterior_sd"][None, :]
        * rng.normal(size=(replicates, cycle_count))
    )

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
        if generation >= cycle_count:
            environmental = (
                epsilon_fit["prior_mean"]
                + epsilon_fit["prior_sd"] * rng.normal(size=(replicates, 1))
            )
        else:
            environmental = shared_epsilon[:, generation, None]
        sampled_fraction = rng.beta(
            fraction_fit["shape1"],
            fraction_fit["shape2"],
            size=(replicates, int(continuing.sum())),
        )
        q_value = np.log(sampled_fraction) + environmental
        growth_time = sample_growth_time(
            current[:, continuing].ravel(),
            math.log(hazard["u"]),
            hazard["omega2"],
            hazard["eta"],
            rng,
        ).reshape(replicates, int(continuing.sum()))
        q[:, continuing, generation] = q_value
        fraction[:, continuing, generation] = sampled_fraction
        time[:, continuing, generation] = growth_time / alpha[:, continuing, generation]
        current[:, continuing] = current[:, continuing] + growth_time + q_value
    return {
        "birth": np.exp(np.clip(log_birth, -700.0, 700.0)),
        "log_birth": log_birth,
        "alpha": alpha,
        "time": time,
        "q": q,
        "fraction": fraction,
        "shared_epsilon": shared_epsilon,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = balanced_arrays()
    epsilon_fit = fit_epsilon_random_effect(data["epsilon"])
    fraction_fit = fit_fraction(data["f"])
    source = json.loads((SOURCE / "fit_results.json").read_text())
    params = source["fit"]
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    generated = simulate(
        params, epsilon_fit, fraction_fit, lengths, args.replicates
    )

    horizon = min(args.horizon, int(lengths.max()) - 1)
    generation = np.arange(horizon + 1)
    balanced = lengths > horizon
    observed_mean = observed_birth[balanced, : horizon + 1].mean(axis=0)
    simulated_means = generated["birth"][:, balanced, : horizon + 1].mean(axis=1)
    simulated_mean = simulated_means.mean(axis=0)
    simulated_low, simulated_high = np.quantile(
        simulated_means, [0.025, 0.975], axis=0
    )
    pd.DataFrame(
        {
            "generation": generation,
            "observed_mean_birth": observed_mean,
            "simulated_mean_birth": simulated_mean,
            "simulated_lo95_birth": simulated_low,
            "simulated_hi95_birth": simulated_high,
        }
    ).to_csv(OUT / "balanced_generation_means.csv", index=False)

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

    posterior = pd.DataFrame(
        {
            "generation": np.arange(len(epsilon_fit["posterior_mean"])),
            "raw_mean_epsilon": epsilon_fit["raw_generation_mean"],
            "posterior_mean_epsilon": epsilon_fit["posterior_mean"],
            "posterior_sd_epsilon": epsilon_fit["posterior_sd"],
        }
    )
    posterior["posterior_lo95_epsilon"] = (
        posterior.posterior_mean_epsilon - 1.96 * posterior.posterior_sd_epsilon
    )
    posterior["posterior_hi95_epsilon"] = (
        posterior.posterior_mean_epsilon + 1.96 * posterior.posterior_sd_epsilon
    )
    posterior.to_csv(OUT / "epsilon_posterior.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axis = axes[0]
    axis.fill_between(generation, simulated_low, simulated_high, color="#c4512c", alpha=0.18)
    axis.plot(generation, simulated_mean, "s-", color="#c4512c", ms=3.5, label="Shared epsilon")
    axis.plot(generation, observed_mean, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title="Balanced-cohort mean birth size", xlabel="Generation", ylabel="Mean birth size")
    axis.legend(frameon=False)

    axis = axes[1]
    axis.fill_between(
        posterior.generation,
        posterior.posterior_lo95_epsilon,
        posterior.posterior_hi95_epsilon,
        color="#c4512c",
        alpha=0.18,
    )
    axis.plot(posterior.generation, posterior.posterior_mean_epsilon, "s-", color="#c4512c", ms=3.5)
    axis.plot(posterior.generation, posterior.raw_mean_epsilon, "o-", color="#23364d", ms=3.5)
    axis.set(title=r"Shared $\epsilon_i$ posterior", xlabel="Generation", ylabel=r"$\epsilon_i$")

    axis = axes[2]
    axis.fill_between(birth_acf.lag, birth_acf.simulated_lo95, birth_acf.simulated_hi95,
                      color="#c4512c", alpha=0.18)
    axis.plot(birth_acf.lag, birth_acf.simulated_mean, "s-", color="#c4512c", ms=3.5, label="Shared epsilon")
    axis.plot(birth_acf.lag, birth_acf.observed, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title="Birth-size ACF", xlabel="Lag (generations)", ylabel="Correlation")
    for axis in axes:
        axis.grid(alpha=0.16)
    fig.suptitle("Glu: iid actual fractions plus posterior shared cycle epsilon")
    fig.savefig(OUT / "shared_cycle_epsilon_diagnostics.png", dpi=190)
    fig.savefig(OUT / "shared_cycle_epsilon_diagnostics.pdf")
    plt.close(fig)

    simulated_slopes = np.polyfit(generation, simulated_means.T, 1)[0]
    serializable_epsilon = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in epsilon_fit.items()
    }
    result = {
        "model": {
            "q": "q[l,i]=log(f[l,i])+epsilon[i]; f is iid Beta; epsilon[i] has an iid Normal prior and is shared across lineages",
            "inference_caveat": "Because measured q-log(f) is not identical across lineages, within-cycle scatter is treated as observation noise when inferring epsilon[i], but is not propagated as a third q component in the requested strict simulation.",
            "fraction": fraction_fit,
            "epsilon": serializable_epsilon,
            "alpha": source["model"]["alpha"],
            "hazard": source["model"]["hazard"],
        },
        "simulation": {
            "replicates": args.replicates,
            "observed_generation_20_change": float(observed_mean[-1] - observed_mean[0]),
            "simulated_generation_20_change": float(simulated_mean[-1] - simulated_mean[0]),
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
