#!/usr/bin/env python3
"""Infer one iid-prior environmental q effect for each observed cell cycle."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from analyze_tau_q_epsilon_correlations import balanced_arrays  # noqa: E402
from autoregressive_residual_model import sample_growth_time, transform_gamma  # noqa: E402
from fit_autoregressive_residual_glu import contiguous_prefixes, observed_matrix, summarize_acf  # noqa: E402
from fit_lineage_offset_ar_q_glu import pooled_acf_replicates, q_matrix  # noqa: E402
from fit_shared_cycle_epsilon_glu import fit_epsilon_random_effect  # noqa: E402
from glu_workflow import load_data  # noqa: E402


BASE = PROJECT_DIR / "outputs/model_comparison/Glu"
SOURCE = BASE / "fast_slow_ar_q_protein_hazard"
OUT = BASE / "shared_cycle_environment_q_protein_hazard"


def fit_environment(q):
    location = fit_epsilon_random_effect(q, mean_bounds=(-1.0, -0.1))
    mu = float(location["prior_mean"])
    return {
        "mu_q": mu,
        "environment_prior_sd": float(location["prior_sd"]),
        "cell_q_sd": float(location["within_cycle_observation_sd"]),
        "posterior_mean_E": np.asarray(location["posterior_mean"]) - mu,
        "posterior_sd_E": np.asarray(location["posterior_sd"]),
        "raw_generation_mean_q": np.asarray(location["raw_generation_mean"]),
        "log_likelihood": float(location["log_likelihood"]),
        "success": bool(location["success"]),
        "message": str(location["message"]),
    }


def simulate(params, qfit, lengths, replicates=500, seed=61379):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages, horizon = len(lengths), int(lengths.max())
    shape = (replicates, lineages)
    log_birth = np.zeros((replicates, lineages, horizon))
    alpha = np.zeros_like(log_birth)
    time = np.full_like(log_birth, np.nan)
    q = np.full_like(log_birth, np.nan)

    current = (
        params["initial_birth"]["log_mean"]
        + params["initial_birth"]["log_sd"] * rng.normal(size=shape)
    )
    alpha_latent = rng.normal(size=shape)
    afit, hazard = params["alpha"], params["hazard"]
    cycle_count = len(qfit["posterior_mean_E"])
    environmental = (
        qfit["posterior_mean_E"][None, :]
        + qfit["posterior_sd_E"][None, :]
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
        if generation < cycle_count:
            effect = environmental[:, generation, None]
        else:
            effect = qfit["environment_prior_sd"] * rng.normal(
                size=(replicates, 1)
            )
        q_value = (
            qfit["mu_q"]
            + effect
            + qfit["cell_q_sd"]
            * rng.normal(size=(replicates, int(continuing.sum())))
        )
        growth_time = sample_growth_time(
            current[:, continuing].ravel(),
            math.log(hazard["u"]),
            hazard["omega2"],
            hazard["eta"],
            rng,
        ).reshape(replicates, int(continuing.sum()))
        q[:, continuing, generation] = q_value
        time[:, continuing, generation] = growth_time / alpha[:, continuing, generation]
        current[:, continuing] = current[:, continuing] + growth_time + q_value
    return {
        "birth": np.exp(np.clip(log_birth, -700.0, 700.0)),
        "log_birth": log_birth,
        "alpha": alpha,
        "time": time,
        "q": q,
        "environmental": environmental,
    }


def serializable(fit):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in fit.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = balanced_arrays()
    qfit = fit_environment(data["q"])
    source = json.loads((SOURCE / "fit_results.json").read_text())
    params = source["fit"]
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    generated = simulate(params, qfit, lengths, args.replicates)

    horizon = min(args.horizon, int(lengths.max()) - 1)
    generation = np.arange(horizon + 1)
    balanced = lengths > horizon
    observed_mean = observed_birth[balanced, : horizon + 1].mean(axis=0)
    simulated_means = generated["birth"][:, balanced, : horizon + 1].mean(axis=1)
    simulated_mean = simulated_means.mean(axis=0)
    simulated_low, simulated_high = np.quantile(
        simulated_means, [0.025, 0.975], axis=0
    )
    original = pd.read_csv(
        SOURCE / "mean_size_drift/balanced_generation_means.csv"
    ).query("generation <= @horizon")
    pd.DataFrame(
        {
            "generation": generation,
            "observed_mean_birth": observed_mean,
            "simulated_mean_birth": simulated_mean,
            "simulated_lo95_birth": simulated_low,
            "simulated_hi95_birth": simulated_high,
            "original_factorized_mean_birth": original.simulated_mean_birth.to_numpy(),
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

    effect = pd.DataFrame(
        {
            "generation": np.arange(len(qfit["posterior_mean_E"])),
            "observed_mean_q": qfit["raw_generation_mean_q"],
            "posterior_mean_E": qfit["posterior_mean_E"],
            "posterior_sd_E": qfit["posterior_sd_E"],
        }
    )
    effect["posterior_mean_q"] = qfit["mu_q"] + effect.posterior_mean_E
    effect["posterior_lo95_E"] = effect.posterior_mean_E - 1.96 * effect.posterior_sd_E
    effect["posterior_hi95_E"] = effect.posterior_mean_E + 1.96 * effect.posterior_sd_E
    effect.to_csv(OUT / "environment_posterior.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    axis = axes[0, 0]
    axis.fill_between(generation, simulated_low, simulated_high, color="#c4512c", alpha=0.18)
    axis.plot(generation, simulated_mean, "s-", color="#c4512c", ms=3.5, label="Posterior shared environment")
    axis.plot(generation, original.simulated_mean_birth, "--", color="#8290a1", label="Original factorized")
    axis.plot(generation, observed_mean, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title="Balanced-cohort mean birth size", xlabel="Generation", ylabel="Mean birth size")
    axis.legend(frameon=False, fontsize=8.3)

    axis = axes[0, 1]
    axis.fill_between(effect.generation, effect.posterior_lo95_E, effect.posterior_hi95_E,
                      color="#c4512c", alpha=0.18)
    axis.plot(effect.generation, effect.posterior_mean_E, "s-", color="#c4512c", ms=3.5)
    axis.axhline(0.0, color="0.6", lw=0.8)
    axis.set(title=r"Posterior environmental effect $E_i$", xlabel="Generation", ylabel=r"$E_i$")

    axis = axes[1, 0]
    axis.fill_between(birth_acf.lag, birth_acf.simulated_lo95, birth_acf.simulated_hi95,
                      color="#c4512c", alpha=0.18)
    axis.plot(birth_acf.lag, birth_acf.simulated_mean, "s-", color="#c4512c", ms=3.5, label="Posterior shared environment")
    axis.plot(birth_acf.lag, birth_acf.observed, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set(title="Birth-size ACF", xlabel="Lag (generations)", ylabel="Correlation")

    axis = axes[1, 1]
    axis.fill_between(q_acf.lag, q_acf.simulated_lo95, q_acf.simulated_hi95,
                      color="#c4512c", alpha=0.18)
    axis.plot(q_acf.lag, q_acf.simulated_mean, "s-", color="#c4512c", ms=3.5,
              label="Posterior shared environment")
    axis.plot(q_acf.lag, q_acf.observed, "o-", color="#23364d", ms=3.5,
              label="Observed")
    axis.set(title=r"Effective-jump ACF ($q$)", xlabel="Lag (generations)", ylabel="Correlation")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    fig.suptitle("Glu: independently regularized environmental q effects by cell cycle")
    fig.savefig(OUT / "shared_cycle_environment_q_diagnostics.png", dpi=190)
    fig.savefig(OUT / "shared_cycle_environment_q_diagnostics.pdf")
    plt.close(fig)

    simulated_slopes = np.polyfit(generation, simulated_means.T, 1)[0]
    result = {
        "model": {
            "q": "q[l,i]=mu_q+E[i]+xi[l,i]; E[i]~iid Normal(0,sigma_E); xi[l,i]~iid Normal(0,sigma_q)",
            "posterior_prediction": "One E[i] is drawn from its inferred posterior per replicate and shared by every lineage.",
            "environment": serializable(qfit),
            "alpha": source["model"]["alpha"],
            "hazard": source["model"]["hazard"],
        },
        "simulation": {
            "replicates": args.replicates,
            "observed_generation_20_change": float(observed_mean[-1] - observed_mean[0]),
            "simulated_generation_20_change": float(simulated_mean[-1] - simulated_mean[0]),
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
