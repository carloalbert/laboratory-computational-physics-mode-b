#!/usr/bin/env python3
"""Infer every Glu lineage and cycle effect, then run conditional and new-lineage PPCs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from autoregressive_residual_model import (  # noqa: E402
    sample_birth_dependent_growth_time,
    sample_growth_time,
    transform_gamma,
)
from fit_all_parameters_birth_acf_glu import marginal_rows  # noqa: E402
from fit_autoregressive_residual_glu import contiguous_prefixes, observed_matrix  # noqa: E402
from fit_decomposed_birth_regularized_glu import residualize_generation_mean  # noqa: E402
from fit_environment_lineage_offset_ar_q_glu import (  # noqa: E402
    acf_summary,
    fit_q,
    serializable,
    slope_statistics,
)
from fit_lineage_offset_ar_q_glu import pooled_acf_replicates, q_matrix  # noqa: E402
from glu_workflow import load_data  # noqa: E402


BASE = PROJECT_DIR / "outputs/model_comparison/Glu"
SOURCE = BASE / "environment_fast_slow_ar_q_protein_hazard"
OUT = BASE / "individual_environment_lineage_effects_q_protein_hazard"


def ar_covariance(length: int, rho: float, stationary_sd: float) -> np.ndarray:
    """Stationary AR(1) covariance for one lineage's residual process."""
    distance = np.abs(np.subtract.outer(np.arange(length), np.arange(length)))
    return stationary_sd**2 * rho**distance


def joint_effect_posterior(q_values, q_lengths, qfit):
    """Exact Gaussian posterior of absolute A_l and E_i at fixed hyperparameters.

    The fitted model is q_li = A_l + E_i + r_li, with
    A_l ~ N(mu_A, sigma_A^2), E_i ~ N(0, sigma_E^2), and stationary AR(1) r.
    """
    q_values = np.asarray(q_values, dtype=float)
    q_lengths = np.asarray(q_lengths, dtype=int)
    lineages, cycles = q_values.shape
    dimension = lineages + cycles
    sigma_a = qfit["sigma_lineage_offset"]
    sigma_e = qfit["environment_prior_sd"]

    precision = np.zeros((dimension, dimension))
    precision[np.arange(lineages), np.arange(lineages)] = 1.0 / sigma_a**2
    environment_indices = lineages + np.arange(cycles)
    precision[environment_indices, environment_indices] = 1.0 / sigma_e**2
    score = np.zeros(dimension)
    score[:lineages] = qfit["mean"] / sigma_a**2

    inverse_cache = {}
    for lineage, length in enumerate(q_lengths):
        if not length:
            continue
        if length not in inverse_cache:
            covariance = ar_covariance(
                int(length), qfit["rho"], qfit["stationary_ar_sd"]
            )
            factor = cho_factor(covariance, lower=True, check_finite=False)
            inverse_cache[int(length)] = cho_solve(
                factor, np.eye(length), check_finite=False
            )
        inverse = inverse_cache[int(length)]
        values = q_values[lineage, :length]
        indices = lineages + np.arange(length)
        ones_inverse = inverse.sum(axis=0)

        precision[lineage, lineage] += float(inverse.sum())
        precision[lineage, indices] += ones_inverse
        precision[indices, lineage] += ones_inverse
        precision[np.ix_(indices, indices)] += inverse
        score[lineage] += float(ones_inverse @ values)
        score[indices] += inverse @ values

    factor = cho_factor(precision, lower=True, check_finite=False)
    posterior_mean = cho_solve(factor, score, check_finite=False)
    posterior_covariance = cho_solve(
        factor, np.eye(dimension), check_finite=False
    )
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    return {
        "mean": posterior_mean,
        "covariance": posterior_covariance,
        "sd": np.sqrt(np.maximum(np.diag(posterior_covariance), 0.0)),
        "lineages": lineages,
        "cycles": cycles,
    }


def apply_environment_gauge(a, e, weights):
    """Set the active-lineage-weighted mean E to zero without changing A_l+E_i."""
    a = np.asarray(a, dtype=float)
    e = np.asarray(e, dtype=float)
    weights = np.asarray(weights, dtype=float)
    shift = np.sum(e * weights, axis=-1, keepdims=True) / weights.sum()
    return a + shift, e - shift


def draw_conditional_effects(posterior, q_lengths, replicates, rng):
    draws = rng.multivariate_normal(
        posterior["mean"], posterior["covariance"], size=replicates
    )
    lineages = posterior["lineages"]
    a, e = draws[:, :lineages], draws[:, lineages:]
    weights = np.asarray(
        [np.sum(np.asarray(q_lengths) > generation) for generation in range(len(e[0]))],
        dtype=float,
    )
    return apply_environment_gauge(a, e, weights)


def posterior_summary_tables(prefixes, q_values, q_lengths, posterior):
    lineages, cycles = posterior["lineages"], posterior["cycles"]
    mean = posterior["mean"]
    covariance = posterior["covariance"]
    weights = np.asarray(
        [np.sum(np.asarray(q_lengths) > generation) for generation in range(cycles)],
        dtype=float,
    )

    # Linear transformation to the reporting gauge sum_i n_i E_i = 0.
    normalized = weights / weights.sum()
    transform = np.eye(lineages + cycles)
    transform[:lineages, lineages:] += normalized
    transform[lineages:, lineages:] -= np.ones((cycles, 1)) @ normalized[None, :]
    gauged_mean = transform @ mean
    gauged_covariance = transform @ covariance @ transform.T
    gauged_sd = np.sqrt(np.maximum(np.diag(gauged_covariance), 0.0))

    lineage_rows = []
    for index, (group, length) in enumerate(zip(prefixes, q_lengths)):
        observed_mean = float(np.mean(q_values[index, :length])) if length else np.nan
        lineage_rows.append(
            {
                "lineage_index": index,
                "lineage_id": float(group.lineage.iloc[0]),
                "q_transitions": int(length),
                "observed_mean_q": observed_mean,
                "posterior_mean_A": float(gauged_mean[index]),
                "posterior_sd_A": float(gauged_sd[index]),
                "posterior_lo95_A": float(gauged_mean[index] - 1.96 * gauged_sd[index]),
                "posterior_hi95_A": float(gauged_mean[index] + 1.96 * gauged_sd[index]),
            }
        )
    environment_rows = []
    for generation in range(cycles):
        index = lineages + generation
        observed = q_values[q_lengths > generation, generation]
        environment_rows.append(
            {
                "generation": generation,
                "active_lineages": int(weights[generation]),
                "observed_mean_q": float(observed.mean()),
                "posterior_mean_E": float(gauged_mean[index]),
                "posterior_sd_E": float(gauged_sd[index]),
                "posterior_lo95_E": float(gauged_mean[index] - 1.96 * gauged_sd[index]),
                "posterior_hi95_E": float(gauged_mean[index] + 1.96 * gauged_sd[index]),
            }
        )
    return pd.DataFrame(lineage_rows), pd.DataFrame(environment_rows), gauged_mean


def simulate(
    params,
    lengths,
    posterior,
    q_lengths,
    replicates=500,
    seed=11937,
    conditional=False,
    observed_initial=None,
    alpha_posterior=None,
    cycle_log_u_draws=None,
    cycle_log_omega2_draws=None,
):
    """Posterior predictive simulation conditional on old or predictive for new lineages."""
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages, horizon = len(lengths), int(lengths.max())
    shape = (replicates, lineages)
    log_birth = np.zeros((replicates, lineages, horizon))
    alpha = np.zeros_like(log_birth)
    growth = np.full_like(log_birth, np.nan)
    time_values = np.full_like(log_birth, np.nan)
    q = np.full_like(log_birth, np.nan)
    qfit, afit, hazard = params["q"], params["alpha"], params["hazard"]

    if conditional:
        if observed_initial is None:
            raise ValueError("conditional simulation requires observed_initial")
        current = np.broadcast_to(np.log(observed_initial), shape).copy()
        lineage_effect, environment = draw_conditional_effects(
            posterior, q_lengths, replicates, rng
        )
    else:
        current = (
            params["initial_birth"]["log_mean"]
            + params["initial_birth"]["log_sd"] * rng.normal(size=shape)
        )
        lineage_effect = (
            qfit["mean"]
            + qfit["sigma_lineage_offset"] * rng.normal(size=shape)
        )
        start = posterior["lineages"]
        environment = rng.multivariate_normal(
            posterior["mean"][start:],
            posterior["covariance"][start:, start:],
            size=replicates,
        )

    if alpha_posterior is None:
        alpha_latent = rng.normal(size=shape)
        alpha_environment = None
    else:
        alpha_latent = afit["stationary_residual_sd"] * rng.normal(size=shape)
        alpha_environment = rng.multivariate_normal(
            alpha_posterior["mean"],
            alpha_posterior["covariance"],
            size=replicates,
        )
    residual = qfit["stationary_ar_sd"] * rng.normal(size=shape)
    for generation in range(horizon):
        active = lengths > generation
        if generation:
            alpha_latent[:, active] = (
                afit["rho"] * alpha_latent[:, active]
                + afit.get(
                    "sigma_residual_innovation",
                    math.sqrt(1.0 - afit["rho"] ** 2),
                )
                * rng.normal(size=(replicates, int(active.sum())))
            )
            residual[:, active] = (
                qfit["rho"] * residual[:, active]
                + qfit["sigma_innovation"]
                * rng.normal(size=(replicates, int(active.sum())))
            )
        log_birth[:, active, generation] = current[:, active]
        alpha_argument = alpha_latent[:, active]
        if alpha_environment is not None:
            if generation < alpha_environment.shape[1]:
                alpha_argument = (
                    alpha_argument + alpha_environment[:, generation, None]
                )
            else:
                alpha_argument = (
                    alpha_argument
                    + afit["environment_prior_sd"]
                    * rng.normal(size=(replicates, 1))
                )
        alpha[:, active, generation] = transform_gamma(
            alpha_argument, afit["shape"], afit["scale"]
        )
        continuing = lengths > generation + 1
        if not continuing.any():
            continue
        effect = (
            environment[:, generation, None]
            if generation < environment.shape[1]
            else qfit["environment_prior_sd"] * rng.normal(size=(replicates, 1))
        )
        q_value = lineage_effect[:, continuing] + effect + residual[:, continuing]
        if cycle_log_u_draws is not None and cycle_log_omega2_draws is not None:
            raise ValueError("Only one cycle-dependent hazard parameter is supported")
        if cycle_log_u_draws is not None:
            log_u = np.repeat(
                cycle_log_u_draws[:, generation], int(continuing.sum())
            )
            growth_value = sample_growth_time(
                current[:, continuing].ravel(),
                log_u,
                hazard["omega2"],
                hazard["eta"],
                rng,
            ).reshape(replicates, int(continuing.sum()))
        elif cycle_log_omega2_draws is not None:
            omega2 = np.repeat(
                np.exp(cycle_log_omega2_draws[:, generation]),
                int(continuing.sum()),
            )
            growth_value = sample_growth_time(
                current[:, continuing].ravel(),
                math.log(hazard["u"]),
                omega2,
                hazard["eta"],
                rng,
            ).reshape(replicates, int(continuing.sum()))
        elif "kappa" in hazard:
            growth_value = sample_birth_dependent_growth_time(
                current[:, continuing].ravel(),
                math.log(hazard["u"]),
                hazard["omega2"],
                hazard["eta"],
                hazard["kappa"],
                math.log(hazard["b_star"]),
                rng,
            ).reshape(replicates, int(continuing.sum()))
        else:
            growth_value = sample_growth_time(
                current[:, continuing].ravel(),
                math.log(hazard["u"]),
                hazard["omega2"],
                hazard["eta"],
                rng,
            ).reshape(replicates, int(continuing.sum()))
        q[:, continuing, generation] = q_value
        growth[:, continuing, generation] = growth_value
        time_values[:, continuing, generation] = (
            growth_value / alpha[:, continuing, generation]
        )
        current[:, continuing] += growth_value + q_value

    return {
        "birth": np.exp(np.clip(log_birth, -700.0, 700.0)),
        "log_birth": log_birth,
        "alpha": alpha,
        "growth": growth,
        "time": time_values,
        "q": q,
        "environment": environment,
        "lineage_effect": lineage_effect,
        "alpha_environment": alpha_environment,
        "cycle_log_u": cycle_log_u_draws,
        "cycle_log_omega2": cycle_log_omega2_draws,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--seed", type=int, default=11937)
    args = parser.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    source = json.loads((SOURCE / "fit_results.json").read_text())
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    q_values, q_lengths = q_matrix(prefixes)
    sequences = [q_values[index, :length] for index, length in enumerate(q_lengths) if length]
    qfit, log_density, optimizer = fit_q(sequences, int(q_values.shape[1]))
    posterior = joint_effect_posterior(q_values, q_lengths, qfit)
    lineage_table, environment_table, gauged_mean = posterior_summary_tables(
        prefixes, q_values, q_lengths, posterior
    )
    lineage_table.to_csv(OUT / "lineage_effect_posterior.csv", index=False)
    environment_table.to_csv(OUT / "environment_effect_posterior.csv", index=False)

    params = {
        "initial_birth": source["fit"]["initial_birth"],
        "alpha": source["fit"]["alpha"],
        "hazard": source["fit"]["hazard"],
        "q": qfit,
    }
    initial = np.asarray([float(group.length_birth.iloc[0]) for group in prefixes])
    generated = {
        "new_lineages": simulate(
            params, lengths, posterior, q_lengths, args.replicates, args.seed
        ),
        "conditional_A_E": simulate(
            params,
            lengths,
            posterior,
            q_lengths,
            args.replicates,
            args.seed + 1,
            conditional=True,
            observed_initial=initial,
        ),
    }

    acf = pd.DataFrame({"lag": np.arange(args.max_lag + 1)})
    for kind, residualized in [("raw", False), ("demeaned", True)]:
        target, _, _, _ = acf_summary(
            observed_birth,
            generated["new_lineages"]["birth"],
            lengths,
            args.max_lag,
            residualized,
        )
        acf[f"observed_{kind}"] = target
        for label, simulation in generated.items():
            _, mean, low, high = acf_summary(
                observed_birth,
                simulation["birth"],
                lengths,
                args.max_lag,
                residualized,
            )
            acf[f"{label}_{kind}"] = mean
            acf[f"{label}_{kind}_lo95"] = low
            acf[f"{label}_{kind}_hi95"] = high
    acf.to_csv(OUT / "birth_acf_comparison.csv", index=False)

    qacf = pd.DataFrame(
        {
            "lag": np.arange(args.max_lag + 1),
            "observed": pooled_acf_replicates(q_values, q_lengths, args.max_lag)[0],
        }
    )
    for label, simulation in generated.items():
        samples = pooled_acf_replicates(simulation["q"], q_lengths, args.max_lag)
        qacf[f"{label}_mean"] = np.nanmean(samples, axis=0)
        qacf[f"{label}_lo95"], qacf[f"{label}_hi95"] = np.nanquantile(
            samples, [0.025, 0.975], axis=0
        )
    qacf.to_csv(OUT / "q_acf_comparison.csv", index=False)

    horizon = min(args.horizon, int(lengths.max()) - 1)
    complete = lengths > horizon
    generations = np.arange(horizon + 1)
    drift = pd.DataFrame(
        {
            "generation": generations,
            "observed": observed_birth[complete, : horizon + 1].mean(axis=0),
        }
    )
    for label, simulation in generated.items():
        paths = simulation["birth"][:, complete, : horizon + 1].mean(axis=1)
        drift[f"{label}_mean"] = paths.mean(axis=0)
        drift[f"{label}_lo95"], drift[f"{label}_hi95"] = np.quantile(
            paths, [0.025, 0.975], axis=0
        )
    drift.to_csv(OUT / "mean_birth_trajectory.csv", index=False)

    marginal = pd.DataFrame(
        sum(
            [
                marginal_rows(prefixes, simulation, lengths, label)
                for label, simulation in generated.items()
            ],
            [],
        )
    )
    marginal.to_csv(OUT / "marginal_comparison.csv", index=False)

    slopes = [
        slope_statistics(observed_birth, lengths).assign(model="observed", replicate=0)
    ]
    for label, simulation in generated.items():
        values = slope_statistics(simulation["birth"], lengths)
        values["model"] = label
        values["replicate"] = np.arange(len(values))
        slopes.append(values)
    slopes = pd.concat(slopes, ignore_index=True)
    slopes.to_csv(OUT / "lineage_slope_statistics.csv", index=False)

    fit_slice = slice(1, args.max_lag + 1)
    metrics = {
        "observed": {
            "generation_20_change": float(drift.observed.iloc[-1] - drift.observed.iloc[0]),
            **slopes.query("model == 'observed'")[
                ["slope_sd", "median_abs_slope", "median_trend_R2"]
            ].iloc[0].to_dict(),
        }
    }
    for label in generated:
        marginal_lookup = marginal.query("model == @label").set_index("variable")
        values = slopes.query("model == @label")[
            ["slope_sd", "median_abs_slope", "median_trend_R2"]
        ].mean()
        metrics[label] = {
            "raw_birth_acf_rmse": float(
                np.sqrt(np.mean((acf.observed_raw.to_numpy()[fit_slice] - acf[f"{label}_raw"].to_numpy()[fit_slice]) ** 2))
            ),
            "demeaned_birth_acf_rmse": float(
                np.sqrt(np.mean((acf.observed_demeaned.to_numpy()[fit_slice] - acf[f"{label}_demeaned"].to_numpy()[fit_slice]) ** 2))
            ),
            "q_acf_rmse": float(
                np.sqrt(np.mean((qacf.observed.to_numpy()[fit_slice] - qacf[f"{label}_mean"].to_numpy()[fit_slice]) ** 2))
            ),
            "mean_birth_rmse": float(np.sqrt(np.mean((drift.observed - drift[f"{label}_mean"]) ** 2))),
            "generation_20_change": float(drift[f"{label}_mean"].iloc[-1] - drift[f"{label}_mean"].iloc[0]),
            "birth_ks": float(marginal_lookup.loc["birth", "ks_distance"]),
            "q_ks": float(marginal_lookup.loc["q", "ks_distance"]),
            "alpha_ks": float(marginal_lookup.loc["alpha", "ks_distance"]),
            "time_ks": float(marginal_lookup.loc["time", "ks_distance"]),
            "raw_birth_acf_lags_inside_95": int(
                np.sum(
                    (acf.observed_raw.to_numpy()[fit_slice] >= acf[f"{label}_raw_lo95"].to_numpy()[fit_slice])
                    & (acf.observed_raw.to_numpy()[fit_slice] <= acf[f"{label}_raw_hi95"].to_numpy()[fit_slice])
                )
            ),
            "demeaned_birth_acf_lags_inside_95": int(
                np.sum(
                    (acf.observed_demeaned.to_numpy()[fit_slice] >= acf[f"{label}_demeaned_lo95"].to_numpy()[fit_slice])
                    & (acf.observed_demeaned.to_numpy()[fit_slice] <= acf[f"{label}_demeaned_hi95"].to_numpy()[fit_slice])
                )
            ),
            **values.to_dict(),
        }

    colors = {"new_lineages": "#718096", "conditional_A_E": "#c4512c"}
    styles = {"new_lineages": "--", "conditional_A_E": "-"}
    labels = {
        "new_lineages": "New lineages",
        "conditional_A_E": r"Observed lineages: inferred $A_\ell,E_i$",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.1), constrained_layout=True)
    for axis, kind, title in [
        (axes[0, 0], "raw", "Raw birth-size ACF"),
        (axes[0, 1], "demeaned", "Generation-demeaned birth-size ACF"),
    ]:
        axis.plot(acf.lag, acf[f"observed_{kind}"], "o-", color="#23364d", ms=3.5, label="Observed")
        for label in generated:
            axis.fill_between(acf.lag, acf[f"{label}_{kind}_lo95"], acf[f"{label}_{kind}_hi95"], color=colors[label], alpha=0.10)
            axis.plot(acf.lag, acf[f"{label}_{kind}"], styles[label], color=colors[label], label=labels[label])
        axis.set(title=title, xlabel="Lag", ylabel="Correlation")
        axis.grid(alpha=0.16)
    axes[0, 0].legend(frameon=False, fontsize=8)

    axis = axes[0, 2]
    axis.plot(drift.generation, drift.observed, "o-", color="#23364d", ms=3.5, label="Observed")
    for label in generated:
        axis.fill_between(drift.generation, drift[f"{label}_lo95"], drift[f"{label}_hi95"], color=colors[label], alpha=0.10)
        axis.plot(drift.generation, drift[f"{label}_mean"], styles[label], color=colors[label])
    axis.set(title="Balanced-cohort mean birth size", xlabel="Generation", ylabel="Mean birth size")
    axis.grid(alpha=0.16)

    axis = axes[1, 0]
    axis.plot(qacf.lag, qacf.observed, "o-", color="#23364d", ms=3.5)
    for label in generated:
        axis.fill_between(qacf.lag, qacf[f"{label}_lo95"], qacf[f"{label}_hi95"], color=colors[label], alpha=0.10)
        axis.plot(qacf.lag, qacf[f"{label}_mean"], styles[label], color=colors[label])
    axis.set(title=r"Effective-jump ACF ($q$)", xlabel="Lag", ylabel="Correlation")
    axis.grid(alpha=0.16)

    axis = axes[1, 1]
    valid = lineage_table.q_transitions > 0
    axis.errorbar(
        lineage_table.loc[valid, "observed_mean_q"],
        lineage_table.loc[valid, "posterior_mean_A"],
        yerr=1.96 * lineage_table.loc[valid, "posterior_sd_A"],
        fmt="o",
        ms=2.5,
        lw=0.5,
        alpha=0.55,
        color="#c4512c",
    )
    limits = [
        float(min(lineage_table.loc[valid, "observed_mean_q"].min(), lineage_table.loc[valid, "posterior_mean_A"].min())),
        float(max(lineage_table.loc[valid, "observed_mean_q"].max(), lineage_table.loc[valid, "posterior_mean_A"].max())),
    ]
    axis.plot(limits, limits, color="0.55", lw=0.8)
    axis.set(title=r"Inferred absolute lineage effect $A_\ell$", xlabel="Observed lineage mean q", ylabel=r"Posterior mean $A_\ell$")
    axis.grid(alpha=0.16)

    axis = axes[1, 2]
    variables = ["birth", "q", "alpha", "time"]
    width = 0.34
    for index, label in enumerate(generated):
        values = marginal.query("model == @label").set_index("variable").loc[variables].ks_distance
        axis.bar(np.arange(4) + (index - 0.5) * width, values, width, color=colors[label], label=labels[label])
    axis.set_xticks(np.arange(4), ["birth", "q", r"$\alpha$", r"$\tau$"])
    axis.set(title="Marginal distributions", ylabel="KS distance")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.16, axis="y")
    fig.suptitle(r"Glu: individual $A_\ell$ and $E_i$ posterior prediction")
    fig.savefig(OUT / "individual_effects_diagnostics.png", dpi=190)
    fig.savefig(OUT / "individual_effects_diagnostics.pdf")
    plt.close(fig)

    fitted_a = gauged_mean[: posterior["lineages"]]
    valid_lineage = lineage_table.q_transitions > 0
    a_mean_q_correlation = float(
        np.corrcoef(
            lineage_table.loc[valid_lineage, "posterior_mean_A"],
            lineage_table.loc[valid_lineage, "observed_mean_q"],
        )[0, 1]
    )
    q_hyperparameters = {
        key: qfit[key]
        for key in [
            "mean",
            "rho",
            "sigma_lineage_offset",
            "stationary_ar_sd",
            "sigma_innovation",
            "environment_prior_sd",
            "offset_variance_fraction",
            "fast_correlation_length_generations",
            "success",
            "message",
        ]
    }
    diagnostics = {
        "model": {
            "equation": "q_li = A_l + E_i + r_li",
            "lineage_prior": "A_l ~ Normal(mu_A, sigma_A^2); mu_A is the former mu_q",
            "environment_prior": "E_i ~ iid Normal(0, sigma_E^2)",
            "residual": "r_li is a stationary Gaussian AR(1)",
            "reporting_gauge": "sum_i n_i E_i = 0, where n_i is the number of active lineages; A_l+E_i is invariant",
            "inference": "exact joint Gaussian posterior of every A_l and E_i conditional on plug-in fitted hyperparameters",
        },
        "fit": {
            "q_hyperparameters": serializable(q_hyperparameters),
            "individual_effects": {
                "lineages": int(posterior["lineages"]),
                "cycles": int(posterior["cycles"]),
                "mean_posterior_A": float(fitted_a.mean()),
                "sd_posterior_mean_A": float(fitted_a.std()),
                "median_posterior_sd_A": float(np.median(lineage_table.posterior_sd_A)),
                "median_posterior_sd_E": float(np.median(environment_table.posterior_sd_E)),
                "correlation_posterior_A_with_observed_lineage_mean_q": a_mean_q_correlation,
            },
            "hyperparameter_log_joint_density": float(log_density),
            "optimizer_success": bool(optimizer.success),
            "optimizer_message": str(optimizer.message),
        },
        "metrics": serializable(metrics),
        "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (OUT / "fit_results.json").write_text(json.dumps(diagnostics, indent=2))

    lines = [
        "# Glu model with individually inferred lineage and environment effects",
        "",
        r"The effective jump is parameterized as $q_{\ell i}=A_\ell+E_i+r_{\ell i}$.",
        r"Here $A_\ell\sim\mathcal N(\mu_A,\sigma_A^2)$ is the absolute lineage mean",
        r"(so the former $\mu_q$ is absorbed into $A_\ell$),",
        r"$E_i\sim\mathcal N(0,\sigma_E^2)$ is shared by all lineages at cycle $i$,",
        r"and $r_{\ell i}$ is a stationary Gaussian AR(1). At fixed fitted",
        "hyperparameters the complete vector of all 303 lineage effects and 47",
        "cycle effects has an exact joint Gaussian posterior. The tables in this",
        "directory report every effect and its posterior uncertainty.",
        "",
        r"For display, the gauge is chosen so $\sum_i n_iE_i=0$, with $n_i$ the",
        r"number of observed transitions at cycle $i$. Shifting all $A_\ell$ by a",
        r"constant and all $E_i$ by its negative leaves every $A_\ell+E_i$ unchanged.",
        "",
        "## Posterior predictive checks",
        "",
        "`conditional A E` draws the fitted effects jointly from their posterior,",
        "uses each measured initial birth size and lineage length, but generates new",
        r"AR residuals, growth rates, and division times. `new lineages` draws new $A_\ell$",
        "and initial sizes, while retaining posterior uncertainty in the shared environment.",
        "",
        "| Check | New lineages | Conditional A,E |",
        "|---|---:|---:|",
    ]
    for key, title in [
        ("raw_birth_acf_rmse", "Raw b-ACF RMSE"),
        ("demeaned_birth_acf_rmse", "Demeaned b-ACF RMSE"),
        ("q_acf_rmse", "q-ACF RMSE"),
        ("mean_birth_rmse", "Mean-b trajectory RMSE"),
        ("birth_ks", "Birth KS"),
        ("q_ks", "q KS"),
        ("alpha_ks", "alpha KS"),
        ("time_ks", "tau KS"),
    ]:
        lines.append(
            f"| {title} | {metrics['new_lineages'][key]:.5f} | {metrics['conditional_A_E'][key]:.5f} |"
        )
    lines.extend(
        [
            "",
            "The conditional check is deliberately in-sample for the persistent",
            "effects and initial sizes. Its improvement measures how much information",
            "the fitted identities contain; only the new-lineage check measures",
            "unconditional generalization.",
            "",
            "## Result",
            "",
            rf"The inferred $A_\ell$ values do track the observed lineage mean q "
            f"(correlation {a_mean_q_correlation:.3f}), but their fitted between-lineage "
            f"SD is only {fitted_a.std():.4f}, versus a residual stationary SD of "
            f"{qfit['stationary_ar_sd']:.4f}. Conditioning on every fitted effect "
            f"therefore changes the raw b-ACF RMSE only from "
            f"{metrics['new_lineages']['raw_birth_acf_rmse']:.5f} to "
            f"{metrics['conditional_A_E']['raw_birth_acf_rmse']:.5f}; the demeaned "
            f"b-ACF RMSE changes from {metrics['new_lineages']['demeaned_birth_acf_rmse']:.5f} "
            f"to {metrics['conditional_A_E']['demeaned_birth_acf_rmse']:.5f}.",
            "",
            f"Only {metrics['conditional_A_E']['raw_birth_acf_lags_inside_95']} of "
            f"{args.max_lag} nonzero raw-ACF lags and "
            f"{metrics['conditional_A_E']['demeaned_birth_acf_lags_inside_95']} of "
            f"{args.max_lag} demeaned-ACF lags lie in the conditional 95% posterior "
            "predictive bands. Thus the failure is not caused by having drawn new",
            "lineage offsets in the earlier check: an individually inferred static",
            "offset shifts a lineage's size setpoint, but the hazard feedback still",
            "removes birth-size memory too quickly.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
