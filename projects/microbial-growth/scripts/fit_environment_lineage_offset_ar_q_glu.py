#!/usr/bin/env python3
"""Fit Glu q with shared cycle effects, a lineage offset, and one fast AR mode."""

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
from scipy.optimize import minimize


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from autoregressive_residual_model import (  # noqa: E402
    lineage_acf,
    sample_growth_time,
    transform_gamma,
)
from fit_all_parameters_birth_acf_glu import marginal_rows  # noqa: E402
from fit_autoregressive_residual_glu import (  # noqa: E402
    contiguous_prefixes,
    observed_matrix,
)
from fit_decomposed_birth_regularized_glu import (  # noqa: E402
    residualize_generation_mean,
)
from fit_environment_fast_slow_ar_q_glu import simulate as simulate_fast_slow  # noqa: E402
from fit_likelihood_acf_regularized_glu import fit_q_pure_map  # noqa: E402
from fit_lineage_offset_ar_q_glu import pooled_acf_replicates, q_matrix  # noqa: E402
from glu_workflow import load_data  # noqa: E402


BASE = PROJECT_DIR / "outputs/model_comparison/Glu"
SOURCE = BASE / "environment_fast_slow_ar_q_protein_hazard"
OUT = BASE / "environment_lineage_offset_ar_q_protein_hazard"


def serializable(value):
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def q_covariance(length, rho, sigma_offset, stationary_ar_sd):
    distance = np.abs(np.subtract.outer(np.arange(length), np.arange(length)))
    return (
        stationary_ar_sd**2 * rho**distance
        + sigma_offset**2 * np.ones((length, length))
    )


def unpack_hyperparameters(theta):
    mu, rho, log_offset_sd, log_ar_sd, log_environment_sd = theta
    return {
        "mean": float(mu),
        "rho": float(rho),
        "sigma_lineage_offset": math.exp(log_offset_sd),
        "stationary_ar_sd": math.exp(log_ar_sd),
        "environment_prior_sd": math.exp(log_environment_sd),
    }


def mode_and_log_density(theta, sequences, environment_count):
    qfit = unpack_hyperparameters(theta)
    precision = np.eye(environment_count) / qfit["environment_prior_sd"] ** 2
    score = np.zeros(environment_count)
    cache = {}
    for sequence in sequences:
        length = len(sequence)
        if length not in cache:
            covariance = q_covariance(
                length,
                qfit["rho"],
                qfit["sigma_lineage_offset"],
                qfit["stationary_ar_sd"],
            )
            cholesky = np.linalg.cholesky(covariance)
            inverse = np.linalg.solve(
                cholesky.T, np.linalg.solve(cholesky, np.eye(length))
            )
            logdet = 2.0 * float(np.log(np.diag(cholesky)).sum())
            cache[length] = cholesky, inverse, logdet
        _, inverse, _ = cache[length]
        precision[:length, :length] += inverse
        score[:length] += inverse @ (sequence - qfit["mean"])

    covariance_e = np.linalg.inv(precision)
    effects = covariance_e @ score
    value = 0.0
    for sequence in sequences:
        length = len(sequence)
        cholesky, _, logdet = cache[length]
        residual = sequence - qfit["mean"] - effects[:length]
        whitened = np.linalg.solve(cholesky, residual)
        value += -0.5 * (
            length * math.log(2.0 * math.pi) + logdet + whitened @ whitened
        )
    sigma_e = qfit["environment_prior_sd"]
    value += float(
        np.sum(
            -0.5
            * (
                math.log(2.0 * math.pi * sigma_e**2)
                + effects**2 / sigma_e**2
            )
        )
    )
    qfit["posterior_mean_E"] = effects
    qfit["posterior_covariance_E"] = 0.5 * (covariance_e + covariance_e.T)
    qfit["posterior_sd_E"] = np.sqrt(np.diag(covariance_e))
    qfit["sigma_innovation"] = qfit["stationary_ar_sd"] * math.sqrt(
        1.0 - qfit["rho"] ** 2
    )
    total_variance = (
        qfit["sigma_lineage_offset"] ** 2
        + qfit["stationary_ar_sd"] ** 2
        + qfit["environment_prior_sd"] ** 2
    )
    qfit["offset_variance_fraction"] = (
        qfit["sigma_lineage_offset"] ** 2 / total_variance
    )
    qfit["fast_correlation_length_generations"] = (
        -1.0 / math.log(abs(qfit["rho"])) if abs(qfit["rho"]) > 0 else 0.0
    )
    qfit["log_joint_density"] = float(value)
    return qfit, float(value)


def fit_q(sequences, environment_count):
    flat = np.concatenate(sequences)
    starts = [
        [-0.52, 0.0, math.log(0.022), math.log(0.080), math.log(0.010)],
        [-0.52, -0.15, math.log(0.020), math.log(0.078), math.log(0.015)],
        [-0.52, 0.20, math.log(0.035), math.log(0.073), math.log(0.010)],
        [float(flat.mean()), 0.05, math.log(0.012), math.log(float(flat.std())), math.log(0.020)],
    ]
    bounds = [
        (-0.70, -0.35),
        (-0.65, 0.65),
        (math.log(0.001), math.log(0.12)),
        (math.log(0.02), math.log(0.18)),
        (math.log(0.002), math.log(0.10)),
    ]

    def objective(theta):
        try:
            return -mode_and_log_density(theta, sequences, environment_count)[1]
        except np.linalg.LinAlgError:
            return 1e100

    fits = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-13},
        )
        for start in starts
    ]
    fitted = min(fits, key=lambda result: result.fun)
    qfit, value = mode_and_log_density(fitted.x, sequences, environment_count)
    qfit["success"] = bool(fitted.success)
    qfit["message"] = str(fitted.message)
    qfit["multistart_log_joint_densities"] = [-float(result.fun) for result in fits]
    return qfit, value, fitted


def draw_environment(qfit, replicates, rng):
    return rng.multivariate_normal(
        qfit["posterior_mean_E"],
        qfit["posterior_covariance_E"],
        size=replicates,
        check_valid="raise",
    )


def simulate(params, lengths, replicates=500, seed=83011):
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
    qfit, afit, hazard = params["q"], params["alpha"], params["hazard"]
    lineage_offset = qfit["sigma_lineage_offset"] * rng.normal(size=shape)
    residual = qfit["stationary_ar_sd"] * rng.normal(size=shape)
    environment = draw_environment(qfit, replicates, rng)

    for generation in range(horizon):
        active = lengths > generation
        if generation:
            alpha_latent[:, active] = (
                afit["rho"] * alpha_latent[:, active]
                + math.sqrt(1.0 - afit["rho"] ** 2)
                * rng.normal(size=(replicates, int(active.sum())))
            )
            residual[:, active] = (
                qfit["rho"] * residual[:, active]
                + qfit["sigma_innovation"]
                * rng.normal(size=(replicates, int(active.sum())))
            )
        log_birth[:, active, generation] = current[:, active]
        alpha[:, active, generation] = transform_gamma(
            alpha_latent[:, active], afit["shape"], afit["scale"]
        )
        continuing = lengths > generation + 1
        if not continuing.any():
            continue
        effect = (
            environment[:, generation, None]
            if generation < environment.shape[1]
            else qfit["environment_prior_sd"] * rng.normal(size=(replicates, 1))
        )
        q_value = (
            qfit["mean"]
            + effect
            + lineage_offset[:, continuing]
            + residual[:, continuing]
        )
        growth_value = sample_growth_time(
            current[:, continuing].ravel(),
            math.log(hazard["u"]),
            hazard["omega2"],
            hazard["eta"],
            rng,
        ).reshape(replicates, int(continuing.sum()))
        q[:, continuing, generation] = q_value
        growth[:, continuing, generation] = growth_value
        time[:, continuing, generation] = growth_value / alpha[:, continuing, generation]
        current[:, continuing] += growth_value + q_value

    return {
        "birth": np.exp(np.clip(log_birth, -700.0, 700.0)),
        "log_birth": log_birth,
        "alpha": alpha,
        "growth": growth,
        "time": time,
        "q": q,
        "environment": environment,
        "lineage_offset": lineage_offset,
    }


def acf_summary(observed, generated, lengths, max_lag, residual=False):
    if residual:
        observed = residualize_generation_mean(observed, lengths)
        generated = residualize_generation_mean(generated, lengths)
    target = lineage_acf(observed, lengths, max_lag)[0]
    samples = lineage_acf(generated, lengths, max_lag)
    low, high = np.nanquantile(samples, [0.025, 0.975], axis=0)
    return target, np.nanmean(samples, axis=0), low, high


def slope_statistics(values, lengths):
    residual = residualize_generation_mean(values, lengths)
    if residual.ndim == 2:
        residual = residual[None, ...]
    rows = []
    for replicate in residual:
        slopes, absolute, r2 = [], [], []
        for lineage, length in enumerate(lengths):
            if length < 6:
                continue
            y = replicate[lineage, :length]
            x = np.arange(length, dtype=float)
            x -= x.mean()
            y = y - y.mean()
            slope = float(x @ y / (x @ x))
            remainder = y - slope * x
            slopes.append(slope)
            absolute.append(abs(slope))
            r2.append(1.0 - float(remainder @ remainder / (y @ y)) if y @ y else 0.0)
        rows.append(
            {
                "slope_sd": float(np.std(slopes)),
                "median_abs_slope": float(np.median(absolute)),
                "median_trend_R2": float(np.median(r2)),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--seed", type=int, default=83011)
    args = parser.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    source = json.loads((SOURCE / "fit_results.json").read_text())
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    q_values, q_lengths = q_matrix(prefixes)
    sequences = [q_values[index, :length] for index, length in enumerate(q_lengths) if length]
    environment_count = int(lengths.max()) - 1

    proposed_q, proposed_log_density, proposed_optimizer = fit_q(sequences, environment_count)
    current_q, current_log_density, current_optimizer = fit_q_pure_map(
        source["fit"]["q"], sequences, environment_count
    )
    common = {
        "initial_birth": source["fit"]["initial_birth"],
        "alpha": source["fit"]["alpha"],
        "hazard": source["fit"]["hazard"],
    }
    proposed = {**common, "q": proposed_q}
    current = {**common, "q": current_q}
    current_generated = simulate_fast_slow(
        current, current_q, lengths, args.replicates, seed=args.seed
    )
    proposed_generated = simulate(
        proposed, lengths, args.replicates, seed=args.seed + 1
    )
    generated = {"current_fast_slow": current_generated, "proposed_offset": proposed_generated}

    acf = pd.DataFrame({"lag": np.arange(args.max_lag + 1)})
    for kind, residual in [("raw", False), ("demeaned", True)]:
        target, _, _, _ = acf_summary(
            observed_birth, current_generated["birth"], lengths, args.max_lag, residual
        )
        acf[f"observed_{kind}"] = target
        for label, simulation in generated.items():
            _, mean, low, high = acf_summary(
                observed_birth, simulation["birth"], lengths, args.max_lag, residual
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
            [marginal_rows(prefixes, simulation, lengths, label) for label, simulation in generated.items()],
            [],
        )
    )
    marginal.to_csv(OUT / "marginal_comparison.csv", index=False)

    slopes = [
        slope_statistics(observed_birth, lengths).assign(model="observed", replicate=0)
    ]
    for label, simulation in generated.items():
        value = slope_statistics(simulation["birth"], lengths)
        value["model"] = label
        value["replicate"] = np.arange(len(value))
        slopes.append(value)
    slopes = pd.concat(slopes, ignore_index=True)
    slopes.to_csv(OUT / "lineage_slope_statistics.csv", index=False)

    fit_slice = slice(1, args.max_lag + 1)
    metrics = {}
    for label in generated:
        marginal_lookup = marginal.query("model == @label").set_index("variable")
        metrics[label] = {
            "raw_birth_acf_rmse": float(
                np.sqrt(np.mean((acf["observed_raw"].to_numpy()[fit_slice] - acf[f"{label}_raw"].to_numpy()[fit_slice]) ** 2))
            ),
            "demeaned_birth_acf_rmse": float(
                np.sqrt(np.mean((acf["observed_demeaned"].to_numpy()[fit_slice] - acf[f"{label}_demeaned"].to_numpy()[fit_slice]) ** 2))
            ),
            "q_acf_rmse": float(
                np.sqrt(np.mean((qacf["observed"].to_numpy()[fit_slice] - qacf[f"{label}_mean"].to_numpy()[fit_slice]) ** 2))
            ),
            "mean_birth_rmse": float(
                np.sqrt(np.mean((drift.observed - drift[f"{label}_mean"]) ** 2))
            ),
            "generation_20_change": float(drift[f"{label}_mean"].iloc[-1] - drift[f"{label}_mean"].iloc[0]),
            "birth_ks": float(marginal_lookup.loc["birth", "ks_distance"]),
            "q_ks": float(marginal_lookup.loc["q", "ks_distance"]),
            "alpha_ks": float(marginal_lookup.loc["alpha", "ks_distance"]),
            "time_ks": float(marginal_lookup.loc["time", "ks_distance"]),
        }
    metrics["observed"] = {
        "generation_20_change": float(drift.observed.iloc[-1] - drift.observed.iloc[0]),
        **slopes.query("model == 'observed'")[["slope_sd", "median_abs_slope", "median_trend_R2"]].iloc[0].to_dict(),
    }
    for label in generated:
        means = slopes.query("model == @label")[["slope_sd", "median_abs_slope", "median_trend_R2"]].mean()
        metrics[label].update(means.to_dict())

    comparison = {
        "proposed_log_joint_density": proposed_log_density,
        "current_log_joint_density": current_log_density,
        "proposed_minus_current_log_density": proposed_log_density - current_log_density,
        "aic_proposed_minus_current": (
            -2.0 * proposed_log_density + 2.0 * 5
            - (-2.0 * current_log_density + 2.0 * 6)
        ),
        "proposed_hyperparameter_count": 5,
        "current_hyperparameter_count": 6,
    }

    colors = {"current_fast_slow": "#718096", "proposed_offset": "#c4512c"}
    styles = {"current_fast_slow": "--", "proposed_offset": "-"}
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.1), constrained_layout=True)
    for axis, kind, title in [
        (axes[0, 0], "raw", "Raw birth-size ACF"),
        (axes[0, 1], "demeaned", "Generation-demeaned birth-size ACF"),
    ]:
        axis.plot(acf.lag, acf[f"observed_{kind}"], "o-", color="#23364d", ms=3.5, label="Observed")
        for label in generated:
            axis.plot(acf.lag, acf[f"{label}_{kind}"], styles[label], color=colors[label], label=label.replace("_", " "))
        axis.set(title=title, xlabel="Lag", ylabel="Correlation")
        axis.grid(alpha=0.16)
    axes[0, 0].legend(frameon=False, fontsize=8)

    axis = axes[0, 2]
    axis.plot(drift.generation, drift.observed, "o-", color="#23364d", ms=3.5)
    for label in generated:
        axis.plot(drift.generation, drift[f"{label}_mean"], styles[label], color=colors[label])
    axis.set(title="Balanced-cohort mean birth size", xlabel="Generation", ylabel="Mean birth size")
    axis.grid(alpha=0.16)

    axis = axes[1, 0]
    axis.plot(qacf.lag, qacf.observed, "o-", color="#23364d", ms=3.5)
    for label in generated:
        axis.plot(qacf.lag, qacf[f"{label}_mean"], styles[label], color=colors[label])
    axis.set(title=r"Effective-jump ACF ($q$)", xlabel="Lag", ylabel="Correlation")
    axis.grid(alpha=0.16)

    axis = axes[1, 1]
    variables = ["birth", "q", "alpha", "time"]
    width = 0.34
    for index, label in enumerate(generated):
        values = marginal.query("model == @label").set_index("variable").loc[variables].ks_distance
        axis.bar(np.arange(4) + (index - 0.5) * width, values, width, color=colors[label], label=label.replace("_", " "))
    axis.set_xticks(np.arange(4), ["birth", "q", r"$\alpha$", r"$\tau$"])
    axis.set(title="Marginal distributions", ylabel="KS distance")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.16, axis="y")

    axis = axes[1, 2]
    slope_names = ["slope_sd", "median_abs_slope", "median_trend_R2"]
    observed_slope = np.asarray([metrics["observed"][name] for name in slope_names])
    for index, label in enumerate(generated):
        ratios = np.asarray([metrics[label][name] for name in slope_names]) / observed_slope
        axis.bar(np.arange(3) + (index - 0.5) * width, ratios, width, color=colors[label])
    axis.axhline(1.0, color="#23364d", lw=1.0, label="Observed")
    axis.set_xticks(np.arange(3), ["slope SD", "median |slope|", "median trend $R^2$"])
    axis.set(title="Residual lineage-trend statistics", ylabel="Simulation / observed")
    axis.grid(alpha=0.16, axis="y")
    fig.suptitle(r"Glu: shared $E_i$ + lineage offset + fast AR(1) in $q$")
    fig.savefig(OUT / "environment_lineage_offset_diagnostics.png", dpi=190)
    fig.savefig(OUT / "environment_lineage_offset_diagnostics.pdf")
    plt.close(fig)

    diagnostics = {
        "model": {
            "equation": "q_li = mu_q + E_i + A_l + r_li; A_l~N(0,sigma_A^2); r_li stationary AR(1); E_i~N(0,sigma_E^2)",
            "likelihood": "A_l integrated exactly; E_i inferred with Gaussian prior; one covariance per observed contiguous lineage length",
        },
        "comparison": comparison,
        "metrics": metrics,
        "proposed_q": serializable(proposed_q),
        "current_q": serializable(current_q),
        "optimizers": {
            "proposed": {"success": bool(proposed_optimizer.success), "message": str(proposed_optimizer.message)},
            "current": {"success": bool(current_optimizer.success), "message": str(current_optimizer.message)},
        },
        "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (OUT / "fit_results.json").write_text(json.dumps(diagnostics, indent=2))

    lines = [
        "# Glu shared-environment plus lineage-offset q model",
        "",
        r"The proposed process is $q_{\ell i}=\mu_q+E_i+A_\ell+r_{\ell i}$,",
        r"where $E_i$ is shared across lineages, $A_\ell$ is a Gaussian lineage",
        "offset integrated out of the likelihood, and `r` is one stationary AR(1).",
        "The comparator is the current shared-environment fast-plus-slow AR model,",
        "refitted to the same variable-length contiguous q sequences.",
        "",
        "## Fitted q parameters",
        "",
        f"- Mean q: {proposed_q['mean']:.6f}",
        f"- Environmental prior SD: {proposed_q['environment_prior_sd']:.6f}",
        f"- Lineage-offset SD: {proposed_q['sigma_lineage_offset']:.6f}",
        f"- Fast AR coefficient: {proposed_q['rho']:.6f}",
        f"- Fast stationary SD: {proposed_q['stationary_ar_sd']:.6f}",
        f"- Offset fraction of total q variance: {proposed_q['offset_variance_fraction']:.3f}",
        "",
        "## Autonomous validation",
        "",
        "| Model | Raw b-ACF RMSE | Demeaned b-ACF RMSE | q-ACF RMSE | Mean-b RMSE | Birth KS | q KS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in generated:
        value = metrics[label]
        lines.append(
            f"| {label.replace('_', ' ')} | {value['raw_birth_acf_rmse']:.5f} | "
            f"{value['demeaned_birth_acf_rmse']:.5f} | {value['q_acf_rmse']:.5f} | "
            f"{value['mean_birth_rmse']:.5f} | {value['birth_ks']:.4f} | {value['q_ks']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Proposed minus current q joint log density: {comparison['proposed_minus_current_log_density']:.3f}.",
            f"AIC(proposed) - AIC(current): {comparison['aic_proposed_minus_current']:.3f} (negative favors proposed).",
            "",
            "The lineage-slope comparison uses birth sizes after removing the",
            "generation-wise common mean in every observed or simulated replicate.",
            "",
            "## Interpretation",
            "",
            "The lineage offset is clearly nonzero and the remaining AR process is",
            "essentially iid. Nevertheless, replacing the slow AR mode by an",
            "independent lineage offset does not improve the raw birth-size ACF. It",
            "slightly improves the generation-demeaned ACF and mean-size trajectory,",
            "but worsens the q-ACF and is disfavored by both joint q density and AIC.",
            "",
            "A static q offset mostly shifts the lineage's size setpoint under the",
            "hazard feedback; the per-lineage centering in the ACF then removes much",
            "of its effect. The observed slope heterogeneity therefore cannot be",
            "represented adequately by an independent q intercept alone.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")

    print(
        json.dumps(
            {
                "qfit": {
                    key: proposed_q[key]
                    for key in [
                        "mean",
                        "rho",
                        "sigma_lineage_offset",
                        "stationary_ar_sd",
                        "environment_prior_sd",
                        "offset_variance_fraction",
                        "log_joint_density",
                    ]
                },
                "comparison": comparison,
                "metrics": metrics,
                "runtime_minutes": diagnostics["runtime_minutes"],
            },
            default=serializable,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
