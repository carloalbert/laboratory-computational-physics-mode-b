#!/usr/bin/env python3
"""Add individually inferred shared cycle effects to both q and alpha for Glu."""

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
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from autoregressive_residual_model import lineage_acf  # noqa: E402
from fit_all_parameters_birth_acf_glu import marginal_rows  # noqa: E402
from fit_autoregressive_residual_glu import contiguous_prefixes, observed_matrix  # noqa: E402
from fit_environment_lineage_offset_ar_q_glu import acf_summary, fit_q  # noqa: E402
from fit_individual_environment_lineage_effects_glu import (  # noqa: E402
    joint_effect_posterior,
    posterior_summary_tables,
    serializable,
    simulate,
)
from fit_lineage_offset_ar_q_glu import q_matrix  # noqa: E402
from glu_workflow import load_data  # noqa: E402


BASE = PROJECT_DIR / "outputs/model_comparison/Glu"
SOURCE = BASE / "environment_fast_slow_ar_q_protein_hazard"
OUT = BASE / "individual_q_alpha_environment_effects_protein_hazard"


def alpha_latent_matrix(prefixes, alpha_fit):
    lengths = np.asarray([len(group) for group in prefixes], dtype=int)
    values = np.full((len(prefixes), int(lengths.max())), np.nan)
    for index, group in enumerate(prefixes):
        cdf = gamma_dist.cdf(
            group.alpha.to_numpy(float),
            a=alpha_fit["shape"],
            scale=alpha_fit["scale"],
        )
        values[index, : lengths[index]] = norm.ppf(
            np.clip(cdf, 1e-10, 1.0 - 1e-10)
        )
    return values, lengths


def alpha_mode_and_log_density(theta, sequences, environment_count):
    rho = float(theta[0])
    sigma_environment = math.exp(float(theta[1]))
    if not (-0.95 < rho < 0.95 and 0.0 < sigma_environment < 1.0):
        return None, -np.inf
    residual_sd = math.sqrt(1.0 - sigma_environment**2)
    precision = np.eye(environment_count) / sigma_environment**2
    score = np.zeros(environment_count)
    cache = {}
    for sequence in sequences:
        length = len(sequence)
        if length not in cache:
            distance = np.abs(
                np.subtract.outer(np.arange(length), np.arange(length))
            )
            covariance = residual_sd**2 * rho**distance
            factor = cho_factor(covariance, lower=True, check_finite=False)
            inverse = cho_solve(factor, np.eye(length), check_finite=False)
            logdet = 2.0 * float(np.log(np.diag(factor[0])).sum())
            cache[length] = factor, inverse, logdet
        _, inverse, _ = cache[length]
        precision[:length, :length] += inverse
        score[:length] += inverse @ sequence

    factor_g = cho_factor(precision, lower=True, check_finite=False)
    covariance_g = cho_solve(
        factor_g, np.eye(environment_count), check_finite=False
    )
    effects = cho_solve(factor_g, score, check_finite=False)
    value = 0.0
    for sequence in sequences:
        length = len(sequence)
        factor, _, logdet = cache[length]
        residual = sequence - effects[:length]
        whitened = np.linalg.solve(np.tril(factor[0]), residual)
        value += -0.5 * (
            length * math.log(2.0 * math.pi) + logdet + whitened @ whitened
        )
    value += float(
        np.sum(
            -0.5
            * (
                math.log(2.0 * math.pi * sigma_environment**2)
                + effects**2 / sigma_environment**2
            )
        )
    )
    sign, posterior_logdet = np.linalg.slogdet(covariance_g)
    if sign <= 0:
        return None, -np.inf
    # Exact Gaussian identity: p(z)=p(z|G_hat)p(G_hat)/p(G_hat|z).
    marginal_value = value + 0.5 * (
        environment_count * math.log(2.0 * math.pi) + posterior_logdet
    )
    result = {
        "rho": rho,
        "environment_prior_sd": sigma_environment,
        "stationary_residual_sd": residual_sd,
        "sigma_residual_innovation": residual_sd * math.sqrt(1.0 - rho**2),
        "posterior_mean_G": effects,
        "posterior_covariance_G": 0.5 * (covariance_g + covariance_g.T),
        "posterior_sd_G": np.sqrt(np.maximum(np.diag(covariance_g), 0.0)),
        "latent_log_joint_density": float(value),
        "latent_log_marginal_likelihood": float(marginal_value),
    }
    return result, float(marginal_value)


def fit_alpha_environment(latent, lengths):
    sequences = [latent[index, :length] for index, length in enumerate(lengths)]
    starts = [
        [0.52, math.log(0.12)],
        [0.45, math.log(0.20)],
        [0.60, math.log(0.08)],
        [0.30, math.log(0.30)],
    ]
    bounds = [(-0.8, 0.9), (math.log(0.001), math.log(0.75))]

    def objective(theta):
        try:
            return -alpha_mode_and_log_density(theta, sequences, latent.shape[1])[1]
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
    result, value = alpha_mode_and_log_density(
        fitted.x, sequences, latent.shape[1]
    )
    result["success"] = bool(fitted.success)
    result["message"] = str(fitted.message)
    result["multistart_log_joint_densities"] = [
        -float(item.fun) for item in fits
    ]
    return result, value, fitted


def alpha_environment_table(prefixes, latent, lengths, fit):
    rows = []
    for generation in range(latent.shape[1]):
        active = lengths > generation
        observed_alpha = np.asarray(
            [group.alpha.iloc[generation] for group in prefixes if len(group) > generation],
            dtype=float,
        )
        mean = fit["posterior_mean_G"][generation]
        sd = fit["posterior_sd_G"][generation]
        rows.append(
            {
                "generation": generation,
                "active_lineages": int(active.sum()),
                "observed_mean_alpha": float(observed_alpha.mean()),
                "observed_mean_latent_z": float(latent[active, generation].mean()),
                "posterior_mean_G": float(mean),
                "posterior_sd_G": float(sd),
                "posterior_lo95_G": float(mean - 1.96 * sd),
                "posterior_hi95_G": float(mean + 1.96 * sd),
            }
        )
    return pd.DataFrame(rows)


def trajectory(prefixes, lengths, generated, horizon):
    complete = lengths > horizon
    observed = {
        "birth": np.asarray(
            [group.length_birth.to_numpy(float)[: horizon + 1] for group, keep in zip(prefixes, complete) if keep]
        ),
        "alpha": np.asarray(
            [group.alpha.to_numpy(float)[:horizon] for group, keep in zip(prefixes, complete) if keep]
        ),
        "time": np.asarray(
            [group.time.to_numpy(float)[:horizon] for group, keep in zip(prefixes, complete) if keep]
        ),
    }
    observed["growth"] = observed["alpha"] * observed["time"]
    rows = []
    for variable in ["birth", "alpha", "time", "growth"]:
        stop = horizon + 1 if variable == "birth" else horizon
        obs = observed[variable].mean(axis=0)
        paths = generated[variable][:, complete, :stop].mean(axis=1)
        low, high = np.quantile(paths, [0.025, 0.975], axis=0)
        for generation in range(stop):
            rows.append(
                {
                    "variable": variable,
                    "generation": generation,
                    "observed": float(obs[generation]),
                    "simulated_mean": float(paths[:, generation].mean()),
                    "simulated_lo95": float(low[generation]),
                    "simulated_hi95": float(high[generation]),
                }
            )
    return pd.DataFrame(rows)


def alpha_acf_summary(prefixes, generated, lengths, max_lag):
    observed = np.full((len(prefixes), int(lengths.max())), np.nan)
    for index, group in enumerate(prefixes):
        observed[index, : len(group)] = group.alpha.to_numpy(float)
    target = lineage_acf(observed, lengths, max_lag)[0]
    samples = lineage_acf(generated["alpha"], lengths, max_lag)
    low, high = np.nanquantile(samples, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "lag": np.arange(max_lag + 1),
            "observed": target,
            "simulated_mean": np.nanmean(samples, axis=0),
            "simulated_lo95": low,
            "simulated_hi95": high,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--seed", type=int, default=77123)
    args = parser.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    source = json.loads((SOURCE / "fit_results.json").read_text())
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    q_values, q_lengths = q_matrix(prefixes)
    qfit, q_density, q_optimizer = fit_q(
        [q_values[index, :length] for index, length in enumerate(q_lengths) if length],
        q_values.shape[1],
    )
    q_posterior = joint_effect_posterior(q_values, q_lengths, qfit)
    lineage_table, q_environment_table, _ = posterior_summary_tables(
        prefixes, q_values, q_lengths, q_posterior
    )
    lineage_table.to_csv(OUT / "lineage_effect_posterior.csv", index=False)
    q_environment_table.to_csv(OUT / "q_environment_effect_posterior.csv", index=False)

    latent_alpha, alpha_lengths = alpha_latent_matrix(prefixes, source["fit"]["alpha"])
    alpha_environment, alpha_density, alpha_optimizer = fit_alpha_environment(
        latent_alpha, alpha_lengths
    )
    alpha_table = alpha_environment_table(
        prefixes, latent_alpha, alpha_lengths, alpha_environment
    )
    alpha_table.to_csv(OUT / "alpha_environment_effect_posterior.csv", index=False)
    alpha_posterior = {
        "mean": alpha_environment["posterior_mean_G"],
        "covariance": alpha_environment["posterior_covariance_G"],
    }
    enhanced_alpha = {
        **source["fit"]["alpha"],
        **{
            key: alpha_environment[key]
            for key in [
                "rho",
                "environment_prior_sd",
                "stationary_residual_sd",
                "sigma_residual_innovation",
            ]
        },
    }
    common = {
        "initial_birth": source["fit"]["initial_birth"],
        "hazard": source["fit"]["hazard"],
        "q": qfit,
    }
    initial = np.asarray([float(group.length_birth.iloc[0]) for group in prefixes])
    generated = {}
    generated["stationary_alpha"] = simulate(
        {**common, "alpha": source["fit"]["alpha"]},
        lengths,
        q_posterior,
        q_lengths,
        args.replicates,
        args.seed,
        conditional=True,
        observed_initial=initial,
    )
    generated["alpha_environment"] = simulate(
        {**common, "alpha": enhanced_alpha},
        lengths,
        q_posterior,
        q_lengths,
        args.replicates,
        args.seed + 1,
        conditional=True,
        observed_initial=initial,
        alpha_posterior=alpha_posterior,
    )

    trajectory_tables = []
    birth_acf = pd.DataFrame({"lag": np.arange(args.max_lag + 1)})
    alpha_acf = pd.DataFrame({"lag": np.arange(args.max_lag + 1)})
    marginals = []
    for label, simulation in generated.items():
        table = trajectory(prefixes, lengths, simulation, args.horizon)
        table["model"] = label
        trajectory_tables.append(table)
        target, mean, low, high = acf_summary(
            observed_birth, simulation["birth"], lengths, args.max_lag
        )
        birth_acf["observed"] = target
        birth_acf[f"{label}_mean"] = mean
        birth_acf[f"{label}_lo95"] = low
        birth_acf[f"{label}_hi95"] = high
        atable = alpha_acf_summary(prefixes, simulation, lengths, args.max_lag)
        alpha_acf["observed"] = atable.observed
        alpha_acf[f"{label}_mean"] = atable.simulated_mean
        alpha_acf[f"{label}_lo95"] = atable.simulated_lo95
        alpha_acf[f"{label}_hi95"] = atable.simulated_hi95
        marginals.extend(marginal_rows(prefixes, simulation, lengths, label))
    trajectories = pd.concat(trajectory_tables, ignore_index=True)
    marginal = pd.DataFrame(marginals)
    trajectories.to_csv(OUT / "balanced_trajectory_comparison.csv", index=False)
    birth_acf.to_csv(OUT / "birth_acf_comparison.csv", index=False)
    alpha_acf.to_csv(OUT / "alpha_acf_comparison.csv", index=False)
    marginal.to_csv(OUT / "marginal_comparison.csv", index=False)

    metrics = {}
    fit_slice = slice(1, args.max_lag + 1)
    for label in generated:
        metrics[label] = {}
        for variable in ["birth", "alpha", "time", "growth"]:
            table = trajectories.query("model == @label and variable == @variable")
            metrics[label][f"{variable}_trajectory_rmse"] = float(
                np.sqrt(np.mean((table.observed - table.simulated_mean) ** 2))
            )
        metrics[label]["birth_acf_rmse"] = float(
            np.sqrt(
                np.mean(
                    (birth_acf.observed.to_numpy()[fit_slice] - birth_acf[f"{label}_mean"].to_numpy()[fit_slice]) ** 2
                )
            )
        )
        metrics[label]["alpha_acf_rmse"] = float(
            np.sqrt(
                np.mean(
                    (alpha_acf.observed.to_numpy()[fit_slice] - alpha_acf[f"{label}_mean"].to_numpy()[fit_slice]) ** 2
                )
            )
        )
        lookup = marginal.query("model == @label").set_index("variable")
        for variable in ["birth", "alpha", "time", "q"]:
            metrics[label][f"{variable}_ks"] = float(
                lookup.loc[variable, "ks_distance"]
            )

    colors = {"stationary_alpha": "#718096", "alpha_environment": "#c4512c"}
    labels = {
        "stationary_alpha": r"Stationary $\alpha$",
        "alpha_environment": r"Inferred $G_i$",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.1), constrained_layout=True)
    axis = axes[0, 0]
    axis.fill_between(alpha_table.generation, alpha_table.posterior_lo95_G, alpha_table.posterior_hi95_G, color="#c4512c", alpha=0.16)
    axis.plot(alpha_table.generation, alpha_table.posterior_mean_G, "s-", color="#c4512c", ms=3)
    axis.axhline(0.0, color="0.55", lw=0.8)
    axis.set(title=r"Alpha environment posterior $G_i$", xlabel="Generation", ylabel="Latent-normal effect")

    for axis, variable, title, ylabel in [
        (axes[0, 1], "alpha", r"Mean growth rate $\alpha$", r"$\alpha$"),
        (axes[0, 2], "time", r"Mean division time $\tau$", r"$\tau$"),
        (axes[1, 0], "growth", r"Mean growth increment $\alpha\tau$", r"$\alpha\tau$"),
        (axes[1, 1], "birth", "Mean birth size", "Birth size"),
    ]:
        for label in generated:
            table = trajectories.query("model == @label and variable == @variable")
            axis.fill_between(table.generation, table.simulated_lo95, table.simulated_hi95, color=colors[label], alpha=0.09)
            axis.plot(table.generation, table.simulated_mean, "--" if label == "stationary_alpha" else "-", color=colors[label], label=labels[label])
        observed = trajectories.query("model == 'stationary_alpha' and variable == @variable")
        axis.plot(observed.generation, observed.observed, "o-", color="#23364d", ms=3, label="Observed")
        axis.set(title=title, xlabel="Generation", ylabel=ylabel)
    axes[0, 1].legend(frameon=False, fontsize=8)

    axis = axes[1, 2]
    axis.plot(birth_acf.lag, birth_acf.observed, "o-", color="#23364d", ms=3, label="Observed")
    for label in generated:
        axis.plot(birth_acf.lag, birth_acf[f"{label}_mean"], "--" if label == "stationary_alpha" else "-", color=colors[label], label=labels[label])
    axis.set(title="Birth-size ACF", xlabel="Lag", ylabel="Correlation")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    fig.suptitle(r"Glu: individually inferred $E_i$ in q and $G_i$ in $\alpha$")
    fig.savefig(OUT / "q_alpha_environment_diagnostics.png", dpi=190)
    fig.savefig(OUT / "q_alpha_environment_diagnostics.pdf")
    plt.close(fig)

    alpha_summary = {
        key: alpha_environment[key]
        for key in [
            "rho",
            "environment_prior_sd",
            "stationary_residual_sd",
            "sigma_residual_innovation",
            "latent_log_joint_density",
            "latent_log_marginal_likelihood",
            "success",
            "message",
        ]
    }
    result = {
        "model": {
            "q": "q_li=A_l+E_i+r_li with every A_l and E_i inferred",
            "alpha": "z_li=G_i+x_li; alpha_li=GammaQuantile(Phi(z_li)); every G_i inferred",
            "variance_constraint": "Var(G_i)+Var(x_li)=1, preserving the fitted Gamma marginal before conditioning",
            "hazard_consequence": "The hazard is alpha-scaled, so G_i changes tau but not the sampled growth increment s=alpha*tau or the birth recursion",
        },
        "fit": {
            "alpha_environment": serializable(alpha_summary),
            "q_log_joint_density": float(q_density),
            "alpha_log_joint_density": float(alpha_density),
            "q_optimizer_success": bool(q_optimizer.success),
            "alpha_optimizer_success": bool(alpha_optimizer.success),
        },
        "metrics": serializable(metrics),
        "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Glu q and alpha environmental effects",
        "",
        r"This model retains $q_{\ell i}=A_\ell+E_i+r_{\ell i}$ and adds",
        r"$z^{(\alpha)}_{\ell i}=G_i+x_{\ell i}$ with",
        r"$\alpha_{\ell i}=F_\Gamma^{-1}[\Phi(z^{(\alpha)}_{\ell i})]$.",
        r"Every $A_\ell$, $E_i$, and $G_i$ is inferred. The constraint",
        r"$\sigma_G^2+\sigma_x^2=1$ preserves the fitted Gamma marginal on the",
        "unconditional latent-normal scale.",
        "",
        "## Fitted alpha process",
        "",
        f"- Shared alpha-environment SD on the latent scale: {alpha_environment['environment_prior_sd']:.5f}",
        f"- Residual alpha AR coefficient: {alpha_environment['rho']:.5f}",
        f"- Residual stationary SD: {alpha_environment['stationary_residual_sd']:.5f}",
        "",
        "## Posterior checks",
        "",
        "| Metric | Stationary alpha | Inferred G_i |",
        "|---|---:|---:|",
    ]
    for key, title in [
        ("alpha_trajectory_rmse", "Alpha trajectory RMSE"),
        ("time_trajectory_rmse", "Tau trajectory RMSE"),
        ("growth_trajectory_rmse", "Alpha-tau trajectory RMSE"),
        ("birth_trajectory_rmse", "Mean-b trajectory RMSE"),
        ("alpha_acf_rmse", "Alpha ACF RMSE"),
        ("birth_acf_rmse", "Birth ACF RMSE"),
        ("alpha_ks", "Alpha KS"),
        ("time_ks", "Tau KS"),
    ]:
        lines.append(
            f"| {title} | {metrics['stationary_alpha'][key]:.5f} | {metrics['alpha_environment'][key]:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            r"The inferred $G_i$ path can reproduce the growth-rate drift and thereby",
            r"change the division-time trajectory. It cannot repair the growth-increment",
            r"or birth-size trajectories in this hazard model: simulation samples",
            r"$s=\alpha\tau$ directly from a distribution depending on birth size and",
            r"the hazard parameters, after which $\tau=s/\alpha$. Consequently changing",
            r"$\alpha$ leaves both $s$ and $b_{\ell,i+1}=b_{\ell i}\exp(s+q)$ invariant",
            "apart from Monte Carlo differences between runs.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
