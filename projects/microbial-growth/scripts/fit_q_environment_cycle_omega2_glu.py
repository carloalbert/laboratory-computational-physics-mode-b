#!/usr/bin/env python3
"""Fit Glu with individual q effects and one inferred omega2_i per cycle."""

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
from scipy.special import expit, logsumexp


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_all_parameters_birth_acf_glu import marginal_rows  # noqa: E402
from fit_autoregressive_residual_glu import (  # noqa: E402
    contiguous_prefixes,
    observed_matrix,
)
from fit_environment_lineage_offset_ar_q_glu import acf_summary, fit_q  # noqa: E402
from fit_individual_environment_lineage_effects_glu import (  # noqa: E402
    joint_effect_posterior,
    posterior_summary_tables,
    serializable,
    simulate,
)
from fit_individual_q_alpha_environment_effects_glu import trajectory  # noqa: E402
from fit_lineage_offset_ar_q_glu import q_matrix  # noqa: E402
from fit_q_environment_birth_hill_glu import time_acf  # noqa: E402
from fit_q_environment_cycle_u_glu import (  # noqa: E402
    CycleThresholdLikelihood,
    weighted_quantile,
)
from glu_workflow import load_data  # noqa: E402


BASE = PROJECT_DIR / "outputs/model_comparison/Glu"
SOURCE = BASE / "environment_fast_slow_ar_q_protein_hazard"
OUT = BASE / "q_environment_cycle_omega2_protein_hazard"


class CycleRateLikelihood(CycleThresholdLikelihood):
    """Random-cycle log-rate likelihood with a common threshold u."""

    def __init__(self, prefixes, integration_order=32, random_effect_order=32):
        super().__init__(prefixes, integration_order, random_effect_order)

    def cycle_rate_log_likelihood(self, group, log_u, log_omega2, eta):
        log_omega2 = np.atleast_1d(np.asarray(log_omega2, dtype=float))
        event_odds = eta * (group["event_log_added"] - log_u)
        integration_odds = eta * (group["integration_log_added"] - log_u)
        integral = group["integration_scale"] * np.sum(
            self.integration_weights[None, :] * expit(integration_odds),
            axis=1,
        )
        return (
            group["sum_log_alpha"]
            + group["n"] * log_omega2
            + np.sum(-np.logaddexp(0.0, -event_odds))
            - np.exp(log_omega2) * np.sum(integral)
        )

    def marginal_log_likelihood(self, theta):
        log_u, mean_log_omega2, log_eta, log_sigma_omega2 = map(float, theta)
        eta = math.exp(log_eta)
        sigma = math.exp(log_sigma_omega2)
        nodes = mean_log_omega2 + math.sqrt(2.0) * sigma * self.hermite_nodes
        return float(
            sum(
                logsumexp(
                    self.cycle_rate_log_likelihood(group, log_u, nodes, eta)
                    + self.log_hermite_weights
                )
                for group in self.groups
            )
        )

    def posterior(self, theta):
        log_u, mean_log_omega2, log_eta, log_sigma_omega2 = map(float, theta)
        eta = math.exp(log_eta)
        sigma = math.exp(log_sigma_omega2)
        nodes = mean_log_omega2 + math.sqrt(2.0) * sigma * self.hermite_nodes
        probabilities = []
        for group in self.groups:
            log_weight = (
                self.cycle_rate_log_likelihood(group, log_u, nodes, eta)
                + self.log_hermite_weights
            )
            log_weight -= logsumexp(log_weight)
            probabilities.append(np.exp(log_weight))
        return nodes, np.asarray(probabilities)


def fit_cycle_rate(prefixes, source_hazard):
    likelihood = CycleRateLikelihood(prefixes)
    shared_bounds = [
        (math.log(0.05), math.log(10.0)),
        (math.log(1.0), math.log(25.0)),
        (math.log(0.2), math.log(30.0)),
    ]
    starts = [
        np.log([source_hazard["u"], source_hazard["omega2"], source_hazard["eta"]]),
        np.log([1.0, 10.0, 6.0]),
        np.log([2.0, 12.0, 12.0]),
    ]
    shared_fits = [
        minimize(
            lambda theta: -likelihood.shared_log_likelihood(theta),
            start,
            method="L-BFGS-B",
            bounds=shared_bounds,
            options={"maxiter": 1500, "ftol": 1e-12},
        )
        for start in starts
    ]
    shared_fit = min(shared_fits, key=lambda item: item.fun)
    bounds = shared_bounds + [(math.log(0.003), math.log(1.0))]
    hierarchical_fits = [
        minimize(
            lambda theta: -likelihood.marginal_log_likelihood(theta),
            np.r_[shared_fit.x, math.log(sigma)],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        for sigma in [0.03, 0.10, 0.25, 0.5]
    ]
    fitted = min(hierarchical_fits, key=lambda item: item.fun)
    nodes, probabilities = likelihood.posterior(fitted.x)
    posterior_mean_log = probabilities @ nodes
    posterior_sd_log = np.sqrt(probabilities @ (nodes**2) - posterior_mean_log**2)
    posterior_mean = probabilities @ np.exp(nodes)
    log_u, mean_log, log_eta, log_sigma = map(float, fitted.x)
    fit = {
        "u": math.exp(log_u),
        "mean_omega2": math.exp(mean_log),
        "mean_log_omega2": mean_log,
        "eta": math.exp(log_eta),
        "sigma_log_omega2": math.exp(log_sigma),
        "log_marginal_likelihood": -float(fitted.fun),
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "posterior_nodes_log_omega2": nodes,
        "posterior_probabilities": probabilities,
        "posterior_mean_log_omega2": posterior_mean_log,
        "posterior_sd_log_omega2": posterior_sd_log,
        "posterior_mean_omega2": posterior_mean,
    }
    shared = {
        "u": math.exp(float(shared_fit.x[0])),
        "omega2": math.exp(float(shared_fit.x[1])),
        "eta": math.exp(float(shared_fit.x[2])),
        "log_likelihood": -float(shared_fit.fun),
        "success": bool(shared_fit.success),
        "message": str(shared_fit.message),
    }
    gain = 2.0 * (fit["log_marginal_likelihood"] - shared["log_likelihood"])
    comparison = {
        "n_transitions": likelihood.n_observations,
        "twice_log_likelihood_gain": gain,
        "delta_bic_hierarchical_minus_shared": math.log(likelihood.n_observations)
        - gain,
    }
    return likelihood, fit, shared, comparison


def rate_table(likelihood, fit):
    nodes = fit["posterior_nodes_log_omega2"]
    rows = []
    for group, probabilities, mean_log, sd_log, mean_rate in zip(
        likelihood.groups,
        fit["posterior_probabilities"],
        fit["posterior_mean_log_omega2"],
        fit["posterior_sd_log_omega2"],
        fit["posterior_mean_omega2"],
    ):
        rows.append(
            {
                "generation": group["generation"],
                "active_lineages": group["n"],
                "observed_mean_growth": group["observed_mean_growth"],
                "posterior_mean_log_omega2": float(mean_log),
                "posterior_sd_log_omega2": float(sd_log),
                "posterior_mean_omega2": float(mean_rate),
                "posterior_median_omega2": math.exp(
                    weighted_quantile(nodes, probabilities, 0.5)
                ),
                "posterior_lo95_omega2": math.exp(
                    weighted_quantile(nodes, probabilities, 0.025)
                ),
                "posterior_hi95_omega2": math.exp(
                    weighted_quantile(nodes, probabilities, 0.975)
                ),
            }
        )
    return pd.DataFrame(rows)


def draw_cycle_log_omega2(fit, replicates, rng):
    nodes = fit["posterior_nodes_log_omega2"]
    probabilities = fit["posterior_probabilities"]
    draws = np.empty((replicates, len(probabilities)))
    for generation, weights in enumerate(probabilities):
        draws[:, generation] = rng.choice(nodes, size=replicates, p=weights)
    return draws


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--seed", type=int, default=91723)
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
    lineage_table, environment_table, _ = posterior_summary_tables(
        prefixes, q_values, q_lengths, q_posterior
    )
    lineage_table.to_csv(OUT / "lineage_effect_posterior.csv", index=False)
    environment_table.to_csv(OUT / "q_environment_effect_posterior.csv", index=False)

    likelihood, rate_fit, shared_hazard, comparison = fit_cycle_rate(
        prefixes, source["fit"]["hazard"]
    )
    rates = rate_table(likelihood, rate_fit)
    rates.to_csv(OUT / "cycle_omega2_posterior.csv", index=False)
    rng = np.random.default_rng(args.seed)
    cycle_log_omega2 = draw_cycle_log_omega2(rate_fit, args.replicates, rng)
    common = {
        "initial_birth": source["fit"]["initial_birth"],
        "alpha": source["fit"]["alpha"],
        "q": qfit,
    }
    cycle_hazard = {
        "u": rate_fit["u"],
        "omega2": rate_fit["mean_omega2"],
        "eta": rate_fit["eta"],
    }
    initial = np.asarray([float(group.length_birth.iloc[0]) for group in prefixes])
    generated = {
        "shared_omega2": simulate(
            {**common, "hazard": shared_hazard}, lengths, q_posterior, q_lengths,
            args.replicates, args.seed + 1, conditional=True, observed_initial=initial,
        ),
        "cycle_omega2": simulate(
            {**common, "hazard": cycle_hazard}, lengths, q_posterior, q_lengths,
            args.replicates, args.seed + 2, conditional=True, observed_initial=initial,
            cycle_log_omega2_draws=cycle_log_omega2,
        ),
    }

    trajectory_tables, marginal_values = [], []
    birth_acf = pd.DataFrame({"lag": np.arange(args.max_lag + 1)})
    tau_acf = pd.DataFrame({"lag": np.arange(args.max_lag + 1)})
    for label, simulation in generated.items():
        values = trajectory(prefixes, lengths, simulation, args.horizon)
        values["model"] = label
        trajectory_tables.append(values)
        target, mean, low, high = acf_summary(
            observed_birth, simulation["birth"], lengths, args.max_lag
        )
        birth_acf["observed"] = target
        birth_acf[f"{label}_mean"] = mean
        birth_acf[f"{label}_lo95"] = low
        birth_acf[f"{label}_hi95"] = high
        target, mean, low, high = time_acf(prefixes, simulation, q_lengths, args.max_lag)
        tau_acf["observed"] = target
        tau_acf[f"{label}_mean"] = mean
        tau_acf[f"{label}_lo95"] = low
        tau_acf[f"{label}_hi95"] = high
        marginal_values.extend(marginal_rows(prefixes, simulation, lengths, label))
    trajectories = pd.concat(trajectory_tables, ignore_index=True)
    marginal = pd.DataFrame(marginal_values)
    trajectories.to_csv(OUT / "balanced_trajectory_comparison.csv", index=False)
    birth_acf.to_csv(OUT / "birth_acf_comparison.csv", index=False)
    tau_acf.to_csv(OUT / "tau_acf_comparison.csv", index=False)
    marginal.to_csv(OUT / "marginal_comparison.csv", index=False)

    metrics = {}
    fit_slice = slice(1, args.max_lag + 1)
    for label in generated:
        metrics[label] = {}
        for variable in ["birth", "alpha", "time", "growth"]:
            values = trajectories.query("model == @label and variable == @variable")
            metrics[label][f"{variable}_trajectory_rmse"] = float(
                np.sqrt(np.mean((values.observed - values.simulated_mean) ** 2))
            )
        for name, table in [("birth_acf", birth_acf), ("tau_acf", tau_acf)]:
            metrics[label][f"{name}_rmse"] = float(
                np.sqrt(np.mean((table.observed.to_numpy()[fit_slice] - table[f"{label}_mean"].to_numpy()[fit_slice]) ** 2))
            )
        lookup = marginal.query("model == @label").set_index("variable")
        for variable in ["birth", "q", "alpha", "time"]:
            metrics[label][f"{variable}_ks"] = float(lookup.loc[variable, "ks_distance"])

    colors = {"shared_omega2": "#718096", "cycle_omega2": "#287a78"}
    labels = {"shared_omega2": r"Shared $\omega_2$", "cycle_omega2": r"Inferred $\omega_{2,i}$"}
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.1), constrained_layout=True)
    axis = axes[0, 0]
    axis.fill_between(rates.generation, rates.posterior_lo95_omega2, rates.posterior_hi95_omega2, color="#287a78", alpha=0.16)
    axis.plot(rates.generation, rates.posterior_mean_omega2, "s-", color="#287a78", ms=3)
    axis.axhline(rate_fit["mean_omega2"], color="0.5", ls="--", lw=1)
    axis.set(title=r"Cycle-rate posterior $\omega_{2,i}$", xlabel="Generation", ylabel=r"$\omega_{2,i}$")
    for axis, variable, title, ylabel in [
        (axes[0, 1], "birth", "Balanced-cohort mean birth size", "Birth size"),
        (axes[0, 2], "growth", r"Mean growth increment $\alpha\tau$", r"$\alpha\tau$"),
        (axes[1, 0], "time", r"Mean division time $\tau$", r"$\tau$"),
    ]:
        for label in generated:
            values = trajectories.query("model == @label and variable == @variable")
            axis.fill_between(values.generation, values.simulated_lo95, values.simulated_hi95, color=colors[label], alpha=0.09)
            axis.plot(values.generation, values.simulated_mean, "--" if label == "shared_omega2" else "-", color=colors[label], label=labels[label])
        observed = trajectories.query("model == 'shared_omega2' and variable == @variable")
        axis.plot(observed.generation, observed.observed, "o-", color="#23364d", ms=3, label="Observed")
        axis.set(title=title, xlabel="Generation", ylabel=ylabel)
    axes[0, 1].legend(frameon=False, fontsize=8)
    for axis, table, title in [(axes[1, 1], birth_acf, "Birth-size ACF"), (axes[1, 2], tau_acf, "Division-time ACF")]:
        axis.plot(table.lag, table.observed, "o-", color="#23364d", ms=3)
        for label in generated:
            axis.fill_between(table.lag, table[f"{label}_lo95"], table[f"{label}_hi95"], color=colors[label], alpha=0.09)
            axis.plot(table.lag, table[f"{label}_mean"], "--" if label == "shared_omega2" else "-", color=colors[label])
        axis.set(title=title, xlabel="Lag", ylabel="Correlation")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    fig.suptitle(r"Glu: inferred $E_i$ in q and cycle-specific $\omega_{2,i}$")
    fig.savefig(OUT / "cycle_omega2_diagnostics.png", dpi=190)
    fig.savefig(OUT / "cycle_omega2_diagnostics.pdf")
    plt.close(fig)

    fit_summary = {key: rate_fit[key] for key in [
        "u", "mean_omega2", "mean_log_omega2", "eta", "sigma_log_omega2",
        "log_marginal_likelihood", "success", "message",
    ]}
    result = {
        "model": {
            "q": "q_li=A_l+E_i+r_li; every A_l and E_i inferred",
            "alpha": "stationary Gamma marginal with Gaussian-copula AR(1); no G_i process",
            "hazard": "h_i(t|b,alpha)=alpha*omega2_i*sigmoid(eta*log(b*(exp(alpha*t)-1)/u))",
            "rate_prior": "log(omega2_i)~Normal(mu_omega2,sigma_omega2^2); every omega2_i integrated for hyperparameter fitting and inferred individually afterward",
            "explicit_static_birth_term": "absent (kappa=0)",
        },
        "fit": {
            "cycle_rate_hazard": serializable(fit_summary),
            "shared_rate_hazard": serializable(shared_hazard),
            "likelihood_comparison": serializable(comparison),
            "q_log_joint_density": float(q_density),
            "q_optimizer_success": bool(q_optimizer.success),
        },
        "metrics": serializable(metrics),
        "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2))

    shared, cycle = metrics["shared_omega2"], metrics["cycle_omega2"]
    drift_gain = 100.0 * (1.0 - cycle["birth_trajectory_rmse"] / shared["birth_trajectory_rmse"])
    acf_gain = 100.0 * (1.0 - cycle["birth_acf_rmse"] / shared["birth_acf_rmse"])
    lines = [
        "# Glu q environment plus cycle-specific omega2", "",
        r"The q model remains $q_{\ell i}=A_\ell+E_i+r_{\ell i}$. There is no",
        r"alpha $G_i$ process and no explicit-birth coefficient $\kappa$. The hazard is", "",
        r"$$h_i(t\mid b,\alpha)=\alpha\omega_{2,i}\,\operatorname{sigmoid}\!\left[",
        r"\eta\log\frac{b(e^{\alpha t}-1)}{u}\right],$$", "",
        r"$$\log\omega_{2,i}\sim\mathcal N(\mu_{\log\omega_2},\sigma_{\log\omega_2}^2).$$", "",
        "Each cycle rate is integrated by Gauss-Hermite quadrature during hyperparameter fitting,",
        "then its posterior is recovered and propagated through simulation.", "", "## Fit", "",
        f"- Shared threshold u: {rate_fit['u']:.6f}",
        f"- Population geometric-mean omega2: {rate_fit['mean_omega2']:.6f}",
        f"- Between-cycle SD of log omega2: {rate_fit['sigma_log_omega2']:.6f}",
        f"- eta: {rate_fit['eta']:.6f}",
        f"- Twice marginal log-likelihood gain: {comparison['twice_log_likelihood_gain']:.3f}",
        f"- Delta BIC: {comparison['delta_bic_hierarchical_minus_shared']:.3f}", "",
        "## Posterior checks", "", "| Metric | Shared omega2 | Inferred omega2_i |", "|---|---:|---:|",
    ]
    for key, title in [
        ("birth_trajectory_rmse", "Mean-b trajectory RMSE"),
        ("growth_trajectory_rmse", "Alpha-tau trajectory RMSE"),
        ("time_trajectory_rmse", "Tau trajectory RMSE"),
        ("birth_acf_rmse", "Birth-ACF RMSE"),
        ("tau_acf_rmse", "Tau-ACF RMSE"),
        ("birth_ks", "Birth KS"), ("time_ks", "Tau KS"),
    ]:
        lines.append(f"| {title} | {shared[key]:.5f} | {cycle[key]:.5f} |")
    lines.extend(["", "## Interpretation", "",
        f"Mean-birth trajectory RMSE changes by {drift_gain:.1f}% and birth-ACF RMSE by {acf_gain:.1f}%",
        "relative to the refitted shared-rate model (positive means improvement). This is an",
        "in-sample empirical-Bayes diagnostic; held-out cycles are needed for a predictive test.",
    ])
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
