#!/usr/bin/env python3
"""Fit Glu with individual q effects and one inferred hazard threshold u_i per cycle."""

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
from fit_individual_q_alpha_environment_effects_glu import trajectory  # noqa: E402
from fit_lineage_offset_ar_q_glu import q_matrix  # noqa: E402
from fit_q_environment_birth_hill_glu import time_acf  # noqa: E402
from glu_workflow import load_data  # noqa: E402


BASE = PROJECT_DIR / "outputs/model_comparison/Glu"
SOURCE = BASE / "environment_fast_slow_ar_q_protein_hazard"
OUT = BASE / "q_environment_cycle_u_protein_hazard"


class CycleThresholdLikelihood:
    """Exact random-cycle-effect likelihood using Gauss-Hermite quadrature."""

    def __init__(self, prefixes, integration_order=32, random_effect_order=24):
        nodes, self.integration_weights = np.polynomial.legendre.leggauss(
            integration_order
        )
        self.groups = []
        for generation in range(max(len(group) for group in prefixes) - 1):
            rows = [
                group.iloc[generation]
                for group in prefixes
                if len(group) > generation + 1
            ]
            birth = np.asarray([float(row.length_birth) for row in rows])
            alpha = np.asarray([float(row.alpha) for row in rows])
            time_values = np.asarray([float(row.time) for row in rows])
            growth = alpha * time_values
            log_birth = np.log(birth)
            event_log_added = log_birth + np.log(
                np.expm1(np.minimum(growth, 700.0))
            )
            integration_time = 0.5 * growth[:, None] * (nodes + 1.0)
            integration_log_added = log_birth[:, None] + np.log(
                np.expm1(np.minimum(integration_time, 700.0))
            )
            self.groups.append(
                {
                    "generation": generation,
                    "n": len(rows),
                    "sum_log_alpha": float(np.log(alpha).sum()),
                    "event_log_added": event_log_added,
                    "integration_log_added": integration_log_added,
                    "integration_scale": 0.5 * growth,
                    "observed_mean_growth": float(growth.mean()),
                }
            )
        self.hermite_nodes, hermite_weights = np.polynomial.hermite.hermgauss(
            random_effect_order
        )
        self.log_hermite_weights = np.log(hermite_weights) - 0.5 * math.log(
            math.pi
        )
        self.n_observations = int(sum(group["n"] for group in self.groups))

    def cycle_log_likelihood(self, group, log_u, log_omega2, eta):
        log_u = np.atleast_1d(np.asarray(log_u, dtype=float))
        event_odds = eta * (
            group["event_log_added"][:, None] - log_u[None, :]
        )
        integration_odds = eta * (
            group["integration_log_added"][:, :, None]
            - log_u[None, None, :]
        )
        integral = group["integration_scale"][:, None] * np.sum(
            self.integration_weights[None, :, None] * expit(integration_odds),
            axis=1,
        )
        return (
            group["sum_log_alpha"]
            + group["n"] * log_omega2
            + np.sum(-np.logaddexp(0.0, -event_odds), axis=0)
            - math.exp(log_omega2) * np.sum(integral, axis=0)
        )

    def shared_log_likelihood(self, theta):
        log_u, log_omega2, log_eta = map(float, theta)
        eta = math.exp(log_eta)
        return float(
            sum(
                self.cycle_log_likelihood(group, [log_u], log_omega2, eta)[0]
                for group in self.groups
            )
        )

    def marginal_log_likelihood(self, theta):
        mean_log_u, log_omega2, log_eta, log_sigma_u = map(float, theta)
        eta, sigma_u = math.exp(log_eta), math.exp(log_sigma_u)
        nodes = mean_log_u + math.sqrt(2.0) * sigma_u * self.hermite_nodes
        value = 0.0
        for group in self.groups:
            conditional = self.cycle_log_likelihood(
                group, nodes, log_omega2, eta
            )
            value += float(logsumexp(conditional + self.log_hermite_weights))
        return value

    def posterior(self, theta):
        mean_log_u, log_omega2, log_eta, log_sigma_u = map(float, theta)
        eta, sigma_u = math.exp(log_eta), math.exp(log_sigma_u)
        nodes = mean_log_u + math.sqrt(2.0) * sigma_u * self.hermite_nodes
        probabilities = []
        for group in self.groups:
            log_weight = self.cycle_log_likelihood(
                group, nodes, log_omega2, eta
            ) + self.log_hermite_weights
            log_weight -= logsumexp(log_weight)
            probabilities.append(np.exp(log_weight))
        return nodes, np.asarray(probabilities)


def weighted_quantile(values, probabilities, quantile):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    probabilities = np.asarray(probabilities)[order]
    return float(values[np.searchsorted(np.cumsum(probabilities), quantile)])


def fit_cycle_threshold(prefixes, source_hazard):
    likelihood = CycleThresholdLikelihood(prefixes)
    shared_bounds = [
        (math.log(0.05), math.log(10.0)),
        (math.log(1.0), math.log(25.0)),
        (math.log(0.2), math.log(30.0)),
    ]
    shared_starts = [
        np.log(
            [source_hazard["u"], source_hazard["omega2"], source_hazard["eta"]]
        ),
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
        for start in shared_starts
    ]
    shared_fit = min(shared_fits, key=lambda item: item.fun)

    hierarchical_bounds = shared_bounds + [(math.log(0.003), math.log(1.0))]
    hierarchical_starts = [
        np.r_[shared_fit.x, math.log(value)] for value in [0.03, 0.10, 0.25, 0.5]
    ]
    hierarchical_fits = [
        minimize(
            lambda theta: -likelihood.marginal_log_likelihood(theta),
            start,
            method="L-BFGS-B",
            bounds=hierarchical_bounds,
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        for start in hierarchical_starts
    ]
    fitted = min(hierarchical_fits, key=lambda item: item.fun)
    nodes, probabilities = likelihood.posterior(fitted.x)
    posterior_mean_log_u = probabilities @ nodes
    posterior_sd_log_u = np.sqrt(
        probabilities @ (nodes**2) - posterior_mean_log_u**2
    )
    posterior_mean_u = probabilities @ np.exp(nodes)

    mean_log_u, log_omega2, log_eta, log_sigma_u = map(float, fitted.x)
    fit = {
        "mean_u": math.exp(mean_log_u),
        "mean_log_u": mean_log_u,
        "omega2": math.exp(log_omega2),
        "eta": math.exp(log_eta),
        "sigma_log_u": math.exp(log_sigma_u),
        "log_marginal_likelihood": -float(fitted.fun),
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "posterior_nodes_log_u": nodes,
        "posterior_probabilities": probabilities,
        "posterior_mean_log_u": posterior_mean_log_u,
        "posterior_sd_log_u": posterior_sd_log_u,
        "posterior_mean_u": posterior_mean_u,
    }
    shared = {
        "u": math.exp(float(shared_fit.x[0])),
        "omega2": math.exp(float(shared_fit.x[1])),
        "eta": math.exp(float(shared_fit.x[2])),
        "log_likelihood": -float(shared_fit.fun),
        "success": bool(shared_fit.success),
        "message": str(shared_fit.message),
    }
    comparison = {
        "n_transitions": likelihood.n_observations,
        "twice_log_likelihood_gain": 2.0
        * (fit["log_marginal_likelihood"] - shared["log_likelihood"]),
        "delta_bic_hierarchical_minus_shared": math.log(likelihood.n_observations)
        - 2.0 * (fit["log_marginal_likelihood"] - shared["log_likelihood"]),
    }
    return likelihood, fit, shared, comparison


def draw_cycle_log_u(fit, replicates, rng):
    probabilities = fit["posterior_probabilities"]
    nodes = fit["posterior_nodes_log_u"]
    draws = np.empty((replicates, len(probabilities)))
    for generation, weights in enumerate(probabilities):
        draws[:, generation] = rng.choice(nodes, size=replicates, p=weights)
    return draws


def threshold_table(likelihood, fit):
    rows = []
    nodes = fit["posterior_nodes_log_u"]
    for group, probabilities, mean_log, sd_log, mean_u in zip(
        likelihood.groups,
        fit["posterior_probabilities"],
        fit["posterior_mean_log_u"],
        fit["posterior_sd_log_u"],
        fit["posterior_mean_u"],
    ):
        rows.append(
            {
                "generation": group["generation"],
                "active_lineages": group["n"],
                "observed_mean_growth": group["observed_mean_growth"],
                "posterior_mean_log_u": float(mean_log),
                "posterior_sd_log_u": float(sd_log),
                "posterior_mean_u": float(mean_u),
                "posterior_median_u": math.exp(
                    weighted_quantile(nodes, probabilities, 0.5)
                ),
                "posterior_lo95_u": math.exp(
                    weighted_quantile(nodes, probabilities, 0.025)
                ),
                "posterior_hi95_u": math.exp(
                    weighted_quantile(nodes, probabilities, 0.975)
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--seed", type=int, default=88231)
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

    threshold_likelihood, threshold_fit, shared_hazard, comparison = fit_cycle_threshold(
        prefixes, source["fit"]["hazard"]
    )
    thresholds = threshold_table(threshold_likelihood, threshold_fit)
    thresholds.to_csv(OUT / "cycle_threshold_posterior.csv", index=False)
    rng = np.random.default_rng(args.seed)
    cycle_log_u = draw_cycle_log_u(threshold_fit, args.replicates, rng)

    common = {
        "initial_birth": source["fit"]["initial_birth"],
        "alpha": source["fit"]["alpha"],
        "q": qfit,
    }
    cycle_hazard = {
        "u": threshold_fit["mean_u"],
        "omega2": threshold_fit["omega2"],
        "eta": threshold_fit["eta"],
    }
    initial = np.asarray([float(group.length_birth.iloc[0]) for group in prefixes])
    generated = {
        "shared_u": simulate(
            {**common, "hazard": shared_hazard},
            lengths,
            q_posterior,
            q_lengths,
            args.replicates,
            args.seed + 1,
            conditional=True,
            observed_initial=initial,
        ),
        "cycle_u": simulate(
            {**common, "hazard": cycle_hazard},
            lengths,
            q_posterior,
            q_lengths,
            args.replicates,
            args.seed + 2,
            conditional=True,
            observed_initial=initial,
            cycle_log_u_draws=cycle_log_u,
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
        target, mean, low, high = time_acf(
            prefixes, simulation, q_lengths, args.max_lag
        )
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
                np.sqrt(
                    np.mean(
                        (table.observed.to_numpy()[fit_slice] - table[f"{label}_mean"].to_numpy()[fit_slice]) ** 2
                    )
                )
            )
        lookup = marginal.query("model == @label").set_index("variable")
        for variable in ["birth", "q", "alpha", "time"]:
            metrics[label][f"{variable}_ks"] = float(
                lookup.loc[variable, "ks_distance"]
            )

    colors = {"shared_u": "#718096", "cycle_u": "#c4512c"}
    labels = {"shared_u": "Shared u", "cycle_u": r"Inferred $u_i$"}
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.1), constrained_layout=True)
    axis = axes[0, 0]
    axis.fill_between(thresholds.generation, thresholds.posterior_lo95_u, thresholds.posterior_hi95_u, color="#c4512c", alpha=0.16)
    axis.plot(thresholds.generation, thresholds.posterior_mean_u, "s-", color="#c4512c", ms=3)
    axis.axhline(threshold_fit["mean_u"], color="0.5", ls="--", lw=1)
    axis.set(title=r"Cycle-threshold posterior $u_i$", xlabel="Generation", ylabel=r"$u_i$")

    for axis, variable, title, ylabel in [
        (axes[0, 1], "birth", "Balanced-cohort mean birth size", "Birth size"),
        (axes[0, 2], "growth", r"Mean growth increment $\alpha\tau$", r"$\alpha\tau$"),
        (axes[1, 0], "time", r"Mean division time $\tau$", r"$\tau$"),
    ]:
        for label in generated:
            values = trajectories.query("model == @label and variable == @variable")
            axis.fill_between(values.generation, values.simulated_lo95, values.simulated_hi95, color=colors[label], alpha=0.09)
            axis.plot(values.generation, values.simulated_mean, "--" if label == "shared_u" else "-", color=colors[label], label=labels[label])
        observed = trajectories.query("model == 'shared_u' and variable == @variable")
        axis.plot(observed.generation, observed.observed, "o-", color="#23364d", ms=3, label="Observed")
        axis.set(title=title, xlabel="Generation", ylabel=ylabel)
    axes[0, 1].legend(frameon=False, fontsize=8)

    for axis, table, title in [
        (axes[1, 1], birth_acf, "Birth-size ACF"),
        (axes[1, 2], tau_acf, "Division-time ACF"),
    ]:
        axis.plot(table.lag, table.observed, "o-", color="#23364d", ms=3)
        for label in generated:
            axis.fill_between(table.lag, table[f"{label}_lo95"], table[f"{label}_hi95"], color=colors[label], alpha=0.09)
            axis.plot(table.lag, table[f"{label}_mean"], "--" if label == "shared_u" else "-", color=colors[label])
        axis.set(title=title, xlabel="Lag", ylabel="Correlation")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    fig.suptitle(r"Glu: inferred $E_i$ in q and cycle-specific hazard threshold $u_i$")
    fig.savefig(OUT / "cycle_u_diagnostics.png", dpi=190)
    fig.savefig(OUT / "cycle_u_diagnostics.pdf")
    plt.close(fig)

    threshold_summary = {
        key: threshold_fit[key]
        for key in [
            "mean_u",
            "mean_log_u",
            "omega2",
            "eta",
            "sigma_log_u",
            "log_marginal_likelihood",
            "success",
            "message",
        ]
    }
    result = {
        "model": {
            "q": "q_li=A_l+E_i+r_li; every A_l and E_i inferred",
            "alpha": "stationary Gamma marginal with Gaussian-copula AR(1); no G_i process",
            "hazard": "h_i(t|b,alpha)=alpha*omega2*sigmoid(eta*log(b*(exp(alpha*t)-1)/u_i))",
            "threshold_prior": "log(u_i)~Normal(mu_u,sigma_u^2); every u_i integrated for hyperparameter fitting and inferred individually afterward",
            "explicit_static_birth_term": "absent (kappa=0)",
        },
        "fit": {
            "cycle_threshold_hazard": serializable(threshold_summary),
            "shared_threshold_hazard": serializable(shared_hazard),
            "likelihood_comparison": serializable(comparison),
            "q_log_joint_density": float(q_density),
            "q_optimizer_success": bool(q_optimizer.success),
        },
        "metrics": serializable(metrics),
        "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Glu q environment plus cycle-specific hazard threshold",
        "",
        r"The q model remains $q_{\ell i}=A_\ell+E_i+r_{\ell i}$, with every",
        r"$A_\ell$ and $E_i$ inferred. There is no alpha $G_i$ process and no",
        r"static explicit-birth coefficient $\kappa$. Instead,",
        "",
        r"$$h_i(t\mid b,\alpha)=\alpha\omega_2\,\operatorname{sigmoid}\!\left[",
        r"\eta\log\frac{b(e^{\alpha t}-1)}{u_i}\right],$$",
        "",
        r"$$\log u_i\sim\mathcal N(\mu_u,\sigma_u^2).$$",
        "",
        r"All $u_i$ are integrated out by Gauss-Hermite quadrature when fitting",
        r"$(\mu_u,\sigma_u,\omega_2,\eta)$, then each posterior $p(u_i\mid\mathrm{data})$",
        "is recovered and propagated through simulation.",
        "",
        "## Fit",
        "",
        f"- Population geometric-mean threshold: {threshold_fit['mean_u']:.6f}",
        f"- Between-cycle SD of log u: {threshold_fit['sigma_log_u']:.6f}",
        f"- omega2: {threshold_fit['omega2']:.6f}",
        f"- eta: {threshold_fit['eta']:.6f}",
        f"- Twice marginal log-likelihood gain: {comparison['twice_log_likelihood_gain']:.3f}",
        f"- Delta BIC (cycle-u minus shared-u): {comparison['delta_bic_hierarchical_minus_shared']:.3f}",
        "",
        "## Posterior checks",
        "",
        "| Metric | Shared u | Inferred u_i |",
        "|---|---:|---:|",
    ]
    for key, title in [
        ("birth_trajectory_rmse", "Mean-b trajectory RMSE"),
        ("growth_trajectory_rmse", "Alpha-tau trajectory RMSE"),
        ("time_trajectory_rmse", "Tau trajectory RMSE"),
        ("birth_acf_rmse", "Birth-ACF RMSE"),
        ("tau_acf_rmse", "Tau-ACF RMSE"),
        ("birth_ks", "Birth KS"),
        ("time_ks", "Tau KS"),
    ]:
        lines.append(
            f"| {title} | {metrics['shared_u'][key]:.5f} | {metrics['cycle_u'][key]:.5f} |"
        )
    birth_drift_gain = 100.0 * (
        1.0
        - metrics["cycle_u"]["birth_trajectory_rmse"]
        / metrics["shared_u"]["birth_trajectory_rmse"]
    )
    birth_acf_gain = 100.0 * (
        1.0
        - metrics["cycle_u"]["birth_acf_rmse"]
        / metrics["shared_u"]["birth_acf_rmse"]
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The fitted log-threshold SD corresponds to about {100.0 * threshold_fit['sigma_log_u']:.1f}% ",
            "multiplicative cycle-to-cycle variation. Relative to the refitted shared-u",
            f"comparator, inferred u_i reduce the mean-birth trajectory RMSE by {birth_drift_gain:.1f}%",
            f"and the birth-ACF RMSE by {birth_acf_gain:.1f}%. The birth marginal is essentially",
            "unchanged, and the division-time marginal does not improve.",
            "",
            "This is an in-sample empirical-Bayes diagnostic: the same division-time data",
            "identify each u_i and assess its posterior replay. A held-out-cycle or future-cycle",
            "check would be required to establish genuine predictive improvement.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
