#!/usr/bin/env python3
"""Fit Glu with lineage-specific u, omega2, and eta in the time hazard.

The lineage log-parameters follow a full-covariance Gaussian population law.
Inference uses a scalable Laplace empirical-Bayes approximation: each lineage
likelihood is approximated locally as Gaussian, and the population mean and
covariance are fitted by marginal maximum likelihood with the lineage
uncertainties retained.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from alpha_scaled_hill_time_hazard import (  # noqa: E402
    EPS, log_time_density, sample_growth_time, sample_lower_truncated_residual,
)
from run_loglogistic_endpoint_residual import load_endpoint_data, mean_acf  # noqa: E402


MODEL_NAME = "hierarchical_alpha_scaled_time_hazard_all_parameters"
NAMES = ["log_u", "log_omega2", "log_eta"]
PHYSICAL_NAMES = ["u", "omega2", "eta"]
LOG_BOUNDS = np.log(np.asarray([[1e-4, 20.0], [0.1, 30.0], [1e-3, 20.0]]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-lineage-observations", type=int, default=5)
    parser.add_argument("--acf-replicates", type=int, default=300)
    parser.add_argument("--acf-max-lag", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4271)
    return parser.parse_args()


def lineage_nll(theta: np.ndarray, group: pd.DataFrame) -> float:
    values = log_time_density(
        group.time.to_numpy(float), group.length_birth.to_numpy(float),
        group.alpha.to_numpy(float), *np.exp(theta),
    )
    return float(-np.sum(values)) if np.all(np.isfinite(values)) else 1e100


def numerical_hessian(function, point: np.ndarray, step: float = 2e-3) -> np.ndarray:
    n = len(point)
    result = np.empty((n, n))
    center = function(point)
    for i in range(n):
        direction = np.zeros(n); direction[i] = step
        result[i, i] = (function(point + direction) - 2 * center + function(point - direction)) / step**2
        for j in range(i):
            second = np.zeros(n); second[j] = step
            value = (
                function(point + direction + second)
                - function(point + direction - second)
                - function(point - direction + second)
                + function(point - direction - second)
            ) / (4 * step**2)
            result[i, j] = result[j, i] = value
    return result


def positive_covariance(hessian: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    values = np.clip(values, 0.25, 1e5)
    covariance = (vectors * (1 / values)) @ vectors.T
    eig, vec = np.linalg.eigh(covariance)
    return (vec * np.clip(eig, 1e-5, 2.0)) @ vec.T


def unpack_population(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values[:3]
    chol = np.zeros((3, 3))
    chol[0, 0], chol[1, 1], chol[2, 2] = np.exp(values[3:6])
    chol[1, 0], chol[2, 0], chol[2, 1] = values[6:9]
    return mean, chol @ chol.T


def population_nll(values: np.ndarray, estimates: np.ndarray, covariances: np.ndarray) -> float:
    mean, population_covariance = unpack_population(values)
    total = 0.0
    for estimate, measurement_covariance in zip(estimates, covariances):
        covariance = population_covariance + measurement_covariance
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return 1e100
        difference = estimate - mean
        total += 0.5 * (logdet + difference @ np.linalg.solve(covariance, difference))
    return float(total)


def conditional_gaussian(
    estimate: np.ndarray, measurement_covariance: np.ndarray,
    mean: np.ndarray, population_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gain = population_covariance @ np.linalg.inv(population_covariance + measurement_covariance)
    posterior_mean = mean + gain @ (estimate - mean)
    posterior_covariance = population_covariance - gain @ population_covariance
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    return posterior_mean, posterior_covariance


def simulate_replay(
    groups: list[pd.DataFrame], effects: np.ndarray, residual: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    mu_epsilon, sigma_epsilon = map(float, residual)
    rows = []
    for lineage_index, (group, physical) in enumerate(zip(groups, np.exp(effects))):
        u, omega2, eta = map(float, physical)
        birth = float(group.length_birth.iloc[0])
        for generation in range(len(group)):
            alpha = float(group.alpha.iloc[generation])
            growth_time = sample_growth_time(birth, u, omega2, eta, rng)
            division_time = growth_time / alpha
            epsilon = sample_lower_truncated_residual(
                -growth_time, mu_epsilon, sigma_epsilon, rng
            )
            final = birth * math.exp(min(growth_time + epsilon, 700))
            rows.append({"lineage": lineage_index, "generation": generation,
                         "time": division_time, "length_birth": birth})
            if generation + 1 < len(group):
                birth = max(float(group.f.iloc[generation + 1]) * final, EPS)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = PROJECT_DIR / "outputs" / "model_comparison" / "Glu" / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    real, _, _, _ = load_endpoint_data("Glu")
    groups = [g.sort_values("generation").reset_index(drop=True)
              for _, g in real.groupby("lineage", sort=False)]
    labels = np.asarray([g.lineage.iloc[0] for g in groups])
    global_summary = pd.read_csv(
        PROJECT_DIR / "outputs" / "model_comparison" / "Glu"
        / "alpha_scaled_hill_time_hazard" / "posterior_summary.csv"
    ).set_index("parameter")
    global_center = np.log(global_summary.loc[PHYSICAL_NAMES, "median"].to_numpy(float))

    estimates, covariance_rows, fit_rows, included = [], [], [], []
    for index, group in enumerate(groups):
        if len(group) < args.min_lineage_observations:
            continue
        result = minimize(
            lineage_nll, global_center, args=(group,), method="L-BFGS-B",
            bounds=[tuple(row) for row in LOG_BOUNDS],
            options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-7},
        )
        theta = np.asarray(result.x)
        hessian = numerical_hessian(lambda x: lineage_nll(x, group), theta)
        covariance = positive_covariance(hessian)
        estimates.append(theta); covariance_rows.append(covariance); included.append(index)
        fit_rows.append({"lineage": labels[index], "n_observations": len(group),
                         "success": bool(result.success), "log_likelihood": -float(result.fun),
                         **dict(zip(PHYSICAL_NAMES, np.exp(theta)))})
    estimates = np.asarray(estimates); covariance_rows = np.asarray(covariance_rows)

    empirical_covariance = np.cov(estimates.T) - np.mean(covariance_rows, axis=0)
    eig, vec = np.linalg.eigh(0.5 * (empirical_covariance + empirical_covariance.T))
    empirical_covariance = (vec * np.clip(eig, 2.5e-4, None)) @ vec.T
    initial_chol = np.linalg.cholesky(empirical_covariance)
    initial = np.r_[np.mean(estimates, axis=0), np.log(np.diag(initial_chol)),
                    initial_chol[1, 0], initial_chol[2, 0], initial_chol[2, 1]]
    population_fit = minimize(
        population_nll, initial, args=(estimates, covariance_rows), method="L-BFGS-B",
        bounds=[tuple(row) for row in LOG_BOUNDS] + [(-7, 1)] * 3 + [(-2, 2)] * 3,
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7},
    )
    population_mean, population_covariance = unpack_population(population_fit.x)
    population_sd = np.sqrt(np.diag(population_covariance))
    population_corr = population_covariance / np.outer(population_sd, population_sd)

    posterior_means = np.tile(population_mean, (len(groups), 1))
    posterior_covariances = np.tile(population_covariance, (len(groups), 1, 1))
    for estimate, covariance, index in zip(estimates, covariance_rows, included):
        posterior_means[index], posterior_covariances[index] = conditional_gaussian(
            estimate, covariance, population_mean, population_covariance
        )
    lineage_table = pd.DataFrame({"lineage": labels, "n_observations": [len(g) for g in groups]})
    for parameter_index, name in enumerate(PHYSICAL_NAMES):
        variance = posterior_covariances[:, parameter_index, parameter_index]
        lineage_table[f"{name}_median"] = np.exp(posterior_means[:, parameter_index])
        lineage_table[f"{name}_q2.5"] = np.exp(posterior_means[:, parameter_index] - 1.96 * np.sqrt(variance))
        lineage_table[f"{name}_q97.5"] = np.exp(posterior_means[:, parameter_index] + 1.96 * np.sqrt(variance))
    lineage_table.to_csv(output_dir / "lineage_parameter_posterior.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "lineage_likelihood_fits.csv", index=False)

    residual_chain = pd.read_csv(
        PROJECT_DIR / "outputs" / "model_comparison" / "Glu"
        / "alpha_scaled_hill_time_hazard" / "residual_flat_chain.csv"
    ).to_numpy(float)
    rng = np.random.default_rng(args.seed)
    max_lag = min(args.acf_max_lag, max(len(g) for g in groups) - 1)
    predictive = {name: np.full((args.acf_replicates, max_lag + 1), np.nan)
                  for name in ["time", "length_birth"]}
    for replicate in range(args.acf_replicates):
        effects = np.vstack([
            rng.multivariate_normal(mean, covariance)
            for mean, covariance in zip(posterior_means, posterior_covariances)
        ])
        effects = np.clip(effects, LOG_BOUNDS[:, 0], LOG_BOUNDS[:, 1])
        simulated = simulate_replay(
            groups, effects, residual_chain[rng.integers(len(residual_chain))], rng
        )
        for name in predictive:
            predictive[name][replicate] = mean_acf(simulated, name, max_lag)

    lags = np.arange(max_lag + 1)
    output = {"lag": lags}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    acf_metrics = {}
    for axis, name, title in zip(axes, predictive, ["Division-time ACF", "Birth-size ACF"]):
        observed = mean_acf(real, name, max_lag)
        median = np.nanmedian(predictive[name], axis=0)
        lower, upper = np.nanpercentile(predictive[name], [2.5, 97.5], axis=0)
        output[f"observed_{name}"] = observed
        output[f"predictive_median_{name}"] = median
        output[f"predictive_q2.5_{name}"] = lower
        output[f"predictive_q97.5_{name}"] = upper
        axis.fill_between(lags, lower, upper, color="tab:orange", alpha=.22, label="95% band")
        axis.plot(lags, median, color="tab:orange", lw=2, label="Hierarchical model")
        axis.plot(lags, observed, color="black", lw=2, label="Glu data")
        axis.axhline(0, color="0.6", lw=.8); axis.set(title=title, xlabel="Lag (generations)", ylabel="Mean lineage ACF")
        acf_metrics[name] = {
            "mae_lags_1_to_30": float(np.nanmean(np.abs(median[1:] - observed[1:]))),
            "covered_lags": int(np.sum((observed[1:] >= lower[1:]) & (observed[1:] <= upper[1:]))),
            "lag1_observed": float(observed[1]), "lag1_median": float(median[1]),
        }
    axes[0].legend(frameon=False); fig.tight_layout()
    fig.savefig(output_dir / "posterior_replay_acf.png", dpi=200)
    fig.savefig(output_dir / "posterior_replay_acf.pdf"); plt.close(fig)
    pd.DataFrame(output).to_csv(output_dir / "posterior_replay_acf.csv", index=False)

    physical_population_median = np.exp(population_mean)
    physical_population_cv = np.sqrt(np.exp(population_sd**2) - 1)
    diagnostics = {
        "model": MODEL_NAME, "sugar": "Glu",
        "hazard": "h(t|b,alpha)=alpha*omega2_l*(p(t)/u_l)^eta_l/(1+(p(t)/u_l)^eta_l)",
        "hierarchy": "(log u_l, log omega2_l, log eta_l) jointly Gaussian with full covariance",
        "inference": "Laplace empirical Bayes using lineage likelihood estimates and Hessians",
        "minimum_observations_for_lineage_likelihood": args.min_lineage_observations,
        "n_lineages": len(groups), "n_lineages_in_population_fit": len(estimates),
        "population_optimizer_success": bool(population_fit.success),
        "population_optimizer_message": str(population_fit.message),
        "population_median": dict(zip(PHYSICAL_NAMES, map(float, physical_population_median))),
        "population_log_sd": dict(zip(PHYSICAL_NAMES, map(float, population_sd))),
        "population_cv": dict(zip(PHYSICAL_NAMES, map(float, physical_population_cv))),
        "population_log_correlation": {
            row: {column: float(population_corr[i, j]) for j, column in enumerate(PHYSICAL_NAMES)}
            for i, row in enumerate(PHYSICAL_NAMES)
        },
        "acf_metrics": acf_metrics, "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60,
    }
    with (output_dir / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
