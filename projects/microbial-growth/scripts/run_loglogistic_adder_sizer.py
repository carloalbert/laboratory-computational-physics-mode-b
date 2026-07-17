#!/usr/bin/env python3
"""Fit the normalized three-parameter log-logistic adder-sizer model."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import emcee
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from loglogistic_adder_sizer import (  # noqa: E402
    log_division_time_density,
    log_likelihood,
    simulate_lineages,
    survival,
)
from run_weibull_adder_sizer import (  # noqa: E402
    PARAMETER_NAMES,
    average_survival_original,
    distribution_metrics,
    load_data,
    make_bounds,
    make_plots,
    mean_acf,
    posterior_summary,
    survival_metrics,
    valid_prior,
)
from weibull_adder_sizer import (  # noqa: E402
    EPS,
    added_size,
    beta_ab,
    gamma_shape_scale,
    log_beta_pdf_mean_sd,
    log_gamma_pdf_mean_sd,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sugar", default="Ace")
    parser.add_argument("--walkers", type=int, default=40)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim-lineages", type=int, default=40)
    parser.add_argument("--sim-generations", type=int, default=80)
    parser.add_argument("--acf-max-lag", type=int, default=30)
    parser.add_argument("--predictive-replicates", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def fit_mle(data, bounds, seed):
    hazard_bounds = [bounds[name] for name in ["u", "omega2", "eta"]]

    def hazard_objective(x):
        values = log_division_time_density(
            data["time"], data["length_birth"], data["alpha"], *x
        )
        return -float(np.sum(values)) if np.all(np.isfinite(values)) else 1e100

    hazard_result = differential_evolution(
        hazard_objective,
        hazard_bounds,
        seed=seed,
        popsize=15,
        maxiter=300,
        tol=1e-9,
        polish=True,
        workers=1,
        updating="immediate",
    )

    def alpha_objective(x):
        values = log_gamma_pdf_mean_sd(data["alpha"], *x)
        return -float(np.sum(values)) if np.all(np.isfinite(values)) else 1e100

    alpha_result = minimize(
        alpha_objective,
        [np.mean(data["alpha"]), np.std(data["alpha"], ddof=1)],
        method="L-BFGS-B",
        bounds=[bounds["mu_alpha"], bounds["sigma_alpha"]],
    )

    def fraction_objective(x):
        values = log_beta_pdf_mean_sd(data["f"], *x)
        return -float(np.sum(values)) if np.all(np.isfinite(values)) else 1e100

    fraction_result = minimize(
        fraction_objective,
        [np.mean(data["f"]), np.std(data["f"], ddof=1)],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-12, "fatol": 1e-9},
    )

    params = np.concatenate([hazard_result.x, alpha_result.x, fraction_result.x])
    if not valid_prior(params, bounds):
        raise RuntimeError(f"Invalid MLE parameters: {params}")
    details = {
        "hazard_success": bool(hazard_result.success),
        "hazard_message": str(hazard_result.message),
        "alpha_success": bool(alpha_result.success),
        "fraction_success": bool(fraction_result.success),
        "log_likelihood": float(log_likelihood(params, data)),
    }
    return params, details


def fit_original_hazard_mle(data, bounds, seed):
    """Re-optimize the original Burr hazard for a fair same-optimizer comparison."""
    p = added_size(data["time"], data["length_birth"], data["alpha"])
    log_p = np.log(np.maximum(p, EPS))
    log_dpdt = (
        np.log(data["length_birth"])
        + np.log(data["alpha"])
        + data["alpha"] * data["time"]
    )

    def objective(params):
        u, v, omega2 = params
        log_x = log_p - math.log(u)
        log_denominator = np.logaddexp(0.0, v * log_x)
        values = (
            math.log(omega2)
            + math.log(v)
            - math.log(u)
            + (v - 1.0) * log_x
            - log_denominator
            + log_dpdt
            - omega2 * log_denominator
        )
        return -float(np.sum(values)) if np.all(np.isfinite(values)) else 1e100

    result = differential_evolution(
        objective,
        [bounds["u"], (0.05, 30.0), (1e-8, 50.0)],
        seed=seed,
        popsize=15,
        maxiter=500,
        tol=1e-10,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return result.x, -float(result.fun), bool(result.success), str(result.message)


def initialize_walkers(center, n_walkers, bounds, data, seed):
    rng = np.random.default_rng(seed)
    scales = np.maximum(np.abs(center) * 0.025, 1e-4)
    walkers = []
    for _ in range(200_000):
        proposal = center + rng.normal(0.0, scales)
        if valid_prior(proposal, bounds) and np.isfinite(log_likelihood(proposal, data)):
            walkers.append(proposal)
            if len(walkers) == n_walkers:
                return np.asarray(walkers)
    raise RuntimeError("Unable to initialize all walkers")


def average_survival(t_grid, data, params):
    u, omega2, eta = params[:3]
    pieces = []
    for start in range(0, len(t_grid), 128):
        time_grid = t_grid[start : start + 128, None]
        birth = data["length_birth"][None, :]
        alpha = data["alpha"][None, :]
        p = birth * np.expm1(np.clip(alpha * time_grid, 0.0, 700.0))
        pieces.append(np.mean(survival(p, birth, u, omega2, eta), axis=1))
    return np.concatenate(pieces)


def simulate_original(params, initial_lengths, sim_lineages, sim_generations, seed):
    """Simulate the original Burr model using its analytic inverse survival."""
    u, v, omega2, mu_alpha, sigma_alpha, mu_f, sigma_f = map(float, params)
    gamma_shape, gamma_scale = gamma_shape_scale(mu_alpha, sigma_alpha)
    beta_a, beta_b = beta_ab(mu_f, sigma_f)
    rng = np.random.default_rng(seed)
    rows = []
    for lineage in range(sim_lineages):
        birth = float(rng.choice(initial_lengths))
        for generation in range(sim_generations):
            alpha = float(rng.gamma(gamma_shape, gamma_scale))
            fraction = float(rng.beta(beta_a, beta_b))
            uniform = max(float(rng.uniform()), np.finfo(float).tiny)
            p = u * (uniform ** (-1.0 / omega2) - 1.0) ** (1.0 / v)
            final = birth + p
            rows.append(
                {
                    "lineage": lineage,
                    "generation": generation,
                    "time": math.log1p(p / birth) / alpha,
                    "alpha": alpha,
                    "f": fraction,
                    "length_birth": birth,
                    "length_final": final,
                    "added_size": p,
                }
            )
            birth = max(fraction * final, np.finfo(float).tiny)
    return pd.DataFrame(rows)


def run_replicated_predictive_comparison(
    real,
    old_params,
    new_params,
    initial_lengths,
    args,
    output_dir,
):
    """Quantify Monte Carlo variability in posterior-median predictive checks."""
    rows = []
    for replicate in range(args.predictive_replicates):
        seed = args.seed + 1000 + replicate
        simulations = {
            "original_burr": simulate_original(
                old_params,
                initial_lengths,
                args.sim_lineages,
                args.sim_generations,
                seed,
            ),
            "loglogistic_adder_sizer": pd.DataFrame(
                simulate_lineages(
                    new_params,
                    initial_lengths,
                    args.sim_lineages,
                    args.sim_generations,
                    seed,
                )
            ),
        }
        for model, simulated in simulations.items():
            row = {"model": model, "replicate": replicate}
            for metric in distribution_metrics(real, simulated, model):
                variable = metric["variable"]
                row[f"{variable}_mean_ratio"] = metric["mean_ratio"]
                row[f"{variable}_sd_ratio"] = metric["sd_ratio"]
                row[f"{variable}_ks"] = metric["ks_statistic"]
            row["time_acf_lag1"] = mean_acf(simulated, "time", args.acf_max_lag)[1]
            row["length_birth_acf_lag1"] = mean_acf(
                simulated, "length_birth", args.acf_max_lag
            )[1]
            rows.append(row)

    detailed = pd.DataFrame(rows)
    detailed.to_csv(output_dir / "replicated_predictive_metrics.csv", index=False)
    summaries = []
    numeric = [column for column in detailed.columns if column not in {"model", "replicate"}]
    for model, sub in detailed.groupby("model"):
        for metric in numeric:
            values = sub[metric].to_numpy(float)
            summaries.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": np.mean(values),
                    "std": np.std(values, ddof=1),
                    "q2.5": np.percentile(values, 2.5),
                    "median": np.median(values),
                    "q97.5": np.percentile(values, 97.5),
                }
            )
    pd.DataFrame(summaries).to_csv(
        output_dir / "replicated_predictive_summary.csv", index=False
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        PROJECT_DIR / "outputs" / "model_comparison" / args.sugar / "loglogistic_adder_sizer"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    real, data, initial_lengths = load_data(args.sugar)
    bounds = make_bounds(data)
    started = time.time()
    mle_params, mle_details = fit_mle(data, bounds, args.seed)
    original_mle_params, original_hazard_ll, original_mle_success, original_mle_message = (
        fit_original_hazard_mle(data, bounds, args.seed)
    )
    new_hazard_ll = float(
        np.sum(
            log_division_time_density(
                data["time"], data["length_birth"], data["alpha"], *mle_params[:3]
            )
        )
    )
    shared_marginal_ll = mle_details["log_likelihood"] - new_hazard_ll
    original_reoptimized_full_ll = shared_marginal_ll + original_hazard_ll
    print("MLE parameters:")
    print(pd.Series(mle_params, index=PARAMETER_NAMES))
    print("MLE log likelihood:", mle_details["log_likelihood"])
    print("Re-optimized original MLE parameters:", original_mle_params)
    print("Re-optimized original full log likelihood:", original_reoptimized_full_ll)

    n_walkers = max(args.walkers, 2 * len(PARAMETER_NAMES) + 2)

    def log_probability(params):
        return log_likelihood(params, data) if valid_prior(params, bounds) else -np.inf

    p0 = initialize_walkers(mle_params, n_walkers, bounds, data, args.seed)
    moves = [
        (emcee.moves.StretchMove(), 0.5),
        (emcee.moves.DEMove(), 0.3),
        (emcee.moves.DESnookerMove(), 0.2),
    ]
    sampler = emcee.EnsembleSampler(n_walkers, len(PARAMETER_NAMES), log_probability, moves=moves)
    print(f"Burn-in: {args.burn} steps with {n_walkers} walkers")
    state = sampler.run_mcmc(p0, args.burn, progress=False)
    sampler.reset()
    print(f"Production: {args.steps} steps")
    sampler.run_mcmc(state, args.steps, progress=False)

    chain = sampler.get_chain()
    flat_chain = sampler.get_chain(thin=args.thin, flat=True)
    flat_logp = sampler.get_log_prob(thin=args.thin, flat=True)
    summary = posterior_summary(flat_chain, flat_logp)
    median_params = summary.set_index("parameter").loc[PARAMETER_NAMES, "median"].to_numpy(float)
    summary.to_csv(output_dir / "posterior_summary.csv", index=False)
    pd.DataFrame(flat_chain, columns=PARAMETER_NAMES).to_csv(output_dir / "flat_chain.csv", index=False)
    pd.DataFrame({"log_probability": flat_logp}).to_csv(output_dir / "flat_log_probability.csv", index=False)

    new_sim = pd.DataFrame(
        simulate_lineages(
            median_params,
            initial_lengths,
            args.sim_lineages,
            args.sim_generations,
            args.seed + 1,
        )
    )
    new_sim.to_csv(output_dir / "posterior_predictive_simulation.csv", index=False)

    old_dir = PROJECT_DIR / "outputs" / "by_sugar" / args.sugar
    old_sim = pd.read_csv(old_dir / "posterior_predictive_simulation.csv")
    old_summary = pd.read_csv(old_dir / "posterior_summary.csv").set_index("parameter")
    old_params = old_summary.loc[
        ["u", "v", "omega2", "mu_alpha", "sigma_alpha", "mu_f", "sigma_f"], "median"
    ].to_numpy(float)
    with (old_dir / "diagnostics.json").open() as handle:
        old_diagnostics = json.load(handle)

    metrics = distribution_metrics(real, old_sim, "original_burr")
    metrics += distribution_metrics(real, new_sim, "loglogistic_adder_sizer")
    pd.DataFrame(metrics).to_csv(output_dir / "comparison_metrics.csv", index=False)
    run_replicated_predictive_comparison(
        real,
        old_params,
        median_params,
        initial_lengths,
        args,
        output_dir,
    )

    times = np.sort(data["time"])
    empirical_survival = 1.0 - np.arange(1, len(times) + 1) / len(times)
    old_survival = average_survival_original(times, data, old_params)
    new_survival = average_survival(times, data, median_params)

    acf_comparison = {}
    for column in ["time", "length_birth"]:
        acf_comparison[column] = {
            "real_lag1": float(mean_acf(real, column, args.acf_max_lag)[1]),
            "original_lag1": float(mean_acf(old_sim, column, args.acf_max_lag)[1]),
            "loglogistic_lag1": float(mean_acf(new_sim, column, args.acf_max_lag)[1]),
        }

    try:
        tau_values = sampler.get_autocorr_time(tol=0)
        tau = {name: float(value) for name, value in zip(PARAMETER_NAMES, tau_values)}
    except Exception as exc:
        tau = {"error": str(exc)}

    comparison = {
        "sugar": args.sugar,
        "n_observations": len(real),
        "n_lineages": int(real.lineage.nunique()),
        "model": "loglogistic_mixed_adder_sizer",
        "parameter_names": PARAMETER_NAMES,
        "mle": mle_details,
        "original_reoptimized_mle": {
            "hazard_parameters": {
                "u": float(original_mle_params[0]),
                "v": float(original_mle_params[1]),
                "omega2": float(original_mle_params[2]),
            },
            "success": original_mle_success,
            "message": original_mle_message,
            "log_likelihood": float(original_reoptimized_full_ll),
        },
        "delta_mle_log_likelihood_new_minus_original": float(
            mle_details["log_likelihood"] - original_reoptimized_full_ll
        ),
        "max_sampled_log_probability": float(np.max(flat_logp)),
        "original_max_sampled_log_probability": float(old_diagnostics["max_log_probability"]),
        "delta_max_log_probability_new_minus_original": float(
            np.max(flat_logp) - old_diagnostics["max_log_probability"]
        ),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": tau,
        "runtime_minutes": (time.time() - started) / 60.0,
        "survival": {
            "original_burr": survival_metrics(times, old_survival),
            "loglogistic_adder_sizer": survival_metrics(times, new_survival),
        },
        "acf_lag1": acf_comparison,
        "run_config": {
            "walkers": n_walkers,
            "burn": args.burn,
            "steps": args.steps,
            "thin": args.thin,
            "seed": args.seed,
            "sim_lineages": args.sim_lineages,
            "sim_generations": args.sim_generations,
            "acf_max_lag": args.acf_max_lag,
            "predictive_replicates": args.predictive_replicates,
        },
        "prior_bounds": {name: list(value) for name, value in bounds.items()},
    }
    with (output_dir / "comparison_summary.json").open("w") as handle:
        json.dump(comparison, handle, indent=2)

    make_plots(
        output_dir,
        args.sugar,
        real,
        old_sim,
        new_sim,
        times,
        empirical_survival,
        old_survival,
        new_survival,
        chain,
        flat_chain,
        args.acf_max_lag,
        alternative_label="Log-logistic adder-sizer",
    )

    print("\nPosterior summary:")
    print(summary.to_string(index=False))
    print("\nComparison summary:")
    print(json.dumps(comparison, indent=2))
    print("\nSaved to", output_dir)


if __name__ == "__main__":
    main()
