#!/usr/bin/env python3
"""Fit and assess the three-parameter Weibull mixed adder-sizer model.

Example
-------
python projects/microbial-growth/scripts/run_weibull_adder_sizer.py --sugar Ace
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
from scipy.optimize import differential_evolution, minimize
from scipy.stats import ks_2samp

import emcee

try:
    import corner
except ImportError:
    corner = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from weibull_adder_sizer import (  # noqa: E402
    EPS,
    added_size,
    beta_ab,
    cumulative_hazard,
    gamma_shape_scale,
    log_beta_pdf_mean_sd,
    log_division_time_density,
    log_gamma_pdf_mean_sd,
    log_likelihood,
    simulate_lineages,
    survival,
)


PARAMETER_NAMES = [
    "u",
    "omega2",
    "eta",
    "mu_alpha",
    "sigma_alpha",
    "mu_f",
    "sigma_f",
]


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
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_data(sugar: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    path = PROJECT_DIR / "data" / "processed" / "processed_input_data.csv"
    df = pd.read_csv(path)
    required = ["source", "lineage", "generation", "time", "alpha", "f", "length_birth"]
    df = df[required].copy()
    for col in ["lineage", "generation", "time", "alpha", "f", "length_birth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df = df[(df.time > 0) & (df.alpha > 0) & (df.f > 0) & (df.f < 1) & (df.length_birth > 0)]
    df = df[df.source == sugar].sort_values(["lineage", "generation"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No data for sugar {sugar!r}")

    gen0 = df[df.generation == 0].groupby("lineage", sort=False).first()
    retained = gen0.index.to_numpy()
    df = df[df.lineage.isin(retained)].sort_values(["lineage", "generation"]).reset_index(drop=True)
    initial_lengths = gen0.loc[retained, "length_birth"].to_numpy(float)
    data = {name: df[name].to_numpy(float) for name in ["time", "alpha", "f", "length_birth"]}
    data["lineage"] = df.lineage.to_numpy()
    data["generation"] = df.generation.to_numpy(float)
    return df, data, initial_lengths


def make_bounds(data: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    p = added_size(data["time"], data["length_birth"], data["alpha"])
    p = p[np.isfinite(p) & (p > 0)]
    u_upper = max(20.0, 5.0 * float(np.percentile(p, 99)))
    alpha_upper = max(5.0, 3.0 * float(np.percentile(data["alpha"], 99)))
    alpha_sd_upper = max(5.0, 3.0 * float(np.std(data["alpha"], ddof=1)))
    return {
        "u": (1e-5, u_upper),
        "omega2": (0.05, 30.0),
        "eta": (1e-6, 1.0 - 1e-6),
        "mu_alpha": (1e-8, alpha_upper),
        "sigma_alpha": (1e-8, alpha_sd_upper),
        "mu_f": (1e-8, 1.0 - 1e-8),
        "sigma_f": (1e-8, 0.5),
    }


def valid_prior(params: np.ndarray, bounds: dict[str, tuple[float, float]]) -> bool:
    for value, name in zip(params, PARAMETER_NAMES):
        lo, hi = bounds[name]
        if not (lo < value < hi):
            return False
    if gamma_shape_scale(params[3], params[4])[0] is None:
        return False
    if beta_ab(params[5], params[6])[0] is None:
        return False
    return True


def fit_mle(
    data: dict[str, np.ndarray], bounds: dict[str, tuple[float, float]], seed: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Exploit likelihood factorization for robust low-dimensional optimization."""
    hazard_bounds = [bounds[name] for name in ["u", "omega2", "eta"]]

    def hazard_objective(x: np.ndarray) -> float:
        vals = log_division_time_density(
            data["time"], data["length_birth"], data["alpha"], x[0], x[1], x[2]
        )
        return -float(np.sum(vals)) if np.all(np.isfinite(vals)) else 1e100

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

    def alpha_objective(x: np.ndarray) -> float:
        vals = log_gamma_pdf_mean_sd(data["alpha"], x[0], x[1])
        return -float(np.sum(vals)) if np.all(np.isfinite(vals)) else 1e100

    alpha_start = np.array([np.mean(data["alpha"]), np.std(data["alpha"], ddof=1)])
    alpha_result = minimize(
        alpha_objective,
        alpha_start,
        method="L-BFGS-B",
        bounds=[bounds["mu_alpha"], bounds["sigma_alpha"]],
    )

    def fraction_objective(x: np.ndarray) -> float:
        vals = log_beta_pdf_mean_sd(data["f"], x[0], x[1])
        return -float(np.sum(vals)) if np.all(np.isfinite(vals)) else 1e100

    fraction_start = np.array([np.mean(data["f"]), np.std(data["f"], ddof=1)])
    fraction_result = minimize(
        fraction_objective,
        fraction_start,
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


def initialize_walkers(
    center: np.ndarray,
    n_walkers: int,
    bounds: dict[str, tuple[float, float]],
    data: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
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


def posterior_summary(flat_chain: np.ndarray, flat_logp: np.ndarray) -> pd.DataFrame:
    mode = flat_chain[int(np.argmax(flat_logp))]
    rows = []
    for idx, name in enumerate(PARAMETER_NAMES):
        vals = flat_chain[:, idx]
        rows.append(
            {
                "parameter": name,
                "mode": mode[idx],
                "mean": np.mean(vals),
                "median": np.median(vals),
                "ci_2.5": np.percentile(vals, 2.5),
                "ci_97.5": np.percentile(vals, 97.5),
                "std": np.std(vals, ddof=1),
            }
        )
    return pd.DataFrame(rows)


def acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = np.full(max_lag + 1, np.nan)
    if len(x) < 3:
        return out
    centered = x - np.mean(x)
    denominator = np.sum(centered**2)
    if denominator <= 0:
        return out
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag < len(centered):
            out[lag] = np.sum(centered[:-lag] * centered[lag:]) / denominator
    return out


def mean_acf(df: pd.DataFrame, column: str, max_lag: int) -> np.ndarray:
    values = []
    for _, sub in df.groupby("lineage"):
        sub = sub.sort_values("generation")
        if len(sub) >= 3:
            values.append(acf_1d(sub[column].to_numpy(float), max_lag))
    array = np.asarray(values)
    counts = np.sum(np.isfinite(array), axis=0)
    totals = np.nansum(array, axis=0)
    return np.divide(totals, counts, out=np.full(max_lag + 1, np.nan), where=counts > 0)


def distribution_metrics(real: pd.DataFrame, simulated: pd.DataFrame, model: str) -> list[dict[str, object]]:
    rows = []
    for column in ["time", "alpha", "f", "length_birth"]:
        x = real[column].to_numpy(float)
        y = simulated[column].to_numpy(float)
        rows.append(
            {
                "model": model,
                "variable": column,
                "real_mean": np.mean(x),
                "simulated_mean": np.mean(y),
                "mean_ratio": np.mean(y) / np.mean(x),
                "real_sd": np.std(x, ddof=1),
                "simulated_sd": np.std(y, ddof=1),
                "sd_ratio": np.std(y, ddof=1) / np.std(x, ddof=1),
                "ks_statistic": ks_2samp(x, y).statistic,
            }
        )
    return rows


def average_survival_new(t_grid: np.ndarray, data: dict[str, np.ndarray], params: np.ndarray) -> np.ndarray:
    u, omega2, eta = params[:3]
    curves = []
    for start in range(0, len(t_grid), 128):
        t = t_grid[start : start + 128, None]
        b = data["length_birth"][None, :]
        a = data["alpha"][None, :]
        p = b * np.expm1(np.clip(a * t, 0.0, 700.0))
        curves.append(np.mean(survival(p, b, u, omega2, eta), axis=1))
    return np.concatenate(curves)


def average_survival_original(t_grid: np.ndarray, data: dict[str, np.ndarray], params: np.ndarray) -> np.ndarray:
    u, v, omega2 = params[:3]
    curves = []
    for start in range(0, len(t_grid), 128):
        t = t_grid[start : start + 128, None]
        b = data["length_birth"][None, :]
        a = data["alpha"][None, :]
        p = b * np.expm1(np.clip(a * t, 0.0, 700.0))
        log_x = np.log(np.maximum(p, EPS)) - math.log(u)
        curves.append(np.mean(np.exp(-omega2 * np.logaddexp(0.0, v * log_x)), axis=1))
    return np.concatenate(curves)


def survival_metrics(times: np.ndarray, model_survival: np.ndarray) -> dict[str, float]:
    empirical = 1.0 - np.arange(1, len(times) + 1) / len(times)
    absolute = np.abs(model_survival - empirical)
    return {
        "mean_absolute_error": float(np.mean(absolute)),
        "max_absolute_error": float(np.max(absolute)),
        "normalized_integrated_absolute_error": float(np.trapezoid(absolute, times) / (times[-1] - times[0])),
    }


def make_plots(
    output_dir: Path,
    sugar: str,
    real: pd.DataFrame,
    old_sim: pd.DataFrame,
    new_sim: pd.DataFrame,
    times: np.ndarray,
    empirical_survival: np.ndarray,
    old_survival: np.ndarray,
    new_survival: np.ndarray,
    chain: np.ndarray,
    flat_chain: np.ndarray,
    max_lag: int,
    alternative_label: str = "Weibull adder-sizer",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(times, empirical_survival, where="post", color="black", label="Empirical")
    ax.plot(times, old_survival, linestyle="--", label="Original Burr")
    ax.plot(times, new_survival, linestyle="-.", label=alternative_label)
    ax.set(xlabel="Division time", ylabel="Survival probability", title=f"Survival comparison: {sugar}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "survival_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    specs = [("time", "Division time"), ("alpha", "Growth rate"), ("f", "Division ratio"), ("length_birth", "Birth length")]
    for ax, (column, title) in zip(axes.ravel(), specs):
        combined = np.concatenate([real[column], old_sim[column], new_sim[column]])
        bins = np.histogram_bin_edges(combined, bins=30)
        ax.hist(real[column], bins=bins, density=True, histtype="step", linewidth=2, color="black", label="Real")
        ax.hist(old_sim[column], bins=bins, density=True, histtype="step", linewidth=1.5, label="Original")
        ax.hist(new_sim[column], bins=bins, density=True, histtype="step", linewidth=1.5, label=alternative_label)
        ax.set_title(title)
        ax.set_ylabel("Density")
    axes[0, 0].legend()
    fig.suptitle(f"Posterior-predictive comparison: {sugar}")
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_predictive_comparison.png", dpi=180)
    plt.close(fig)

    lags = np.arange(max_lag + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, title in [
        (axes[0], "time", "Division-time ACF"),
        (axes[1], "length_birth", "Birth-length ACF"),
    ]:
        ax.plot(lags, mean_acf(real, column, max_lag), marker="o", label="Real")
        ax.plot(lags, mean_acf(old_sim, column, max_lag), marker="o", label="Original")
        ax.plot(lags, mean_acf(new_sim, column, max_lag), marker="o", label=alternative_label)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set(title=title, xlabel="Lag", ylabel="ACF")
        ax.legend()
    fig.suptitle(f"Lineage correlation comparison: {sugar}")
    fig.tight_layout()
    fig.savefig(output_dir / "acf_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(PARAMETER_NAMES), 1, figsize=(10, 12), sharex=True)
    for idx, (ax, name) in enumerate(zip(axes, PARAMETER_NAMES)):
        ax.plot(chain[:, :, idx], alpha=0.25, linewidth=0.5)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("Production step")
    fig.tight_layout()
    fig.savefig(output_dir / "trace_plots.png", dpi=160)
    plt.close(fig)

    if corner is not None:
        fig = corner.corner(flat_chain, labels=PARAMETER_NAMES, show_titles=True)
        fig.savefig(output_dir / "corner_plot.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        PROJECT_DIR / "outputs" / "model_comparison" / args.sugar / "weibull_adder_sizer"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    real, data, initial_lengths = load_data(args.sugar)
    bounds = make_bounds(data)
    start_time = time.time()
    mle_params, mle_details = fit_mle(data, bounds, args.seed)
    print("MLE parameters:")
    print(pd.Series(mle_params, index=PARAMETER_NAMES))
    print("MLE log likelihood:", mle_details["log_likelihood"])

    n_walkers = max(args.walkers, 2 * len(PARAMETER_NAMES) + 2)

    def log_probability(params: np.ndarray) -> float:
        if not valid_prior(params, bounds):
            return -np.inf
        return log_likelihood(params, data)

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
    old_params = old_summary.loc[["u", "v", "omega2", "mu_alpha", "sigma_alpha", "mu_f", "sigma_f"], "median"].to_numpy(float)
    with (old_dir / "diagnostics.json").open() as handle:
        old_diagnostics = json.load(handle)

    metric_rows = distribution_metrics(real, old_sim, "original_burr")
    metric_rows += distribution_metrics(real, new_sim, "weibull_adder_sizer")
    pd.DataFrame(metric_rows).to_csv(output_dir / "comparison_metrics.csv", index=False)

    times = np.sort(data["time"])
    empirical_survival = 1.0 - np.arange(1, len(times) + 1) / len(times)
    old_survival = average_survival_original(times, data, old_params)
    new_survival = average_survival_new(times, data, median_params)

    acf_comparison = {}
    for column in ["time", "length_birth"]:
        acf_comparison[column] = {
            "real_lag1": float(mean_acf(real, column, args.acf_max_lag)[1]),
            "original_lag1": float(mean_acf(old_sim, column, args.acf_max_lag)[1]),
            "weibull_lag1": float(mean_acf(new_sim, column, args.acf_max_lag)[1]),
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
        "model": "weibull_mixed_adder_sizer",
        "parameter_names": PARAMETER_NAMES,
        "mle": mle_details,
        "max_sampled_log_probability": float(np.max(flat_logp)),
        "original_max_sampled_log_probability": float(old_diagnostics["max_log_probability"]),
        "delta_max_log_probability_new_minus_original": float(np.max(flat_logp) - old_diagnostics["max_log_probability"]),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": tau,
        "runtime_minutes": (time.time() - start_time) / 60.0,
        "survival": {
            "original_burr": survival_metrics(times, old_survival),
            "weibull_adder_sizer": survival_metrics(times, new_survival),
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
    )

    print("\nPosterior summary:")
    print(summary.to_string(index=False))
    print("\nComparison summary:")
    print(json.dumps(comparison, indent=2))
    print("\nSaved to", output_dir)


if __name__ == "__main__":
    main()
