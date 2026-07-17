#!/usr/bin/env python3
"""Fit the endpoint-residual log-logistic model from processed_input_data.csv.

The script needs no sugar-specific raw file.  It reconstructs measured final
length, added size and endpoint residual from consecutive lineage rows using

    L_final,n = b_(n+1) / f_(n+1).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import ks_2samp, t as student_t

try:
    import corner
except ImportError:
    corner = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from loglogistic_endpoint_residual import (  # noqa: E402
    RESIDUAL_DF,
    log_added_size_density,
    log_endpoint_residual_density,
    log_likelihood_pivoted,
    pivoted_to_physical_params,
    simulate_lineages,
    survival,
)
from weibull_adder_sizer import (  # noqa: E402
    beta_ab,
    gamma_shape_scale,
    log_beta_pdf_mean_sd,
    log_gamma_pdf_mean_sd,
)


PARAMETER_NAMES = [
    "p_star",
    "omega2",
    "eta",
    "mu_alpha",
    "sigma_alpha",
    "mu_f",
    "sigma_f",
    "mu_epsilon",
    "sigma_epsilon",
]

CORNER_LABELS = [
    r"$p_\star$",
    r"$\omega_2$",
    r"$\eta$",
    r"$\mu_\alpha$",
    r"$\sigma_\alpha$",
    r"$\mu_f$",
    r"$\sigma_f$",
    r"$\mu_\epsilon$",
    r"$\sigma_\epsilon$",
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
    parser.add_argument("--predictive-replicates", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_endpoint_data(
    sugar: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Load one sugar and reconstruct endpoints from consecutive rows only."""
    path = PROJECT_DIR / "data" / "processed" / "processed_input_data.csv"
    required = [
        "source",
        "lineage",
        "generation",
        "time",
        "alpha",
        "f",
        "length_birth",
    ]
    frame = pd.read_csv(path, usecols=required)
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    frame = frame[
        (frame.time > 0.0)
        & (frame.alpha > 0.0)
        & (frame.f > 0.0)
        & (frame.f < 1.0)
        & (frame.length_birth > 0.0)
    ]
    frame = frame[frame.source.eq(sugar)].sort_values(
        ["lineage", "generation"]
    )
    if frame.empty:
        raise ValueError(f"No valid rows for sugar {sugar!r}")

    generation_zero = frame[frame.generation.eq(0)].groupby(
        "lineage", sort=False
    ).first()
    retained = generation_zero.index.to_numpy()
    frame = frame[frame.lineage.isin(retained)].sort_values(
        ["lineage", "generation"]
    ).reset_index(drop=True)
    initial_lengths = generation_zero.loc[retained, "length_birth"].to_numpy(float)

    grouped = frame.groupby("lineage", sort=False)
    transition = frame.copy()
    transition["next_generation"] = grouped.generation.shift(-1)
    transition["next_length_birth"] = grouped.length_birth.shift(-1)
    transition["next_f"] = grouped.f.shift(-1)
    transition = transition[
        transition.next_generation.eq(transition.generation + 1.0)
    ].dropna(subset=["next_length_birth", "next_f"])
    transition["length_final"] = (
        transition.next_length_birth / transition.next_f
    )
    transition["added_size"] = (
        transition.length_final - transition.length_birth
    )
    transition["epsilon"] = (
        np.log(transition.length_final / transition.length_birth)
        - transition.alpha * transition.time
    )
    transition = transition[
        (transition.length_final > transition.length_birth)
        & np.isfinite(transition.added_size)
        & np.isfinite(transition.epsilon)
    ].reset_index(drop=True)
    if transition.empty:
        raise ValueError(f"No valid consecutive transitions for sugar {sugar!r}")

    data = {
        "alpha": frame.alpha.to_numpy(float),
        "f": frame.f.to_numpy(float),
        "transition_birth": transition.length_birth.to_numpy(float),
        "added_size": transition.added_size.to_numpy(float),
        "epsilon": transition.epsilon.to_numpy(float),
        "transition_log_jacobian": float(
            np.sum(np.log(transition.alpha) - np.log(transition.next_f))
        ),
    }
    return frame, transition, data, initial_lengths


def make_bounds(
    data: dict[str, np.ndarray], birth_pivot: float
) -> dict[str, tuple[float, float]]:
    added = data["added_size"]
    alpha = data["alpha"]
    u_bounds = (1e-5, max(20.0, 5.0 * float(np.percentile(added, 99))))
    eta_bounds = (1e-6, 1.0 - 1e-6)
    return {
        # Rectangular outer bounds for p_star; valid_prior also enforces the
        # original slanted support u_low < p_star + eta*b_star < u_high.
        "p_star": (
            u_bounds[0] - eta_bounds[1] * birth_pivot,
            u_bounds[1] - eta_bounds[0] * birth_pivot,
        ),
        "u": u_bounds,
        "omega2": (0.05, 30.0),
        "eta": eta_bounds,
        "mu_alpha": (1e-8, max(5.0, 3.0 * float(np.percentile(alpha, 99)))),
        "sigma_alpha": (1e-8, max(5.0, 3.0 * float(np.std(alpha, ddof=1)))),
        "mu_f": (1e-8, 1.0 - 1e-8),
        "sigma_f": (1e-8, 0.5),
        "mu_epsilon": (-1.0, 1.0),
        "sigma_epsilon": (1e-5, 1.0),
    }


def valid_prior(
    params: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    birth_pivot: float,
) -> bool:
    if len(params) != len(PARAMETER_NAMES):
        return False
    for value, name in zip(params, PARAMETER_NAMES):
        low, high = bounds[name]
        if not (low < value < high):
            return False
    physical_u = params[0] + params[2] * birth_pivot
    if not (bounds["u"][0] < physical_u < bounds["u"][1]):
        return False
    if gamma_shape_scale(params[3], params[4])[0] is None:
        return False
    if beta_ab(params[5], params[6])[0] is None:
        return False
    return True


def fit_mle(
    data: dict[str, np.ndarray],
    bounds: dict[str, tuple[float, float]],
    birth_pivot: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    def hazard_objective(values: np.ndarray) -> float:
        physical_u = values[0] + values[2] * birth_pivot
        if not (bounds["u"][0] < physical_u < bounds["u"][1]):
            return 1e100
        log_density = log_added_size_density(
            data["added_size"],
            data["transition_birth"],
            physical_u,
            values[1],
            values[2],
        )
        return -float(np.sum(log_density)) if np.all(np.isfinite(log_density)) else 1e100

    hazard = differential_evolution(
        hazard_objective,
        [bounds[name] for name in ["p_star", "omega2", "eta"]],
        seed=seed,
        popsize=15,
        maxiter=400,
        tol=1e-10,
        polish=True,
        workers=1,
        updating="immediate",
    )

    def alpha_objective(values: np.ndarray) -> float:
        density = log_gamma_pdf_mean_sd(data["alpha"], *values)
        return -float(np.sum(density)) if np.all(np.isfinite(density)) else 1e100

    alpha = minimize(
        alpha_objective,
        [np.mean(data["alpha"]), np.std(data["alpha"], ddof=1)],
        method="L-BFGS-B",
        bounds=[bounds["mu_alpha"], bounds["sigma_alpha"]],
    )

    def fraction_objective(values: np.ndarray) -> float:
        density = log_beta_pdf_mean_sd(data["f"], *values)
        return -float(np.sum(density)) if np.all(np.isfinite(density)) else 1e100

    fraction = minimize(
        fraction_objective,
        [np.mean(data["f"]), np.std(data["f"], ddof=1)],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-12, "fatol": 1e-9},
    )

    def residual_objective(values: np.ndarray) -> float:
        density = log_endpoint_residual_density(
            data["epsilon"],
            np.log1p(data["added_size"] / data["transition_birth"]),
            *values,
        )
        return -float(np.sum(density)) if np.all(np.isfinite(density)) else 1e100

    median = float(np.median(data["epsilon"]))
    mad_scale = max(
        1e-3,
        1.4826 * float(np.median(np.abs(data["epsilon"] - median))),
    )
    residual = minimize(
        residual_objective,
        [median, mad_scale],
        method="L-BFGS-B",
        bounds=[bounds["mu_epsilon"], bounds["sigma_epsilon"]],
    )

    params = np.concatenate([hazard.x, alpha.x, fraction.x, residual.x])
    if not valid_prior(params, bounds, birth_pivot):
        raise RuntimeError(f"Invalid MLE parameters: {params}")
    details = {
        "hazard_success": bool(hazard.success),
        "hazard_message": str(hazard.message),
        "alpha_success": bool(alpha.success),
        "fraction_success": bool(fraction.success),
        "residual_success": bool(residual.success),
        "log_likelihood": float(
            log_likelihood_pivoted(params, data, birth_pivot)
        ),
    }
    return params, details


def initialize_walkers(
    center: np.ndarray,
    n_walkers: int,
    bounds: dict[str, tuple[float, float]],
    data: dict[str, np.ndarray],
    birth_pivot: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = np.maximum(np.abs(center) * 0.025, 1e-4)
    walkers: list[np.ndarray] = []
    for _ in range(300_000):
        proposal = center + rng.normal(0.0, scales)
        if valid_prior(proposal, bounds, birth_pivot) and np.isfinite(
            log_likelihood_pivoted(proposal, data, birth_pivot)
        ):
            walkers.append(proposal)
            if len(walkers) == n_walkers:
                return np.asarray(walkers)
    raise RuntimeError("Unable to initialize all MCMC walkers")


def posterior_summary(
    flat_chain: np.ndarray, flat_log_probability: np.ndarray
) -> pd.DataFrame:
    mode = flat_chain[int(np.argmax(flat_log_probability))]
    rows = []
    for index, name in enumerate(PARAMETER_NAMES):
        values = flat_chain[:, index]
        rows.append(
            {
                "parameter": name,
                "mode": mode[index],
                "mean": np.mean(values),
                "median": np.median(values),
                "ci_2.5": np.percentile(values, 2.5),
                "ci_97.5": np.percentile(values, 97.5),
                "std": np.std(values, ddof=1),
            }
        )
    return pd.DataFrame(rows)


def acf_1d(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    result = np.full(max_lag + 1, np.nan)
    if len(values) < 3:
        return result
    centered = values - np.mean(values)
    denominator = np.sum(centered**2)
    if denominator <= 0.0:
        return result
    result[0] = 1.0
    for lag in range(1, min(max_lag + 1, len(centered))):
        result[lag] = np.sum(centered[:-lag] * centered[lag:]) / denominator
    return result


def mean_acf(frame: pd.DataFrame, column: str, max_lag: int) -> np.ndarray:
    pieces = []
    for _, group in frame.groupby("lineage"):
        group = group.sort_values("generation")
        if len(group) >= 3:
            pieces.append(acf_1d(group[column].to_numpy(float), max_lag))
    array = np.asarray(pieces)
    counts = np.sum(np.isfinite(array), axis=0)
    return np.divide(
        np.nansum(array, axis=0),
        counts,
        out=np.full(max_lag + 1, np.nan),
        where=counts > 0,
    )


def metric_row(
    model: str, variable: str, observed: np.ndarray, simulated: np.ndarray
) -> dict[str, object]:
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    observed = observed[np.isfinite(observed)]
    simulated = simulated[np.isfinite(simulated)]
    observed_sd = float(np.std(observed, ddof=1))
    return {
        "model": model,
        "variable": variable,
        "observed_mean": float(np.mean(observed)),
        "simulated_mean": float(np.mean(simulated)),
        "mean_ratio": float(np.mean(simulated) / np.mean(observed)),
        "observed_sd": observed_sd,
        "simulated_sd": float(np.std(simulated, ddof=1)),
        "sd_ratio": float(np.std(simulated, ddof=1) / observed_sd),
        "ks_statistic": float(ks_2samp(observed, simulated).statistic),
    }


def predictive_metrics(
    real: pd.DataFrame,
    transition: pd.DataFrame,
    simulated: pd.DataFrame,
    model: str,
) -> list[dict[str, object]]:
    observed = {
        "time": real.time.to_numpy(float),
        "alpha": real.alpha.to_numpy(float),
        "f": real.f.to_numpy(float),
        "length_birth": real.length_birth.to_numpy(float),
        "length_final": transition.length_final.to_numpy(float),
        "added_size": transition.added_size.to_numpy(float),
        "epsilon": transition.epsilon.to_numpy(float),
    }
    return [
        metric_row(model, variable, values, simulated[variable].to_numpy(float))
        for variable, values in observed.items()
    ]


def run_replicated_predictive_checks(
    real: pd.DataFrame,
    transition: pd.DataFrame,
    params: np.ndarray,
    birth_pivot: float,
    initial_lengths: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for replicate in range(args.predictive_replicates):
        simulated = pd.DataFrame(
            simulate_lineages(
                pivoted_to_physical_params(params, birth_pivot),
                initial_lengths,
                args.sim_lineages,
                args.sim_generations,
                args.seed + 1000 + replicate,
            )
        )
        row: dict[str, object] = {"replicate": replicate}
        for metric in predictive_metrics(
            real, transition, simulated, "endpoint_residual"
        ):
            variable = str(metric["variable"])
            row[f"{variable}_mean_ratio"] = metric["mean_ratio"]
            row[f"{variable}_sd_ratio"] = metric["sd_ratio"]
            row[f"{variable}_ks"] = metric["ks_statistic"]
        row["time_acf_lag1"] = mean_acf(
            simulated, "time", args.acf_max_lag
        )[1]
        row["length_birth_acf_lag1"] = mean_acf(
            simulated, "length_birth", args.acf_max_lag
        )[1]
        rows.append(row)
    detailed = pd.DataFrame(rows)
    detailed.to_csv(output_dir / "replicated_predictive_metrics.csv", index=False)

    summaries = []
    for metric in detailed.columns.drop("replicate"):
        values = detailed[metric].to_numpy(float)
        summaries.append(
            {
                "metric": metric,
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "q2.5": np.percentile(values, 2.5),
                "median": np.median(values),
                "q97.5": np.percentile(values, 97.5),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "replicated_predictive_summary.csv", index=False)
    return summary


def average_survival(
    p_grid: np.ndarray,
    births: np.ndarray,
    params: np.ndarray,
    birth_pivot: float,
) -> np.ndarray:
    u, omega2, eta = pivoted_to_physical_params(params, birth_pivot)[:3]
    pieces = []
    for start in range(0, len(p_grid), 128):
        p = p_grid[start : start + 128, None]
        pieces.append(
            np.mean(survival(p, births[None, :], u, omega2, eta), axis=1)
        )
    return np.concatenate(pieces)


def make_observation_matched_acf_plot(
    output_dir: Path,
    sugar: str,
    real: pd.DataFrame,
    flat_chain: np.ndarray,
    long_simulation: pd.DataFrame,
    birth_pivot: float,
    max_lag: int,
    seed: int,
    n_replicates: int = 500,
) -> pd.DataFrame:
    """Posterior-predictive ACF with the empirical lineage-length design.

    Every replicate has the same number and lengths of lineages as the data.
    The same per-lineage, sample-mean-subtracted estimator is then applied to
    observed and simulated traces, so its finite-length bias is reproduced.
    """
    ordered = real.sort_values(["lineage", "generation"])
    grouped = ordered.groupby("lineage", sort=False)
    design = grouped.size().to_numpy(int)
    paired_initial_lengths = grouped.length_birth.first().to_numpy(float)
    matched_max_lag = min(max_lag, int(np.max(design)) - 1)
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(flat_chain), size=n_replicates)
    simulated_acf = {
        "time": np.full((n_replicates, matched_max_lag + 1), np.nan),
        "length_birth": np.full(
            (n_replicates, matched_max_lag + 1), np.nan
        ),
    }

    for replicate, draw_index in enumerate(draw_indices):
        full = pd.DataFrame(
            simulate_lineages(
                pivoted_to_physical_params(
                    flat_chain[draw_index], birth_pivot
                ),
                paired_initial_lengths,
                len(design),
                int(np.max(design)),
                seed + replicate + 1,
                fixed_initial_lengths=True,
            )
        )
        pieces = []
        for lineage, length in enumerate(design):
            pieces.append(full[full.lineage.eq(lineage)].iloc[:length])
        matched = pd.concat(pieces, ignore_index=True)
        for variable in simulated_acf:
            simulated_acf[variable][replicate] = mean_acf(
                matched, variable, matched_max_lag
            )

    contributors = np.asarray(
        [
            np.sum((design >= 3) & (design > lag))
            for lag in range(matched_max_lag + 1)
        ],
        dtype=int,
    )
    lags = np.arange(matched_max_lag + 1)
    rows: dict[str, np.ndarray] = {
        "lag": lags,
        "contributing_lineages": contributors,
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    specifications = [
        ("time", "Division-time ACF"),
        ("length_birth", "Birth-length ACF"),
    ]
    for axis, (variable, title) in zip(axes, specifications):
        observed = mean_acf(real, variable, matched_max_lag)
        median = np.nanmedian(simulated_acf[variable], axis=0)
        lower = np.nanpercentile(simulated_acf[variable], 2.5, axis=0)
        upper = np.nanpercentile(simulated_acf[variable], 97.5, axis=0)
        long_acf = mean_acf(long_simulation, variable, matched_max_lag)
        rows[f"observed_{variable}"] = observed
        rows[f"predictive_median_{variable}"] = median
        rows[f"predictive_q2.5_{variable}"] = lower
        rows[f"predictive_q97.5_{variable}"] = upper
        rows[f"long_lineage_{variable}"] = long_acf

        axis.fill_between(
            lags,
            lower,
            upper,
            color="tab:orange",
            alpha=0.25,
            label="Matched-length 95% predictive band",
        )
        axis.plot(
            lags,
            median,
            color="tab:orange",
            marker="o",
            ms=3.5,
            label="Matched-length predictive median",
        )
        axis.plot(
            lags,
            observed,
            color="tab:blue",
            marker="o",
            ms=3.5,
            lw=2,
            label=f"{sugar} data",
        )
        axis.plot(
            lags,
            long_acf,
            color="0.35",
            ls="--",
            label="80-generation simulation",
        )
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_xlabel("Generation lag")
        axis.set_ylabel("Mean sample ACF")
        axis.set_title(title)
        tick_step = 1 if lags.max() <= 10 else 5
        axis.set_xticks(np.arange(0, lags.max() + 1, tick_step))
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        f"{sugar}: exact-pair observation-matched ACF check"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "acf_observation_matched.png", dpi=180)
    plt.close(fig)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "acf_observation_matched.csv", index=False)
    return result


def make_plots(
    output_dir: Path,
    sugar: str,
    real: pd.DataFrame,
    transition: pd.DataFrame,
    simulated: pd.DataFrame,
    params: np.ndarray,
    birth_pivot: float,
    chain: np.ndarray,
    flat_chain: np.ndarray,
    acf_max_lag: int,
) -> None:
    observed = {
        "time": real.time.to_numpy(float),
        "alpha": real.alpha.to_numpy(float),
        "f": real.f.to_numpy(float),
        "length_birth": real.length_birth.to_numpy(float),
        "length_final": transition.length_final.to_numpy(float),
        "added_size": transition.added_size.to_numpy(float),
        "epsilon": transition.epsilon.to_numpy(float),
    }
    labels = {
        "time": "Division time",
        "alpha": "Growth rate",
        "f": "Division fraction",
        "length_birth": "Birth length",
        "length_final": "Final length",
        "added_size": "Measured added length",
        "epsilon": "Endpoint residual",
    }
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, variable in zip(axes.flat, observed):
        values = observed[variable]
        combined = np.concatenate([values, simulated[variable].to_numpy(float)])
        low, high = np.percentile(combined[np.isfinite(combined)], [0.5, 99.5])
        bins = np.linspace(low, high, 36)
        axis.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            label=f"{sugar} data",
        )
        axis.hist(
            simulated[variable],
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            label="Endpoint-residual model",
        )
        axis.set_xlabel(labels[variable])
        axis.set_ylabel("Density")
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"{sugar}: posterior-predictive marginal distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_predictive_comparison.png", dpi=180)
    plt.close(fig)

    p_grid = np.sort(transition.added_size.to_numpy(float))
    empirical = 1.0 - np.arange(1, len(p_grid) + 1) / len(p_grid)
    fitted = average_survival(
        p_grid,
        transition.length_birth.to_numpy(float),
        params,
        birth_pivot,
    )
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.step(p_grid, empirical, where="post", lw=2, label="Empirical measured added size")
    axis.plot(p_grid, fitted, lw=2, label="Posterior-median model")
    axis.set_xlabel("Measured added length p")
    axis.set_ylabel("Survival probability")
    axis.set_ylim(-0.02, 1.02)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "added_size_survival.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, variable, title in [
        (axes[0], "time", "Division-time ACF"),
        (axes[1], "length_birth", "Birth-length ACF"),
    ]:
        axis.plot(
            mean_acf(real, variable, acf_max_lag),
            marker="o",
            label=f"{sugar} data",
        )
        axis.plot(
            mean_acf(simulated, variable, acf_max_lag),
            marker="o",
            label="Endpoint-residual model",
        )
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_xlabel("Generation lag")
        axis.set_ylabel("Mean ACF")
        axis.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "acf_comparison.png", dpi=180)
    plt.close(fig)

    mu_epsilon, sigma_epsilon = params[-2:]
    epsilon_grid = np.linspace(
        np.percentile(transition.epsilon, 0.5),
        np.percentile(transition.epsilon, 99.5),
        400,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(transition.epsilon, bins=40, density=True, alpha=0.45)
    axes[0].plot(
        epsilon_grid,
        student_t.pdf(
            (epsilon_grid - mu_epsilon) / sigma_epsilon, RESIDUAL_DF
        )
        / sigma_epsilon,
        lw=2,
    )
    axes[0].set_xlabel("Endpoint residual epsilon")
    axes[0].set_ylabel("Density")
    axes[1].scatter(transition.length_birth, transition.epsilon, s=10, alpha=0.5)
    axes[1].set_xlabel("Birth length")
    axes[1].set_ylabel("Endpoint residual epsilon")
    axes[2].scatter(transition.added_size, transition.epsilon, s=10, alpha=0.5)
    axes[2].set_xlabel("Measured added length")
    axes[2].set_ylabel("Endpoint residual epsilon")
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_residual_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(13, 13), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, PARAMETER_NAMES)):
        axis.plot(chain[:, :, index], alpha=0.35, lw=0.45)
        axis.set_ylabel(name)
    axes.flat[-1].axis("off")
    axes[-1, 0].set_xlabel("Production step")
    fig.tight_layout()
    fig.savefig(output_dir / "trace_plots.png", dpi=180)
    plt.close(fig)

    if corner is not None:
        figure = corner.corner(
            flat_chain,
            labels=CORNER_LABELS,
            truths=params,
            show_titles=True,
            title_fmt=".3g",
            quantiles=[0.16, 0.5, 0.84],
            max_n_ticks=3,
        )
        figure.suptitle(
            f"{sugar}: re-parameterized posterior", y=1.0, fontsize=16
        )
        figure.savefig(
            output_dir / "corner_plot.png", dpi=160, bbox_inches="tight"
        )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / "loglogistic_endpoint_residual"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    real, transition, data, initial_lengths = load_endpoint_data(args.sugar)
    birth_pivot = float(np.mean(data["transition_birth"]))
    bounds = make_bounds(data, birth_pivot)
    mle_params, mle_details = fit_mle(
        data, bounds, birth_pivot, args.seed
    )
    mle_details["physical_u"] = float(
        pivoted_to_physical_params(mle_params, birth_pivot)[0]
    )
    print("MLE parameters")
    print(pd.Series(mle_params, index=PARAMETER_NAMES).to_string())
    print("MLE log likelihood", mle_details["log_likelihood"])

    n_walkers = max(args.walkers, 2 * len(PARAMETER_NAMES) + 2)

    def log_probability(params: np.ndarray) -> float:
        return (
            log_likelihood_pivoted(params, data, birth_pivot)
            if valid_prior(params, bounds, birth_pivot)
            else -np.inf
        )

    initial = initialize_walkers(
        mle_params, n_walkers, bounds, data, birth_pivot, args.seed
    )
    sampler = emcee.EnsembleSampler(
        n_walkers,
        len(PARAMETER_NAMES),
        log_probability,
        moves=[
            (emcee.moves.StretchMove(), 0.5),
            (emcee.moves.DEMove(), 0.3),
            (emcee.moves.DESnookerMove(), 0.2),
        ],
    )
    print(f"Burn-in: {args.burn} steps with {n_walkers} walkers")
    state = sampler.run_mcmc(initial, args.burn, progress=False)
    sampler.reset()
    print(f"Production: {args.steps} steps")
    sampler.run_mcmc(state, args.steps, progress=False)

    chain = sampler.get_chain()
    flat_chain = sampler.get_chain(thin=args.thin, flat=True)
    flat_log_probability = sampler.get_log_prob(thin=args.thin, flat=True)
    summary = posterior_summary(flat_chain, flat_log_probability)
    median_params = summary.set_index("parameter").loc[
        PARAMETER_NAMES, "median"
    ].to_numpy(float)
    flat_physical = pivoted_to_physical_params(flat_chain, birth_pivot)
    physical_u = flat_physical[:, 0]
    mode_index = int(np.argmax(flat_log_probability))
    derived_u_summary = pd.DataFrame(
        [
            {
                "parameter": "u",
                "mode": physical_u[mode_index],
                "mean": np.mean(physical_u),
                "median": np.median(physical_u),
                "ci_2.5": np.percentile(physical_u, 2.5),
                "ci_97.5": np.percentile(physical_u, 97.5),
                "std": np.std(physical_u, ddof=1),
            }
        ]
    )
    summary.to_csv(output_dir / "posterior_summary.csv", index=False)
    derived_u_summary.to_csv(
        output_dir / "derived_physical_posterior_summary.csv", index=False
    )
    pd.DataFrame(flat_chain, columns=PARAMETER_NAMES).to_csv(
        output_dir / "flat_chain.csv", index=False
    )
    pd.DataFrame({"log_probability": flat_log_probability}).to_csv(
        output_dir / "flat_log_probability.csv", index=False
    )
    transition.to_csv(output_dir / "reconstructed_transitions.csv", index=False)

    simulated = pd.DataFrame(
        simulate_lineages(
            pivoted_to_physical_params(median_params, birth_pivot),
            initial_lengths,
            args.sim_lineages,
            args.sim_generations,
            args.seed + 1,
        )
    )
    simulated.to_csv(
        output_dir / "posterior_predictive_simulation.csv", index=False
    )
    metrics = predictive_metrics(
        real, transition, simulated, "loglogistic_endpoint_residual"
    )

    previous_path = (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / "loglogistic_adder_sizer"
        / "posterior_predictive_simulation.csv"
    )
    if previous_path.exists():
        previous = pd.read_csv(previous_path)
        for variable in ["time", "alpha", "f", "length_birth"]:
            metrics.append(
                metric_row(
                    "previous_surrogate_model",
                    variable,
                    real[variable].to_numpy(float),
                    previous[variable].to_numpy(float),
                )
            )
    pd.DataFrame(metrics).to_csv(output_dir / "comparison_metrics.csv", index=False)
    replicated_summary = run_replicated_predictive_checks(
        real,
        transition,
        median_params,
        birth_pivot,
        initial_lengths,
        args,
        output_dir,
    )
    matched_acf = make_observation_matched_acf_plot(
        output_dir,
        args.sugar,
        real,
        flat_chain,
        simulated,
        birth_pivot,
        args.acf_max_lag,
        args.seed + 20_000,
    )

    try:
        autocorrelation = {
            name: float(value)
            for name, value in zip(
                PARAMETER_NAMES, sampler.get_autocorr_time(tol=0)
            )
        }
    except Exception as exc:
        autocorrelation = {"error": str(exc)}

    diagnostics = {
        "sugar": args.sugar,
        "input_file": "data/processed/processed_input_data.csv",
        "model": "loglogistic_mixed_adder_sizer_with_endpoint_residual",
        "parameterization": {
            "sampled_scale": "p_star",
            "birth_pivot": birth_pivot,
            "physical_scale_relation": "u = p_star + eta * birth_pivot",
            "transformation_jacobian": 1.0,
            "posterior_corr_p_star_eta": float(
                np.corrcoef(flat_chain[:, 0], flat_chain[:, 2])[0, 1]
            ),
            "posterior_corr_derived_u_eta": float(
                np.corrcoef(physical_u, flat_chain[:, 2])[0, 1]
            ),
            "derived_physical_u": {
                key: float(derived_u_summary.iloc[0][key])
                for key in [
                    "mode",
                    "mean",
                    "median",
                    "ci_2.5",
                    "ci_97.5",
                    "std",
                ]
            },
        },
        "residual_distribution": {
            "family": "Student-t location-scale",
            "fixed_degrees_of_freedom": RESIDUAL_DF,
        },
        "n_complete_observations": int(len(real)),
        "n_consecutive_transitions": int(len(transition)),
        "n_transition_lineages": int(transition.lineage.nunique()),
        "parameter_names": PARAMETER_NAMES,
        "mle": mle_details,
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": autocorrelation,
        "max_sampled_log_probability": float(np.max(flat_log_probability)),
        "runtime_minutes": float((time.time() - started) / 60.0),
        "observed_endpoint_residual": {
            "mean": float(np.mean(transition.epsilon)),
            "median": float(np.median(transition.epsilon)),
            "std": float(np.std(transition.epsilon, ddof=1)),
        },
        "acf_lag1": {
            "time": {
                "observed": float(mean_acf(real, "time", args.acf_max_lag)[1]),
                "simulated": float(mean_acf(simulated, "time", args.acf_max_lag)[1]),
            },
            "length_birth": {
                "observed": float(
                    mean_acf(real, "length_birth", args.acf_max_lag)[1]
                ),
                "simulated": float(
                    mean_acf(simulated, "length_birth", args.acf_max_lag)[1]
                ),
            },
        },
        "observation_matched_acf_lag1": {
            "time": {
                "observed": float(matched_acf.loc[1, "observed_time"]),
                "predictive_median": float(
                    matched_acf.loc[1, "predictive_median_time"]
                ),
                "predictive_q2.5": float(
                    matched_acf.loc[1, "predictive_q2.5_time"]
                ),
                "predictive_q97.5": float(
                    matched_acf.loc[1, "predictive_q97.5_time"]
                ),
            },
            "length_birth": {
                "observed": float(
                    matched_acf.loc[1, "observed_length_birth"]
                ),
                "predictive_median": float(
                    matched_acf.loc[1, "predictive_median_length_birth"]
                ),
                "predictive_q2.5": float(
                    matched_acf.loc[1, "predictive_q2.5_length_birth"]
                ),
                "predictive_q97.5": float(
                    matched_acf.loc[1, "predictive_q97.5_length_birth"]
                ),
            },
        },
        "observation_matched_design": {
            "posterior_replicates": 500,
            "preserves_initial_size_lineage_length_pairing": True,
            "description": (
                "Each simulated lineage uses its observed "
                "(b0, lineage length) pair."
            ),
        },
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
        "prior_bounds": {name: list(values) for name, values in bounds.items()},
        "replicated_predictive_medians": {
            row.metric: float(row.median)
            for row in replicated_summary.itertuples(index=False)
        },
    }
    with (output_dir / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)

    make_plots(
        output_dir,
        args.sugar,
        real,
        transition,
        simulated,
        median_params,
        birth_pivot,
        chain,
        flat_chain,
        args.acf_max_lag,
    )

    print("\nPosterior summary")
    print(summary.to_string(index=False))
    print("\nDiagnostics")
    print(json.dumps(diagnostics, indent=2))
    print("\nSaved to", output_dir)


if __name__ == "__main__":
    main()
