#!/usr/bin/env python3
"""Fit the alpha-scaled saturating time-hazard model to one sugar."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import corner
import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import ks_2samp


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from alpha_scaled_hill_time_hazard import (  # noqa: E402
    RESIDUAL_DF,
    log_lower_truncated_residual_density,
    log_time_density,
    simulate_lineages,
)
from run_loglogistic_endpoint_residual import (  # noqa: E402
    acf_1d,
    load_endpoint_data,
    mean_acf,
)


PARAMETER_NAMES = [
    "u",
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
    r"$u$",
    r"$\omega_2$",
    r"$\eta$",
    r"$\mu_\alpha$",
    r"$\sigma_\alpha$",
    r"$\mu_f$",
    r"$\sigma_f$",
    r"$\mu_\epsilon$",
    r"$\sigma_\epsilon$",
]
HAZARD_BOUNDS = np.asarray(
    [[1e-4, 20.0], [0.1, 30.0], [1e-3, 20.0]], dtype=float
)
RESIDUAL_BOUNDS = np.asarray([[-1.0, 1.0], [1e-5, 1.0]], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sugar", default="Glu")
    parser.add_argument("--hazard-walkers", type=int, default=24)
    parser.add_argument("--hazard-burn", type=int, default=750)
    parser.add_argument("--hazard-steps", type=int, default=2500)
    parser.add_argument("--residual-walkers", type=int, default=16)
    parser.add_argument("--residual-burn", type=int, default=500)
    parser.add_argument("--residual-steps", type=int, default=2000)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--acf-replicates", type=int, default=500)
    parser.add_argument("--acf-max-lag", type=int, default=30)
    parser.add_argument("--sim-lineages", type=int, default=40)
    parser.add_argument("--sim-generations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def sum_time_log_likelihood(
    physical: np.ndarray, real: pd.DataFrame
) -> float:
    values = log_time_density(
        real.time.to_numpy(float),
        real.length_birth.to_numpy(float),
        real.alpha.to_numpy(float),
        *physical,
    )
    return float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf


def sum_residual_log_likelihood(
    physical: np.ndarray, transition: pd.DataFrame
) -> float:
    location, scale = map(float, physical)
    values = log_lower_truncated_residual_density(
        transition.epsilon.to_numpy(float),
        -(transition.alpha * transition.time).to_numpy(float),
        location,
        scale,
    )
    return float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf


def fit_block_mle(
    objective,
    log_bounds: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    result = differential_evolution(
        lambda values: -objective(values),
        bounds=[tuple(row) for row in log_bounds],
        seed=seed,
        popsize=12,
        maxiter=120,
        polish=True,
        workers=1,
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"MLE failed: {result.message}")
    return result.x, float(-result.fun)


def initialize_walkers(
    center: np.ndarray,
    n_walkers: int,
    bounds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    scale = np.maximum(0.015 * (bounds[:, 1] - bounds[:, 0]), 1e-3)
    walkers: list[np.ndarray] = []
    for _ in range(100_000):
        candidate = center + rng.normal(0.0, scale)
        if np.all(candidate > bounds[:, 0]) and np.all(
            candidate < bounds[:, 1]
        ):
            walkers.append(candidate)
            if len(walkers) == n_walkers:
                return np.asarray(walkers)
    raise RuntimeError("Unable to initialize MCMC walkers")


def run_block_sampler(
    log_probability,
    center: np.ndarray,
    bounds: np.ndarray,
    n_walkers: int,
    burn: int,
    steps: int,
    thin: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, emcee.EnsembleSampler]:
    rng = np.random.default_rng(seed)
    initial = initialize_walkers(center, n_walkers, bounds, rng)
    sampler = emcee.EnsembleSampler(
        n_walkers,
        len(center),
        log_probability,
        moves=[
            (emcee.moves.StretchMove(), 0.6),
            (emcee.moves.DEMove(), 0.4),
        ],
    )
    state = sampler.run_mcmc(initial, burn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, steps, progress=False)
    return (
        sampler.get_chain(),
        sampler.get_chain(thin=thin, flat=True),
        sampler.get_log_prob(thin=thin, flat=True),
        sampler,
    )


def posterior_summary(
    flat_chain: np.ndarray, mode_values: dict[str, float]
) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(PARAMETER_NAMES):
        values = flat_chain[:, index]
        rows.append(
            {
                "parameter": name,
                "mode": mode_values[name],
                "mean": np.mean(values),
                "median": np.median(values),
                "ci_2.5": np.percentile(values, 2.5),
                "ci_97.5": np.percentile(values, 97.5),
                "std": np.std(values, ddof=1),
            }
        )
    return pd.DataFrame(rows)


def make_trace_plot(
    output_dir: Path,
    hazard_chain: np.ndarray,
    residual_chain: np.ndarray,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=False)
    physical_hazard = np.exp(hazard_chain)
    for index, label in enumerate(CORNER_LABELS[:3]):
        axes[index].plot(physical_hazard[:, :, index], alpha=0.35, lw=0.45)
        axes[index].set_ylabel(label)
    physical_residual = residual_chain.copy()
    physical_residual[:, :, 1] = np.exp(physical_residual[:, :, 1])
    for offset, label in enumerate(CORNER_LABELS[-2:]):
        axes[3 + offset].plot(
            physical_residual[:, :, offset], alpha=0.35, lw=0.45
        )
        axes[3 + offset].set_ylabel(label)
    axes[-1].set_xlabel("Production step")
    fig.tight_layout()
    fig.savefig(output_dir / "trace_plot.png", dpi=180)
    plt.close(fig)


def make_corner_plot(
    output_dir: Path, flat_chain: np.ndarray, medians: np.ndarray
) -> None:
    figure = corner.corner(
        flat_chain,
        labels=CORNER_LABELS,
        truths=medians,
        show_titles=True,
        title_fmt=".3g",
        quantiles=[0.16, 0.5, 0.84],
        max_n_ticks=3,
    )
    figure.suptitle(
        "Glu: alpha-scaled time-hazard posterior", y=1.0, fontsize=15
    )
    figure.savefig(output_dir / "corner_plot.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def make_marginal_plot(
    output_dir: Path,
    real: pd.DataFrame,
    transition: pd.DataFrame,
    simulated: pd.DataFrame,
    current: pd.DataFrame | None,
) -> pd.DataFrame:
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
    rows: list[dict[str, object]] = []
    for axis, variable in zip(axes.flat, observed):
        obs = observed[variable]
        new = simulated[variable].to_numpy(float)
        combined = np.concatenate([obs, new])
        low, high = np.percentile(combined[np.isfinite(combined)], [0.5, 99.5])
        bins = np.linspace(low, high, 36)
        axis.hist(
            obs,
            bins=bins,
            density=True,
            histtype="step",
            color="tab:blue",
            lw=2,
            label="Glu data",
        )
        axis.hist(
            new,
            bins=bins,
            density=True,
            color="tab:orange",
            alpha=0.32,
            label="Alpha-scaled hazard",
        )
        current_values = None
        if current is not None and variable in current:
            current_values = current[variable].to_numpy(float)
            axis.hist(
                current_values,
                bins=bins,
                density=True,
                histtype="step",
                color="0.45",
                ls="--",
                lw=1.4,
                label="Current log-logistic model",
            )
        axis.set_title(labels[variable])
        axis.set_ylabel("Density")
        row = {
            "variable": variable,
            "observed_mean": np.mean(obs),
            "alpha_scaled_mean": np.mean(new),
            "current_model_mean": (
                np.mean(current_values)
                if current_values is not None
                else np.nan
            ),
            "observed_sd": np.std(obs, ddof=1),
            "alpha_scaled_sd": np.std(new, ddof=1),
            "current_model_sd": (
                np.std(current_values, ddof=1)
                if current_values is not None
                else np.nan
            ),
            "alpha_scaled_ks": ks_2samp(obs, new).statistic,
            "current_model_ks": (
                ks_2samp(obs, current_values).statistic
                if current_values is not None
                else np.nan
            ),
        }
        rows.append(row)
    axes.flat[-1].axis("off")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "posterior_predictive_comparison.png", dpi=180)
    plt.close(fig)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "comparison_metrics.csv", index=False)
    return metrics


def make_matched_acf_plot(
    output_dir: Path,
    real: pd.DataFrame,
    flat_chain: np.ndarray,
    long_simulation: pd.DataFrame,
    current_table: pd.DataFrame | None,
    max_lag: int,
    n_replicates: int,
    seed: int,
) -> pd.DataFrame:
    ordered = real.sort_values(["lineage", "generation"])
    groups = [
        group for _, group in ordered.groupby("lineage", sort=False)
    ]
    design = np.asarray([len(group) for group in groups], dtype=int)
    initial = np.asarray(
        [group.length_birth.iloc[0] for group in groups], dtype=float
    )
    max_lag = min(max_lag, int(design.max()) - 1)
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(flat_chain), size=n_replicates)
    predictive = {
        "time": np.full((n_replicates, max_lag + 1), np.nan),
        "length_birth": np.full((n_replicates, max_lag + 1), np.nan),
    }
    for replicate, draw_index in enumerate(draw_indices):
        full = pd.DataFrame(
            simulate_lineages(
                flat_chain[draw_index],
                initial,
                len(design),
                int(design.max()),
                int(rng.integers(np.iinfo(np.int32).max)),
                fixed_initial_lengths=True,
            )
        )
        for variable in predictive:
            pieces = [
                acf_1d(
                    full.loc[full.lineage.eq(lineage), variable].to_numpy(
                        float
                    )[:length],
                    max_lag,
                )
                for lineage, length in enumerate(design)
                if length >= 3
            ]
            predictive[variable][replicate] = np.nanmean(
                np.asarray(pieces), axis=0
            )

    lags = np.arange(max_lag + 1)
    result: dict[str, np.ndarray] = {
        "lag": lags,
        "contributing_lineages": np.asarray(
            [np.sum((design >= 3) & (design > lag)) for lag in lags],
            dtype=int,
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    for axis, (variable, title) in zip(
        axes,
        [("time", "Division-time ACF"), ("length_birth", "Birth-length ACF")],
    ):
        observed = mean_acf(real, variable, max_lag)
        median = np.nanmedian(predictive[variable], axis=0)
        lower = np.nanpercentile(predictive[variable], 2.5, axis=0)
        upper = np.nanpercentile(predictive[variable], 97.5, axis=0)
        result[f"observed_{variable}"] = observed
        result[f"predictive_median_{variable}"] = median
        result[f"predictive_q2.5_{variable}"] = lower
        result[f"predictive_q97.5_{variable}"] = upper
        result[f"long_lineage_{variable}"] = mean_acf(
            long_simulation, variable, max_lag
        )
        if current_table is not None:
            current = current_table[current_table.lag.le(max_lag)]
            current_median = current[
                f"predictive_median_{variable}"
            ].to_numpy(float)
            current_lower = current[
                f"predictive_q2.5_{variable}"
            ].to_numpy(float)
            current_upper = current[
                f"predictive_q97.5_{variable}"
            ].to_numpy(float)
            result[f"current_median_{variable}"] = current_median
            result[f"current_q2.5_{variable}"] = current_lower
            result[f"current_q97.5_{variable}"] = current_upper
            axis.fill_between(
                lags,
                current_lower,
                current_upper,
                color="0.65",
                alpha=0.18,
                label="Current-model 95% band",
            )
            axis.plot(
                lags,
                current_median,
                color="0.45",
                ls="--",
                lw=1.3,
                label="Current-model median",
            )
        axis.fill_between(
            lags,
            lower,
            upper,
            color="tab:orange",
            alpha=0.25,
            label="New-model 95% band",
        )
        axis.plot(
            lags,
            median,
            color="tab:orange",
            marker="o",
            ms=3,
            label="New-model median",
        )
        axis.plot(
            lags,
            observed,
            color="tab:blue",
            marker="o",
            ms=3,
            lw=2,
            label="Glu data",
        )
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_title(title)
        axis.set_xlabel("Generation lag")
        axis.set_ylabel("Mean sample ACF")
        axis.set_xticks(np.arange(0, max_lag + 1, 5))
        axis.grid(axis="y", alpha=0.13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Glu: observation-matched ACF comparison", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(output_dir / "acf_observation_matched.png", dpi=180)
    plt.close(fig)
    table = pd.DataFrame(result)
    table.to_csv(output_dir / "acf_observation_matched.csv", index=False)
    return table


def sampler_diagnostics(sampler: emcee.EnsembleSampler) -> dict[str, object]:
    try:
        tau: object = sampler.get_autocorr_time(tol=0).tolist()
    except Exception as exc:
        tau = {"error": str(exc)}
    return {
        "mean_acceptance_fraction": float(
            np.mean(sampler.acceptance_fraction)
        ),
        "autocorrelation_time": tau,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / "alpha_scaled_hill_time_hazard"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    real, transition, _, initial_lengths = load_endpoint_data(args.sugar)

    log_hazard_bounds = np.log(HAZARD_BOUNDS)
    hazard_mle_log, hazard_mle_ll = fit_block_mle(
        lambda values: sum_time_log_likelihood(np.exp(values), real),
        log_hazard_bounds,
        args.seed,
    )
    residual_log_bounds = np.asarray(
        [RESIDUAL_BOUNDS[0], np.log(RESIDUAL_BOUNDS[1])]
    )
    residual_mle_transformed, residual_mle_ll = fit_block_mle(
        lambda values: sum_residual_log_likelihood(
            np.asarray([values[0], math.exp(values[1])]), transition
        ),
        residual_log_bounds,
        args.seed + 1,
    )
    print("Hazard MLE", np.exp(hazard_mle_log), hazard_mle_ll)
    print(
        "Residual MLE",
        [residual_mle_transformed[0], math.exp(residual_mle_transformed[1])],
        residual_mle_ll,
    )

    def hazard_log_probability(values: np.ndarray) -> float:
        if not (
            np.all(values > log_hazard_bounds[:, 0])
            and np.all(values < log_hazard_bounds[:, 1])
        ):
            return -np.inf
        likelihood = sum_time_log_likelihood(np.exp(values), real)
        return likelihood + float(np.sum(values))

    def residual_log_probability(values: np.ndarray) -> float:
        if not (
            np.all(values > residual_log_bounds[:, 0])
            and np.all(values < residual_log_bounds[:, 1])
        ):
            return -np.inf
        likelihood = sum_residual_log_likelihood(
            np.asarray([values[0], math.exp(values[1])]), transition
        )
        return likelihood + float(values[1])

    print("Sampling hazard block")
    hazard_chain, hazard_flat_log, hazard_log_prob, hazard_sampler = (
        run_block_sampler(
            hazard_log_probability,
            hazard_mle_log,
            log_hazard_bounds,
            args.hazard_walkers,
            args.hazard_burn,
            args.hazard_steps,
            args.thin,
            args.seed + 10,
        )
    )
    print("Sampling endpoint-residual block")
    residual_chain, residual_flat_transformed, residual_log_prob, residual_sampler = (
        run_block_sampler(
            residual_log_probability,
            residual_mle_transformed,
            residual_log_bounds,
            args.residual_walkers,
            args.residual_burn,
            args.residual_steps,
            args.thin,
            args.seed + 20,
        )
    )
    hazard_flat = np.exp(hazard_flat_log)
    residual_flat = residual_flat_transformed.copy()
    residual_flat[:, 1] = np.exp(residual_flat[:, 1])

    current_dir = (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / "loglogistic_endpoint_residual"
    )
    current_chain = pd.read_csv(current_dir / "flat_chain.csv")
    nuisance_names = ["mu_alpha", "sigma_alpha", "mu_f", "sigma_f"]
    rng = np.random.default_rng(args.seed + 30)
    n_combined = min(len(hazard_flat), len(residual_flat), len(current_chain))
    hazard_indices = rng.choice(len(hazard_flat), n_combined, replace=False)
    residual_indices = rng.choice(len(residual_flat), n_combined, replace=False)
    nuisance_indices = rng.choice(len(current_chain), n_combined, replace=False)
    combined = np.column_stack(
        [
            hazard_flat[hazard_indices],
            current_chain.loc[nuisance_indices, nuisance_names].to_numpy(float),
            residual_flat[residual_indices],
        ]
    )

    current_summary = pd.read_csv(current_dir / "posterior_summary.csv").set_index(
        "parameter"
    )
    hazard_mode = hazard_flat[int(np.argmax(hazard_log_prob))]
    residual_mode = residual_flat[int(np.argmax(residual_log_prob))]
    mode_values = {
        "u": hazard_mode[0],
        "omega2": hazard_mode[1],
        "eta": hazard_mode[2],
        **{
            name: float(current_summary.loc[name, "mode"])
            for name in nuisance_names
        },
        "mu_epsilon": residual_mode[0],
        "sigma_epsilon": residual_mode[1],
    }
    summary = posterior_summary(combined, mode_values)
    summary.to_csv(output_dir / "posterior_summary.csv", index=False)
    pd.DataFrame(combined, columns=PARAMETER_NAMES).to_csv(
        output_dir / "flat_chain.csv", index=False
    )
    pd.DataFrame(hazard_flat, columns=PARAMETER_NAMES[:3]).to_csv(
        output_dir / "hazard_flat_chain.csv", index=False
    )
    pd.DataFrame(
        residual_flat, columns=PARAMETER_NAMES[-2:]
    ).to_csv(output_dir / "residual_flat_chain.csv", index=False)
    transition.to_csv(output_dir / "reconstructed_transitions.csv", index=False)

    median_params = summary.set_index("parameter").loc[
        PARAMETER_NAMES, "median"
    ].to_numpy(float)
    long_simulation = pd.DataFrame(
        simulate_lineages(
            median_params,
            initial_lengths,
            args.sim_lineages,
            args.sim_generations,
            args.seed + 100,
        )
    )
    long_simulation.to_csv(
        output_dir / "posterior_predictive_simulation.csv", index=False
    )
    current_simulation_path = current_dir / "posterior_predictive_simulation.csv"
    current_simulation = (
        pd.read_csv(current_simulation_path)
        if current_simulation_path.exists()
        else None
    )
    metrics = make_marginal_plot(
        output_dir,
        real,
        transition,
        long_simulation,
        current_simulation,
    )
    current_acf_path = current_dir / "acf_observation_matched.csv"
    current_acf = (
        pd.read_csv(current_acf_path) if current_acf_path.exists() else None
    )
    matched = make_matched_acf_plot(
        output_dir,
        real,
        combined,
        long_simulation,
        current_acf,
        args.acf_max_lag,
        args.acf_replicates,
        args.seed + 200,
    )
    make_trace_plot(output_dir, hazard_chain, residual_chain)
    make_corner_plot(output_dir, combined, median_params)

    diagnostics = {
        "sugar": args.sugar,
        "model": "alpha_scaled_hill_time_hazard_with_endpoint_residual",
        "input_file": "data/processed/processed_input_data.csv",
        "hazard": (
            "h(t|b,alpha)=alpha*omega2*(p(t)/u)^eta/"
            "(1+(p(t)/u)^eta), p(t)=b*(exp(alpha*t)-1)"
        ),
        "endpoint_model": "L_final=b*exp(alpha*t+epsilon)",
        "residual_distribution": {
            "family": "lower-truncated Student-t location-scale",
            "degrees_of_freedom": RESIDUAL_DF,
            "lower_bound": "epsilon > -alpha*t",
        },
        "inference": {
            "strategy": "independent posterior blocks",
            "hazard_parameters_sampled_here": PARAMETER_NAMES[:3],
            "residual_parameters_sampled_here": PARAMETER_NAMES[-2:],
            "reused_unchanged_nuisance_block": nuisance_names,
            "nuisance_source": str(current_dir / "flat_chain.csv"),
            "combined_samples": int(n_combined),
        },
        "mle": {
            "hazard": dict(zip(PARAMETER_NAMES[:3], np.exp(hazard_mle_log))),
            "hazard_log_likelihood": hazard_mle_ll,
            "residual": {
                "mu_epsilon": residual_mle_transformed[0],
                "sigma_epsilon": math.exp(residual_mle_transformed[1]),
            },
            "residual_log_likelihood": residual_mle_ll,
        },
        "hazard_sampler": sampler_diagnostics(hazard_sampler),
        "residual_sampler": sampler_diagnostics(residual_sampler),
        "run_config": vars(args) | {"output_dir": str(output_dir)},
        "runtime_minutes": (time.time() - started) / 60.0,
        "acf_lag1": {
            variable: {
                "observed": float(matched.loc[1, f"observed_{variable}"]),
                "predictive_median": float(
                    matched.loc[1, f"predictive_median_{variable}"]
                ),
                "predictive_q2.5": float(
                    matched.loc[1, f"predictive_q2.5_{variable}"]
                ),
                "predictive_q97.5": float(
                    matched.loc[1, f"predictive_q97.5_{variable}"]
                ),
            }
            for variable in ["time", "length_birth"]
        },
        "predictive_metrics": metrics.to_dict(orient="records"),
    }
    with (output_dir / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(summary.to_string(index=False))
    print("Runtime minutes", diagnostics["runtime_minutes"])


if __name__ == "__main__":
    main()
