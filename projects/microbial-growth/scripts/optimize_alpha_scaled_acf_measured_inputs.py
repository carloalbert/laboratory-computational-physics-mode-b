#!/usr/bin/env python3
"""Calibrate the alpha-scaled hazard directly to Glu finite-lineage ACFs.

This is deliberately not a likelihood fit.  It replays each lineage's
measured alpha and f sequences and minimizes the birth-length ACF error, with
an option to include the division-time ACF, over a configurable lag window.
Endpoint-residual parameters remain fixed at their likelihood-posterior
medians, so only u, omega2 and eta are optimized.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import ks_2samp
from scipy.stats import t as student_t


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from alpha_scaled_hill_time_hazard import (  # noqa: E402
    EPS,
    RESIDUAL_DF,
    cumulative_hazard_growth_time,
    log_time_density,
)
from validate_acf_with_measured_inputs import (  # noqa: E402
    acf_1d,
    average_acfs,
    load_sugar,
    observed_acf,
)


PARAMETER_NAMES = ["u", "omega2", "eta"]
PHYSICAL_BOUNDS = np.asarray(
    [[1e-4, 20.0], [0.1, 30.0], [1e-3, 20.0]], dtype=float
)


@dataclass(frozen=True)
class ReplayDesign:
    initial_birth: np.ndarray
    alpha: np.ndarray
    fraction: np.ndarray
    lengths: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.alpha.shape


@dataclass(frozen=True)
class ReplayRandomness:
    exponential_threshold: np.ndarray
    residual_uniform: np.ndarray

    @property
    def replicates(self) -> int:
        return self.exponential_threshold.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sugar", default="Glu", choices=["Glu"])
    parser.add_argument("--fit-max-lag", type=int, default=15)
    parser.add_argument("--plot-max-lag", type=int, default=30)
    parser.add_argument(
        "--objective",
        choices=["birth", "both"],
        default="birth",
        help="Optimize birth ACF only, or birth and time ACFs equally.",
    )
    parser.add_argument("--training-replicates", type=int, default=3)
    parser.add_argument("--validation-replicates", type=int, default=300)
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--seed", type=int, default=731_003)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def build_design(real: pd.DataFrame) -> ReplayDesign:
    groups = [
        group.sort_values("generation")
        for _, group in real.groupby("lineage", sort=False)
        if len(group) >= 3
    ]
    lengths = np.asarray([len(group) for group in groups], dtype=int)
    max_length = int(lengths.max())
    alpha = np.full((len(groups), max_length), np.nan)
    fraction = np.full_like(alpha, np.nan)
    initial_birth = np.empty(len(groups), dtype=float)
    for index, group in enumerate(groups):
        length = len(group)
        alpha[index, :length] = group.alpha.to_numpy(float)
        fraction[index, :length] = group.f.to_numpy(float)
        initial_birth[index] = float(group.length_birth.iloc[0])
    return ReplayDesign(initial_birth, alpha, fraction, lengths)


def make_randomness(
    replicates: int, design: ReplayDesign, seed: int
) -> ReplayRandomness:
    rng = np.random.default_rng(seed)
    shape = (replicates, *design.shape)
    return ReplayRandomness(
        exponential_threshold=rng.exponential(size=shape),
        residual_uniform=rng.uniform(size=shape),
    )


def inverse_cumulative_hazard(
    birth: np.ndarray,
    threshold: np.ndarray,
    u: float,
    omega2: float,
    eta: float,
) -> np.ndarray:
    """Invert H(s|b)=E in parallel using safeguarded bisection."""
    birth = np.asarray(birth, dtype=float)
    threshold = np.asarray(threshold, dtype=float)
    low = np.zeros_like(birth)
    switch_time = np.log1p(u / np.maximum(birth, EPS))
    high = switch_time + 2.0 * (threshold + 2.0) / omega2 + 1.0
    for _ in range(12):
        cumulative = cumulative_hazard_growth_time(
            high, birth, u, omega2, eta
        )
        unresolved = cumulative < threshold
        if not np.any(unresolved):
            break
        high[unresolved] *= 2.0
    else:
        raise FloatingPointError("Could not bracket a growth-time event")
    for _ in range(32):
        middle = 0.5 * (low + high)
        cumulative = cumulative_hazard_growth_time(
            middle, birth, u, omega2, eta
        )
        below = cumulative < threshold
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    return 0.5 * (low + high)


def sample_truncated_residual_from_uniform(
    growth_time: np.ndarray,
    uniform: np.ndarray,
    location: float,
    scale: float,
) -> np.ndarray:
    lower_standardized = (-growth_time - location) / scale
    lower_probability = student_t.cdf(lower_standardized, RESIDUAL_DF)
    probability = lower_probability + uniform * (1.0 - lower_probability)
    probability = np.clip(
        probability, np.finfo(float).eps, 1.0 - np.finfo(float).eps
    )
    return location + scale * student_t.ppf(probability, RESIDUAL_DF)


def simulate_one_replay(
    physical_params: np.ndarray,
    residual_params: np.ndarray,
    design: ReplayDesign,
    exponential_threshold: np.ndarray,
    residual_uniform: np.ndarray,
    max_lag: int,
    return_values: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    u, omega2, eta = map(float, physical_params)
    residual_location, residual_scale = map(float, residual_params)
    n_lineages, max_length = design.shape
    births = np.full((n_lineages, max_length), np.nan)
    times = np.full_like(births, np.nan)
    finals = np.full_like(births, np.nan)
    growth_times = np.full_like(births, np.nan)
    current_birth = design.initial_birth.copy()
    for generation in range(max_length):
        active = design.lengths > generation
        if not np.any(active):
            break
        birth = current_birth[active]
        growth_time = inverse_cumulative_hazard(
            birth,
            exponential_threshold[active, generation],
            u,
            omega2,
            eta,
        )
        epsilon = sample_truncated_residual_from_uniform(
            growth_time,
            residual_uniform[active, generation],
            residual_location,
            residual_scale,
        )
        endpoint_log_growth = growth_time + epsilon
        if (
            np.any(~np.isfinite(endpoint_log_growth))
            or np.any(np.abs(endpoint_log_growth) > 50.0)
        ):
            raise FloatingPointError("Unstable endpoint growth")
        final = birth * np.exp(endpoint_log_growth)
        births[active, generation] = birth
        times[active, generation] = (
            growth_time / design.alpha[active, generation]
        )
        finals[active, generation] = final
        growth_times[active, generation] = growth_time
        continues = design.lengths > generation + 1
        if np.any(continues):
            next_birth = (
                design.fraction[continues, generation + 1]
                * finals[continues, generation]
            )
            if (
                np.any(~np.isfinite(next_birth))
                or np.any(next_birth <= 0.0)
                or np.any(next_birth > 1e8)
            ):
                raise FloatingPointError("Unstable birth-size recursion")
            current_birth[continues] = np.maximum(next_birth, EPS)

    time_pieces = [
        acf_1d(times[index, :length], max_lag)
        for index, length in enumerate(design.lengths)
    ]
    birth_pieces = [
        acf_1d(births[index, :length], max_lag)
        for index, length in enumerate(design.lengths)
    ]
    values = None
    if return_values:
        mask = np.arange(max_length)[None, :] < design.lengths[:, None]
        values = {
            "time": times[mask],
            "length_birth": births[mask],
            "length_final": finals[mask],
            "growth_time": growth_times[mask],
        }
        values["added_size"] = (
            values["length_final"] - values["length_birth"]
        )
    return (
        average_acfs(time_pieces, max_lag),
        average_acfs(birth_pieces, max_lag),
        values,
    )


def simulate_replays(
    physical_params: np.ndarray,
    residual_params: np.ndarray,
    design: ReplayDesign,
    randomness: ReplayRandomness,
    max_lag: int,
) -> dict[str, np.ndarray]:
    result = {
        "time": np.full((randomness.replicates, max_lag + 1), np.nan),
        "length_birth": np.full(
            (randomness.replicates, max_lag + 1), np.nan
        ),
    }
    for replicate in range(randomness.replicates):
        time_acf, birth_acf, _ = simulate_one_replay(
            physical_params,
            residual_params,
            design,
            randomness.exponential_threshold[replicate],
            randomness.residual_uniform[replicate],
            max_lag,
        )
        result["time"][replicate] = time_acf
        result["length_birth"][replicate] = birth_acf
    return result


def loss_components(
    replay: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
    fit_max_lag: int,
    objective: str,
) -> dict[str, float]:
    lag_slice = slice(1, fit_max_lag + 1)
    components = {}
    for variable in ["time", "length_birth"]:
        prediction = np.nanmean(replay[variable], axis=0)
        difference = (
            prediction[lag_slice] - observed[variable][lag_slice]
        )
        components[variable] = float(np.mean(difference**2))
    components["total"] = components["length_birth"]
    if objective == "both":
        components["total"] += components["time"]
    return components


def summarize_curve(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmedian(matrix, axis=0),
        np.nanpercentile(matrix, 2.5, axis=0),
        np.nanpercentile(matrix, 97.5, axis=0),
    )


def curve_metrics(
    observed: np.ndarray,
    matrix: np.ndarray,
    fit_max_lag: int,
    plot_max_lag: int,
) -> dict[str, float]:
    median, lower, upper = summarize_curve(matrix)
    row: dict[str, float] = {
        "lag1_observed": float(observed[1]),
        "lag1_median": float(median[1]),
        "lag1_q2.5": float(lower[1]),
        "lag1_q97.5": float(upper[1]),
    }
    for label, final_lag in [
        ("fit", fit_max_lag),
        ("all", plot_max_lag),
    ]:
        difference = median[1 : final_lag + 1] - observed[1 : final_lag + 1]
        row[f"mae_{label}"] = float(np.mean(np.abs(difference)))
        row[f"rmse_{label}"] = float(np.sqrt(np.mean(difference**2)))
        row[f"coverage_{label}"] = float(
            np.mean(
                (observed[1 : final_lag + 1] >= lower[1 : final_lag + 1])
                & (observed[1 : final_lag + 1] <= upper[1 : final_lag + 1])
            )
        )
    return row


def make_plot(
    output_dir: Path,
    observed: dict[str, np.ndarray],
    likelihood_replay: dict[str, np.ndarray],
    optimized_replay: dict[str, np.ndarray],
    optimized_params: np.ndarray,
    fit_max_lag: int,
    max_lag: int,
    objective: str,
) -> pd.DataFrame:
    lags = np.arange(max_lag + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    fig.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
    rows: list[dict[str, object]] = []
    for axis, (variable, title) in zip(
        axes,
        [
            ("time", "Division-time ACF"),
            ("length_birth", "Birth-length ACF"),
        ],
    ):
        fitted_median, fitted_lower, fitted_upper = summarize_curve(
            likelihood_replay[variable]
        )
        optimized_median, optimized_lower, optimized_upper = summarize_curve(
            optimized_replay[variable]
        )
        axis.fill_between(
            lags,
            fitted_lower,
            fitted_upper,
            color="tab:orange",
            alpha=0.18,
            label="Likelihood-fit 95% band",
        )
        axis.plot(
            lags,
            fitted_median,
            color="tab:orange",
            ls="--",
            lw=1.4,
            label="Likelihood-fit median",
        )
        axis.fill_between(
            lags,
            optimized_lower,
            optimized_upper,
            color="tab:green",
            alpha=0.22,
            label="ACF-fit 95% band",
        )
        axis.plot(
            lags,
            optimized_median,
            color="tab:green",
            marker="o",
            ms=3,
            lw=1.5,
            label="ACF-fit median",
        )
        axis.plot(
            lags,
            observed[variable],
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
        for fit_name, replay in [
            ("likelihood", likelihood_replay),
            ("acf_optimized", optimized_replay),
        ]:
            rows.append(
                {
                    "variable": variable,
                    "fit": fit_name,
                    **curve_metrics(
                        observed[variable],
                        replay[variable],
                        fit_max_lag,
                        max_lag,
                    ),
                }
            )
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
    fig.suptitle(
        "Glu: alpha-scaled hazard calibrated to "
        + ("birth ACF" if objective == "birth" else "measured-input ACFs")
        + "\n"
        + rf"$u={optimized_params[0]:.3g}$, "
        + rf"$\omega_2={optimized_params[1]:.3g}$, "
        + rf"$\eta={optimized_params[2]:.3g}$",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.77))
    fig.savefig(
        output_dir / "acf_optimized_measured_inputs.png",
        dpi=180,
        facecolor="white",
        transparent=False,
    )
    fig.savefig(
        output_dir / "acf_optimized_measured_inputs.pdf",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "acf_optimization_metrics.csv", index=False)
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / "alpha_scaled_hill_time_hazard"
        / (
            "acf_optimization_birth_measured_inputs"
            if args.objective == "birth"
            else "acf_optimization_measured_inputs"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    real, _ = load_sugar(args.sugar)
    design = build_design(real)
    max_lag = min(args.plot_max_lag, int(design.lengths.max()) - 1)
    if not 1 <= args.fit_max_lag <= max_lag:
        raise ValueError("fit-max-lag must be between 1 and plot-max-lag")
    observed = {
        variable: observed_acf(real, variable, max_lag)
        for variable in ["time", "length_birth"]
    }
    if not all(np.isclose(values[0], 1.0) for values in observed.values()):
        raise RuntimeError("Observed ACFs are not normalized to one at lag zero")

    posterior = pd.read_csv(
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / "alpha_scaled_hill_time_hazard"
        / "posterior_summary.csv"
    ).set_index("parameter")
    likelihood_params = posterior.loc[PARAMETER_NAMES, "median"].to_numpy(float)
    residual_params = posterior.loc[
        ["mu_epsilon", "sigma_epsilon"], "median"
    ].to_numpy(float)
    training_randomness = make_randomness(
        args.training_replicates, design, args.seed
    )
    history: list[dict[str, float]] = []

    def objective(log_params: np.ndarray) -> float:
        physical = np.exp(log_params)
        try:
            replay = simulate_replays(
                physical,
                residual_params,
                design,
                training_randomness,
                args.fit_max_lag,
            )
            components = loss_components(
                replay, observed, args.fit_max_lag, args.objective
            )
            return components["total"]
        except (FloatingPointError, OverflowError, ValueError):
            return 1e6

    def callback(values: np.ndarray, convergence: float) -> bool:
        physical = np.exp(values)
        history.append(
            {
                **dict(zip(PARAMETER_NAMES, physical)),
                "loss": objective(values),
                "convergence": float(convergence),
            }
        )
        print(
            f"generation={len(history):02d} loss={history[-1]['loss']:.6g} "
            f"u={physical[0]:.5g} omega2={physical[1]:.5g} "
            f"eta={physical[2]:.5g}",
            flush=True,
        )
        return False

    result = differential_evolution(
        objective,
        bounds=[tuple(row) for row in np.log(PHYSICAL_BOUNDS)],
        seed=args.seed + 1,
        popsize=args.popsize,
        maxiter=args.maxiter,
        tol=0.003,
        polish=True,
        workers=1,
        updating="immediate",
        callback=callback,
    )
    optimized_params = np.exp(result.x)
    pd.DataFrame(history).to_csv(
        output_dir / "optimization_history.csv", index=False
    )

    validation_randomness = make_randomness(
        args.validation_replicates, design, args.seed + 2
    )
    likelihood_replay = simulate_replays(
        likelihood_params,
        residual_params,
        design,
        validation_randomness,
        max_lag,
    )
    optimized_replay = simulate_replays(
        optimized_params,
        residual_params,
        design,
        validation_randomness,
        max_lag,
    )
    metrics = make_plot(
        output_dir,
        observed,
        likelihood_replay,
        optimized_replay,
        optimized_params,
        args.fit_max_lag,
        max_lag,
        args.objective,
    )
    table: dict[str, np.ndarray] = {"lag": np.arange(max_lag + 1)}
    for variable in ["time", "length_birth"]:
        table[f"observed_{variable}"] = observed[variable]
        for name, replay in [
            ("likelihood", likelihood_replay),
            ("acf_optimized", optimized_replay),
        ]:
            median, lower, upper = summarize_curve(replay[variable])
            table[f"{name}_median_{variable}"] = median
            table[f"{name}_q2.5_{variable}"] = lower
            table[f"{name}_q97.5_{variable}"] = upper
    pd.DataFrame(table).to_csv(
        output_dir / "acf_optimized_measured_inputs.csv", index=False
    )

    _, _, fitted_values = simulate_one_replay(
        likelihood_params,
        residual_params,
        design,
        validation_randomness.exponential_threshold[0],
        validation_randomness.residual_uniform[0],
        max_lag,
        return_values=True,
    )
    _, _, optimized_values = simulate_one_replay(
        optimized_params,
        residual_params,
        design,
        validation_randomness.exponential_threshold[0],
        validation_randomness.residual_uniform[0],
        max_lag,
        return_values=True,
    )
    assert fitted_values is not None and optimized_values is not None
    marginal_rows = []
    observed_values = {
        "time": np.concatenate(
            [
                group.time.to_numpy(float)
                for _, group in real.groupby("lineage", sort=False)
                if len(group) >= 3
            ]
        ),
        "length_birth": np.concatenate(
            [
                group.length_birth.to_numpy(float)
                for _, group in real.groupby("lineage", sort=False)
                if len(group) >= 3
            ]
        ),
    }
    for variable in ["time", "length_birth"]:
        marginal_rows.append(
            {
                "variable": variable,
                "observed_mean": np.mean(observed_values[variable]),
                "observed_sd": np.std(observed_values[variable], ddof=1),
                "likelihood_mean": np.mean(fitted_values[variable]),
                "likelihood_sd": np.std(fitted_values[variable], ddof=1),
                "likelihood_ks": ks_2samp(
                    observed_values[variable], fitted_values[variable]
                ).statistic,
                "acf_optimized_mean": np.mean(optimized_values[variable]),
                "acf_optimized_sd": np.std(
                    optimized_values[variable], ddof=1
                ),
                "acf_optimized_ks": ks_2samp(
                    observed_values[variable], optimized_values[variable]
                ).statistic,
            }
        )
    pd.DataFrame(marginal_rows).to_csv(
        output_dir / "acf_optimization_marginal_check.csv", index=False
    )

    fitted_hazard_log_likelihood = float(
        np.sum(
            log_time_density(
                real.time.to_numpy(float),
                real.length_birth.to_numpy(float),
                real.alpha.to_numpy(float),
                *likelihood_params,
            )
        )
    )
    optimized_hazard_log_likelihood = float(
        np.sum(
            log_time_density(
                real.time.to_numpy(float),
                real.length_birth.to_numpy(float),
                real.alpha.to_numpy(float),
                *optimized_params,
            )
        )
    )
    diagnostics = {
        "sugar": args.sugar,
        "purpose": "ACF-targeted sanity check; not Bayesian inference",
        "objective": (
            (
                "mean squared birth-ACF error"
                if args.objective == "birth"
                else "equal-weight sum of mean squared time-ACF and "
                "birth-ACF errors"
            )
            + f" over lags 1..{args.fit_max_lag}"
        ),
        "measured_inputs_replayed": ["alpha", "f"],
        "lineages": int(len(design.lengths)),
        "observations": int(np.sum(design.lengths)),
        "likelihood_posterior_median": dict(
            zip(PARAMETER_NAMES, likelihood_params)
        ),
        "optimized_parameters": dict(zip(PARAMETER_NAMES, optimized_params)),
        "fixed_endpoint_residual_parameters": dict(
            zip(["mu_epsilon", "sigma_epsilon"], residual_params)
        ),
        "hazard_log_likelihood": {
            "likelihood_posterior_median": fitted_hazard_log_likelihood,
            "acf_optimized": optimized_hazard_log_likelihood,
            "difference": (
                optimized_hazard_log_likelihood
                - fitted_hazard_log_likelihood
            ),
        },
        "optimizer": {
            "success": bool(result.success),
            "message": result.message,
            "objective": float(result.fun),
            "evaluations": int(result.nfev),
            "iterations": int(result.nit),
        },
        "run_config": vars(args) | {"output_dir": str(output_dir)},
        "metrics": metrics.to_dict(orient="records"),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    with (output_dir / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
