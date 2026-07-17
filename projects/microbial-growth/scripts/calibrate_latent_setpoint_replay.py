#!/usr/bin/env python3
"""Calibrate a two-parameter latent set point to Glu and Lac summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from analyze_lineage_size_persistence import half_means  # noqa: E402
from latent_setpoint_endpoint_residual import (  # noqa: E402
    simulate_measured_input_replay,
)
from loglogistic_endpoint_residual import pivoted_to_physical_params  # noqa: E402
from run_loglogistic_endpoint_residual import (  # noqa: E402
    load_endpoint_data,
    mean_acf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sugars", nargs="+", default=["Glu", "Lac"])
    parser.add_argument("--fit-max-lag", type=int, default=15)
    parser.add_argument("--plot-max-lag", type=int, default=30)
    parser.add_argument("--training-replicates", type=int, default=4)
    parser.add_argument("--validation-replicates", type=int, default=300)
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--seed", type=int, default=517_200)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def model_dir(sugar: str) -> Path:
    return (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / sugar
        / "loglogistic_endpoint_residual"
    )


def lineage_inputs(real: pd.DataFrame) -> list[tuple[float, np.ndarray, np.ndarray]]:
    return [
        (
            float(group.length_birth.iloc[0]),
            group.alpha.to_numpy(float),
            group.f.to_numpy(float),
        )
        for _, group in real.sort_values(["lineage", "generation"]).groupby(
            "lineage", sort=False
        )
    ]


def half_correlation(frame: pd.DataFrame, min_length: int = 8) -> float:
    groups = [
        group
        for _, group in frame.groupby("lineage", sort=False)
        if len(group) >= min_length
    ]
    first, second = half_means(groups)
    return float(pearsonr(first, second).statistic)


def simulate_summary(
    params: np.ndarray,
    phi: float,
    sigma_z: float,
    inputs: list[tuple[float, np.ndarray, np.ndarray]],
    max_lag: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    frame = pd.DataFrame(
        simulate_measured_input_replay(params, phi, sigma_z, inputs, seed)
    )
    return (
        frame,
        mean_acf(frame, "length_birth", max_lag),
        mean_acf(frame, "time", max_lag),
        half_correlation(frame),
    )


def calibrate_sugar(
    sugar: str,
    args: argparse.Namespace,
    output: Path,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    real, _, _, _ = load_endpoint_data(sugar)
    inputs = lineage_inputs(real)
    directory = model_dir(sugar)
    chain = pd.read_csv(directory / "flat_chain.csv").to_numpy(float)
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    pivot = float(diagnostics["parameterization"]["birth_pivot"])
    base_params = pivoted_to_physical_params(np.median(chain, axis=0), pivot)
    max_lag = min(args.plot_max_lag, max(len(item[1]) for item in inputs) - 1)
    observed_birth = mean_acf(real, "length_birth", max_lag)
    observed_time = mean_acf(real, "time", max_lag)
    observed_half = half_correlation(real)
    observed_birth_mean = float(real.length_birth.mean())
    observed_birth_sd = float(real.length_birth.std(ddof=1))
    training_seeds = [seed + index for index in range(args.training_replicates)]

    def objective(values: np.ndarray) -> float:
        phi, log_sigma = map(float, values)
        sigma_z = math.exp(log_sigma)
        birth_curves = []
        time_curves = []
        half_values = []
        means = []
        sds = []
        try:
            for replay_seed in training_seeds:
                frame, birth_acf, time_acf, half = simulate_summary(
                    base_params,
                    phi,
                    sigma_z,
                    inputs,
                    args.fit_max_lag,
                    replay_seed,
                )
                birth_curves.append(birth_acf)
                time_curves.append(time_acf)
                half_values.append(half)
                means.append(float(frame.length_birth.mean()))
                sds.append(float(frame.length_birth.std(ddof=1)))
        except (FloatingPointError, OverflowError, ValueError):
            return 1e6
        birth = np.mean(birth_curves, axis=0)
        time_curve = np.mean(time_curves, axis=0)
        lag_slice = slice(1, args.fit_max_lag + 1)
        birth_mse = np.mean(
            (birth[lag_slice] - observed_birth[lag_slice]) ** 2
        )
        time_mse = np.mean(
            (time_curve[lag_slice] - observed_time[lag_slice]) ** 2
        )
        half_error = (np.mean(half_values) - observed_half) ** 2
        mean_error = (
            (np.mean(means) - observed_birth_mean) / observed_birth_sd
        ) ** 2
        sd_error = math.log(max(np.mean(sds), 1e-8) / observed_birth_sd) ** 2
        return float(
            birth_mse
            + 0.25 * time_mse
            + 0.02 * half_error
            + 0.005 * mean_error
            + 0.005 * sd_error
        )

    history: list[dict[str, float]] = []

    def callback(values: np.ndarray, convergence: float) -> bool:
        row = {
            "phi": float(values[0]),
            "sigma_z": math.exp(float(values[1])),
            "objective": objective(values),
            "convergence": float(convergence),
        }
        history.append(row)
        print(
            f"{sugar} generation={len(history):02d} "
            f"loss={row['objective']:.6g} phi={row['phi']:.4f} "
            f"sigma_z={row['sigma_z']:.4f}",
            flush=True,
        )
        return False

    result = differential_evolution(
        objective,
        bounds=[(0.0, 0.999), (math.log(1e-3), math.log(0.8))],
        seed=seed + 100,
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=0.004,
        polish=True,
        callback=callback,
        workers=1,
        updating="immediate",
    )
    phi = float(result.x[0])
    sigma_z = math.exp(float(result.x[1]))
    pd.DataFrame(history).to_csv(
        output / f"{sugar}_latent_setpoint_optimization_history.csv",
        index=False,
    )

    rng = np.random.default_rng(seed + 200)
    latent_birth = np.full((args.validation_replicates, max_lag + 1), np.nan)
    latent_time = np.full_like(latent_birth, np.nan)
    latent_half = np.full(args.validation_replicates, np.nan)
    baseline_birth = np.full_like(latent_birth, np.nan)
    baseline_time = np.full_like(latent_birth, np.nan)
    baseline_half = np.full(args.validation_replicates, np.nan)
    latent_means = np.full(args.validation_replicates, np.nan)
    latent_sds = np.full(args.validation_replicates, np.nan)
    baseline_means = np.full(args.validation_replicates, np.nan)
    baseline_sds = np.full(args.validation_replicates, np.nan)
    for replicate in range(args.validation_replicates):
        replay_seed = int(rng.integers(np.iinfo(np.int32).max))
        frame, birth, time_curve, half = simulate_summary(
            base_params, phi, sigma_z, inputs, max_lag, replay_seed
        )
        baseline, base_birth, base_time, base_half = simulate_summary(
            base_params, 0.0, 0.0, inputs, max_lag, replay_seed
        )
        latent_birth[replicate] = birth
        latent_time[replicate] = time_curve
        latent_half[replicate] = half
        baseline_birth[replicate] = base_birth
        baseline_time[replicate] = base_time
        baseline_half[replicate] = base_half
        latent_means[replicate] = frame.length_birth.mean()
        latent_sds[replicate] = frame.length_birth.std(ddof=1)
        baseline_means[replicate] = baseline.length_birth.mean()
        baseline_sds[replicate] = baseline.length_birth.std(ddof=1)

    table: dict[str, np.ndarray] = {"lag": np.arange(max_lag + 1)}
    for variable, observed, latent, baseline in [
        ("length_birth", observed_birth, latent_birth, baseline_birth),
        ("time", observed_time, latent_time, baseline_time),
    ]:
        table[f"observed_{variable}"] = observed
        for name, matrix in [("latent", latent), ("baseline", baseline)]:
            table[f"{name}_median_{variable}"] = np.nanmedian(matrix, axis=0)
            table[f"{name}_q2.5_{variable}"] = np.nanpercentile(
                matrix, 2.5, axis=0
            )
            table[f"{name}_q97.5_{variable}"] = np.nanpercentile(
                matrix, 97.5, axis=0
            )
    table_frame = pd.DataFrame(table)
    table_frame.to_csv(
        output / f"{sugar}_latent_setpoint_acf.csv", index=False
    )
    row: dict[str, object] = {
        "sugar": sugar,
        "phi": phi,
        "sigma_z": sigma_z,
        "latent_correlation_time_generations": (
            -1.0 / math.log(phi) if phi > 0.0 else 0.0
        ),
        "objective": float(result.fun),
        "optimizer_success": bool(result.success),
        "observed_half_correlation": observed_half,
        "baseline_half_median": np.median(baseline_half),
        "latent_half_median": np.median(latent_half),
        "latent_half_q2.5": np.percentile(latent_half, 2.5),
        "latent_half_q97.5": np.percentile(latent_half, 97.5),
        "observed_birth_mean": observed_birth_mean,
        "observed_birth_sd": observed_birth_sd,
        "baseline_birth_mean_median": np.median(baseline_means),
        "baseline_birth_sd_median": np.median(baseline_sds),
        "latent_birth_mean_median": np.median(latent_means),
        "latent_birth_sd_median": np.median(latent_sds),
    }
    for variable, observed, latent, baseline in [
        ("birth", observed_birth, latent_birth, baseline_birth),
        ("time", observed_time, latent_time, baseline_time),
    ]:
        finite = slice(1, max_lag + 1)
        latent_median = np.nanmedian(latent, axis=0)
        baseline_median = np.nanmedian(baseline, axis=0)
        row[f"latent_{variable}_acf_mae"] = float(
            np.mean(np.abs(latent_median[finite] - observed[finite]))
        )
        row[f"baseline_{variable}_acf_mae"] = float(
            np.mean(np.abs(baseline_median[finite] - observed[finite]))
        )
    return row, table_frame


def make_plot(summary: pd.DataFrame, tables: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(len(tables), 2, figsize=(12.5, 8.2), sharex=True)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor("white")
    handles = labels = None
    for row_index, (sugar, table) in enumerate(tables.items()):
        lags = table.lag.to_numpy(int)
        for column, (variable, title) in enumerate(
            [("length_birth", "Birth-length ACF"), ("time", "Division-time ACF")]
        ):
            axis = axes[row_index, column]
            axis.set_facecolor("white")
            observed = table[f"observed_{variable}"].to_numpy(float)
            latent_median = table[f"latent_median_{variable}"].to_numpy(float)
            latent_lower = table[f"latent_q2.5_{variable}"].to_numpy(float)
            latent_upper = table[f"latent_q97.5_{variable}"].to_numpy(float)
            baseline = table[f"baseline_median_{variable}"].to_numpy(float)
            axis.fill_between(
                lags,
                latent_lower,
                latent_upper,
                color="tab:green",
                alpha=0.22,
                label="Latent-set-point 95% band",
            )
            axis.plot(
                lags,
                latent_median,
                color="tab:green",
                marker="o",
                ms=2.6,
                label="Latent-set-point median",
            )
            axis.plot(
                lags,
                baseline,
                color="0.48",
                ls="--",
                label="Measured-input baseline",
            )
            axis.plot(
                lags,
                observed,
                color="tab:blue",
                marker="o",
                ms=2.6,
                lw=1.8,
                label="Observed",
            )
            axis.axhline(0.0, color="black", lw=0.7)
            axis.set_title(f"{sugar}: {title}")
            axis.set_ylabel("Mean sample ACF")
            axis.grid(axis="y", alpha=0.12)
            if row_index == len(tables) - 1:
                axis.set_xlabel("Generation lag")
            if handles is None:
                handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        fontsize=8,
    )
    fig.suptitle("Slow latent division-scale replay", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        output / "Glu_Lac_latent_setpoint_acf.png",
        dpi=180,
        facecolor="white",
        transparent=False,
    )
    fig.savefig(
        output / "Glu_Lac_latent_setpoint_acf.pdf",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir or PROJECT_DIR / "outputs" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    tables = {}
    for index, sugar in enumerate(args.sugars):
        row, table = calibrate_sugar(
            sugar, args, output, args.seed + 100_000 * index
        )
        rows.append(row)
        tables[sugar] = table
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "Glu_Lac_latent_setpoint_summary.csv", index=False)
    make_plot(summary, tables, output)
    with (output / "Glu_Lac_latent_setpoint_config.json").open("w") as handle:
        json.dump(vars(args) | {"output_dir": str(output)}, handle, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
