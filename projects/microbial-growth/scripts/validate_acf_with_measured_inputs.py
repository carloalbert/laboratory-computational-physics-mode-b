#!/usr/bin/env python3
"""Validate state ACFs by replaying the measured alpha and f sequences.

The fitted division-size and endpoint-residual mechanisms remain stochastic,
but each simulated lineage uses its observed initial birth length, lineage
length, growth-rate sequence and division-factor sequence.  The resulting
conditional ACF band is compared with the ordinary IID-input posterior-
predictive band saved by the inference pipeline.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from loglogistic_endpoint_residual import (  # noqa: E402
    EPS,
    pivoted_to_physical_params,
    sample_added_size,
    sample_valid_residual,
)


SUGARS = ["Ace", "DMa", "Glu", "Gly", "Lac", "Pyr", "Rha", "Suc", "Xyl"]
MODEL_NAME = "loglogistic_endpoint_residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sugar",
        choices=["all", *SUGARS],
        default="all",
        help="Sugar to process; default processes all nine.",
    )
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--seed", type=int, default=170_000)
    return parser.parse_args()


def output_dir(sugar: str) -> Path:
    return (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / sugar
        / MODEL_NAME
    )


def load_sugar(sugar: str) -> tuple[pd.DataFrame, float]:
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
        & frame.source.eq(sugar)
    ].sort_values(["lineage", "generation"])
    generation_zero = frame[frame.generation.eq(0)].groupby(
        "lineage", sort=False
    ).first()
    frame = frame[
        frame.lineage.isin(generation_zero.index)
    ].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No valid observations for {sugar}")

    grouped = frame.groupby("lineage", sort=False)
    transition = frame.copy()
    transition["next_generation"] = grouped.generation.shift(-1)
    transition["next_length_birth"] = grouped.length_birth.shift(-1)
    transition["next_f"] = grouped.f.shift(-1)
    transition = transition[
        transition.next_generation.eq(transition.generation + 1.0)
    ].dropna(subset=["next_length_birth", "next_f"])
    final = transition.next_length_birth / transition.next_f
    added = final - transition.length_birth
    valid = (
        (final > transition.length_birth)
        & np.isfinite(final)
        & np.isfinite(added)
    )
    birth_pivot = float(transition.loc[valid, "length_birth"].mean())
    if not np.isfinite(birth_pivot):
        raise ValueError(f"Could not reconstruct the birth pivot for {sugar}")
    return frame, birth_pivot


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
        result[lag] = (
            np.sum(centered[:-lag] * centered[lag:]) / denominator
        )
    return result


def average_acfs(pieces: list[np.ndarray], max_lag: int) -> np.ndarray:
    array = np.asarray(pieces)
    counts = np.sum(np.isfinite(array), axis=0)
    return np.divide(
        np.nansum(array, axis=0),
        counts,
        out=np.full(max_lag + 1, np.nan),
        where=counts > 0,
    )


def observed_acf(frame: pd.DataFrame, variable: str, max_lag: int) -> np.ndarray:
    pieces = [
        acf_1d(group[variable].to_numpy(float), max_lag)
        for _, group in frame.groupby("lineage", sort=False)
        if len(group) >= 3
    ]
    return average_acfs(pieces, max_lag)


def measured_input_replay_acf(
    physical_params: np.ndarray,
    lineage_inputs: list[tuple[float, np.ndarray, np.ndarray]],
    max_lag: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate p and epsilon while replaying observed alpha and f in order."""
    u, omega2, eta = map(float, physical_params[:3])
    mu_epsilon, sigma_epsilon = map(float, physical_params[-2:])
    time_pieces: list[np.ndarray] = []
    birth_pieces: list[np.ndarray] = []
    for initial_birth, alphas, fractions in lineage_inputs:
        n_observations = len(alphas)
        births = np.empty(n_observations, dtype=float)
        times = np.empty(n_observations, dtype=float)
        birth = float(initial_birth)
        for generation in range(n_observations):
            births[generation] = birth
            p = sample_added_size(birth, u, omega2, eta, rng)
            endpoint_log_growth = math.log1p(p / birth)
            epsilon = sample_valid_residual(
                endpoint_log_growth, mu_epsilon, sigma_epsilon, rng
            )
            times[generation] = (
                endpoint_log_growth - epsilon
            ) / alphas[generation]
            if generation + 1 < n_observations:
                # f on row n+1 is the factor that produces b_(n+1).
                birth = max(fractions[generation + 1] * (birth + p), EPS)
        if n_observations >= 3:
            time_pieces.append(acf_1d(times, max_lag))
            birth_pieces.append(acf_1d(births, max_lag))
    return (
        average_acfs(time_pieces, max_lag),
        average_acfs(birth_pieces, max_lag),
    )


def summarize_replays(
    replay: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmedian(replay, axis=0),
        np.nanpercentile(replay, 2.5, axis=0),
        np.nanpercentile(replay, 97.5, axis=0),
    )


def analyze_sugar(
    sugar: str, n_replicates: int, requested_max_lag: int, seed: int
) -> pd.DataFrame:
    real, birth_pivot = load_sugar(sugar)
    groups = [
        group.sort_values("generation")
        for _, group in real.groupby("lineage", sort=False)
    ]
    design = np.asarray([len(group) for group in groups], dtype=int)
    max_lag = min(requested_max_lag, int(design.max()) - 1)
    lineage_inputs = [
        (
            float(group.length_birth.iloc[0]),
            group.alpha.to_numpy(float),
            group.f.to_numpy(float),
        )
        for group in groups
    ]

    chain = pd.read_csv(output_dir(sugar) / "flat_chain.csv").to_numpy(float)
    baseline = pd.read_csv(output_dir(sugar) / "acf_observation_matched.csv")
    baseline = baseline[baseline.lag.le(max_lag)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(chain), size=n_replicates)
    replay = {
        "time": np.full((n_replicates, max_lag + 1), np.nan),
        "length_birth": np.full((n_replicates, max_lag + 1), np.nan),
    }
    for replicate, draw_index in enumerate(draw_indices):
        physical = pivoted_to_physical_params(
            chain[draw_index], birth_pivot
        )
        time_acf, birth_acf = measured_input_replay_acf(
            physical, lineage_inputs, max_lag, rng
        )
        replay["time"][replicate] = time_acf
        replay["length_birth"][replicate] = birth_acf

    lags = np.arange(max_lag + 1)
    contributors = np.asarray(
        [np.sum((design >= 3) & (design > lag)) for lag in lags], dtype=int
    )
    result: dict[str, np.ndarray] = {
        "lag": lags,
        "contributing_lineages": contributors,
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    for axis, (variable, title) in zip(
        axes,
        [
            ("time", "Division-time ACF"),
            ("length_birth", "Birth-length ACF"),
        ],
    ):
        observed = observed_acf(real, variable, max_lag)
        replay_median, replay_lower, replay_upper = summarize_replays(
            replay[variable]
        )
        iid_median = baseline[f"predictive_median_{variable}"].to_numpy(float)
        iid_lower = baseline[f"predictive_q2.5_{variable}"].to_numpy(float)
        iid_upper = baseline[f"predictive_q97.5_{variable}"].to_numpy(float)

        result[f"observed_{variable}"] = observed
        result[f"iid_median_{variable}"] = iid_median
        result[f"iid_q2.5_{variable}"] = iid_lower
        result[f"iid_q97.5_{variable}"] = iid_upper
        result[f"measured_inputs_median_{variable}"] = replay_median
        result[f"measured_inputs_q2.5_{variable}"] = replay_lower
        result[f"measured_inputs_q97.5_{variable}"] = replay_upper

        axis.fill_between(
            lags, iid_lower, iid_upper, color="0.65", alpha=0.22,
            label="IID-input 95% band",
        )
        axis.plot(
            lags, iid_median, color="0.45", ls="--", lw=1.5,
            label="IID-input median",
        )
        axis.fill_between(
            lags, replay_lower, replay_upper, color="tab:green", alpha=0.22,
            label=r"Measured $f,\alpha$: 95% band",
        )
        axis.plot(
            lags, replay_median, color="tab:green", marker="o", ms=3,
            lw=1.5, label=r"Measured $f,\alpha$: median",
        )
        axis.plot(
            lags, observed, color="tab:blue", marker="o", ms=3, lw=2,
            label=f"{sugar} data",
        )
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_title(title)
        axis.set_xlabel("Generation lag")
        axis.set_ylabel("Mean sample ACF")
        axis.set_xticks(np.arange(0, max_lag + 1, 5))
        axis.grid(axis="y", alpha=0.14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
        ncol=3, frameon=False, fontsize=8,
    )
    fig.suptitle(
        f"{sugar}: ACF validation with measured input sequences", y=0.98
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(output_dir(sugar) / "acf_measured_inputs_validation.png", dpi=180)
    plt.close(fig)

    table = pd.DataFrame(result)
    table.to_csv(
        output_dir(sugar) / "acf_measured_inputs_validation.csv", index=False
    )
    return table


def plot_all_sugars(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fig, axes = plt.subplots(
        len(SUGARS), 2, figsize=(12.5, 22), sharex=True, sharey=True
    )
    lag1_rows: list[dict[str, object]] = []
    all_lag_rows: list[dict[str, object]] = []
    handles = labels = None
    for row, sugar in enumerate(SUGARS):
        table = tables[sugar]
        lags = table.lag.to_numpy(int)
        for column, (variable, title) in enumerate(
            [("time", "Division-time ACF"), ("length_birth", "Birth-length ACF")]
        ):
            axis = axes[row, column]
            observed = table[f"observed_{variable}"].to_numpy(float)
            iid_median = table[f"iid_median_{variable}"].to_numpy(float)
            iid_lower = table[f"iid_q2.5_{variable}"].to_numpy(float)
            iid_upper = table[f"iid_q97.5_{variable}"].to_numpy(float)
            replay_median = table[
                f"measured_inputs_median_{variable}"
            ].to_numpy(float)
            replay_lower = table[
                f"measured_inputs_q2.5_{variable}"
            ].to_numpy(float)
            replay_upper = table[
                f"measured_inputs_q97.5_{variable}"
            ].to_numpy(float)
            axis.fill_between(
                lags, iid_lower, iid_upper, color="0.65", alpha=0.20,
                label="IID-input 95% band",
            )
            axis.plot(
                lags, iid_median, color="0.45", ls="--", lw=1.1,
                label="IID-input median",
            )
            axis.fill_between(
                lags, replay_lower, replay_upper, color="tab:green", alpha=0.20,
                label=r"Measured $f,\alpha$: 95% band",
            )
            axis.plot(
                lags, replay_median, color="tab:green", marker="o", ms=2.2,
                lw=1.2, label=r"Measured $f,\alpha$: median",
            )
            axis.plot(
                lags, observed, color="tab:blue", marker="o", ms=2.2,
                lw=1.6, label="Observed",
            )
            axis.axhline(0.0, color="black", lw=0.6)
            axis.set_xlim(0, 30)
            axis.set_ylim(-0.25, 1.05)
            axis.set_xticks(np.arange(0, 31, 5))
            axis.grid(axis="y", alpha=0.12)
            if row == 0:
                axis.set_title(title, fontsize=12)
            if column == 0:
                axis.annotate(
                    sugar, xy=(-0.13, 0.5), xycoords="axes fraction",
                    ha="right", va="center", fontsize=11,
                )
            if handles is None:
                handles, labels = axis.get_legend_handles_labels()
            lag1_rows.append(
                {
                    "sugar": sugar,
                    "variable": variable,
                    "observed_lag1": observed[1],
                    "iid_median_lag1": iid_median[1],
                    "iid_q2.5_lag1": iid_lower[1],
                    "iid_q97.5_lag1": iid_upper[1],
                    "iid_covers_lag1": bool(
                        iid_lower[1] <= observed[1] <= iid_upper[1]
                    ),
                    "measured_inputs_median_lag1": replay_median[1],
                    "measured_inputs_q2.5_lag1": replay_lower[1],
                    "measured_inputs_q97.5_lag1": replay_upper[1],
                    "measured_inputs_cover_lag1": bool(
                        replay_lower[1] <= observed[1] <= replay_upper[1]
                    ),
                }
            )
            nonzero = lags > 0
            finite = (
                nonzero
                & np.isfinite(observed)
                & np.isfinite(iid_median)
                & np.isfinite(replay_median)
            )
            all_lag_rows.append(
                {
                    "sugar": sugar,
                    "variable": variable,
                    "n_nonzero_lags": int(np.sum(finite)),
                    "iid_mean_absolute_error": float(
                        np.mean(np.abs(observed[finite] - iid_median[finite]))
                    ),
                    "measured_inputs_mean_absolute_error": float(
                        np.mean(
                            np.abs(observed[finite] - replay_median[finite])
                        )
                    ),
                    "iid_coverage_fraction": float(
                        np.mean(
                            (observed[finite] >= iid_lower[finite])
                            & (observed[finite] <= iid_upper[finite])
                        )
                    ),
                    "measured_inputs_coverage_fraction": float(
                        np.mean(
                            (observed[finite] >= replay_lower[finite])
                            & (observed[finite] <= replay_upper[finite])
                        )
                    ),
                }
            )

    fig.suptitle(
        "ACF validation using measured growth-rate and division-factor sequences",
        y=0.995, fontsize=15,
    )
    fig.supxlabel("Generation lag", y=0.015)
    fig.supylabel("Mean sample ACF", x=0.015)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.977),
        ncol=3, frameon=False, fontsize=8,
    )
    fig.tight_layout(rect=(0.055, 0.025, 1.0, 0.95), h_pad=0.50)
    summary_dir = PROJECT_DIR / "outputs" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary_dir / "all_sugars_acf_measured_inputs.png", dpi=180)
    fig.savefig(summary_dir / "all_sugars_acf_measured_inputs.pdf")
    plt.close(fig)

    lag1 = pd.DataFrame(lag1_rows)
    lag1.to_csv(
        summary_dir / "all_sugars_acf_measured_inputs_lag1.csv", index=False
    )
    pd.DataFrame(all_lag_rows).to_csv(
        summary_dir / "all_sugars_acf_measured_inputs_metrics.csv", index=False
    )
    return lag1


def main() -> None:
    args = parse_args()
    sugars = SUGARS if args.sugar == "all" else [args.sugar]
    tables = {
        sugar: analyze_sugar(
            sugar,
            args.replicates,
            args.max_lag,
            args.seed + 10_000 * SUGARS.index(sugar),
        )
        for sugar in sugars
    }
    for sugar, table in tables.items():
        print(
            f"{sugar}: birth lag1 observed={table.observed_length_birth.iloc[1]:.3f}, "
            f"measured-input median="
            f"{table.measured_inputs_median_length_birth.iloc[1]:.3f}"
        )
    if sugars == SUGARS:
        lag1 = plot_all_sugars(tables)
        covered = lag1.groupby("variable").measured_inputs_cover_lag1.sum()
        print(
            "Measured-input lag1 coverage: "
            + ", ".join(
                f"{variable} {int(count)}/{len(SUGARS)}"
                for variable, count in covered.items()
            )
        )


if __name__ == "__main__":
    main()
