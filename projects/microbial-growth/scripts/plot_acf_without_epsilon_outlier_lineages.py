#!/usr/bin/env python3
"""Plot Glu/Lac ACF checks after removing epsilon-outlier lineages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from loglogistic_endpoint_residual import (  # noqa: E402
    pivoted_to_physical_params,
    simulate_lineages,
)
from validate_acf_with_measured_inputs import (  # noqa: E402
    acf_1d,
    average_acfs,
    load_sugar,
)


SUGARS = ["Glu", "Lac"]
MODEL_NAME = "loglogistic_endpoint_residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=-5.0)
    parser.add_argument("--seed", type=int, default=220_000)
    return parser.parse_args()


def output_dir(sugar: str) -> Path:
    return (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / sugar
        / MODEL_NAME
    )


def mean_acf(frame: pd.DataFrame, variable: str, max_lag: int) -> np.ndarray:
    pieces = [
        acf_1d(
            group.sort_values("generation")[variable].to_numpy(float),
            max_lag,
        )
        for _, group in frame.groupby("lineage", sort=False)
        if len(group) >= 3
    ]
    return average_acfs(pieces, max_lag)


def epsilon_outlier_lineages(
    real: pd.DataFrame,
    mu_epsilon: float,
    sigma_epsilon: float,
    threshold: float,
) -> set[float]:
    grouped = real.groupby("lineage", sort=False)
    transition = real.copy()
    transition["next_generation"] = grouped.generation.shift(-1)
    transition["next_length_birth"] = grouped.length_birth.shift(-1)
    transition["next_f"] = grouped.f.shift(-1)
    transition = transition[
        transition.next_generation.eq(transition.generation + 1.0)
    ].dropna(subset=["next_length_birth", "next_f"])
    transition["length_final"] = (
        transition.next_length_birth / transition.next_f
    )
    transition = transition[
        transition.length_final.gt(transition.length_birth)
    ].copy()
    transition["epsilon"] = (
        np.log(transition.length_final / transition.length_birth)
        - transition.alpha * transition.time
    )
    transition["epsilon_z"] = (
        transition.epsilon - mu_epsilon
    ) / sigma_epsilon
    return set(
        transition.loc[transition.epsilon_z.lt(threshold), "lineage"]
    )


def analyze_sugar(
    sugar: str,
    n_replicates: int,
    requested_max_lag: int,
    threshold: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    real, birth_pivot = load_sugar(sugar)
    chain_frame = pd.read_csv(output_dir(sugar) / "flat_chain.csv")
    mu_epsilon = float(chain_frame.mu_epsilon.median())
    sigma_epsilon = float(chain_frame.sigma_epsilon.median())
    removed = epsilon_outlier_lineages(
        real, mu_epsilon, sigma_epsilon, threshold
    )
    retained = real[~real.lineage.isin(removed)].copy()
    groups = [
        group.sort_values("generation")
        for _, group in retained.groupby("lineage", sort=False)
        if len(group) >= 3
    ]
    design = np.asarray([len(group) for group in groups], dtype=int)
    initial_lengths = np.asarray(
        [group.length_birth.iloc[0] for group in groups], dtype=float
    )
    max_lag = min(requested_max_lag, int(design.max()) - 1)
    chain = chain_frame.to_numpy(float)
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(chain), size=n_replicates)
    predictive = {
        "time": np.full((n_replicates, max_lag + 1), np.nan),
        "length_birth": np.full((n_replicates, max_lag + 1), np.nan),
    }

    for replicate, draw_index in enumerate(draw_indices):
        simulated = pd.DataFrame(
            simulate_lineages(
                pivoted_to_physical_params(
                    chain[draw_index], birth_pivot
                ),
                initial_lengths,
                len(design),
                int(design.max()),
                int(rng.integers(np.iinfo(np.int32).max)),
                fixed_initial_lengths=True,
            )
        )
        for variable in predictive:
            pieces = [
                acf_1d(
                    simulated.loc[
                        simulated.lineage.eq(lineage), variable
                    ].to_numpy(float)[:length],
                    max_lag,
                )
                for lineage, length in enumerate(design)
            ]
            predictive[variable][replicate] = average_acfs(
                pieces, max_lag
            )

    lags = np.arange(max_lag + 1)
    result: dict[str, np.ndarray] = {
        "lag": lags,
        "contributing_lineages": np.asarray(
            [np.sum(design > lag) for lag in lags], dtype=int
        ),
    }
    for variable in predictive:
        result[f"unfiltered_observed_{variable}"] = mean_acf(
            real, variable, max_lag
        )
        result[f"filtered_observed_{variable}"] = mean_acf(
            retained, variable, max_lag
        )
        result[f"predictive_median_{variable}"] = np.nanmedian(
            predictive[variable], axis=0
        )
        result[f"predictive_q2.5_{variable}"] = np.nanpercentile(
            predictive[variable], 2.5, axis=0
        )
        result[f"predictive_q97.5_{variable}"] = np.nanpercentile(
            predictive[variable], 97.5, axis=0
        )

    table = pd.DataFrame(result)
    table.to_csv(
        output_dir(sugar) / "acf_without_epsilon_outlier_lineages.csv",
        index=False,
    )
    counts = {
        "total": int(real.lineage.nunique()),
        "removed": len(removed),
        "retained": int(retained.lineage.nunique()),
    }
    return table, counts


def make_plot(
    tables: dict[str, pd.DataFrame], counts: dict[str, dict[str, int]], threshold: float
) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True
    )
    specifications = [
        ("time", "Division-time ACF"),
        ("length_birth", "Birth-length ACF"),
    ]
    handles = labels = None
    for row, sugar in enumerate(SUGARS):
        table = tables[sugar]
        lags = table.lag.to_numpy(int)
        for column, (variable, title) in enumerate(specifications):
            axis = axes[row, column]
            unfiltered = table[
                f"unfiltered_observed_{variable}"
            ].to_numpy(float)
            filtered = table[f"filtered_observed_{variable}"].to_numpy(float)
            median = table[f"predictive_median_{variable}"].to_numpy(float)
            lower = table[f"predictive_q2.5_{variable}"].to_numpy(float)
            upper = table[f"predictive_q97.5_{variable}"].to_numpy(float)
            axis.fill_between(
                lags,
                lower,
                upper,
                color="tab:orange",
                alpha=0.23,
                label="Subset-matched 95% predictive band",
            )
            axis.plot(
                lags,
                median,
                color="tab:orange",
                marker="o",
                ms=2.8,
                lw=1.4,
                label="Subset-matched predictive median",
            )
            axis.plot(
                lags,
                unfiltered,
                color="0.60",
                ls="--",
                lw=1.5,
                label="All observed lineages",
            )
            axis.plot(
                lags,
                filtered,
                color="tab:blue",
                marker="o",
                ms=3,
                lw=1.9,
                label="Observed after omission",
            )
            axis.axhline(0.0, color="black", lw=0.7)
            axis.set_xlim(0, 30)
            axis.set_ylim(-0.25, 1.05)
            axis.set_xticks(np.arange(0, 31, 5))
            axis.grid(axis="y", alpha=0.13)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                info = counts[sugar]
                axis.set_ylabel(
                    f"{sugar}  ({info['removed']} of {info['total']} omitted)\n"
                    "Mean sample ACF"
                )
            if row == 1:
                axis.set_xlabel("Generation lag")
            if handles is None:
                handles, labels = axis.get_legend_handles_labels()

    fig.suptitle(
        r"ACFs after omitting lineages with "
        rf"$(\epsilon-\mu_\epsilon)/\sigma_\epsilon < {threshold:g}$",
        y=0.985,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.87))
    summary = PROJECT_DIR / "outputs" / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        summary / "Glu_Lac_acf_without_epsilon_outlier_lineages.png",
        dpi=180,
    )
    fig.savefig(
        summary / "Glu_Lac_acf_without_epsilon_outlier_lineages.pdf"
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    tables: dict[str, pd.DataFrame] = {}
    counts: dict[str, dict[str, int]] = {}
    for index, sugar in enumerate(SUGARS):
        tables[sugar], counts[sugar] = analyze_sugar(
            sugar,
            args.replicates,
            args.max_lag,
            args.threshold,
            args.seed + 10_000 * index,
        )
        table = tables[sugar]
        print(
            f"{sugar}: removed {counts[sugar]['removed']}/"
            f"{counts[sugar]['total']} lineages; birth ACF(1) "
            f"{table.filtered_observed_length_birth.iloc[1]:.4f}, "
            f"predictive median "
            f"{table.predictive_median_length_birth.iloc[1]:.4f}"
        )
    make_plot(tables, counts, args.threshold)


if __name__ == "__main__":
    main()
