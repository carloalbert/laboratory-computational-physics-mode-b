#!/usr/bin/env python3
"""Test for slow lineage-level birth-size persistence beyond the fitted model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from loglogistic_endpoint_residual import (  # noqa: E402
    pivoted_to_physical_params,
    simulate_lineages,
)
from run_loglogistic_endpoint_residual import load_endpoint_data  # noqa: E402


SUGARS = ["Ace", "DMa", "Glu", "Gly", "Lac", "Pyr", "Rha", "Suc", "Xyl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=842_100)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def output_dir(sugar: str) -> Path:
    return (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / sugar
        / "loglogistic_endpoint_residual"
    )


def half_means(
    groups: list[pd.DataFrame], column: str = "length_birth"
) -> tuple[np.ndarray, np.ndarray]:
    first = np.empty(len(groups), dtype=float)
    second = np.empty(len(groups), dtype=float)
    for index, group in enumerate(groups):
        values = np.log(group.sort_values("generation")[column].to_numpy(float))
        half = len(values) // 2
        first[index] = np.mean(values[:half])
        second[index] = np.mean(values[-half:])
    return first, second


def correlations(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    return (
        float(pearsonr(first, second).statistic),
        float(spearmanr(first, second).statistic),
    )


def bootstrap_correlation(
    first: np.ndarray,
    second: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        indices = rng.integers(0, len(first), size=len(first))
        result[replicate] = pearsonr(
            first[indices], second[indices]
        ).statistic
    return result


def analyze_sugar(
    sugar: str,
    min_length: int,
    n_replicates: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    real, _, _, _ = load_endpoint_data(sugar)
    groups = [
        group.sort_values("generation")
        for _, group in real.groupby("lineage", sort=False)
        if len(group) >= min_length
    ]
    if len(groups) < 4:
        raise ValueError(f"Too few {sugar} lineages with length >= {min_length}")
    lengths = np.asarray([len(group) for group in groups], dtype=int)
    initial = np.asarray(
        [group.length_birth.iloc[0] for group in groups], dtype=float
    )
    observed_first, observed_second = half_means(groups)
    observed_pearson, observed_spearman = correlations(
        observed_first, observed_second
    )
    rng = np.random.default_rng(seed)
    bootstrap = bootstrap_correlation(
        observed_first, observed_second, n_bootstrap, rng
    )

    directory = output_dir(sugar)
    chain = pd.read_csv(directory / "flat_chain.csv").to_numpy(float)
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    birth_pivot = float(diagnostics["parameterization"]["birth_pivot"])
    draw_indices = rng.integers(0, len(chain), size=n_replicates)
    null_pearson = np.empty(n_replicates, dtype=float)
    null_spearman = np.empty(n_replicates, dtype=float)
    for replicate, draw_index in enumerate(draw_indices):
        simulated = pd.DataFrame(
            simulate_lineages(
                pivoted_to_physical_params(chain[draw_index], birth_pivot),
                initial,
                len(groups),
                int(lengths.max()),
                int(rng.integers(np.iinfo(np.int32).max)),
                fixed_initial_lengths=True,
            )
        )
        pieces = [
            simulated.loc[simulated.lineage.eq(index)].iloc[:length]
            for index, length in enumerate(lengths)
        ]
        simulated_first, simulated_second = half_means(pieces)
        null_pearson[replicate], null_spearman[replicate] = correlations(
            simulated_first, simulated_second
        )

    exceedance = (1.0 + np.sum(null_pearson >= observed_pearson)) / (
        n_replicates + 1.0
    )
    row: dict[str, object] = {
        "sugar": sugar,
        "lineages": len(groups),
        "min_length": min_length,
        "observed_pearson": observed_pearson,
        "observed_spearman": observed_spearman,
        "observed_bootstrap_q2.5": np.percentile(bootstrap, 2.5),
        "observed_bootstrap_q97.5": np.percentile(bootstrap, 97.5),
        "null_pearson_median": np.median(null_pearson),
        "null_pearson_q2.5": np.percentile(null_pearson, 2.5),
        "null_pearson_q97.5": np.percentile(null_pearson, 97.5),
        "null_spearman_median": np.median(null_spearman),
        "posterior_predictive_upper_p": exceedance,
    }
    scatter = pd.DataFrame(
        {
            "sugar": sugar,
            "first_half_mean_log_birth": observed_first,
            "second_half_mean_log_birth": observed_second,
        }
    )
    return row, scatter


def make_plot(summary: pd.DataFrame, scatter: pd.DataFrame, output: Path) -> None:
    fig = plt.figure(figsize=(13, 8.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    forest = fig.add_subplot(grid[0, :])
    positions = np.arange(len(summary))
    null_median = summary.null_pearson_median.to_numpy(float)
    null_lower = summary.null_pearson_q2_5.to_numpy(float)
    null_upper = summary.null_pearson_q97_5.to_numpy(float)
    observed = summary.observed_pearson.to_numpy(float)
    observed_lower = summary.observed_bootstrap_q2_5.to_numpy(float)
    observed_upper = summary.observed_bootstrap_q97_5.to_numpy(float)
    forest.errorbar(
        positions - 0.08,
        null_median,
        yerr=[null_median - null_lower, null_upper - null_median],
        fmt="o",
        color="0.48",
        capsize=3,
        label="Current-model posterior prediction",
    )
    forest.errorbar(
        positions + 0.08,
        observed,
        yerr=[observed - observed_lower, observed_upper - observed],
        fmt="o",
        color="tab:blue",
        capsize=3,
        label="Observed (bootstrap 95% CI)",
    )
    forest.set_xticks(positions, summary.sugar)
    forest.set_ylabel("First-half vs second-half Pearson correlation")
    forest.set_ylim(-0.25, 1.02)
    forest.grid(axis="y", alpha=0.14)
    forest.legend(frameon=False, ncol=2, loc="lower right")
    forest.set_title("Persistence of lineage mean log birth size")

    for axis, sugar in zip(
        [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ["Glu", "Lac"],
    ):
        subset = scatter[scatter.sugar.eq(sugar)]
        x = subset.first_half_mean_log_birth.to_numpy(float)
        y = subset.second_half_mean_log_birth.to_numpy(float)
        axis.scatter(x, y, s=17, alpha=0.45, color="tab:blue")
        slope, intercept = np.polyfit(x, y, 1)
        line = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 100)
        axis.plot(line, intercept + slope * line, color="tab:orange", lw=2)
        axis.plot(line, line, color="0.55", ls="--", lw=1)
        r = float(summary.loc[summary.sugar.eq(sugar), "observed_pearson"].iloc[0])
        axis.set_title(f"{sugar}: r = {r:.3f}")
        axis.set_xlabel("First-half mean log birth size")
        axis.set_ylabel("Second-half mean log birth size")
        axis.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(
        output / "lineage_half_size_persistence.png",
        dpi=180,
        facecolor="white",
        transparent=False,
    )
    fig.savefig(
        output / "lineage_half_size_persistence.pdf",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir or PROJECT_DIR / "outputs" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    scatters = []
    for index, sugar in enumerate(SUGARS):
        print(f"Analyzing {sugar}", flush=True)
        row, scatter = analyze_sugar(
            sugar,
            args.min_length,
            args.replicates,
            args.bootstrap,
            args.seed + 100_000 * index,
        )
        rows.append(row)
        scatters.append(scatter)
    summary = pd.DataFrame(rows)
    # Attribute-safe names for plotting; saved CSV keeps conventional dots.
    summary.columns = [column.replace(".", "_") for column in summary.columns]
    summary.to_csv(output / "lineage_half_size_persistence.csv", index=False)
    scatter = pd.concat(scatters, ignore_index=True)
    scatter.to_csv(
        output / "lineage_half_size_persistence_scatter.csv", index=False
    )
    make_plot(summary, scatter, output)
    with (output / "lineage_half_size_persistence_config.json").open("w") as handle:
        json.dump(vars(args) | {"output_dir": str(output)}, handle, indent=2)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
