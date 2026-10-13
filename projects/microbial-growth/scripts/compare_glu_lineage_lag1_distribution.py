#!/usr/bin/env python3
"""Compare observed Glu lineage birth ACFs with hierarchical-u replay."""

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
from scipy.stats import wasserstein_distance


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from hierarchical_lineage_u_endpoint_residual import (  # noqa: E402
    hermite_rule,
    prepare_lineage_transitions,
)
from run_hierarchical_lineage_u import (  # noqa: E402
    NUISANCE_NAMES,
    draw_scales_for_observed_groups,
    lineage_groups,
    simulate_groups,
)
from run_loglogistic_endpoint_residual import acf_1d, load_endpoint_data  # noqa: E402


HIERARCHY_DIR = (
    PROJECT_DIR
    / "outputs"
    / "model_comparison"
    / "Glu"
    / "hierarchical_lineage_u_endpoint_residual"
)
COMMON_DIR = (
    PROJECT_DIR
    / "outputs"
    / "model_comparison"
    / "Glu"
    / "loglogistic_endpoint_residual"
)
SUMMARY_DIR = PROJECT_DIR / "outputs" / "summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument("--quadrature-order", type=int, default=24)
    return parser.parse_args()


def lineage_acf(frame: pd.DataFrame, lag: int) -> pd.DataFrame:
    rows = []
    for lineage, group in frame.groupby("lineage", sort=False):
        ordered = group.sort_values("generation")
        if len(ordered) <= lag + 1:
            continue
        value = float(acf_1d(ordered.length_birth.to_numpy(float), lag)[lag])
        rows.append(
            {
                "lineage": lineage,
                "n_observations": len(ordered),
                "birth_acf": value,
            }
        )
    result = pd.DataFrame(rows)
    if not np.all(np.isfinite(result.birth_acf)):
        raise RuntimeError(f"Non-finite lineage lag-{lag} ACF encountered")
    return result


def distribution_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "q2.5": float(np.percentile(values, 2.5)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "q97.5": float(np.percentile(values, 97.5)),
        "fraction_negative": float(np.mean(values < 0)),
        "fraction_above_0.5": float(np.mean(values > 0.5)),
    }


def main() -> None:
    args = parse_args()
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    if args.lag < 1:
        raise ValueError("--lag must be positive")

    real, transition, data, _ = load_endpoint_data("Glu")
    groups = lineage_groups(real)
    observed = lineage_acf(real, args.lag)
    observed_values = observed.birth_acf.to_numpy(float)
    birth_pivot = float(np.mean(data["transition_birth"]))
    prepared = prepare_lineage_transitions(
        transition.length_birth.to_numpy(float),
        transition.added_size.to_numpy(float),
        transition.lineage.to_numpy(),
    )
    z_nodes, log_weights = hermite_rule(args.quadrature_order)
    hierarchy_chain = pd.read_csv(HIERARCHY_DIR / "hierarchy_flat_chain.csv").to_numpy(float)
    nuisance_chain = pd.read_csv(COMMON_DIR / "flat_chain.csv")[NUISANCE_NAMES].to_numpy(float)

    rng = np.random.default_rng(args.seed)
    simulated_rows: list[pd.DataFrame] = []
    replicate_stats = []
    for replicate in range(args.replicates):
        hierarchy = hierarchy_chain[rng.integers(len(hierarchy_chain))]
        nuisance = nuisance_chain[rng.integers(len(nuisance_chain))]
        lineage_u = draw_scales_for_observed_groups(
            hierarchy, prepared, groups, birth_pivot, z_nodes, log_weights, rng
        )
        simulated = simulate_groups(
            groups=groups,
            lineage_u=lineage_u,
            lineage_omega2=np.full(len(groups), float(hierarchy[1])),
            nuisance=nuisance,
            eta=float(hierarchy[2]),
            rng=rng,
            replay_inputs=True,
        )
        lineage_values = lineage_acf(simulated, args.lag)
        lineage_values.insert(0, "replicate", replicate)
        simulated_rows.append(lineage_values)
        values = lineage_values.birth_acf.to_numpy(float)
        stats = distribution_stats(values)
        stats["replicate"] = replicate
        stats["wasserstein_to_observed"] = float(
            wasserstein_distance(observed_values, values)
        )
        replicate_stats.append(stats)

    simulated_acfs = pd.concat(simulated_rows, ignore_index=True)
    stats_frame = pd.DataFrame(replicate_stats)
    observed_stats = distribution_stats(observed_values)
    predictive = {}
    for column in [key for key in observed_stats] + ["wasserstein_to_observed"]:
        values = stats_frame[column].to_numpy(float)
        predictive[column] = {
            "median": float(np.median(values)),
            "q2.5": float(np.percentile(values, 2.5)),
            "q97.5": float(np.percentile(values, 97.5)),
        }
    summary = {
        "sugar": "Glu",
        "lag": args.lag,
        "model": "hierarchical_lineage_u_endpoint_residual",
        "simulation": "observed lineage lengths, initial birth sizes, alpha sequences, and f sequences; posterior conditional u_l draws",
        "replicates": args.replicates,
        "seed": args.seed,
        "n_observed_lineages": int(len(observed)),
        "observed": observed_stats,
        "posterior_predictive": predictive,
        "observed_statistic_percentiles": {
            key: float(np.mean(stats_frame[key].to_numpy(float) <= value))
            for key, value in observed_stats.items()
        },
    }

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"Glu_lineage_birth_lag{args.lag}_hierarchical_u_replay"
    observed.to_csv(SUMMARY_DIR / f"Glu_lineage_birth_lag{args.lag}_observed.csv", index=False)
    simulated_acfs.to_csv(
        SUMMARY_DIR / f"{stem}.csv", index=False
    )
    stats_frame.to_csv(
        SUMMARY_DIR / f"{stem}_stats.csv",
        index=False,
    )
    with (SUMMARY_DIR / f"{stem}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    bins = np.linspace(-0.72, 0.92, 34)
    histogram_samples = np.vstack(
        [
            np.histogram(group.birth_acf, bins=bins, density=True)[0]
            for _, group in simulated_acfs.groupby("replicate", sort=True)
        ]
    )
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist_median = np.median(histogram_samples, axis=0)
    hist_low, hist_high = np.percentile(histogram_samples, [2.5, 97.5], axis=0)

    grid = np.linspace(-0.72, 0.92, 250)
    observed_ecdf = np.searchsorted(np.sort(observed_values), grid, side="right") / len(observed_values)
    ecdf_samples = np.vstack(
        [
            np.searchsorted(np.sort(group.birth_acf), grid, side="right") / len(group)
            for _, group in simulated_acfs.groupby("replicate", sort=True)
        ]
    )
    ecdf_median = np.median(ecdf_samples, axis=0)
    ecdf_low, ecdf_high = np.percentile(ecdf_samples, [2.5, 97.5], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    axes[0].hist(
        observed_values, bins=bins, density=True, histtype="step", color="black",
        linewidth=2.0, label="Observed lineages",
    )
    axes[0].fill_between(
        centers, hist_low, hist_high, color="tab:blue", alpha=0.22,
        label="Hierarchical-u replay: 95% band",
    )
    axes[0].plot(centers, hist_median, color="tab:blue", linewidth=1.8,
                 label="Hierarchical-u replay: median")
    axes[0].set_xlabel(f"Individual-lineage birth-size ACF at lag {args.lag}")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(grid, observed_ecdf, color="black", linewidth=2.0,
                 label="Observed lineages")
    axes[1].fill_between(grid, ecdf_low, ecdf_high, color="tab:blue", alpha=0.22,
                         label="Hierarchical-u replay: 95% band")
    axes[1].plot(grid, ecdf_median, color="tab:blue", linewidth=1.8,
                 label="Hierarchical-u replay: median")
    axes[1].set_xlabel(f"Individual-lineage birth-size ACF at lag {args.lag}")
    axes[1].set_ylabel("Cumulative probability")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.set_xlim(-0.72, 0.92)
        axis.axvline(0, color="0.6", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(
        SUMMARY_DIR / f"{stem}.png",
        dpi=200,
        facecolor="white",
    )
    fig.savefig(
        SUMMARY_DIR / f"{stem}.pdf",
        facecolor="white",
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
