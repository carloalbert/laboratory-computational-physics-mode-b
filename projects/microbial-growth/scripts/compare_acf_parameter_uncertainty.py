#!/usr/bin/env python3
"""Compare fixed-MAP and full-posterior matched ACF predictive bands."""

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
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from run_loglogistic_endpoint_residual import (  # noqa: E402
    acf_1d,
    load_endpoint_data,
)
from loglogistic_endpoint_residual import (  # noqa: E402
    pivoted_to_physical_params,
    simulate_lineages,
)


SUGARS = ["Ace", "DMa", "Glu", "Gly", "Lac", "Pyr", "Rha", "Suc", "Xyl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sugar", default="Glu", choices=SUGARS)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=70_000)
    return parser.parse_args()


def matrix_mean_acf(
    matrix: np.ndarray, design: np.ndarray, max_lag: int
) -> np.ndarray:
    pieces = [
        acf_1d(matrix[index, :length], max_lag)
        for index, length in enumerate(design)
        if length >= 3
    ]
    array = np.asarray(pieces)
    counts = np.sum(np.isfinite(array), axis=0)
    return np.divide(
        np.nansum(array, axis=0),
        counts,
        out=np.full(max_lag + 1, np.nan),
        where=counts > 0,
    )


def main() -> None:
    args = parse_args()
    directory = (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / "loglogistic_endpoint_residual"
    )
    real, _, data, _ = load_endpoint_data(args.sugar)
    ordered = real.sort_values(["lineage", "generation"])
    grouped = ordered.groupby("lineage", sort=False)
    design = grouped.size().to_numpy(int)
    starts = grouped.length_birth.first().to_numpy(float)
    max_lag = min(30, int(design.max()) - 1)

    flat_chain = pd.read_csv(directory / "flat_chain.csv").to_numpy(float)
    flat_log_probability = pd.read_csv(
        directory / "flat_log_probability.csv"
    ).log_probability.to_numpy(float)
    sampled_map = flat_chain[int(np.argmax(flat_log_probability))]
    physical_map = pivoted_to_physical_params(
        sampled_map, float(data["transition_birth"].mean())
    )

    fixed_map_acf = {
        "time": np.full((args.replicates, max_lag + 1), np.nan),
        "length_birth": np.full((args.replicates, max_lag + 1), np.nan),
    }
    n_lineages = len(design)
    max_generations = int(design.max())
    for replicate in range(args.replicates):
        rows = simulate_lineages(
            physical_map,
            starts,
            n_lineages,
            max_generations,
            args.seed + replicate,
            fixed_initial_lengths=True,
        )
        for variable in fixed_map_acf:
            matrix = np.asarray(
                [row[variable] for row in rows], dtype=float
            ).reshape(n_lineages, max_generations)
            fixed_map_acf[variable][replicate] = matrix_mean_acf(
                matrix, design, max_lag
            )

    posterior = pd.read_csv(directory / "acf_observation_matched.csv")
    posterior = posterior[posterior.lag.le(max_lag)].reset_index(drop=True)
    lags = posterior.lag.to_numpy(int)
    result: dict[str, np.ndarray] = {"lag": lags}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    specifications = [
        ("time", "Division-time ACF"),
        ("length_birth", "Birth-length ACF"),
    ]
    for axis, (variable, title) in zip(axes, specifications):
        fixed_median = np.nanmedian(fixed_map_acf[variable], axis=0)
        fixed_lower = np.nanpercentile(fixed_map_acf[variable], 2.5, axis=0)
        fixed_upper = np.nanpercentile(fixed_map_acf[variable], 97.5, axis=0)
        observed = posterior[f"observed_{variable}"].to_numpy(float)
        post_median = posterior[
            f"predictive_median_{variable}"
        ].to_numpy(float)
        post_lower = posterior[f"predictive_q2.5_{variable}"].to_numpy(float)
        post_upper = posterior[f"predictive_q97.5_{variable}"].to_numpy(float)

        result[f"observed_{variable}"] = observed
        result[f"map_median_{variable}"] = fixed_median
        result[f"map_q2.5_{variable}"] = fixed_lower
        result[f"map_q97.5_{variable}"] = fixed_upper
        result[f"posterior_median_{variable}"] = post_median
        result[f"posterior_q2.5_{variable}"] = post_lower
        result[f"posterior_q97.5_{variable}"] = post_upper

        axis.fill_between(
            lags,
            fixed_lower,
            fixed_upper,
            color="tab:blue",
            alpha=0.16,
            label="Fixed MAP: 95% process band",
        )
        axis.plot(
            lags,
            fixed_median,
            color="tab:blue",
            ls="--",
            label="Fixed MAP median",
        )
        axis.fill_between(
            lags,
            post_lower,
            post_upper,
            color="tab:orange",
            alpha=0.24,
            label="Posterior draws: 95% predictive band",
        )
        axis.plot(
            lags,
            post_median,
            color="tab:orange",
            label="Posterior-draw median",
        )
        axis.plot(
            lags,
            observed,
            color="black",
            marker="o",
            ms=3.5,
            lw=1.8,
            label=f"{args.sugar} data",
        )
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_title(title)
        axis.set_xlabel("Generation lag")
        axis.set_ylabel("Mean sample ACF")
        axis.set_xticks(np.arange(0, max_lag + 1, 5))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        f"{args.sugar}: ACF uncertainty from fixed MAP versus posterior draws",
        y=1.03,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure_path = directory / "acf_parameter_uncertainty_comparison.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    result_frame = pd.DataFrame(result)
    result_frame.to_csv(
        directory / "acf_parameter_uncertainty_comparison.csv", index=False
    )
    for variable in ["time", "length_birth"]:
        valid = result_frame.lag.gt(0)
        map_covered = (
            result_frame[f"observed_{variable}"]
            .between(
                result_frame[f"map_q2.5_{variable}"],
                result_frame[f"map_q97.5_{variable}"],
            )
            .loc[valid]
            .sum()
        )
        posterior_covered = (
            result_frame[f"observed_{variable}"]
            .between(
                result_frame[f"posterior_q2.5_{variable}"],
                result_frame[f"posterior_q97.5_{variable}"],
            )
            .loc[valid]
            .sum()
        )
        print(
            f"{variable}: covered lags 1-{max_lag}: "
            f"MAP={map_covered}, posterior={posterior_covered}"
        )
    print("Saved", figure_path)


if __name__ == "__main__":
    main()
