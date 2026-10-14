#!/usr/bin/env python3
"""Check Glu lineages for an early-to-late drift in division time.

The proposed deterministic memory variable is initialized at the beginning of
each observed lineage and decays with absolute time.  A necessary empirical
signature is therefore a common time-aligned drift in inter-division time.
This script measures that signature without fitting the memory model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "processed_input_data.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "summary" / "Glu_interdivision_time_drift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-observations", type=int, default=6)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_rep: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_rep, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def exact_two_sided_sign_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    n = len(values)
    k = int(np.sum(values > 0))
    lower = min(k, n - k)
    tail = sum(math.comb(n, j) for j in range(lower + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def cluster_sufficient_statistics(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    x_time = group["elapsed_start"].to_numpy(float)
    x_alpha = group["alpha"].to_numpy(float)
    y = group["time"].to_numpy(float)
    x_time -= x_time.mean()
    x_alpha -= x_alpha.mean()
    y -= y.mean()
    design = np.column_stack((x_time, x_alpha))
    return design.T @ design, design.T @ y, float(x_time @ x_time), float(x_time @ y)


def fixed_effect_slopes(
    groups: list[pd.DataFrame], rng: np.random.Generator, n_rep: int
) -> dict[str, object]:
    stats = [cluster_sufficient_statistics(group) for group in groups if len(group) >= 3]
    xtx = np.stack([item[0] for item in stats])
    xty = np.stack([item[1] for item in stats])
    sxx = np.asarray([item[2] for item in stats])
    sxy = np.asarray([item[3] for item in stats])

    controlled = np.linalg.solve(xtx.sum(axis=0), xty.sum(axis=0))[0]
    unadjusted = sxy.sum() / sxx.sum()
    indices = rng.integers(0, len(stats), size=(n_rep, len(stats)))
    controlled_draws = np.empty(n_rep)
    unadjusted_draws = np.empty(n_rep)
    for index, sample in enumerate(indices):
        controlled_draws[index] = np.linalg.solve(
            xtx[sample].sum(axis=0), xty[sample].sum(axis=0)
        )[0]
        unadjusted_draws[index] = sxy[sample].sum() / sxx[sample].sum()
    return {
        "unadjusted_hours_per_elapsed_hour": float(unadjusted),
        "unadjusted_ci95": np.quantile(unadjusted_draws, [0.025, 0.975]).tolist(),
        "alpha_adjusted_hours_per_elapsed_hour": float(controlled),
        "alpha_adjusted_ci95": np.quantile(controlled_draws, [0.025, 0.975]).tolist(),
    }


def binned_trend(
    data: pd.DataFrame, value: str, rng: np.random.Generator, n_rep: int
) -> pd.DataFrame:
    lineage_bins = (
        data.groupby(["lineage", "progress_bin"], observed=True)[value]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, float | int]] = []
    for bin_number, group in lineage_bins.groupby("progress_bin", observed=True):
        values = group[value].to_numpy(float)
        low, high = bootstrap_mean(values, rng, n_rep)
        rows.append(
            {
                "progress_bin": int(bin_number),
                "relative_position": (int(bin_number) + 0.5) / 10,
                "mean": float(values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "n_lineages": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data = pd.read_csv(args.data)
    glu = (
        data.loc[data["source"].eq("Glu")]
        .sort_values(["lineage", "generation"])
        .reset_index(drop=True)
    )
    glu = glu.loc[np.isfinite(glu["time"]) & glu["time"].gt(0)].copy()
    glu["elapsed_start"] = glu.groupby("lineage")["time"].cumsum() - glu["time"]
    glu["growth_exponent"] = glu["alpha"] * glu["time"]

    eligible_labels = glu.groupby("lineage").size()
    eligible_labels = eligible_labels[eligible_labels >= args.min_observations].index
    eligible = glu.loc[glu["lineage"].isin(eligible_labels)].copy()
    eligible["lineage_size"] = eligible.groupby("lineage")["time"].transform("size")
    eligible["relative_position"] = (
        eligible.groupby("lineage").cumcount() / (eligible["lineage_size"] - 1)
    )
    eligible["progress_bin"] = np.minimum(
        (10 * eligible["relative_position"]).astype(int), 9
    )
    for value in ("time", "growth_exponent", "alpha"):
        eligible[f"{value}_centered"] = (
            eligible[value] - eligible.groupby("lineage")[value].transform("mean")
        )

    lineage_rows: list[dict[str, float | int]] = []
    for lineage, group in eligible.groupby("lineage", sort=False):
        n = len(group)
        elapsed = group["elapsed_start"].to_numpy(float)
        time = group["time"].to_numpy(float)
        growth = group["growth_exponent"].to_numpy(float)
        alpha = group["alpha"].to_numpy(float)
        third = max(2, n // 3)
        lineage_rows.append(
            {
                "lineage": lineage,
                "n_observations": n,
                "time_slope_per_elapsed_hour": float(
                    np.linalg.lstsq(
                        np.column_stack((np.ones(n), elapsed)), time, rcond=None
                    )[0][1]
                ),
                "early_time_mean_hours": float(time[:third].mean()),
                "late_time_mean_hours": float(time[-third:].mean()),
                "late_minus_early_time_hours": float(
                    time[-third:].mean() - time[:third].mean()
                ),
                "late_minus_early_growth_exponent": float(
                    growth[-third:].mean() - growth[:third].mean()
                ),
                "late_minus_early_alpha": float(
                    alpha[-third:].mean() - alpha[:third].mean()
                ),
            }
        )
    lineage = pd.DataFrame(lineage_rows)

    metric_summary: dict[str, object] = {}
    for column in (
        "late_minus_early_time_hours",
        "late_minus_early_growth_exponent",
        "late_minus_early_alpha",
        "time_slope_per_elapsed_hour",
    ):
        values = lineage[column].to_numpy(float)
        low, high = bootstrap_mean(values, rng, args.bootstrap_replicates)
        metric_summary[column] = {
            "mean": float(values.mean()),
            "mean_ci95": [float(low), float(high)],
            "median": float(np.median(values)),
            "fraction_negative": float(np.mean(values < 0)),
            "sign_test_two_sided_p": exact_two_sided_sign_p(values),
        }

    groups = [group.copy() for _, group in eligible.groupby("lineage", sort=False)]
    groups_drop_first = [group.iloc[1:].copy() for group in groups if len(group) >= 7]
    fixed_effect = fixed_effect_slopes(groups, rng, args.bootstrap_replicates)
    fixed_effect_drop_first = fixed_effect_slopes(
        groups_drop_first, rng, args.bootstrap_replicates
    )

    time_trend = binned_trend(
        eligible, "time_centered", rng, args.bootstrap_replicates
    ).rename(
        columns={
            "mean": "time_mean_hours",
            "ci95_low": "time_ci95_low",
            "ci95_high": "time_ci95_high",
        }
    )
    growth_trend = binned_trend(
        eligible, "growth_exponent_centered", rng, args.bootstrap_replicates
    ).rename(
        columns={
            "mean": "growth_exponent_mean",
            "ci95_low": "growth_exponent_ci95_low",
            "ci95_high": "growth_exponent_ci95_high",
            "n_lineages": "growth_n_lineages",
        }
    )
    trend = time_trend.merge(
        growth_trend[
            [
                "progress_bin",
                "growth_exponent_mean",
                "growth_exponent_ci95_low",
                "growth_exponent_ci95_high",
                "growth_n_lineages",
            ]
        ],
        on="progress_bin",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lineage.to_csv(args.output_dir / "lineage_drift_metrics.csv", index=False)
    trend.to_csv(args.output_dir / "relative_position_trend.csv", index=False)
    eligible[
        [
            "lineage",
            "generation",
            "time",
            "alpha",
            "growth_exponent",
            "elapsed_start",
            "relative_position",
        ]
    ].to_csv(args.output_dir / "analysis_data.csv", index=False)

    summary = {
        "question": "Do Glu inter-division times drift across each observed lineage?",
        "interpretation": (
            "A negative early-to-late change means divisions become faster as "
            "absolute time from lineage start increases."
        ),
        "n_observations": int(len(glu)),
        "n_lineages_total": int(glu["lineage"].nunique()),
        "n_lineages_analyzed": int(len(lineage)),
        "minimum_observations": args.min_observations,
        "early_late_definition": "means of the first and final thirds of each lineage",
        "metrics": metric_summary,
        "fixed_effect_time_trend": fixed_effect,
        "fixed_effect_time_trend_after_dropping_first_cycle": fixed_effect_drop_first,
        "run_config": {
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "data": str(args.data),
        },
    }
    with (args.output_dir / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
