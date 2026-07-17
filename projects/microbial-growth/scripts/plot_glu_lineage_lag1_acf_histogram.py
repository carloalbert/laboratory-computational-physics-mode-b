#!/usr/bin/env python3
"""Plot the distribution of individual-lineage Glu birth-size lag-1 ACFs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_loglogistic_endpoint_residual import acf_1d, load_endpoint_data  # noqa: E402


def main() -> None:
    frame, _, _, _ = load_endpoint_data("Glu")
    rows = []
    for lineage, group in frame.groupby("lineage", sort=False):
        ordered = group.sort_values("generation")
        if len(ordered) < 3:
            continue
        acf = acf_1d(
            ordered.length_birth.to_numpy(float), max_lag=1
        )
        rows.append(
            {
                "lineage": lineage,
                "n_observations": len(ordered),
                "birth_lag1_acf": float(acf[1]),
                "mean_birth_length": float(ordered.length_birth.mean()),
                "sd_birth_length": float(
                    ordered.length_birth.std(ddof=1)
                ),
            }
        )
    result = pd.DataFrame(rows)
    values = result.birth_lag1_acf.to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Non-finite lineage ACF encountered")

    summary = {
        "sugar": "Glu",
        "variable": "length_birth",
        "lag": 1,
        "estimator": (
            "sum((x_t-xbar)(x_t+1-xbar)) / "
            "sum((x_t-xbar)^2)"
        ),
        "n_total_lineages": int(frame.lineage.nunique()),
        "minimum_observations": 3,
        "n_included_lineages": int(len(result)),
        "n_excluded_short_lineages": int(
            frame.lineage.nunique() - len(result)
        ),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "q2.5": float(np.percentile(values, 2.5)),
        "q25": float(np.percentile(values, 25.0)),
        "q75": float(np.percentile(values, 75.0)),
        "q97.5": float(np.percentile(values, 97.5)),
        "max": float(np.max(values)),
        "n_negative": int(np.sum(values < 0.0)),
        "fraction_negative": float(np.mean(values < 0.0)),
        "n_above_0.5": int(np.sum(values > 0.5)),
        "fraction_above_0.5": float(np.mean(values > 0.5)),
    }

    output_dir = PROJECT_DIR / "outputs" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "Glu_lineage_birth_lag1_acf.csv", index=False)
    with (output_dir / "Glu_lineage_birth_lag1_acf_summary.json").open(
        "w"
    ) as handle:
        json.dump(summary, handle, indent=2)

    bins = np.histogram_bin_edges(values, bins="fd")
    fig, axis = plt.subplots(figsize=(7.2, 5.1))
    axis.hist(
        values,
        bins=bins,
        color="tab:blue",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.8,
    )
    axis.axvline(
        summary["mean"],
        color="black",
        linewidth=1.8,
        label=f'Mean = {summary["mean"]:.3f}',
    )
    axis.axvline(
        summary["median"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f'Median = {summary["median"]:.3f}',
    )
    axis.axvline(0.0, color="0.5", linewidth=0.9)
    axis.set_xlabel("Individual-lineage birth-length ACF at lag 1")
    axis.set_ylabel("Number of lineages")
    axis.set_xlim(-0.72, 0.92)
    axis.text(
        0.98,
        0.95,
        f"n = {len(values)} lineages",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(
        output_dir / "Glu_lineage_birth_lag1_acf_histogram.png",
        dpi=200,
        facecolor="white",
    )
    fig.savefig(
        output_dir / "Glu_lineage_birth_lag1_acf_histogram.pdf",
        facecolor="white",
    )
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
