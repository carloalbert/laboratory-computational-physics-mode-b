#!/usr/bin/env python3
"""Plot individual Glu birth-size lag-1 ACF against inferred lineage u_l."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.stats import linregress, pearsonr, spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[1]
SUMMARY_DIR = PROJECT_DIR / "outputs" / "summary"
HIERARCHY_DIR = (
    PROJECT_DIR
    / "outputs"
    / "model_comparison"
    / "Glu"
    / "hierarchical_lineage_u_endpoint_residual"
)


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(values)), controls])
    coefficients = lstsq(design, values)[0]
    return values - design @ coefficients


def main() -> None:
    acf = pd.read_csv(SUMMARY_DIR / "Glu_lineage_birth_lag1_acf.csv")
    scales = pd.read_csv(HIERARCHY_DIR / "lineage_scale_posterior.csv")
    scale_columns = [
        "lineage",
        "u_lineage_mean",
        "u_lineage_median",
        "u_lineage_q2.5",
        "u_lineage_q97.5",
    ]
    joined = acf.merge(
        scales[scale_columns],
        on="lineage",
        how="inner",
        validate="one_to_one",
    ).sort_values("lineage")
    if len(joined) != len(acf):
        raise RuntimeError(
            f"Expected {len(acf)} matched lineages, found {len(joined)}"
        )

    u = joined.u_lineage_median.to_numpy(float)
    lag1 = joined.birth_lag1_acf.to_numpy(float)
    mean_birth = joined.mean_birth_length.to_numpy(float)
    pearson = pearsonr(u, lag1)
    spearman = spearmanr(u, lag1)
    regression = linregress(u, lag1)
    partial = pearsonr(
        residualize(u, mean_birth[:, None]),
        residualize(lag1, mean_birth[:, None]),
    )
    controls = np.column_stack(
        [
            mean_birth,
            joined.sd_birth_length.to_numpy(float),
            joined.n_observations.to_numpy(float),
        ]
    )
    adjusted = pearsonr(
        residualize(u, controls), residualize(lag1, controls)
    )
    summary = {
        "sugar": "Glu",
        "n_lineages": int(len(joined)),
        "x": "posterior median u_l",
        "y": "individual-lineage birth-length lag-1 ACF",
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "linear_slope": float(regression.slope),
        "linear_intercept": float(regression.intercept),
        "linear_r_squared": float(regression.rvalue**2),
        "partial_pearson_controlling_mean_birth_r": float(
            partial.statistic
        ),
        "partial_pearson_controlling_mean_birth_p": float(partial.pvalue),
        "partial_pearson_controlling_mean_sd_and_length_r": float(
            adjusted.statistic
        ),
        "partial_pearson_controlling_mean_sd_and_length_p": float(
            adjusted.pvalue
        ),
    }
    joined.to_csv(
        SUMMARY_DIR / "Glu_lineage_birth_lag1_acf_vs_u.csv", index=False
    )
    with (SUMMARY_DIR / "Glu_lineage_birth_lag1_acf_vs_u_summary.json").open(
        "w"
    ) as handle:
        json.dump(summary, handle, indent=2)

    fig, axis = plt.subplots(figsize=(7.3, 5.4))
    scatter = axis.scatter(
        u,
        lag1,
        c=mean_birth,
        cmap="viridis",
        s=32,
        alpha=0.78,
        linewidth=0.3,
        edgecolor="white",
    )
    grid = np.linspace(np.min(u), np.max(u), 200)
    axis.plot(
        grid,
        regression.intercept + regression.slope * grid,
        color="black",
        linewidth=1.8,
        label="Linear fit",
    )
    axis.axhline(0.0, color="0.55", linewidth=0.8)
    axis.set_xlabel(r"Posterior median lineage scale $u_\ell$")
    axis.set_ylabel("Individual-lineage birth-length ACF at lag 1")
    axis.text(
        0.03,
        0.96,
        (
            f"Pearson r = {pearson.statistic:.3f}\n"
            f"Spearman ρ = {spearman.statistic:.3f}\n"
            f"n = {len(joined)}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("Observed lineage mean birth length")
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(
        SUMMARY_DIR / "Glu_lineage_birth_lag1_acf_vs_u.png",
        dpi=200,
        facecolor="white",
    )
    fig.savefig(
        SUMMARY_DIR / "Glu_lineage_birth_lag1_acf_vs_u.pdf",
        facecolor="white",
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
