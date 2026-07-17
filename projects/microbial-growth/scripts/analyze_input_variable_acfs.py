#!/usr/bin/env python3
"""ACF checks for observed growth rates and division factors."""

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

from run_loglogistic_endpoint_residual import (  # noqa: E402
    acf_1d,
    load_endpoint_data,
    mean_acf,
)
from weibull_adder_sizer import beta_ab, gamma_shape_scale  # noqa: E402


SUGARS = ["Ace", "DMa", "Glu", "Gly", "Lac", "Pyr", "Rha", "Suc", "Xyl"]
MODEL_NAME = "loglogistic_endpoint_residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=130_000)
    return parser.parse_args()


def output_dir(sugar: str) -> Path:
    return (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / sugar
        / MODEL_NAME
    )


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


def analyze_sugar(
    sugar: str, n_replicates: int, seed: int
) -> pd.DataFrame:
    real, _, _, _ = load_endpoint_data(sugar)
    ordered = real.sort_values(["lineage", "generation"])
    design = ordered.groupby("lineage", sort=False).size().to_numpy(int)
    max_lag = min(30, int(design.max()) - 1)
    max_generations = int(design.max())
    chain = pd.read_csv(output_dir(sugar) / "flat_chain.csv")
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(chain), size=n_replicates)
    predictive = {
        "alpha": np.full((n_replicates, max_lag + 1), np.nan),
        "f": np.full((n_replicates, max_lag + 1), np.nan),
    }

    for replicate, draw_index in enumerate(draw_indices):
        draw = chain.iloc[draw_index]
        gamma_shape, gamma_scale = gamma_shape_scale(
            float(draw.mu_alpha), float(draw.sigma_alpha)
        )
        beta_a, beta_b = beta_ab(float(draw.mu_f), float(draw.sigma_f))
        if gamma_shape is None or beta_a is None:
            raise ValueError(f"{sugar}: invalid posterior distribution parameters")
        alpha = rng.gamma(
            gamma_shape,
            gamma_scale,
            size=(len(design), max_generations),
        )
        fraction = rng.beta(
            beta_a,
            beta_b,
            size=(len(design), max_generations),
        )
        predictive["alpha"][replicate] = matrix_mean_acf(
            alpha, design, max_lag
        )
        predictive["f"][replicate] = matrix_mean_acf(
            fraction, design, max_lag
        )

    lags = np.arange(max_lag + 1)
    result: dict[str, np.ndarray] = {"lag": lags}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    specifications = [
        ("alpha", r"Growth-rate ACF ($\alpha$)"),
        ("f", r"Division-factor ACF ($f$)"),
    ]
    for axis, (variable, title) in zip(axes, specifications):
        observed = mean_acf(real, variable, max_lag)
        median = np.nanmedian(predictive[variable], axis=0)
        lower = np.nanpercentile(predictive[variable], 2.5, axis=0)
        upper = np.nanpercentile(predictive[variable], 97.5, axis=0)
        result[f"observed_{variable}"] = observed
        result[f"predictive_median_{variable}"] = median
        result[f"predictive_q2.5_{variable}"] = lower
        result[f"predictive_q97.5_{variable}"] = upper
        axis.fill_between(
            lags,
            lower,
            upper,
            color="tab:orange",
            alpha=0.25,
            label="IID model: 95% predictive interval",
        )
        axis.plot(
            lags,
            median,
            color="tab:orange",
            marker="o",
            ms=3,
            label="IID model median",
        )
        axis.plot(
            lags,
            observed,
            color="tab:blue",
            marker="o",
            ms=3,
            lw=1.8,
            label=f"{sugar} data",
        )
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_title(title)
        axis.set_xlabel("Generation lag")
        axis.set_ylabel("Mean sample ACF")
        axis.set_xticks(np.arange(0, max_lag + 1, 5))
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{sugar}: observed input-variable ACFs versus fitted IID laws"
    )
    fig.tight_layout()
    fig.savefig(output_dir(sugar) / "acf_growth_division_inputs.png", dpi=180)
    plt.close(fig)

    frame = pd.DataFrame(result)
    frame.to_csv(output_dir(sugar) / "acf_growth_division_inputs.csv", index=False)
    return frame


def plot_four_variable_sheet(tables: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(
        len(SUGARS),
        4,
        figsize=(20, 22),
        sharex=True,
        sharey=True,
    )
    specifications = [
        ("time", "Division-time ACF", "matched"),
        ("length_birth", "Birth-length ACF", "matched"),
        ("alpha", r"Growth-rate ACF ($\alpha$)", "input"),
        ("f", r"Division-factor ACF ($f$)", "input"),
    ]
    handles = labels = None
    lag1_rows = []
    for row, sugar in enumerate(SUGARS):
        matched = pd.read_csv(output_dir(sugar) / "acf_observation_matched.csv")
        inputs = tables[sugar]
        for column, (variable, title, source) in enumerate(specifications):
            table = matched if source == "matched" else inputs
            lags = table.lag.to_numpy(int)
            observed = table[f"observed_{variable}"].to_numpy(float)
            median = table[f"predictive_median_{variable}"].to_numpy(float)
            lower = table[f"predictive_q2.5_{variable}"].to_numpy(float)
            upper = table[f"predictive_q97.5_{variable}"].to_numpy(float)
            axis = axes[row, column]
            axis.fill_between(
                lags,
                lower,
                upper,
                color="tab:orange",
                alpha=0.25,
                label="95% predictive interval",
            )
            axis.plot(
                lags,
                median,
                color="tab:orange",
                marker="o",
                ms=2.5,
                lw=1.3,
                label="Predictive median",
            )
            axis.plot(
                lags,
                observed,
                color="tab:blue",
                marker="o",
                ms=2.5,
                lw=1.7,
                label="Observed",
            )
            axis.axhline(0.0, color="black", lw=0.7)
            axis.set_xlim(0, 30)
            axis.set_ylim(-0.25, 1.05)
            axis.set_xticks(np.arange(0, 31, 5))
            axis.grid(axis="y", alpha=0.14)
            if row == 0:
                axis.set_title(title, fontsize=12)
            if column == 0:
                axis.annotate(
                    sugar,
                    xy=(-0.15, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
            if handles is None:
                handles, labels = axis.get_legend_handles_labels()
            lag1_rows.append(
                {
                    "sugar": sugar,
                    "variable": variable,
                    "observed_lag1": observed[1],
                    "predictive_median_lag1": median[1],
                    "predictive_q2.5_lag1": lower[1],
                    "predictive_q97.5_lag1": upper[1],
                    "lag1_covered": bool(lower[1] <= observed[1] <= upper[1]),
                }
            )

    fig.suptitle(
        "Observation-matched ACF checks for state and input variables",
        y=0.995,
        fontsize=16,
    )
    fig.supxlabel("Generation lag", y=0.015)
    fig.supylabel("Mean sample ACF", x=0.015)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.979),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.045, 0.025, 1.0, 0.955), h_pad=0.55)
    summary_dir = PROJECT_DIR / "outputs" / "summary"
    fig.savefig(summary_dir / "all_sugars_acf_four_variables.png", dpi=180)
    fig.savefig(summary_dir / "all_sugars_acf_four_variables.pdf")
    plt.close(fig)
    pd.DataFrame(lag1_rows).to_csv(
        summary_dir / "all_sugars_acf_four_variables_lag1.csv", index=False
    )


def main() -> None:
    args = parse_args()
    tables = {
        sugar: analyze_sugar(
            sugar, args.replicates, args.seed + 10_000 * index
        )
        for index, sugar in enumerate(SUGARS)
    }
    plot_four_variable_sheet(tables)
    print("Updated growth-rate and division-factor ACF checks for all sugars")


if __name__ == "__main__":
    main()
