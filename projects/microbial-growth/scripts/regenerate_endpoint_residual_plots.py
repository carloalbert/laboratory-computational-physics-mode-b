#!/usr/bin/env python3
"""Regenerate the principal diagnostic figures from saved per-sugar fits.

No inference is repeated.  The script reads the saved chains, posterior
predictive simulations and observation-matched ACF tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import corner
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_loglogistic_endpoint_residual import (  # noqa: E402
    PARAMETER_NAMES,
    load_endpoint_data,
)


SUGARS = ["Ace", "DMa", "Glu", "Gly", "Lac", "Pyr", "Rha", "Suc", "Xyl"]
MODEL_NAME = "loglogistic_endpoint_residual"
CORNER_LABELS = [
    r"$p_\star$",
    r"$\omega_2$",
    r"$\eta$",
    r"$\mu_\alpha$",
    r"$\sigma_\alpha$",
    r"$\mu_f$",
    r"$\sigma_f$",
    r"$\mu_\epsilon$",
    r"$\sigma_\epsilon$",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sugar",
        action="append",
        choices=SUGARS,
        help="Sugar to plot; repeat for several. Defaults to all nine.",
    )
    return parser.parse_args()


def output_dir(sugar: str) -> Path:
    return PROJECT_DIR / "outputs" / "model_comparison" / sugar / MODEL_NAME


def plot_acfs(sugar: str, directory: Path) -> None:
    table = pd.read_csv(directory / "acf_observation_matched.csv")
    lags = table["lag"].to_numpy(int)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    specifications = [
        ("time", "Division-time ACF"),
        ("length_birth", "Birth-length ACF"),
    ]
    for axis, (variable, title) in zip(axes, specifications):
        observed = table[f"observed_{variable}"].to_numpy(float)
        median = table[f"predictive_median_{variable}"].to_numpy(float)
        lower = table[f"predictive_q2.5_{variable}"].to_numpy(float)
        upper = table[f"predictive_q97.5_{variable}"].to_numpy(float)
        long_lineage = table[f"long_lineage_{variable}"].to_numpy(float)
        axis.fill_between(
            lags,
            lower,
            upper,
            color="tab:orange",
            alpha=0.25,
            label="Matched-length 95% predictive interval",
        )
        axis.plot(
            lags,
            median,
            color="tab:orange",
            marker="o",
            ms=3.5,
            label="Matched-length predictive median",
        )
        axis.plot(
            lags,
            observed,
            color="tab:blue",
            marker="o",
            ms=3.5,
            lw=2,
            label=f"{sugar} data",
        )
        axis.plot(
            lags,
            long_lineage,
            color="0.35",
            ls="--",
            label="80-generation simulation",
        )
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_xlabel("Generation lag")
        axis.set_ylabel("Mean sample ACF")
        axis.set_title(title)
        tick_step = 1 if lags.max() <= 10 else 5
        axis.set_xticks(np.arange(0, lags.max() + 1, tick_step))
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        f"{sugar}: exact-pair observation-matched ACF check"
    )
    fig.tight_layout()
    fig.savefig(directory / "acf_observation_matched.png", dpi=180)
    plt.close(fig)


def plot_predictive_marginals(sugar: str, directory: Path) -> None:
    real, transition, _, _ = load_endpoint_data(sugar)
    simulated = pd.read_csv(directory / "posterior_predictive_simulation.csv")
    observed = {
        "time": real.time.to_numpy(float),
        "alpha": real.alpha.to_numpy(float),
        "f": real.f.to_numpy(float),
        "length_birth": real.length_birth.to_numpy(float),
        "length_final": transition.length_final.to_numpy(float),
        "added_size": transition.added_size.to_numpy(float),
        "epsilon": transition.epsilon.to_numpy(float),
    }
    labels = {
        "time": "Division time",
        "alpha": "Growth rate",
        "f": "Division fraction",
        "length_birth": "Birth length",
        "length_final": "Final length",
        "added_size": "Measured added length",
        "epsilon": r"Endpoint residual $\epsilon$",
    }
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, variable in zip(axes.flat, observed):
        data_values = observed[variable]
        model_values = simulated[variable].to_numpy(float)
        combined = np.concatenate([data_values, model_values])
        finite = combined[np.isfinite(combined)]
        low, high = np.percentile(finite, [0.5, 99.5])
        bins = np.linspace(low, high, 36)
        axis.hist(
            data_values,
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            label=f"{sugar} data",
        )
        axis.hist(
            model_values,
            bins=bins,
            density=True,
            histtype="step",
            lw=2,
            label="Endpoint-residual model",
        )
        axis.set_xlabel(labels[variable])
        axis.set_ylabel("Density")
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"{sugar}: posterior-predictive marginal distributions")
    fig.tight_layout()
    fig.savefig(directory / "posterior_predictive_comparison.png", dpi=180)
    plt.close(fig)


def plot_corner(sugar: str, directory: Path) -> None:
    chain = pd.read_csv(directory / "flat_chain.csv")[PARAMETER_NAMES]
    posterior = pd.read_csv(directory / "posterior_summary.csv").set_index(
        "parameter"
    )
    reference = posterior.loc[PARAMETER_NAMES, "median"].to_numpy(float)
    figure = corner.corner(
        chain.to_numpy(float),
        labels=CORNER_LABELS,
        truths=reference,
        show_titles=True,
        title_fmt=".3g",
        quantiles=[0.16, 0.5, 0.84],
        max_n_ticks=3,
        label_kwargs={"fontsize": 10},
        title_kwargs={"fontsize": 9},
    )
    figure.suptitle(f"{sugar}: re-parameterized posterior", y=1.0, fontsize=16)
    figure.savefig(directory / "corner_plot.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_all_sugars_acf_sheet() -> None:
    """Put both full ACF diagnostics for all sugars on one sheet."""
    fig, axes = plt.subplots(
        len(SUGARS),
        2,
        figsize=(13, 22),
        sharex=True,
        sharey=True,
    )
    specifications = [
        ("time", "Division-time ACF"),
        ("length_birth", "Birth-length ACF"),
    ]
    legend_handles = None
    legend_labels = None
    for row, sugar in enumerate(SUGARS):
        table = pd.read_csv(output_dir(sugar) / "acf_observation_matched.csv")
        lags = table["lag"].to_numpy(int)
        for column, (variable, title) in enumerate(specifications):
            axis = axes[row, column]
            observed = table[f"observed_{variable}"].to_numpy(float)
            median = table[f"predictive_median_{variable}"].to_numpy(float)
            lower = table[f"predictive_q2.5_{variable}"].to_numpy(float)
            upper = table[f"predictive_q97.5_{variable}"].to_numpy(float)
            long_lineage = table[f"long_lineage_{variable}"].to_numpy(float)
            axis.fill_between(
                lags,
                lower,
                upper,
                color="tab:orange",
                alpha=0.25,
                label="Matched-length 95% predictive interval",
            )
            axis.plot(
                lags,
                median,
                color="tab:orange",
                marker="o",
                ms=2.8,
                lw=1.4,
                label="Matched-length predictive median",
            )
            axis.plot(
                lags,
                observed,
                color="tab:blue",
                marker="o",
                ms=2.8,
                lw=1.7,
                label="Observed",
            )
            axis.plot(
                lags,
                long_lineage,
                color="0.35",
                ls="--",
                lw=1.2,
                label="80-generation simulation",
            )
            axis.axhline(0.0, color="black", lw=0.7)
            axis.set_xlim(0, 30)
            axis.set_ylim(-0.25, 1.05)
            axis.set_xticks(np.arange(0, 31, 5))
            axis.grid(axis="y", alpha=0.15)
            if row == 0:
                axis.set_title(title, fontsize=13)
            if column == 0:
                axis.annotate(
                    sugar,
                    xy=(-0.14, 0.5),
                    xycoords="axes fraction",
                    ha="right",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
            if legend_handles is None:
                legend_handles, legend_labels = axis.get_legend_handles_labels()

    fig.suptitle(
        "Exact-pair observation-matched ACF posterior predictive checks",
        y=0.995,
        fontsize=16,
    )
    fig.supxlabel("Generation lag", y=0.015)
    fig.supylabel("Mean sample ACF", x=0.015)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.979),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.055, 0.025, 1.0, 0.955), h_pad=0.55)
    summary_dir = PROJECT_DIR / "outputs" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary_dir / "all_sugars_acf_sheet.png", dpi=180)
    fig.savefig(summary_dir / "all_sugars_acf_sheet.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sugars = args.sugar or SUGARS
    for sugar in sugars:
        directory = output_dir(sugar)
        required = [
            "acf_observation_matched.csv",
            "flat_chain.csv",
            "posterior_summary.csv",
            "posterior_predictive_simulation.csv",
        ]
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            raise FileNotFoundError(f"{sugar}: missing saved results: {missing}")
        plot_acfs(sugar, directory)
        plot_predictive_marginals(sugar, directory)
        plot_corner(sugar, directory)
        print(f"Updated {sugar}: ACFs, predictive marginals, corner plot")
    if set(sugars) == set(SUGARS):
        plot_all_sugars_acf_sheet()
        print("Updated all-sugars ACF sheet")


if __name__ == "__main__":
    main()
