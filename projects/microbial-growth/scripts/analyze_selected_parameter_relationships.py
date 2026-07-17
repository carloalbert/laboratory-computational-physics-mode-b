#!/usr/bin/env python3
"""Pairwise cross-sugar relationships among u, omega2, eta and mu_alpha."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from validate_acf_with_measured_inputs import load_sugar  # noqa: E402


SUMMARY_DIR = PROJECT_DIR / "outputs" / "summary"
SUGARS = ["Ace", "DMa", "Glu", "Gly", "Lac", "Pyr", "Rha", "Suc", "Xyl"]
PARAMETERS = ["u", "omega2", "eta", "mu_alpha"]
PAIRS = [
    ("eta", "omega2"),
    ("mu_alpha", "omega2"),
    ("u", "omega2"),
    ("mu_alpha", "eta"),
    ("u", "eta"),
    ("u", "mu_alpha"),
]
LABELS = {
    "u": r"$u$",
    "omega2": r"$\omega_2$",
    "eta": r"$\eta$",
    "mu_alpha": r"$\mu_\alpha$",
}
ANNOTATION_OFFSETS = {
    ("eta", "omega2", "Xyl"): (-20, -10),
    ("eta", "omega2", "Pyr"): (4, -11),
    ("eta", "omega2", "Glu"): (4, -10),
    ("mu_alpha", "omega2", "Pyr"): (4, -11),
    ("mu_alpha", "omega2", "Glu"): (4, -10),
    ("mu_alpha", "eta", "Gly"): (-20, 4),
    ("mu_alpha", "eta", "Lac"): (4, -11),
}


def partial_correlation(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray
) -> float:
    controls = np.asarray(controls, dtype=float)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones(len(controls)), controls])
    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(pearsonr(residual_x, residual_y).statistic)


def posterior_correlation_interval(
    x_name: str,
    y_name: str,
    birth_pivots: dict[str, float],
    rng: np.random.Generator,
    n_draws: int = 20_000,
) -> tuple[float, float, float, float]:
    x = np.empty((n_draws, len(SUGARS)))
    y = np.empty_like(x)
    for column, sugar in enumerate(SUGARS):
        requested_columns: set[str] = set()
        for name in [x_name, y_name]:
            requested_columns.update(["p_star", "eta"] if name == "u" else [name])
        chain = pd.read_csv(
            PROJECT_DIR
            / "outputs"
            / "model_comparison"
            / sugar
            / "loglogistic_endpoint_residual"
            / "flat_chain.csv",
            usecols=sorted(requested_columns),
        )
        indices = rng.integers(0, len(chain), size=n_draws)
        values: dict[str, np.ndarray] = {}
        for name in [x_name, y_name]:
            if name == "u":
                values[name] = (
                    chain.p_star.to_numpy(float)
                    + birth_pivots[sugar] * chain.eta.to_numpy(float)
                )
            else:
                values[name] = chain[name].to_numpy(float)
        x[:, column] = values[x_name][indices]
        y[:, column] = values[y_name][indices]
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    correlation = np.sum(x * y, axis=1) / np.sqrt(
        np.sum(x * x, axis=1) * np.sum(y * y, axis=1)
    )
    lower, median, upper = np.percentile(correlation, [2.5, 50.0, 97.5])
    sign_probability = float(
        np.mean(correlation > 0.0)
        if median > 0.0
        else np.mean(correlation < 0.0)
    )
    return float(lower), float(median), float(upper), sign_probability


def main() -> None:
    long = pd.read_csv(SUMMARY_DIR / "all_sugars_posterior_summary.csv")
    selected = long[long.parameter.isin(PARAMETERS)].copy()
    mean = selected.pivot(index="sugar", columns="parameter", values="mean").loc[
        SUGARS
    ]
    lower = selected.pivot(
        index="sugar", columns="parameter", values="ci_2.5"
    ).loc[SUGARS]
    upper = selected.pivot(
        index="sugar", columns="parameter", values="ci_97.5"
    ).loc[SUGARS]
    birth_pivots = {sugar: load_sugar(sugar)[1] for sugar in SUGARS}

    rng = np.random.default_rng(314_159)
    statistics: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5))
    for axis, (x_name, y_name) in zip(axes.flat, PAIRS):
        x = mean[x_name].to_numpy(float)
        y = mean[y_name].to_numpy(float)
        control_names = [
            name for name in PARAMETERS if name not in [x_name, y_name]
        ]
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        rank_partial = partial_correlation(
            rankdata(x),
            rankdata(y),
            np.column_stack(
                [
                    rankdata(mean[name].to_numpy(float))
                    for name in control_names
                ]
            ),
        )
        posterior_lower, posterior_median, posterior_upper, sign_probability = (
            posterior_correlation_interval(
                x_name, y_name, birth_pivots, rng
            )
        )
        leave_one_out = np.asarray(
            [
                spearmanr(np.delete(x, index), np.delete(y, index)).statistic
                for index in range(len(SUGARS))
            ]
        )
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        r_squared = 1.0 - np.sum((y - fitted) ** 2) / np.sum(
            (y - np.mean(y)) ** 2
        )
        statistics.append(
            {
                "x": x_name,
                "y": y_name,
                "controls": ";".join(control_names),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "rank_partial_correlation": rank_partial,
                "leave_one_out_spearman_min": np.min(leave_one_out),
                "leave_one_out_spearman_max": np.max(leave_one_out),
                "posterior_pearson_q2.5": posterior_lower,
                "posterior_pearson_median": posterior_median,
                "posterior_pearson_q97.5": posterior_upper,
                "posterior_sign_probability": sign_probability,
                "linear_intercept": intercept,
                "linear_slope": slope,
                "linear_r_squared": r_squared,
            }
        )

        xerr = np.vstack(
            [x - lower[x_name].to_numpy(float), upper[x_name].to_numpy(float) - x]
        )
        yerr = np.vstack(
            [y - lower[y_name].to_numpy(float), upper[y_name].to_numpy(float) - y]
        )
        axis.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            ms=5,
            color="tab:blue",
            ecolor="0.70",
            elinewidth=1,
            capsize=2,
            zorder=2,
        )
        grid = np.linspace(np.min(x), np.max(x), 100)
        axis.plot(
            grid,
            intercept + slope * grid,
            color="0.25",
            ls="--",
            lw=1.2,
            zorder=1,
        )
        for sugar, x_value, y_value in zip(SUGARS, x, y):
            axis.annotate(
                sugar,
                (x_value, y_value),
                xytext=ANNOTATION_OFFSETS.get(
                    (x_name, y_name, sugar), (4, 4)
                ),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel(LABELS[x_name])
        axis.set_ylabel(LABELS[y_name])
        p_text = (
            r"p<0.001"
            if spearman.pvalue < 0.001
            else rf"p={spearman.pvalue:.3f}"
        )
        axis.set_title(rf"$\rho_S={spearman.statistic:.2f}$, ${p_text}$")
        axis.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(SUMMARY_DIR / "selected_parameter_relationships.png", dpi=180)
    fig.savefig(SUMMARY_DIR / "selected_parameter_relationships.pdf")
    plt.close(fig)
    pd.DataFrame(statistics).to_csv(
        SUMMARY_DIR / "selected_parameter_relationships.csv", index=False
    )


if __name__ == "__main__":
    main()
