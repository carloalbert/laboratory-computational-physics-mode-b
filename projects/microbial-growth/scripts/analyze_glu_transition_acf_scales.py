#!/usr/bin/env python3
"""Estimate fast and slow ACF scales of Glu q, alpha, tau, and epsilon."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_autoregressive_residual_glu import contiguous_prefixes  # noqa: E402
from glu_workflow import load_data  # noqa: E402


OUT = PROJECT_DIR / "outputs" / "summary" / "Glu_transition_acf_scales"
VARIABLES = ["q", "alpha", "tau", "epsilon"]


def transition_sequences():
    _, groups = load_data()
    result = {name: [] for name in VARIABLES}
    for group in contiguous_prefixes(groups):
        if len(group) < 2:
            continue
        birth = group.length_birth.to_numpy(float)
        alpha = group.alpha.to_numpy(float)[:-1]
        tau = group.time.to_numpy(float)[:-1]
        fraction = group.f.to_numpy(float)[1:]
        q = np.diff(np.log(birth)) - alpha * tau
        epsilon = q - np.log(fraction)
        for name, values in {
            "q": q,
            "alpha": alpha,
            "tau": tau,
            "epsilon": epsilon,
        }.items():
            result[name].append(values)
    return result


def pooled_acf(sequences, max_lag):
    mean = float(np.mean(np.concatenate(sequences)))
    result = np.full(max_lag + 1, np.nan)
    counts = np.zeros(max_lag + 1, dtype=int)
    result[0] = 1.0
    counts[0] = len(sequences)
    for lag in range(1, max_lag + 1):
        left, right = [], []
        for values in sequences:
            if len(values) > lag:
                left.append(values[:-lag])
                right.append(values[lag:])
        if left:
            x = np.concatenate(left) - mean
            y = np.concatenate(right) - mean
            result[lag] = float(
                np.sum(x * y) / math.sqrt(np.sum(x * x) * np.sum(y * y))
            )
            counts[lag] = len(left)
    return result, counts


def within_lineage_acf(sequences, max_lag):
    result = np.full(max_lag + 1, np.nan)
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        estimates = []
        for values in sequences:
            if len(values) <= lag or len(values) < 3:
                continue
            centered = values - values.mean()
            denominator = float(np.sum(centered * centered))
            if denominator > 0.0:
                estimates.append(
                    float(np.sum(centered[:-lag] * centered[lag:]) / denominator)
                )
        if estimates:
            result[lag] = float(np.mean(estimates))
    return result


def two_scale_curve(theta, lags):
    fast_weight = 1.0 / (1.0 + np.exp(-theta[0]))
    fast_length = np.exp(theta[1])
    slow_length = fast_length + np.exp(theta[2])
    return fast_weight * np.exp(-lags / fast_length) + (
        1.0 - fast_weight
    ) * np.exp(-lags / slow_length)


def fit_scales(acf, max_fit_lag=15):
    lags = np.arange(1, min(max_fit_lag, len(acf) - 1) + 1)
    observed = acf[lags]
    valid = np.isfinite(observed)
    lags, observed = lags[valid], observed[valid]
    fitted = least_squares(
        lambda theta: two_scale_curve(theta, lags) - observed,
        x0=[2.0, math.log(0.8), math.log(15.0)],
        bounds=(
            [-8.0, math.log(0.05), math.log(0.05)],
            [8.0, math.log(50.0), math.log(1000.0)],
        ),
    )
    fast_weight = float(1.0 / (1.0 + np.exp(-fitted.x[0])))
    fast_length = float(np.exp(fitted.x[1]))
    slow_length = float(fast_length + np.exp(fitted.x[2]))
    residual = two_scale_curve(fitted.x, lags) - observed
    sse_two = float(np.sum(residual * residual))
    one = minimize_scalar(
        lambda log_length: float(
            np.sum((np.exp(-lags / np.exp(log_length)) - observed) ** 2)
        ),
        bounds=(math.log(0.05), math.log(1000.0)),
        method="bounded",
    )
    n = len(lags)
    sse_one = max(float(one.fun), 1e-15)
    sse_two = max(sse_two, 1e-15)
    # One versus three free parameters; positive means two scales are favored.
    delta_aic = n * math.log(sse_one / sse_two) - 4.0
    return {
        "fast_weight": fast_weight,
        "slow_weight": 1.0 - fast_weight,
        "fast_length_generations": fast_length,
        "slow_length_generations": slow_length,
        "one_scale_length_generations": float(np.exp(one.x)),
        "delta_aic_one_minus_two": delta_aic,
        "rmse_two_scale": math.sqrt(sse_two / n),
        "curve": two_scale_curve(fitted.x, np.arange(len(acf))),
    }


def bootstrap_scales(sequences, replicates=500, seed=73194):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(replicates):
        sampled = [sequences[index] for index in rng.integers(0, len(sequences), len(sequences))]
        acf, _ = pooled_acf(sampled, 15)
        rows.append({key: value for key, value in fit_scales(acf).items() if key != "curve"})
    return pd.DataFrame(rows)


def make_plot(acf_table, scale_table):
    labels = {
        "q": r"Effective jump $q$",
        "alpha": r"Growth rate $\alpha$",
        "tau": r"Division time $\tau$",
        "epsilon": r"Endpoint residual $\epsilon$",
    }
    colors = {"pooled": "#23364d", "within": "#73849a", "fit": "#c4512c"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True, sharey=True, constrained_layout=True)
    for axis, name in zip(axes.ravel(), VARIABLES):
        table = acf_table.loc[acf_table.variable.eq(name)]
        row = scale_table.loc[scale_table.variable.eq(name)].iloc[0]
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.plot(table.lag, table.pooled_acf, color=colors["pooled"], marker="o", ms=3.5, label="Pooled")
        axis.plot(table.lag, table.within_lineage_acf, color=colors["within"], marker="s", ms=3.0, label="Lineage-centered")
        axis.plot(table.lag, table.two_scale_fit, color=colors["fit"], lw=2.0, label="Two-scale fit")
        axis.set_title(labels[name])
        axis.set_xlabel("Lag (generations)")
        axis.set_ylabel("ACF")
        axis.grid(alpha=0.16)
        axis.text(
            0.98,
            0.92,
            rf"$\xi_f={row.fast_length_generations:.2f}$, $\xi_s={row.slow_length_generations:.1f}$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Glu transition variables: short and long correlation scales", fontsize=15)
    fig.savefig(OUT / "transition_acf_scales.png", dpi=190)
    fig.savefig(OUT / "transition_acf_scales.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sequences = transition_sequences()
    acf_rows, scale_rows, bootstrap_rows, long_rows = [], [], [], []
    for index, name in enumerate(VARIABLES):
        pooled, counts = pooled_acf(sequences[name], 15)
        within = within_lineage_acf(sequences[name], 15)
        fitted = fit_scales(pooled)
        bootstrap = bootstrap_scales(sequences[name], seed=73194 + 1000 * index)
        for column in [
            "fast_weight",
            "slow_weight",
            "fast_length_generations",
            "slow_length_generations",
            "delta_aic_one_minus_two",
        ]:
            low, high = np.quantile(bootstrap[column], [0.025, 0.975])
            fitted[f"{column}_lo95"] = float(low)
            fitted[f"{column}_hi95"] = float(high)
        curve = fitted.pop("curve")
        scale_rows.append({"variable": name, **fitted})
        bootstrap.insert(0, "variable", name)
        bootstrap.insert(1, "replicate", np.arange(len(bootstrap)))
        bootstrap_rows.append(bootstrap)
        long_sequences = [values for values in sequences[name] if len(values) >= 20]
        long_acf, _ = pooled_acf(long_sequences, 15)
        long_fit = fit_scales(long_acf)
        long_rows.append(
            {
                "variable": name,
                "n_lineages": len(long_sequences),
                **{key: value for key, value in long_fit.items() if key != "curve"},
            }
        )
        for lag in range(len(pooled)):
            acf_rows.append(
                {
                    "variable": name,
                    "lag": lag,
                    "pooled_acf": pooled[lag],
                    "within_lineage_acf": within[lag],
                    "two_scale_fit": curve[lag],
                    "n_lineages": counts[lag],
                }
            )
    acf_table = pd.DataFrame(acf_rows)
    scale_table = pd.DataFrame(scale_rows)
    acf_table.to_csv(OUT / "transition_acfs.csv", index=False)
    scale_table.to_csv(OUT / "correlation_scales.csv", index=False)
    pd.DataFrame(long_rows).to_csv(OUT / "long_lineage_sensitivity.csv", index=False)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(OUT / "bootstrap_scales.csv", index=False)
    make_plot(acf_table, scale_table)
    summary = {
        "definitions": {
            "q": "log(b[i+1]/b[i]) - alpha[i]*tau[i] = log(f[i]) + epsilon[i]",
            "epsilon": "q[i] - log(f[i]), using the actual daughter-row division fraction",
            "two_scale_curve": "w_fast exp(-lag/xi_fast) + (1-w_fast) exp(-lag/xi_slow)",
        },
        "design": "303 generation-zero contiguous prefixes; gaps are never bridged; 500 whole-lineage bootstrap fits. A separate sensitivity analysis keeps the 172 lineages with at least 20 transitions.",
        "interpretation_warning": "The pooled slow component includes lineage-to-lineage mean heterogeneity and nonstationarity. It is not by itself proof of a dynamic slow AR process.",
        "scales": scale_table.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(scale_table.to_string(index=False))


if __name__ == "__main__":
    main()
