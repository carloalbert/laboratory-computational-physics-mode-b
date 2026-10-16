#!/usr/bin/env python3
"""Fit and validate a joint semiparametric q-alpha-tau model for Glu.

The model is deliberately statistical: it contains no division hazard.  Each
marginal is represented empirically, while dependence is described by a
Gaussian-copula error-correcting VAR(2).  The measured identity

    log b[i+1] = log b[i] + q[i] + alpha[i] * tau[i]

is used unchanged in autonomous simulations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_autoregressive_residual_glu import contiguous_prefixes  # noqa: E402
from fit_fast_slow_ar_q_glu import fit_q  # noqa: E402
from glu_workflow import load_data  # noqa: E402


OUT = PROJECT_DIR / "outputs/model_comparison/Glu/joint_q_alpha_tau"
NAMES = ("q", "alpha", "tau")
COLORS = {"observed": "#23364d", "simulated": "#c4512c"}


def empirical_reference(values):
    return np.sort(np.asarray(values, dtype=float))


def normal_score(values, reference):
    values = np.asarray(values, dtype=float)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    probability = 0.5 * (left + right) / len(reference)
    return norm.ppf(np.clip(probability, 1e-10, 1.0 - 1e-10))


def empirical_quantile(probability, reference):
    # Invert the mid-rank plotting position p_i=(i+1/2)/n.
    position = np.clip(probability * len(reference) - 0.5, 0.0, len(reference) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(reference) - 1)
    weight = position - lower
    return (1.0 - weight) * reference[lower] + weight * reference[upper]


def transition_sequences(groups):
    sequences = []
    for group in contiguous_prefixes(groups):
        if len(group) < 2:
            continue
        birth = group.length_birth.to_numpy(float)
        alpha = group.alpha.to_numpy(float)[:-1]
        tau = group.time.to_numpy(float)[:-1]
        q = np.diff(np.log(birth)) - alpha * tau
        x = np.log(birth)
        sequences.append(
            {
                "x": x,
                "x_setpoint": float(x.mean()),
                "y": np.column_stack([q, alpha, tau]),
            }
        )
    return sequences


def gaussian_log_likelihood(residual):
    residual = np.asarray(residual, dtype=float)
    covariance = residual.T @ residual / len(residual)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Residual covariance is not positive definite")
    dimension = residual.shape[1]
    value = -0.5 * len(residual) * (
        dimension * (1.0 + math.log(2.0 * math.pi)) + log_determinant
    )
    return float(value), covariance


def fit_regression(design, target):
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    log_likelihood, covariance = gaussian_log_likelihood(residual)
    return coefficients, covariance, log_likelihood, residual


def alpha_tau_scores(y, references):
    return np.column_stack(
        [normal_score(y[:, 1], references["alpha"]), normal_score(y[:, 2], references["tau"])]
    )


def collect_dynamics(sequences, references, q_mean, order):
    design, target = [], []
    for sequence in sequences:
        y = sequence["y"]
        if len(y) <= order:
            continue
        z = alpha_tau_scores(y, references)
        q = y[:, 0] - q_mean
        for index in range(order, len(y)):
            row = [1.0]
            for lag in range(1, order + 1):
                row.extend(z[index - lag])
            row.extend(q[index - np.arange(order + 1)])
            row.append(sequence["x"][index] - sequence["x_setpoint"])
            design.append(row)
            target.append(z[index])
    return np.asarray(design), np.asarray(target)


def fit_model(sequences, groups):
    pooled = np.concatenate([sequence["y"] for sequence in sequences])
    references = {
        name: empirical_reference(pooled[:, column])
        for column, name in enumerate(NAMES)
    }
    setpoint_reference = empirical_reference(
        [sequence["x_setpoint"] for sequence in sequences]
    )
    initial_delta_reference = empirical_reference(
        [sequence["x"][0] - sequence["x_setpoint"] for sequence in sequences]
    )
    initial_size_scores = np.column_stack(
        [
            normal_score(
                [sequence["x_setpoint"] for sequence in sequences], setpoint_reference
            ),
            normal_score(
                [sequence["x"][0] - sequence["x_setpoint"] for sequence in sequences],
                initial_delta_reference,
            ),
        ]
    )
    initial_size_mean = initial_size_scores.mean(axis=0)
    initial_size_covariance = np.cov(initial_size_scores, rowvar=False, ddof=0)

    q_fit = fit_q(groups)

    # Initial alpha-tau state conditional on the independently generated q_0
    # and the sampled initial size.
    initial_design, initial_target = [], []
    for sequence in sequences:
        initial_design.append(
            [
                1.0,
                sequence["y"][0, 0] - q_fit["mean"],
                sequence["x"][0] - sequence["x_setpoint"],
            ]
        )
        initial_target.append(alpha_tau_scores(sequence["y"][:1], references)[0])
    initial_coefficients, initial_covariance, initial_log_likelihood, initial_residual = fit_regression(
        np.asarray(initial_design), np.asarray(initial_target)
    )

    design1, target1 = collect_dynamics(
        sequences, references, q_fit["mean"], order=1
    )
    coefficients1, covariance1, log_likelihood1, residual1 = fit_regression(design1, target1)

    design2, target2 = collect_dynamics(
        sequences, references, q_fit["mean"], order=2
    )
    coefficients2, covariance2, log_likelihood2, residual2 = fit_regression(design2, target2)

    # The no-feedback comparator differs only by the final x column.
    coefficients_no_x, covariance_no_x, log_likelihood_no_x, _ = fit_regression(
        design2[:, :-1], target2
    )
    aic_gain_x = 2.0 * (log_likelihood2 - log_likelihood_no_x) - 2.0 * target2.shape[1]

    return {
        "references": references,
        "setpoint_reference": setpoint_reference,
        "initial_delta_reference": initial_delta_reference,
        "initial_size_mean": initial_size_mean,
        "initial_size_covariance": initial_size_covariance,
        "q": q_fit,
        "initial_coefficients": initial_coefficients,
        "initial_covariance": initial_covariance,
        "initial_residual": initial_residual,
        "initial_log_likelihood": initial_log_likelihood,
        "var1_coefficients": coefficients1,
        "var1_covariance": covariance1,
        "var1_residual": residual1,
        "var1_log_likelihood": log_likelihood1,
        "var2_coefficients": coefficients2,
        "var2_covariance": covariance2,
        "var2_residual": residual2,
        "var2_log_likelihood": log_likelihood2,
        "no_x_coefficients": coefficients_no_x,
        "no_x_covariance": covariance_no_x,
        "no_x_log_likelihood": log_likelihood_no_x,
        "aic_gain_from_size_feedback": float(aic_gain_x),
        "n_var2": int(len(target2)),
    }


def transform_scores(z, references):
    return np.stack(
        [
            empirical_quantile(norm.cdf(z[..., column]), references[name])
            for column, name in enumerate(("alpha", "tau"))
        ],
        axis=-1,
    )


def simulate(model, lengths, replicates=300, seed=93841):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages = len(lengths)
    horizon = int(lengths.max())
    shape = (replicates, lineages)

    def resample_residual(name):
        residual = model[name]
        return residual[rng.integers(0, len(residual), size=shape)]

    initial_size_scores = rng.multivariate_normal(
        model["initial_size_mean"], model["initial_size_covariance"], size=shape
    )
    setpoint = empirical_quantile(
        norm.cdf(initial_size_scores[..., 0]), model["setpoint_reference"]
    )
    x = setpoint + empirical_quantile(
        norm.cdf(initial_size_scores[..., 1]), model["initial_delta_reference"]
    )
    q_fit = model["q"]
    fast = q_fit["stationary_fast_sd"] * rng.normal(size=shape)
    slow = q_fit["stationary_slow_sd"] * rng.normal(size=shape)
    q_previous = q_fit["mean"] + fast + slow
    initial_mean = (
        model["initial_coefficients"][0]
        + (q_previous - q_fit["mean"])[..., None] * model["initial_coefficients"][1]
        + (x - setpoint)[..., None] * model["initial_coefficients"][2]
    )
    z_previous = initial_mean + resample_residual("initial_residual")
    alpha_tau_previous = transform_scores(z_previous, model["references"])
    y_previous = np.concatenate([q_previous[..., None], alpha_tau_previous], axis=-1)

    result_y = np.full((replicates, lineages, horizon, 3), np.nan)
    result_x = np.full((replicates, lineages, horizon + 1), np.nan)
    result_x[:, :, 0] = x
    result_y[:, :, 0, :] = y_previous

    # Advance size using the exact identity, then generate the first lagged state.
    increment = y_previous[..., 0] + y_previous[..., 1] * y_previous[..., 2]
    x = x + increment
    result_x[:, :, 1] = x

    fast = q_fit["rho_fast"] * fast + q_fit["sigma_fast_innovation"] * rng.normal(size=shape)
    slow = q_fit["rho_slow"] * slow + q_fit["sigma_slow_innovation"] * rng.normal(size=shape)
    q_current = q_fit["mean"] + fast + slow
    coefficient1 = model["var1_coefficients"]
    mean1 = (
        coefficient1[0]
        + np.einsum("...j,jk->...k", z_previous, coefficient1[1:3])
        + (q_current - q_fit["mean"])[..., None] * coefficient1[3]
        + (q_previous - q_fit["mean"])[..., None] * coefficient1[4]
        + (x - setpoint)[..., None] * coefficient1[5]
    )
    z_current = mean1 + resample_residual("var1_residual")
    if horizon > 1:
        alpha_tau_current = transform_scores(z_current, model["references"])
        y_current = np.concatenate([q_current[..., None], alpha_tau_current], axis=-1)
        result_y[:, :, 1, :] = y_current
    else:
        y_current = np.zeros_like(y_previous)

    coefficient2 = model["var2_coefficients"]
    for generation in range(1, horizon):
        if generation > 1:
            alpha_tau_current = transform_scores(z_current, model["references"])
            y_current = np.concatenate([q_current[..., None], alpha_tau_current], axis=-1)
            result_y[:, :, generation, :] = y_current
        increment = y_current[..., 0] + y_current[..., 1] * y_current[..., 2]
        x = x + increment
        result_x[:, :, generation + 1] = x
        if generation + 1 >= horizon:
            continue
        fast = q_fit["rho_fast"] * fast + q_fit["sigma_fast_innovation"] * rng.normal(size=shape)
        slow = q_fit["rho_slow"] * slow + q_fit["sigma_slow_innovation"] * rng.normal(size=shape)
        q_next = q_fit["mean"] + fast + slow
        mean2 = (
            coefficient2[0]
            + np.einsum("...j,jk->...k", z_current, coefficient2[1:3])
            + np.einsum("...j,jk->...k", z_previous, coefficient2[3:5])
            + (q_next - q_fit["mean"])[..., None] * coefficient2[5]
            + (q_current - q_fit["mean"])[..., None] * coefficient2[6]
            + (q_previous - q_fit["mean"])[..., None] * coefficient2[7]
            + (x - setpoint)[..., None] * coefficient2[8]
        )
        z_next = mean2 + resample_residual("var2_residual")
        z_previous, z_current = z_current, z_next
        q_previous, q_current = q_current, q_next

    # Entries beyond each observed lineage length are missing, not zeros.
    for lineage, length in enumerate(lengths):
        result_y[:, lineage, length:, :] = np.nan
        result_x[:, lineage, length + 1 :] = np.nan
    return {"y": result_y, "x": result_x}


def observed_arrays(sequences):
    lengths = np.asarray([len(sequence["y"]) for sequence in sequences], dtype=int)
    horizon = int(lengths.max())
    y = np.full((len(sequences), horizon, 3), np.nan)
    x = np.full((len(sequences), horizon + 1), np.nan)
    for lineage, sequence in enumerate(sequences):
        length = lengths[lineage]
        y[lineage, :length] = sequence["y"]
        x[lineage, : length + 1] = sequence["x"]
    return y, x, lengths


def acf_replicates(values, lengths, max_lag=15):
    if values.ndim == 2:
        values = values[None, ...]
    result = np.full((len(values), max_lag + 1), np.nan)
    result[:, 0] = 1.0
    for replicate in range(len(values)):
        mean = float(np.nanmean(values[replicate]))
        for lag in range(1, max_lag + 1):
            left, right = [], []
            for lineage, length in enumerate(lengths):
                if length > lag:
                    left.append(values[replicate, lineage, : length - lag] - mean)
                    right.append(values[replicate, lineage, lag:length] - mean)
            if left:
                a, b = np.concatenate(left), np.concatenate(right)
                result[replicate, lag] = np.sum(a * b) / math.sqrt(
                    np.sum(a * a) * np.sum(b * b)
                )
    return result


def cross_correlation_replicates(left, right, lengths, lags):
    if left.ndim == 2:
        left = left[None, ...]
    if right.ndim == 2:
        right = right[None, ...]
    result = np.full((len(left), len(lags)), np.nan)
    for replicate in range(len(left)):
        for column, lag in enumerate(lags):
            a, b = [], []
            for lineage, length in enumerate(lengths):
                if length <= abs(lag):
                    continue
                x = left[replicate, lineage, :length]
                y = right[replicate, lineage, :length]
                if lag > 0:
                    a.append(x[:-lag])
                    b.append(y[lag:])
                elif lag < 0:
                    a.append(x[-lag:])
                    b.append(y[:lag])
                else:
                    a.append(x)
                    b.append(y)
            if a:
                result[replicate, column] = np.corrcoef(
                    np.concatenate(a), np.concatenate(b)
                )[0, 1]
    return result


def diagnostic_table(observed, simulated, lags):
    low, high = np.nanquantile(simulated, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "lag": lags,
            "observed": observed,
            "simulated_mean": np.nanmean(simulated, axis=0),
            "simulated_lo95": low,
            "simulated_hi95": high,
        }
    )


def correlation_matrix(values):
    flat = values.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    return np.corrcoef(flat, rowvar=False)


def summarize_and_plot(observed_y, observed_x, simulated, lengths, max_lag=15):
    OUT.mkdir(parents=True, exist_ok=True)
    lags = np.arange(max_lag + 1)
    cross_lags = np.arange(-8, 9)
    acf_tables, cross_tables = {}, {}
    acf_rmse, cross_rmse = {}, {}
    for column, name in enumerate(NAMES):
        observed = acf_replicates(observed_y[..., column], lengths, max_lag)[0]
        generated = acf_replicates(simulated["y"][..., column], lengths, max_lag)
        table = diagnostic_table(observed, generated, lags)
        table.to_csv(OUT / f"{name}_acf.csv", index=False)
        acf_tables[name] = table
        acf_rmse[name] = float(
            np.sqrt(np.nanmean((table.observed[1:] - table.simulated_mean[1:]) ** 2))
        )

    for left_index, right_index in [(0, 1), (0, 2), (1, 2)]:
        left_name, right_name = NAMES[left_index], NAMES[right_index]
        key = f"{left_name}_{right_name}"
        observed = cross_correlation_replicates(
            observed_y[..., left_index], observed_y[..., right_index], lengths, cross_lags
        )[0]
        generated = cross_correlation_replicates(
            simulated["y"][..., left_index],
            simulated["y"][..., right_index],
            lengths,
            cross_lags,
        )
        table = diagnostic_table(observed, generated, cross_lags)
        table.to_csv(OUT / f"{key}_cross_correlation.csv", index=False)
        cross_tables[key] = table
        cross_rmse[key] = float(
            np.sqrt(np.nanmean((table.observed - table.simulated_mean) ** 2))
        )

    birth_lengths = lengths + 1
    observed_birth = acf_replicates(np.exp(observed_x), birth_lengths, max_lag)[0]
    generated_birth = acf_replicates(np.exp(simulated["x"]), birth_lengths, max_lag)
    birth_table = diagnostic_table(observed_birth, generated_birth, lags)
    birth_table.to_csv(OUT / "birth_size_acf.csv", index=False)
    birth_rmse = float(
        np.sqrt(np.nanmean((birth_table.observed[1:] - birth_table.simulated_mean[1:]) ** 2))
    )

    observed_correlation = correlation_matrix(observed_y)
    simulated_correlations = np.stack(
        [correlation_matrix(simulated["y"][replicate]) for replicate in range(len(simulated["y"]))]
    )
    simulated_correlation = simulated_correlations.mean(axis=0)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    for column, name in enumerate(NAMES):
        table = acf_tables[name]
        axis = axes[0, column]
        axis.fill_between(table.lag, table.simulated_lo95, table.simulated_hi95,
                          color=COLORS["simulated"], alpha=0.18)
        axis.plot(table.lag, table.simulated_mean, "s-", ms=3.5,
                  color=COLORS["simulated"], label="Joint model")
        axis.plot(table.lag, table.observed, "o-", ms=3.5,
                  color=COLORS["observed"], label="Observed")
        axis.axhline(0.0, color="0.65", lw=0.8)
        axis.set(title=rf"ACF of ${name}$", xlabel="Lag (generations)", ylabel="Correlation")
        axis.grid(alpha=0.15)
    axes[0, 0].legend(frameon=False)

    labels = {"q_alpha": r"Corr($q_i,\alpha_{i+k}$)",
              "q_tau": r"Corr($q_i,\tau_{i+k}$)",
              "alpha_tau": r"Corr($\alpha_i,\tau_{i+k}$)"}
    for axis, key in zip(axes[1], ("q_alpha", "q_tau", "alpha_tau")):
        table = cross_tables[key]
        axis.fill_between(table.lag, table.simulated_lo95, table.simulated_hi95,
                          color=COLORS["simulated"], alpha=0.18)
        axis.plot(table.lag, table.simulated_mean, "s-", ms=3.5,
                  color=COLORS["simulated"])
        axis.plot(table.lag, table.observed, "o-", ms=3.5,
                  color=COLORS["observed"])
        axis.axhline(0.0, color="0.65", lw=0.8)
        axis.set(title=labels[key], xlabel="k (generations)", ylabel="Correlation")
        axis.grid(alpha=0.15)

    axis = axes[2, 0]
    axis.fill_between(birth_table.lag, birth_table.simulated_lo95, birth_table.simulated_hi95,
                      color=COLORS["simulated"], alpha=0.18)
    axis.plot(birth_table.lag, birth_table.simulated_mean, "s-", ms=3.5,
              color=COLORS["simulated"])
    axis.plot(birth_table.lag, birth_table.observed, "o-", ms=3.5,
              color=COLORS["observed"])
    axis.axhline(0.0, color="0.65", lw=0.8)
    axis.set(title="Implied birth-size ACF", xlabel="Lag (generations)", ylabel="Correlation")
    axis.grid(alpha=0.15)

    for axis, matrix, title in zip(
        axes[2, 1:],
        (observed_correlation, simulated_correlation),
        ("Observed contemporaneous correlation", "Simulated contemporaneous correlation"),
    ):
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(range(3), NAMES)
        axis.set_yticks(range(3), NAMES)
        axis.set_title(title)
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=axes[2, 1:], shrink=0.8)
    fig.suptitle("Glu: error-correcting semiparametric joint model, no division hazard", fontsize=16)
    fig.savefig(OUT / "joint_q_alpha_tau_diagnostics.png", dpi=190)
    fig.savefig(OUT / "joint_q_alpha_tau_diagnostics.pdf")
    plt.close(fig)

    observed_flat = observed_y.reshape(-1, 3)
    observed_flat = observed_flat[np.isfinite(observed_flat).all(axis=1)]
    simulated_flat = simulated["y"].reshape(-1, 3)
    simulated_flat = simulated_flat[np.isfinite(simulated_flat).all(axis=1)]
    marginal = {}
    for column, name in enumerate(NAMES):
        marginal[name] = {
            "observed_mean": float(observed_flat[:, column].mean()),
            "observed_sd": float(observed_flat[:, column].std()),
            "simulated_mean": float(simulated_flat[:, column].mean()),
            "simulated_sd": float(simulated_flat[:, column].std()),
        }
    growth_observed = observed_flat[:, 1] * observed_flat[:, 2]
    growth_simulated = simulated_flat[:, 1] * simulated_flat[:, 2]
    observed_rank_correlation = np.asarray(spearmanr(observed_flat, axis=0).statistic)
    simulated_rank_correlation = np.asarray(spearmanr(simulated_flat, axis=0).statistic)
    return {
        "marginals": marginal,
        "acf_rmse_lags_1_to_max": acf_rmse,
        "cross_correlation_rmse_lags_minus8_to_8": cross_rmse,
        "birth_acf_rmse_lags_1_to_max": birth_rmse,
        "observed_contemporaneous_correlation": observed_correlation.tolist(),
        "simulated_contemporaneous_correlation": simulated_correlation.tolist(),
        "observed_contemporaneous_rank_correlation": observed_rank_correlation.tolist(),
        "simulated_contemporaneous_rank_correlation": simulated_rank_correlation.tolist(),
        "observed_corr_q_alpha_tau": float(np.corrcoef(observed_flat[:, 0], growth_observed)[0, 1]),
        "simulated_corr_q_alpha_tau": float(np.corrcoef(simulated_flat[:, 0], growth_simulated)[0, 1]),
    }


def serializable_model(model):
    coefficients = model["var2_coefficients"]
    innovation_sd = np.sqrt(np.diag(model["var2_covariance"]))
    innovation_correlation = model["var2_covariance"] / np.outer(
        innovation_sd, innovation_sd
    )
    return {
        "variables": list(NAMES),
        "marginals": "two-mode Gaussian q; empirical alpha and tau quantile functions with a conditional semiparametric copula",
        "factorization": "p(size setpoint) p(q path) p(alpha,tau path | q path, birth-size deviations)",
        "lineage_size_scale": "m_l is drawn from the empirical distribution of lineage mean log birth size; feedback uses log(b_li)-m_l",
        "q_dynamics": "q_i=mu_q+r_fast_i+r_slow_i; each r is a stationary Gaussian AR(1)",
        "dynamics": "w_li=c+B1 w_l(i-1)+B2 w_l(i-2)+D0 qtilde_li+D1 qtilde_l(i-1)+D2 qtilde_l(i-2)+K(log(b_li)-m_l)+e_li, where w=(z_alpha,z_tau)",
        "size_update": "log(b_(i+1))=log(b_i)+q_i+alpha_i*tau_i",
        "q": model["q"],
        "intercept": coefficients[0].tolist(),
        "B1": coefficients[1:3].T.tolist(),
        "B2": coefficients[3:5].T.tolist(),
        "D_q_current_lag1_lag2": coefficients[5:8].T.tolist(),
        "K_size_feedback": coefficients[8].tolist(),
        "innovation_distribution": "paired empirical residual bootstrap; Gaussian quasi-likelihood is used only to estimate and compare conditional means",
        "innovation_covariance": model["var2_covariance"].tolist(),
        "innovation_correlation": innovation_correlation.tolist(),
        "n_var2_observations": model["n_var2"],
        "log_likelihood": model["var2_log_likelihood"],
        "no_size_feedback_log_likelihood": model["no_x_log_likelihood"],
        "aic_gain_from_size_feedback": model["aic_gain_from_size_feedback"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--max-lag", type=int, default=15)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    _, groups = load_data()
    sequences = transition_sequences(groups)
    observed_y, observed_x, lengths = observed_arrays(sequences)
    model = fit_model(sequences, groups)
    simulated = simulate(model, lengths, replicates=args.replicates)
    diagnostics = summarize_and_plot(
        observed_y, observed_x, simulated, lengths, max_lag=args.max_lag
    )
    result = {
        "model": serializable_model(model),
        "simulation": {
            "replicates": args.replicates,
            "replayed_quantities": "none; only contiguous-prefix lineage lengths are reused",
            **diagnostics,
        },
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
