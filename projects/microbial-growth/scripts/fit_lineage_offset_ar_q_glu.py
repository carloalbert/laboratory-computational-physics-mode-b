#!/usr/bin/env python3
"""Fit Glu with q represented by an AR(1) plus a lineage random offset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from autoregressive_residual_model import (  # noqa: E402
    lineage_acf,
    random_intercept_ar1_log_likelihood,
    sample_growth_time,
    transform_gamma,
)
from fit_autoregressive_residual_glu import (  # noqa: E402
    build_transitions,
    contiguous_prefixes,
    fit_alpha,
    fit_hazard,
    fit_initial_birth,
    observed_matrix,
    summarize_acf,
)
from glu_workflow import load_data  # noqa: E402


OUT = (
    PROJECT_DIR
    / "outputs"
    / "model_comparison"
    / "Glu"
    / "lineage_offset_ar_q_protein_hazard"
)


def q_sequences(groups):
    values, generations = [], []
    for group in groups:
        q, generation = [], []
        for index in range(len(group) - 1):
            current, daughter = group.iloc[index], group.iloc[index + 1]
            if int(daughter.generation) != int(current.generation) + 1:
                continue
            q.append(
                math.log(float(daughter.length_birth) / float(current.length_birth))
                - float(current.alpha) * float(current.time)
            )
            generation.append(int(current.generation))
        values.append(np.asarray(q, dtype=float))
        generations.append(np.asarray(generation, dtype=int))
    return values, generations


def fit_q(groups):
    sequences, generations = q_sequences(groups)

    def unpack(theta):
        return math.exp(theta[0]), math.tanh(theta[1]), math.exp(theta[2])

    def objective(theta):
        sigma_offset, rho, sigma_innovation = unpack(theta)
        value, _ = random_intercept_ar1_log_likelihood(
            sequences, generations, sigma_offset, rho, sigma_innovation
        )
        return -value

    flat = np.concatenate(sequences)
    starts = [
        [math.log(0.03), np.arctanh(0.0), math.log(np.std(flat))],
        [math.log(0.05), np.arctanh(0.2), math.log(0.08)],
        [math.log(0.08), np.arctanh(-0.2), math.log(0.06)],
    ]
    bounds = [(math.log(1e-4), math.log(1.0)), (-2.0, 2.0), (math.log(1e-4), math.log(1.0))]
    fits = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        for start in starts
    ]
    fitted = min(fits, key=lambda result: result.fun)
    sigma_offset, rho, sigma_innovation = unpack(fitted.x)
    log_likelihood, mean = random_intercept_ar1_log_likelihood(
        sequences, generations, sigma_offset, rho, sigma_innovation
    )
    def no_offset_objective(theta):
        value, _ = random_intercept_ar1_log_likelihood(
            sequences,
            generations,
            1e-8,
            math.tanh(theta[0]),
            math.exp(theta[1]),
        )
        return -value

    no_offset = minimize(
        no_offset_objective,
        [np.arctanh(0.05), math.log(np.std(flat))],
        method="L-BFGS-B",
        bounds=[(-2.0, 2.0), (math.log(1e-4), math.log(1.0))],
    )
    no_offset_log_likelihood = -float(no_offset.fun)
    stationary_sd = sigma_innovation / math.sqrt(1.0 - rho**2)
    total_sd = math.sqrt(sigma_offset**2 + stationary_sd**2)
    return {
        "mean": mean,
        "sigma_lineage_offset": sigma_offset,
        "rho": rho,
        "sigma_innovation": sigma_innovation,
        "stationary_ar_sd": stationary_sd,
        "total_marginal_sd": total_sd,
        "offset_variance_fraction": sigma_offset**2 / total_sd**2,
        "ar_correlation_length_generations": (
            -1.0 / math.log(abs(rho)) if abs(rho) > 0.0 else 0.0
        ),
        "log_likelihood": log_likelihood,
        "no_offset_log_likelihood": no_offset_log_likelihood,
        "likelihood_ratio_twice_log_likelihood_gain": 2.0
        * (log_likelihood - no_offset_log_likelihood),
        "aic_gain_over_no_offset": 2.0
        * (log_likelihood - no_offset_log_likelihood)
        - 2.0,
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "multistart_log_likelihoods": [-float(result.fun) for result in fits],
    }


def simulate(params, lengths, replicates, seed=62519):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages, horizon = len(lengths), int(lengths.max())
    log_birth = np.zeros((replicates, lineages, horizon))
    alpha = np.zeros_like(log_birth)
    growth = np.full_like(log_birth, np.nan)
    time = np.full_like(log_birth, np.nan)
    q = np.full_like(log_birth, np.nan)
    b0, afit, hazard, qfit = (
        params["initial_birth"],
        params["alpha"],
        params["hazard"],
        params["q"],
    )
    current = b0["log_mean"] + b0["log_sd"] * rng.normal(
        size=(replicates, lineages)
    )
    alpha_latent = rng.normal(size=(replicates, lineages))
    lineage_offset = qfit["sigma_lineage_offset"] * rng.normal(
        size=(replicates, lineages)
    )
    q_residual = qfit["stationary_ar_sd"] * rng.normal(
        size=(replicates, lineages)
    )
    for generation in range(horizon):
        active = lengths > generation
        if generation:
            alpha_latent[:, active] = afit["rho"] * alpha_latent[:, active] + math.sqrt(
                1.0 - afit["rho"] ** 2
            ) * rng.normal(size=(replicates, int(active.sum())))
        log_birth[:, active, generation] = current[:, active]
        alpha[:, active, generation] = transform_gamma(
            alpha_latent[:, active], afit["shape"], afit["scale"]
        )
        continuing = lengths > generation + 1
        if not continuing.any():
            continue
        if generation:
            q_residual[:, continuing] = (
                qfit["rho"] * q_residual[:, continuing]
                + qfit["sigma_innovation"]
                * rng.normal(size=(replicates, int(continuing.sum())))
            )
        q_value = (
            qfit["mean"]
            + lineage_offset[:, continuing]
            + q_residual[:, continuing]
        )
        growth_time = sample_growth_time(
            current[:, continuing].ravel(),
            math.log(hazard["u"]),
            hazard["omega2"],
            hazard["eta"],
            rng,
        ).reshape(replicates, int(continuing.sum()))
        q[:, continuing, generation] = q_value
        growth[:, continuing, generation] = growth_time
        time[:, continuing, generation] = (
            growth_time / alpha[:, continuing, generation]
        )
        current[:, continuing] = current[:, continuing] + growth_time + q_value
    return {
        "log_birth": log_birth,
        "birth": np.exp(np.clip(log_birth, -700.0, 700.0)),
        "alpha": alpha,
        "q": q,
        "growth": growth,
        "time": time,
        "lineage_offset": lineage_offset,
    }


def q_matrix(prefixes):
    lengths = np.asarray([max(len(group) - 1, 0) for group in prefixes], dtype=int)
    matrix = np.zeros((len(prefixes), int(lengths.max())))
    for index, group in enumerate(prefixes):
        if lengths[index] == 0:
            continue
        birth = group.length_birth.to_numpy(float)
        matrix[index, : lengths[index]] = (
            np.diff(np.log(birth))
            - group.alpha.to_numpy(float)[:-1] * group.time.to_numpy(float)[:-1]
        )
    return matrix, lengths


def pooled_acf_replicates(values, lengths, max_lag):
    if values.ndim == 2:
        values = values[None, ...]
    result = np.full((len(values), max_lag + 1), np.nan)
    result[:, 0] = 1.0
    for replicate in range(len(values)):
        all_values = np.concatenate(
            [values[replicate, index, :length] for index, length in enumerate(lengths) if length]
        )
        mean = all_values.mean()
        for lag in range(1, max_lag + 1):
            left, right = [], []
            for index, length in enumerate(lengths):
                if length > lag:
                    sequence = values[replicate, index, :length]
                    left.append(sequence[:-lag])
                    right.append(sequence[lag:])
            if left:
                x = np.concatenate(left) - mean
                y = np.concatenate(right) - mean
                result[replicate, lag] = np.sum(x * y) / math.sqrt(
                    np.sum(x * x) * np.sum(y * y)
                )
    return result


def plot_diagnostics(birth_acf, q_acf, previous):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    x = birth_acf.lag.to_numpy()
    axis = axes[0]
    axis.fill_between(x, birth_acf.simulated_lo95, birth_acf.simulated_hi95, color="#c4512c", alpha=0.20, label="New model, 95%")
    axis.plot(x, birth_acf.simulated_mean, color="#c4512c", marker="s", ms=3.5, label="New model")
    if previous is not None:
        axis.plot(previous.lag, previous.simulated_mean, color="#8a98a8", ls="--", lw=1.7, label="Previous AR-$\\epsilon$ model")
    axis.plot(x, birth_acf.observed, color="#23364d", marker="o", ms=3.5, label="Observed")
    axis.set_title("Birth-size ACF")
    axis.legend(frameon=False, fontsize=8.5)

    axis = axes[1]
    axis.fill_between(q_acf.lag, q_acf.simulated_lo95, q_acf.simulated_hi95, color="#c4512c", alpha=0.20, label="New model, 95%")
    axis.plot(q_acf.lag, q_acf.simulated_mean, color="#c4512c", marker="s", ms=3.5, label="New model")
    axis.plot(q_acf.lag, q_acf.observed, color="#23364d", marker="o", ms=3.5, label="Observed")
    axis.set_title(r"Pooled effective-jump ACF ($q$)")
    for axis in axes:
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set(xlabel="Lag (generations)", ylabel="ACF")
        axis.set_xticks(np.arange(0, 16))
        axis.grid(alpha=0.16)
    fig.suptitle("Glu: autonomous lineage-offset + AR(1) q model", fontsize=15)
    fig.savefig(OUT / "birth_and_q_acf_autonomous.png", dpi=190)
    fig.savefig(OUT / "birth_and_q_acf_autonomous.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--max-lag", type=int, default=15)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    transition = build_transitions(groups)
    params = {
        "initial_birth": fit_initial_birth(groups),
        "alpha": fit_alpha(groups),
        "hazard": fit_hazard(transition),
        "q": fit_q(groups),
    }
    for name, values in params.items():
        print(name, values, flush=True)

    observed_birth, lengths = observed_matrix(prefixes)
    generated = simulate(params, lengths, args.replicates)
    birth_acf = summarize_acf(
        observed_birth, generated["birth"], lengths, args.max_lag
    )
    birth_acf.to_csv(OUT / "birth_size_acf.csv", index=False)

    observed_q, q_lengths = q_matrix(prefixes)
    observed_q_acf = pooled_acf_replicates(observed_q, q_lengths, args.max_lag)[0]
    simulated_q_acf = pooled_acf_replicates(generated["q"], q_lengths, args.max_lag)
    q_low, q_high = np.nanquantile(simulated_q_acf, [0.025, 0.975], axis=0)
    q_table = pd.DataFrame(
        {
            "lag": np.arange(args.max_lag + 1),
            "observed": observed_q_acf,
            "simulated_mean": np.nanmean(simulated_q_acf, axis=0),
            "simulated_lo95": q_low,
            "simulated_hi95": q_high,
        }
    )
    q_table.to_csv(OUT / "q_acf.csv", index=False)
    previous_path = (
        PROJECT_DIR
        / "outputs/model_comparison/Glu/autoregressive_residual_protein_hazard/birth_size_acf.csv"
    )
    previous = pd.read_csv(previous_path) if previous_path.exists() else None
    plot_diagnostics(birth_acf, q_table, previous)

    example_rows = []
    for replicate in range(min(3, args.replicates)):
        for lineage in range(min(20, len(lengths))):
            for generation in range(lengths[lineage]):
                example_rows.append(
                    {
                        "replicate": replicate,
                        "lineage_index": lineage,
                        "generation": generation,
                        "birth": generated["birth"][replicate, lineage, generation],
                        "alpha": generated["alpha"][replicate, lineage, generation],
                        "time": generated["time"][replicate, lineage, generation],
                        "growth_time": generated["growth"][replicate, lineage, generation],
                        "q": generated["q"][replicate, lineage, generation],
                        "lineage_q_offset": generated["lineage_offset"][replicate, lineage],
                    }
                )
    pd.DataFrame(example_rows).to_csv(OUT / "autonomous_example_trajectories.csv", index=False)

    observed_flat = np.concatenate(
        [observed_birth[index, :length] for index, length in enumerate(lengths)]
    )
    generated_flat = np.concatenate(
        [generated["birth"][:, index, :length].ravel() for index, length in enumerate(lengths)]
    )
    result = {
        "model": {
            "recursion": "log b[i+1] = log b[i] + alpha[i]*tau[i] + q[i]",
            "q": "q[l,i]=mu_q+a[l]+r[l,i]; a[l]~Normal(0,sigma_a^2); r[l,i]=rho_q*r[l,i-1]+sigma_q*z[l,i]",
            "q_likelihood": "Exact Gaussian marginal likelihood after integrating each lineage offset; stationary AR residual, reset across gaps",
            "hazard": "h(t|b,alpha)=alpha*omega2*sigmoid(eta*log(b*(exp(alpha*t)-1)/u))",
            "alpha": "Gamma marginal, Gaussian-copula AR(1)",
        },
        "fit": params,
        "simulation": {
            "replicates": args.replicates,
            "replayed_quantities": "none; only contiguous-prefix lineage lengths are reused",
            "observed_birth_mean": float(observed_flat.mean()),
            "observed_birth_sd": float(observed_flat.std()),
            "simulated_birth_mean": float(generated_flat.mean()),
            "simulated_birth_sd": float(generated_flat.std()),
            "birth_acf_rmse_lags_1_to_max": float(
                np.sqrt(np.mean((birth_acf.observed[1:] - birth_acf.simulated_mean[1:]) ** 2))
            ),
            "q_acf_rmse_lags_1_to_max": float(
                np.sqrt(np.mean((q_table.observed[1:] - q_table.simulated_mean[1:]) ** 2))
            ),
        },
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2))
    print(birth_acf.to_string(index=False), flush=True)
    print(q_table.to_string(index=False), flush=True)
    print(json.dumps(result["simulation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
