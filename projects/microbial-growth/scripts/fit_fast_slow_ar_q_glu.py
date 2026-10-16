#!/usr/bin/env python3
"""Fit Glu with q represented by independent fast and slow AR(1) modes."""

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
    sample_growth_time,
    transform_gamma,
    two_ar1_log_likelihood,
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
from fit_lineage_offset_ar_q_glu import (  # noqa: E402
    pooled_acf_replicates,
    q_matrix,
    q_sequences,
)
from glu_workflow import load_data  # noqa: E402


OUT = (
    PROJECT_DIR
    / "outputs"
    / "model_comparison"
    / "Glu"
    / "fast_slow_ar_q_protein_hazard"
)


def fit_q(groups):
    sequences, generations = q_sequences(groups)

    def objective(theta):
        value, _ = two_ar1_log_likelihood(
            sequences,
            generations,
            theta[0],
            math.exp(theta[2]),
            theta[1],
            math.exp(theta[3]),
        )
        return -value

    flat_sd = float(np.std(np.concatenate(sequences)))
    starts = [
        [0.0, 0.92, math.log(0.95 * flat_sd), math.log(0.012)],
        [-0.10, 0.85, math.log(0.95 * flat_sd), math.log(0.018)],
        [0.10, 0.97, math.log(0.90 * flat_sd), math.log(0.008)],
        [-0.05, 0.70, math.log(0.90 * flat_sd), math.log(0.030)],
    ]
    bounds = [
        (-0.70, 0.70),
        (0.50, 0.995),
        (math.log(1e-4), math.log(0.5)),
        (math.log(1e-4), math.log(0.5)),
    ]
    fits = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        for start in starts
    ]
    fitted = min(fits, key=lambda result: result.fun)
    rho_fast, rho_slow = map(float, fitted.x[:2])
    sigma_fast, sigma_slow = map(math.exp, fitted.x[2:])
    log_likelihood, mean = two_ar1_log_likelihood(
        sequences,
        generations,
        rho_fast,
        sigma_fast,
        rho_slow,
        sigma_slow,
    )
    fast_sd = sigma_fast / math.sqrt(1.0 - rho_fast**2)
    slow_sd = sigma_slow / math.sqrt(1.0 - rho_slow**2)
    total_variance = fast_sd**2 + slow_sd**2
    static_path = (
        PROJECT_DIR
        / "outputs/model_comparison/Glu/lineage_offset_ar_q_protein_hazard/fit_results.json"
    )
    static_fit = json.loads(static_path.read_text())["fit"]["q"] if static_path.exists() else None
    result = {
        "mean": mean,
        "rho_fast": rho_fast,
        "sigma_fast_innovation": sigma_fast,
        "stationary_fast_sd": fast_sd,
        "fast_correlation_length_generations": (
            -1.0 / math.log(abs(rho_fast)) if abs(rho_fast) > 0.0 else 0.0
        ),
        "rho_slow": rho_slow,
        "sigma_slow_innovation": sigma_slow,
        "stationary_slow_sd": slow_sd,
        "slow_correlation_length_generations": -1.0 / math.log(rho_slow),
        "slow_variance_fraction": slow_sd**2 / total_variance,
        "total_marginal_sd": math.sqrt(total_variance),
        "log_likelihood": log_likelihood,
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "multistart_log_likelihoods": [-float(value.fun) for value in fits],
    }
    if static_fit is not None:
        gain = log_likelihood - float(static_fit["log_likelihood"])
        result.update(
            {
                "static_offset_log_likelihood": float(static_fit["log_likelihood"]),
                "twice_log_likelihood_gain_over_static_offset": 2.0 * gain,
                "aic_gain_over_static_offset": 2.0 * gain - 2.0,
            }
        )
    return result


def simulate(params, lengths, replicates, seed=73521):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages, horizon = len(lengths), int(lengths.max())
    log_birth = np.zeros((replicates, lineages, horizon))
    alpha = np.zeros_like(log_birth)
    growth = np.full_like(log_birth, np.nan)
    time = np.full_like(log_birth, np.nan)
    q = np.full_like(log_birth, np.nan)
    q_fast = np.full_like(log_birth, np.nan)
    q_slow = np.full_like(log_birth, np.nan)
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
    fast = qfit["stationary_fast_sd"] * rng.normal(size=(replicates, lineages))
    slow = qfit["stationary_slow_sd"] * rng.normal(size=(replicates, lineages))
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
            fast[:, continuing] = (
                qfit["rho_fast"] * fast[:, continuing]
                + qfit["sigma_fast_innovation"]
                * rng.normal(size=(replicates, int(continuing.sum())))
            )
            slow[:, continuing] = (
                qfit["rho_slow"] * slow[:, continuing]
                + qfit["sigma_slow_innovation"]
                * rng.normal(size=(replicates, int(continuing.sum())))
            )
        q_value = qfit["mean"] + fast[:, continuing] + slow[:, continuing]
        growth_time = sample_growth_time(
            current[:, continuing].ravel(),
            math.log(hazard["u"]),
            hazard["omega2"],
            hazard["eta"],
            rng,
        ).reshape(replicates, int(continuing.sum()))
        q[:, continuing, generation] = q_value
        q_fast[:, continuing, generation] = fast[:, continuing]
        q_slow[:, continuing, generation] = slow[:, continuing]
        growth[:, continuing, generation] = growth_time
        time[:, continuing, generation] = growth_time / alpha[:, continuing, generation]
        current[:, continuing] = current[:, continuing] + growth_time + q_value
    return {
        "log_birth": log_birth,
        "birth": np.exp(np.clip(log_birth, -700.0, 700.0)),
        "alpha": alpha,
        "growth": growth,
        "time": time,
        "q": q,
        "q_fast": q_fast,
        "q_slow": q_slow,
    }


def plot_diagnostics(birth_acf, q_acf, static_birth, static_q):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    axis = axes[0]
    axis.fill_between(birth_acf.lag, birth_acf.simulated_lo95, birth_acf.simulated_hi95, color="#c4512c", alpha=0.20, label="Fast + slow AR, 95%")
    axis.plot(birth_acf.lag, birth_acf.simulated_mean, color="#c4512c", marker="s", ms=3.5, label="Fast + slow AR")
    if static_birth is not None:
        axis.plot(static_birth.lag, static_birth.simulated_mean, color="#8a98a8", ls="--", lw=1.7, label="Static-offset q")
    axis.plot(birth_acf.lag, birth_acf.observed, color="#23364d", marker="o", ms=3.5, label="Observed")
    axis.set_title("Birth-size ACF")
    axis.legend(frameon=False, fontsize=8.5)
    axis = axes[1]
    axis.fill_between(q_acf.lag, q_acf.simulated_lo95, q_acf.simulated_hi95, color="#c4512c", alpha=0.20, label="Fast + slow AR, 95%")
    axis.plot(q_acf.lag, q_acf.simulated_mean, color="#c4512c", marker="s", ms=3.5, label="Fast + slow AR")
    if static_q is not None:
        axis.plot(static_q.lag, static_q.simulated_mean, color="#8a98a8", ls="--", lw=1.7, label="Static-offset q")
    axis.plot(q_acf.lag, q_acf.observed, color="#23364d", marker="o", ms=3.5, label="Observed")
    axis.set_title(r"Pooled effective-jump ACF ($q$)")
    for axis in axes:
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set(xlabel="Lag (generations)", ylabel="ACF")
        axis.set_xticks(np.arange(0, 16))
        axis.grid(alpha=0.16)
    fig.suptitle("Glu: autonomous fast + slow AR(1) q model", fontsize=15)
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
    birth_acf = summarize_acf(observed_birth, generated["birth"], lengths, args.max_lag)
    birth_acf.to_csv(OUT / "birth_size_acf.csv", index=False)
    observed_q, q_lengths = q_matrix(prefixes)
    observed_q_acf = pooled_acf_replicates(observed_q, q_lengths, args.max_lag)[0]
    simulated_q_acf = pooled_acf_replicates(generated["q"], q_lengths, args.max_lag)
    low, high = np.nanquantile(simulated_q_acf, [0.025, 0.975], axis=0)
    q_acf = pd.DataFrame(
        {
            "lag": np.arange(args.max_lag + 1),
            "observed": observed_q_acf,
            "simulated_mean": np.nanmean(simulated_q_acf, axis=0),
            "simulated_lo95": low,
            "simulated_hi95": high,
        }
    )
    q_acf.to_csv(OUT / "q_acf.csv", index=False)
    static_dir = PROJECT_DIR / "outputs/model_comparison/Glu/lineage_offset_ar_q_protein_hazard"
    static_birth = pd.read_csv(static_dir / "birth_size_acf.csv") if (static_dir / "birth_size_acf.csv").exists() else None
    static_q = pd.read_csv(static_dir / "q_acf.csv") if (static_dir / "q_acf.csv").exists() else None
    plot_diagnostics(birth_acf, q_acf, static_birth, static_q)

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
                        "q_fast": generated["q_fast"][replicate, lineage, generation],
                        "q_slow": generated["q_slow"][replicate, lineage, generation],
                    }
                )
    pd.DataFrame(example_rows).to_csv(OUT / "autonomous_example_trajectories.csv", index=False)
    observed_flat = np.concatenate([observed_birth[index, :length] for index, length in enumerate(lengths)])
    generated_flat = np.concatenate([generated["birth"][:, index, :length].ravel() for index, length in enumerate(lengths)])
    result = {
        "model": {
            "recursion": "log b[i+1] = log b[i] + alpha[i]*tau[i] + q[i]",
            "q": "q[l,i]=mu_q+r_fast[l,i]+r_slow[l,i], with two independent stationary Gaussian AR(1) modes",
            "q_likelihood": "Exact Gaussian lineage likelihood; both modes reset across generation gaps",
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
            "birth_acf_rmse_lags_1_to_max": float(np.sqrt(np.mean((birth_acf.observed[1:] - birth_acf.simulated_mean[1:]) ** 2))),
            "q_acf_rmse_lags_1_to_max": float(np.sqrt(np.mean((q_acf.observed[1:] - q_acf.simulated_mean[1:]) ** 2))),
        },
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2))
    print(birth_acf.to_string(index=False), flush=True)
    print(q_acf.to_string(index=False), flush=True)
    print(json.dumps(result["simulation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
