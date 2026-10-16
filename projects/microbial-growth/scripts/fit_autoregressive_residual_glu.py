#!/usr/bin/env python3
"""Fit and autonomously simulate the Glu AR-residual protein-hazard model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from autoregressive_residual_model import (  # noqa: E402
    RESIDUAL_DF,
    beta_regression_log_likelihood,
    gaussian_copula_ar_log_likelihood,
    lineage_acf,
    profile_hazard_log_likelihood,
    sample_growth_time,
    transform_gamma,
    transform_student_t,
)
from glu_workflow import load_data  # noqa: E402


OUT = (
    PROJECT_DIR
    / "outputs"
    / "model_comparison"
    / "Glu"
    / "autoregressive_residual_protein_hazard"
)


def contiguous_prefixes(groups):
    result = []
    for group in groups:
        generations = group.generation.to_numpy(int)
        expected = np.arange(len(group))
        failures = np.flatnonzero(generations != expected)
        stop = int(failures[0]) if len(failures) else len(group)
        if stop:
            result.append(group.iloc[:stop].reset_index(drop=True))
    return result


def build_transitions(groups):
    rows = []
    for group in groups:
        for index in range(len(group) - 1):
            current, daughter = group.iloc[index], group.iloc[index + 1]
            if int(daughter.generation) != int(current.generation) + 1:
                continue
            epsilon = (
                math.log(float(daughter.length_birth) / float(current.length_birth))
                - math.log(float(daughter.f))
                - float(current.alpha) * float(current.time)
            )
            rows.append(
                {
                    "lineage": float(current.lineage),
                    "generation": int(current.generation),
                    "birth": float(current.length_birth),
                    "alpha": float(current.alpha),
                    "time": float(current.time),
                    "fraction": float(daughter.f),
                    "next_birth": float(daughter.length_birth),
                    "epsilon": epsilon,
                }
            )
    return pd.DataFrame(rows)


def predecessor_indices(lineage, generation):
    predecessor = np.full(len(lineage), -1, dtype=int)
    latest = {}
    for index, (identity, value) in enumerate(zip(lineage, generation)):
        previous = latest.get(identity)
        if previous is not None and generation[previous] + 1 == value:
            predecessor[index] = previous
        latest[identity] = index
    return predecessor


def fit_hazard(transition):
    log_birth = np.log(transition.birth.to_numpy(float))
    alpha = transition.alpha.to_numpy(float)
    time = transition.time.to_numpy(float)

    def objective(theta, order=48):
        value, _ = profile_hazard_log_likelihood(
            log_birth, alpha, time, theta[0], theta[1], order=order
        )
        return -value

    bounds = [(math.log(0.01), math.log(30.0)), (math.log(0.05), math.log(50.0))]
    global_fit = differential_evolution(
        objective, bounds, seed=71421, popsize=12, maxiter=80, polish=False
    )
    fitted = minimize(objective, global_fit.x, method="L-BFGS-B", bounds=bounds)
    log_likelihood, omega2 = profile_hazard_log_likelihood(
        log_birth, alpha, time, fitted.x[0], fitted.x[1], order=96
    )
    return {
        "u": float(math.exp(fitted.x[0])),
        "omega2": omega2,
        "eta": float(math.exp(fitted.x[1])),
        "log_likelihood": log_likelihood,
        "success": bool(fitted.success),
        "message": str(fitted.message),
    }


def fit_alpha(groups):
    frame = pd.concat(
        [g[["lineage", "generation", "alpha"]] for g in groups], ignore_index=True
    )
    values = frame.alpha.to_numpy(float)
    predecessor = predecessor_indices(
        frame.lineage.to_numpy(), frame.generation.to_numpy(int)
    )

    def objective(theta):
        shape, scale, rho = math.exp(theta[0]), math.exp(theta[1]), math.tanh(theta[2])
        return -gaussian_copula_ar_log_likelihood(
            values, predecessor, "gamma", (shape, scale), rho
        )

    mean, variance = float(values.mean()), float(values.var())
    initial = np.array(
        [math.log(mean * mean / variance), math.log(variance / mean), np.arctanh(0.2)]
    )
    bounds = [(-5, 10), (-10, 4), (-2.25, 2.25)]
    global_fit = differential_evolution(
        objective, bounds, seed=71422, popsize=10, maxiter=50, polish=False
    )
    fitted = minimize(
        objective, global_fit.x, method="L-BFGS-B", bounds=bounds
    )
    shape, scale, rho = math.exp(fitted.x[0]), math.exp(fitted.x[1]), math.tanh(fitted.x[2])
    return {
        "shape": shape,
        "scale": scale,
        "mean": shape * scale,
        "sd": math.sqrt(shape) * scale,
        "rho": rho,
        "log_likelihood": -float(fitted.fun),
        "success": bool(fitted.success),
    }


def fit_epsilon(transition):
    values = transition.epsilon.to_numpy(float)
    predecessor = predecessor_indices(
        transition.lineage.to_numpy(), transition.generation.to_numpy(int)
    )

    def objective(theta):
        location, scale, rho = theta[0], math.exp(theta[1]), math.tanh(theta[2])
        return -gaussian_copula_ar_log_likelihood(
            values, predecessor, "student_t", (location, scale), rho
        )

    initial = np.array([np.median(values), math.log(np.std(values) / math.sqrt(2.0)), np.arctanh(0.2)])
    bounds = [(-2, 2), (-8, 1), (-2.25, 2.25)]
    global_fit = differential_evolution(
        objective, bounds, seed=71423, popsize=10, maxiter=50, polish=False
    )
    fitted = minimize(
        objective, global_fit.x, method="L-BFGS-B", bounds=bounds
    )
    location, scale, rho = fitted.x[0], math.exp(fitted.x[1]), math.tanh(fitted.x[2])
    return {
        "df": RESIDUAL_DF,
        "location": float(location),
        "scale": scale,
        "rho": rho,
        "log_likelihood": -float(fitted.fun),
        "success": bool(fitted.success),
    }


def fit_fraction(transition):
    fraction = transition.fraction.to_numpy(float)
    log_birth = np.log(transition.birth.to_numpy(float))
    pivot = float(np.mean(log_birth))

    def objective(theta):
        return -beta_regression_log_likelihood(
            fraction, log_birth, pivot, theta[0], theta[1], theta[2]
        )

    mean = float(fraction.mean())
    initial = [math.log(mean / (1.0 - mean)), 0.0, math.log(50.0)]
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-6, 6), (-5, 5), (math.log(1.01), math.log(1e4))],
    )
    return {
        "log_birth_pivot": pivot,
        "beta0": float(fitted.x[0]),
        "beta_birth": float(fitted.x[1]),
        "concentration": float(math.exp(fitted.x[2])),
        "mean_at_pivot": float(expit(fitted.x[0])),
        "log_likelihood": -float(fitted.fun),
        "success": bool(fitted.success),
    }


def fit_initial_birth(groups):
    values = np.log([float(g.length_birth.iloc[0]) for g in groups])
    return {
        "log_mean": float(values.mean()),
        "log_sd": float(values.std(ddof=0)),
        "mean": float(np.exp(values.mean() + 0.5 * values.var())),
    }


def simulate(params, lengths, replicates, seed=39517):
    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=int)
    lineages, horizon = len(lengths), int(lengths.max())
    active0 = lengths > 0
    log_birth = np.zeros((replicates, lineages, horizon))
    alpha = np.zeros_like(log_birth)
    epsilon = np.zeros_like(log_birth)
    growth = np.zeros_like(log_birth)
    time = np.full_like(log_birth, np.nan)
    fraction = np.full_like(log_birth, np.nan)

    b0 = params["initial_birth"]
    current = b0["log_mean"] + b0["log_sd"] * rng.normal(size=(replicates, lineages))
    za = rng.normal(size=(replicates, lineages))
    ze = rng.normal(size=(replicates, lineages))
    hazard, afit, efit, ffit = (
        params["hazard"],
        params["alpha"],
        params["epsilon"],
        params["fraction"],
    )
    negative_endpoint = 0
    transitions = 0
    for generation in range(horizon):
        active = lengths > generation
        if generation:
            za[:, active] = afit["rho"] * za[:, active] + math.sqrt(
                1.0 - afit["rho"] ** 2
            ) * rng.normal(size=(replicates, int(active.sum())))
            ze[:, active] = efit["rho"] * ze[:, active] + math.sqrt(
                1.0 - efit["rho"] ** 2
            ) * rng.normal(size=(replicates, int(active.sum())))
        log_birth[:, active, generation] = current[:, active]
        alpha[:, active, generation] = transform_gamma(
            za[:, active], afit["shape"], afit["scale"]
        )
        epsilon[:, active, generation] = transform_student_t(
            ze[:, active], efit["location"], efit["scale"]
        )
        continuing = lengths > generation + 1
        if not continuing.any():
            continue
        s = sample_growth_time(
            current[:, continuing].ravel(),
            math.log(hazard["u"]),
            hazard["omega2"],
            hazard["eta"],
            rng,
        ).reshape(replicates, int(continuing.sum()))
        growth[:, continuing, generation] = s
        time[:, continuing, generation] = s / alpha[:, continuing, generation]
        mean_f = expit(
            ffit["beta0"]
            + ffit["beta_birth"]
            * (current[:, continuing] - ffit["log_birth_pivot"])
        )
        aa = mean_f * ffit["concentration"]
        bb = (1.0 - mean_f) * ffit["concentration"]
        f = rng.beta(aa, bb)
        fraction[:, continuing, generation] = f
        residual = epsilon[:, continuing, generation]
        negative_endpoint += int(np.sum(s + residual <= 0.0))
        transitions += s.size
        current[:, continuing] = current[:, continuing] + np.log(f) + s + residual
    return {
        "log_birth": log_birth,
        "birth": np.exp(np.clip(log_birth, -700, 700)),
        "alpha": alpha,
        "epsilon": epsilon,
        "growth": growth,
        "time": time,
        "fraction": fraction,
        "fraction_nonpositive_endpoint_growth": negative_endpoint / transitions,
    }


def observed_matrix(groups):
    lengths = np.array([len(g) for g in groups], dtype=int)
    matrix = np.full((len(groups), int(lengths.max())), np.nan)
    for index, group in enumerate(groups):
        matrix[index, : len(group)] = group.length_birth.to_numpy(float)
    return matrix, lengths


def summarize_acf(observed, simulated, lengths, max_lag):
    observed_acf = lineage_acf(observed, lengths, max_lag=max_lag)[0]
    simulated_acf = lineage_acf(simulated, lengths, max_lag=max_lag)
    low, high = np.nanquantile(simulated_acf, [0.025, 0.975], axis=0)
    mean = np.nanmean(simulated_acf, axis=0)
    return pd.DataFrame(
        {
            "lag": np.arange(max_lag + 1),
            "observed": observed_acf,
            "simulated_mean": mean,
            "simulated_lo95": low,
            "simulated_hi95": high,
            "observed_minus_simulated": observed_acf - mean,
            "n_lineages": [
                len(lengths)
                if lag == 0
                else int(np.sum((lengths > lag) & (lengths >= 3)))
                for lag in range(max_lag + 1)
            ],
        }
    )


def plot_acf(table, output):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    x = table.lag.to_numpy()
    ax.fill_between(
        x,
        table.simulated_lo95,
        table.simulated_hi95,
        color="#cf6f45",
        alpha=0.22,
        label="Autonomous simulations, 95% interval",
    )
    ax.plot(x, table.simulated_mean, color="#c4512c", marker="s", ms=4, label="Autonomous mean")
    ax.plot(x, table.observed, color="#23364d", marker="o", ms=4, label="Observed Glu")
    ax.axhline(0, color="0.55", lw=0.8)
    ax.set(
        xlabel="Lag (generations)",
        ylabel="Mean within-lineage ACF",
        title="Glu birth-size ACF: fully generative AR-residual model",
    )
    ax.set_xticks(x)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.savefig(output / "birth_size_acf_autonomous.png", dpi=190)
    fig.savefig(output / "birth_size_acf_autonomous.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--max-lag", type=int, default=15)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _, all_groups = load_data()
    prefixes = contiguous_prefixes(all_groups)
    transition = build_transitions(all_groups)
    print(f"Fitting {len(all_groups)} lineages and {len(transition)} consecutive transitions", flush=True)

    params = {
        "initial_birth": fit_initial_birth(all_groups),
        "alpha": fit_alpha(all_groups),
        "epsilon": fit_epsilon(transition),
        "fraction": fit_fraction(transition),
        "hazard": fit_hazard(transition),
    }
    for name, values in params.items():
        print(name, values, flush=True)

    observed, lengths = observed_matrix(prefixes)
    generated = simulate(params, lengths, args.replicates)
    acf = summarize_acf(observed, generated["birth"], lengths, args.max_lag)
    acf.to_csv(OUT / "birth_size_acf.csv", index=False)
    transition.to_csv(OUT / "reconstructed_transitions.csv", index=False)
    plot_acf(acf, OUT)

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
                        "epsilon": generated["epsilon"][replicate, lineage, generation],
                        "growth_time": generated["growth"][replicate, lineage, generation],
                        "time": generated["time"][replicate, lineage, generation],
                        "fraction": generated["fraction"][replicate, lineage, generation],
                    }
                )
    pd.DataFrame(example_rows).to_csv(OUT / "autonomous_example_trajectories.csv", index=False)
    observed_flat = np.concatenate([observed[i, :length] for i, length in enumerate(lengths)])
    simulated_flat = np.concatenate(
        [generated["birth"][:, i, :length].ravel() for i, length in enumerate(lengths)]
    )
    result = {
        "model": {
            "recursion": "b[i+1] = b[i] f[i] exp(alpha[i] tau[i] + epsilon[i])",
            "hazard": "h(t|b,alpha)=alpha*omega2*sigmoid(eta*log(b*(exp(alpha*t)-1)/u))",
            "initial_birth": "lognormal MLE",
            "alpha": "Gamma marginal, Gaussian-copula AR(1)",
            "epsilon": "Student-t(df=4) marginal, Gaussian-copula AR(1)",
            "fraction": "Beta with logit mean linear in current log birth",
        },
        "fit": params,
        "data": {
            "lineages": len(all_groups),
            "consecutive_transitions": len(transition),
            "acf_prefix_lineages": len(prefixes),
            "acf_births": int(lengths.sum()),
            "generation_gap_lineages": int(sum(len(a) != len(b) for a, b in zip(prefixes, all_groups))),
        },
        "simulation": {
            "replicates": args.replicates,
            "replayed_quantities": "none; only observed contiguous-prefix lengths are reused as observation design",
            "negative_endpoint_log_growth_fraction": generated[
                "fraction_nonpositive_endpoint_growth"
            ],
            "observed_birth_mean": float(observed_flat.mean()),
            "observed_birth_sd": float(observed_flat.std()),
            "simulated_birth_mean": float(simulated_flat.mean()),
            "simulated_birth_sd": float(simulated_flat.std()),
            "acf_rmse_lags_1_to_max": float(
                np.sqrt(np.nanmean(acf.observed_minus_simulated.to_numpy()[1:] ** 2))
            ),
        },
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(acf.to_string(index=False), flush=True)
    print(json.dumps(result["simulation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
