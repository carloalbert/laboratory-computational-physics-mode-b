#!/usr/bin/env python3
"""ACF-only calibration of all Glu model parameters, including every E_i."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from autoregressive_residual_model import (  # noqa: E402
    gaussian_copula_ar_log_likelihood,
    lineage_acf,
    profile_hazard_log_likelihood,
)
from fit_autoregressive_residual_glu import (  # noqa: E402
    build_transitions,
    contiguous_prefixes,
    observed_matrix,
    predecessor_indices,
    summarize_acf,
)
from fit_environment_fast_slow_ar_q_glu import simulate  # noqa: E402
from fit_lineage_offset_ar_q_glu import pooled_acf_replicates, q_matrix  # noqa: E402
from glu_workflow import load_data  # noqa: E402


SOURCE = PROJECT_DIR / "outputs/model_comparison/Glu/environment_fast_slow_ar_q_protein_hazard"
OUT = PROJECT_DIR / "outputs/model_comparison/Glu/all_parameters_birth_acf_fit"


GLOBAL_NAMES = [
    "log_u",
    "log_omega2",
    "log_eta",
    "log_alpha_mean",
    "log_alpha_sd",
    "rho_alpha",
    "mu_q",
    "rho_fast",
    "rho_slow",
    "log_stationary_fast_sd",
    "log_stationary_slow_sd",
    "initial_log_mean",
    "log_initial_log_sd",
]


def parameter_bounds(environment_count):
    global_bounds = np.asarray(
        [
            [math.log(0.25), math.log(5.0)],
            [math.log(2.0), math.log(30.0)],
            [math.log(1.0), math.log(25.0)],
            [math.log(0.32), math.log(0.80)],
            [math.log(0.07), math.log(0.28)],
            [-0.30, 0.90],
            [-0.70, -0.35],
            [-0.65, 0.65],
            [0.35, 0.995],
            [math.log(0.02), math.log(0.18)],
            [math.log(0.002), math.log(0.16)],
            [0.55, 1.05],
            [math.log(0.07), math.log(0.35)],
        ],
        dtype=float,
    )
    environment_bounds = np.tile([-0.12, 0.12], (environment_count, 1))
    return np.vstack([global_bounds, environment_bounds])


def start_vector(fitted, environment_count):
    qfit = fitted["fit"]["q"]
    alpha = fitted["fit"]["alpha"]
    hazard = fitted["fit"]["hazard"]
    initial = fitted["fit"]["initial_birth"]
    effects = np.zeros(environment_count)
    saved = np.asarray(qfit["posterior_mean_E"], dtype=float)
    effects[: min(len(saved), environment_count)] = saved[:environment_count]
    effect_mean = float(effects.mean())
    effects -= effect_mean
    return np.r_[
        math.log(hazard["u"]),
        math.log(hazard["omega2"]),
        math.log(hazard["eta"]),
        math.log(alpha["mean"]),
        math.log(alpha["sd"]),
        alpha["rho"],
        qfit["mean"] + effect_mean,
        qfit["rho_fast"],
        qfit["rho_slow"],
        math.log(qfit["stationary_fast_sd"]),
        math.log(qfit["stationary_slow_sd"]),
        initial["log_mean"],
        math.log(initial["log_sd"]),
        effects,
    ]


def normalize(vector, bounds):
    return 2.0 * (vector - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1.0


def denormalize(vector, bounds):
    return bounds[:, 0] + 0.5 * (vector + 1.0) * (bounds[:, 1] - bounds[:, 0])


def unpack(vector, environment_count):
    (
        log_u,
        log_omega2,
        log_eta,
        log_alpha_mean,
        log_alpha_sd,
        rho_alpha,
        mu_q,
        rho_fast,
        rho_slow,
        log_fast_sd,
        log_slow_sd,
        initial_log_mean,
        log_initial_sd,
    ) = vector[: len(GLOBAL_NAMES)]
    environment = np.asarray(vector[len(GLOBAL_NAMES) : len(GLOBAL_NAMES) + environment_count])
    environment = environment - environment.mean()
    alpha_mean, alpha_sd = math.exp(log_alpha_mean), math.exp(log_alpha_sd)
    alpha_shape = (alpha_mean / alpha_sd) ** 2
    alpha_scale = alpha_sd**2 / alpha_mean
    fast_sd, slow_sd = math.exp(log_fast_sd), math.exp(log_slow_sd)
    return {
        "initial_birth": {
            "log_mean": float(initial_log_mean),
            "log_sd": math.exp(log_initial_sd),
        },
        "alpha": {
            "shape": alpha_shape,
            "scale": alpha_scale,
            "mean": alpha_mean,
            "sd": alpha_sd,
            "rho": float(rho_alpha),
        },
        "hazard": {
            "u": math.exp(log_u),
            "omega2": math.exp(log_omega2),
            "eta": math.exp(log_eta),
        },
        "q": {
            "mean": float(mu_q),
            "rho_fast": float(rho_fast),
            "rho_slow": float(rho_slow),
            "stationary_fast_sd": fast_sd,
            "stationary_slow_sd": slow_sd,
            "sigma_fast_innovation": fast_sd * math.sqrt(1.0 - rho_fast**2),
            "sigma_slow_innovation": slow_sd * math.sqrt(1.0 - rho_slow**2),
            "environment_prior_sd": float(np.sqrt(np.mean(environment**2))),
            "posterior_mean_E": environment,
            "posterior_covariance_E": np.eye(environment_count) * 1e-12,
            "posterior_sd_E": np.full(environment_count, 1e-6),
        },
    }


def spsa(objective, initial, iterations, seed, label):
    rng = np.random.default_rng(seed)
    x = np.clip(initial.copy(), -1.0, 1.0)
    best_x, best_score = x.copy(), objective(x)
    first_moment = np.zeros_like(x)
    second_moment = np.zeros_like(x)
    history, candidates = [], [(best_score, best_x.copy())]
    for iteration in range(1, iterations + 1):
        perturbation = rng.choice([-1.0, 1.0], size=len(x))
        radius = 0.10 / (iteration + 8.0) ** 0.101
        plus = np.clip(x + radius * perturbation, -1.0, 1.0)
        minus = np.clip(x - radius * perturbation, -1.0, 1.0)
        plus_score, minus_score = objective(plus), objective(minus)
        gradient = (plus_score - minus_score) * perturbation / (2.0 * radius)
        first_moment = 0.90 * first_moment + 0.10 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.90**iteration)
        corrected_second = second_moment / (1.0 - 0.999**iteration)
        learning_rate = 0.035 / (1.0 + iteration / 80.0) ** 0.35
        proposal = np.clip(
            x - learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-9),
            -1.0,
            1.0,
        )
        if iteration % 4 == 0:
            score = objective(proposal)
            candidates.append((score, proposal.copy()))
            if score < best_score:
                best_score, best_x = score, proposal.copy()
            # Pull back toward the best point if an update wanders badly.
            x = proposal if score < 1.35 * best_score else 0.5 * (proposal + best_x)
        else:
            x = proposal
        if iteration % 10 == 0 or iteration == 1:
            history.append(
                {
                    "phase": label,
                    "iteration": iteration,
                    "best_training_mse": best_score,
                    "plus_mse": plus_score,
                    "minus_mse": minus_score,
                }
            )
            print(
                f"{label} iteration {iteration:3d}: best birth-ACF MSE={best_score:.7g}",
                flush=True,
            )
    return best_x, best_score, history, candidates


def observed_and_simulated_values(prefixes, generated, lengths):
    observed = {name: [] for name in ["birth", "q", "alpha", "time"]}
    for group in prefixes:
        length = len(group)
        birth = group.length_birth.to_numpy(float)
        observed["birth"].extend(birth)
        observed["alpha"].extend(group.alpha.to_numpy(float))
        if length > 1:
            observed["time"].extend(group.time.to_numpy(float)[:-1])
            observed["q"].extend(
                np.diff(np.log(birth))
                - group.alpha.to_numpy(float)[:-1] * group.time.to_numpy(float)[:-1]
            )
    observed = {key: np.asarray(value) for key, value in observed.items()}
    simulated = {
        "birth": np.concatenate(
            [generated["birth"][:, i, :length].ravel() for i, length in enumerate(lengths)]
        ),
        "alpha": np.concatenate(
            [generated["alpha"][:, i, :length].ravel() for i, length in enumerate(lengths)]
        ),
        "time": np.concatenate(
            [generated["time"][:, i, : length - 1].ravel() for i, length in enumerate(lengths) if length > 1]
        ),
        "q": np.concatenate(
            [generated["q"][:, i, : length - 1].ravel() for i, length in enumerate(lengths) if length > 1]
        ),
    }
    return observed, simulated


def marginal_rows(observed, generated, lengths, model):
    data, simulation = observed_and_simulated_values(observed, generated, lengths)
    rows = []
    for variable in ["birth", "q", "alpha", "time"]:
        rows.append(
            {
                "model": model,
                "variable": variable,
                "observed_mean": float(data[variable].mean()),
                "simulated_mean": float(simulation[variable].mean()),
                "observed_sd": float(data[variable].std()),
                "simulated_sd": float(simulation[variable].std()),
                "ks_distance": float(ks_2samp(data[variable], simulation[variable]).statistic),
            }
        )
    return rows


def component_log_likelihood(params, prefixes, transition, balanced_q, environment_prior_sd):
    # The q score is evaluated conditionally on the calibrated E_i, with the
    # likelihood-fit sigma_E used only to expose the cost of the fixed effects.
    q = params["q"]
    horizon = balanced_q.shape[1]
    distance = np.abs(np.subtract.outer(np.arange(horizon), np.arange(horizon)))
    covariance = (
        q["stationary_fast_sd"] ** 2 * q["rho_fast"] ** distance
        + q["stationary_slow_sd"] ** 2 * q["rho_slow"] ** distance
    )
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        q_ll = -np.inf
    else:
        inverse = np.linalg.inv(covariance)
        mean = q["mean"] + q["posterior_mean_E"][:horizon]
        residual = balanced_q - mean
        q_ll = -0.5 * (
            balanced_q.shape[0] * (horizon * math.log(2.0 * math.pi) + logdet)
            + np.sum((residual @ inverse) * residual)
        )
        effects = q["posterior_mean_E"][:horizon]
        q_ll += float(
            np.sum(
                -0.5
                * (
                    math.log(2.0 * math.pi * environment_prior_sd**2)
                    + effects**2 / environment_prior_sd**2
                )
            )
        )
    frame = pd.concat(
        [group[["lineage", "generation", "alpha"]] for group in prefixes],
        ignore_index=True,
    )
    predecessor = predecessor_indices(
        frame.lineage.to_numpy(), frame.generation.to_numpy(int)
    )
    alpha = params["alpha"]
    alpha_ll = gaussian_copula_ar_log_likelihood(
        frame.alpha.to_numpy(float),
        predecessor,
        "gamma",
        (alpha["shape"], alpha["scale"]),
        alpha["rho"],
    )
    hazard = params["hazard"]
    hazard_ll, profiled_omega = profile_hazard_log_likelihood(
        np.log(transition.birth.to_numpy(float)),
        transition.alpha.to_numpy(float),
        transition.time.to_numpy(float),
        math.log(hazard["u"]),
        math.log(hazard["eta"]),
        order=64,
    )
    # Replace the profiled omega term by the candidate omega exactly.
    growth = transition.alpha.to_numpy(float) * transition.time.to_numpy(float)
    hazard_ll += len(growth) * math.log(hazard["omega2"] / profiled_omega)
    hazard_ll -= (hazard["omega2"] - profiled_omega) * len(growth) / profiled_omega
    initial_values = np.log([float(group.length_birth.iloc[0]) for group in prefixes])
    initial = params["initial_birth"]
    initial_ll = float(
        np.sum(
            -0.5 * math.log(2.0 * math.pi * initial["log_sd"] ** 2)
            - 0.5 * ((initial_values - initial["log_mean"]) / initial["log_sd"]) ** 2
        )
    )
    return {
        "q_complete_data": float(q_ll),
        "alpha": float(alpha_ll),
        "hazard": float(hazard_ll),
        "initial_birth": initial_ll,
        "total_comparable": float(q_ll + alpha_ll + hazard_ll + initial_ll),
    }


def serializable(params):
    result = {}
    for block, values in params.items():
        result[block] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in values.items()
            if key != "posterior_covariance_E"
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-max-lag", type=int, default=15)
    parser.add_argument("--training-replicates", type=int, default=6)
    parser.add_argument("--validation-replicates", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--seed", type=int, default=85271)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    fitted = json.loads((SOURCE / "fit_results.json").read_text())
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    environment_count = int(lengths.max()) - 1
    bounds = parameter_bounds(environment_count)
    starting_physical = np.clip(start_vector(fitted, environment_count), bounds[:, 0], bounds[:, 1])
    starting_normalized = normalize(starting_physical, bounds)
    target = lineage_acf(observed_birth, lengths, args.fit_max_lag)[0]
    fit_slice = slice(1, args.fit_max_lag + 1)

    def objective(normalized, replicates=args.training_replicates, seed=args.seed):
        try:
            physical = denormalize(np.clip(normalized, -1.0, 1.0), bounds)
            params = unpack(physical, environment_count)
            generated = simulate(params, params["q"], lengths, replicates, seed=seed)
            predicted = np.nanmean(
                lineage_acf(generated["birth"], lengths, args.fit_max_lag), axis=0
            )
            value = float(np.mean((predicted[fit_slice] - target[fit_slice]) ** 2))
            return value if np.isfinite(value) else 1e6
        except (FloatingPointError, OverflowError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return 1e6

    baseline_training = objective(starting_normalized)
    first, _, history1, candidates1 = spsa(
        objective, starting_normalized, args.iterations, args.seed + 1, "global"
    )
    second, _, history2, candidates2 = spsa(
        objective, first, max(60, args.iterations // 2), args.seed + 2, "polish"
    )
    candidates = candidates1 + candidates2 + [(baseline_training, starting_normalized)]
    candidates = sorted(candidates, key=lambda item: item[0])[:18]
    refinement_scores = [
        objective(vector, replicates=32, seed=args.seed + 3)
        for _, vector in candidates
    ]
    chosen_normalized = candidates[int(np.argmin(refinement_scores))][1]
    chosen_physical = denormalize(chosen_normalized, bounds)
    likelihood_params = unpack(starting_physical, environment_count)
    acf_params = unpack(chosen_physical, environment_count)

    likelihood_generated = simulate(
        likelihood_params,
        likelihood_params["q"],
        lengths,
        args.validation_replicates,
        seed=args.seed + 10,
    )
    acf_generated = simulate(
        acf_params,
        acf_params["q"],
        lengths,
        args.validation_replicates,
        seed=args.seed + 11,
    )
    likelihood_birth_acf = summarize_acf(
        observed_birth, likelihood_generated["birth"], lengths, args.fit_max_lag
    )
    acf_birth_acf = summarize_acf(
        observed_birth, acf_generated["birth"], lengths, args.fit_max_lag
    )
    birth_acf = pd.DataFrame(
        {
            "lag": likelihood_birth_acf.lag,
            "observed": likelihood_birth_acf.observed,
            "likelihood_mean": likelihood_birth_acf.simulated_mean,
            "acf_fit_mean": acf_birth_acf.simulated_mean,
            "acf_fit_lo95": acf_birth_acf.simulated_lo95,
            "acf_fit_hi95": acf_birth_acf.simulated_hi95,
        }
    )
    birth_acf.to_csv(OUT / "birth_acf_comparison.csv", index=False)

    observed_q, q_lengths = q_matrix(prefixes)
    observed_q_acf = pooled_acf_replicates(observed_q, q_lengths, args.fit_max_lag)[0]
    likelihood_q = pooled_acf_replicates(
        likelihood_generated["q"], q_lengths, args.fit_max_lag
    )
    acf_q = pooled_acf_replicates(acf_generated["q"], q_lengths, args.fit_max_lag)
    q_acf = pd.DataFrame(
        {
            "lag": np.arange(args.fit_max_lag + 1),
            "observed": observed_q_acf,
            "likelihood_mean": np.nanmean(likelihood_q, axis=0),
            "acf_fit_mean": np.nanmean(acf_q, axis=0),
        }
    )
    q_acf.to_csv(OUT / "q_acf_comparison.csv", index=False)

    horizon = min(20, int(lengths.max()) - 1)
    complete = lengths > horizon
    generation = np.arange(horizon + 1)
    observed_mean = observed_birth[complete, : horizon + 1].mean(axis=0)
    likelihood_paths = likelihood_generated["birth"][:, complete, : horizon + 1].mean(axis=1)
    acf_paths = acf_generated["birth"][:, complete, : horizon + 1].mean(axis=1)
    drift = pd.DataFrame(
        {
            "generation": generation,
            "observed": observed_mean,
            "likelihood_mean": likelihood_paths.mean(axis=0),
            "acf_fit_mean": acf_paths.mean(axis=0),
        }
    )
    drift.to_csv(OUT / "mean_birth_drift_comparison.csv", index=False)

    marginal = pd.DataFrame(
        marginal_rows(prefixes, likelihood_generated, lengths, "likelihood")
        + marginal_rows(prefixes, acf_generated, lengths, "birth_acf")
    )
    marginal.to_csv(OUT / "marginal_comparison.csv", index=False)

    transition = build_transitions(groups)
    balanced_q = q_matrix([group.iloc[:21] for group in prefixes if len(group) > 20])[0][:, :20]
    sigma_environment = float(fitted["fit"]["q"]["environment_prior_sd"])
    likelihood_scores = component_log_likelihood(
        likelihood_params, prefixes, transition, balanced_q, sigma_environment
    )
    acf_scores = component_log_likelihood(
        acf_params, prefixes, transition, balanced_q, sigma_environment
    )

    fit_range = np.arange(1, args.fit_max_lag + 1)
    metrics = {
        "birth_acf_rmse_likelihood": float(np.sqrt(np.mean((birth_acf.observed.iloc[fit_range] - birth_acf.likelihood_mean.iloc[fit_range]) ** 2))),
        "birth_acf_rmse_acf_fit": float(np.sqrt(np.mean((birth_acf.observed.iloc[fit_range] - birth_acf.acf_fit_mean.iloc[fit_range]) ** 2))),
        "q_acf_rmse_likelihood": float(np.sqrt(np.mean((q_acf.observed.iloc[fit_range] - q_acf.likelihood_mean.iloc[fit_range]) ** 2))),
        "q_acf_rmse_acf_fit": float(np.sqrt(np.mean((q_acf.observed.iloc[fit_range] - q_acf.acf_fit_mean.iloc[fit_range]) ** 2))),
        "mean_birth_rmse_likelihood": float(np.sqrt(np.mean((drift.observed - drift.likelihood_mean) ** 2))),
        "mean_birth_rmse_acf_fit": float(np.sqrt(np.mean((drift.observed - drift.acf_fit_mean) ** 2))),
        "generation_20_change_observed": float(drift.observed.iloc[-1] - drift.observed.iloc[0]),
        "generation_20_change_likelihood": float(drift.likelihood_mean.iloc[-1] - drift.likelihood_mean.iloc[0]),
        "generation_20_change_acf_fit": float(drift.acf_fit_mean.iloc[-1] - drift.acf_fit_mean.iloc[0]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(birth_acf.lag, birth_acf.observed, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.plot(birth_acf.lag, birth_acf.likelihood_mean, "--", color="#718096", label="Likelihood fit")
    axis.fill_between(birth_acf.lag, birth_acf.acf_fit_lo95, birth_acf.acf_fit_hi95, color="#c4512c", alpha=0.18)
    axis.plot(birth_acf.lag, birth_acf.acf_fit_mean, "s-", color="#c4512c", ms=3.5, label="All-parameter ACF fit")
    axis.set(title="Birth-size ACF", xlabel="Lag", ylabel="Correlation")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    axis.plot(q_acf.lag, q_acf.observed, "o-", color="#23364d", ms=3.5)
    axis.plot(q_acf.lag, q_acf.likelihood_mean, "--", color="#718096")
    axis.plot(q_acf.lag, q_acf.acf_fit_mean, "s-", color="#c4512c", ms=3.5)
    axis.set(title=r"Effective-jump ACF ($q$)", xlabel="Lag", ylabel="Correlation")

    axis = axes[1, 0]
    axis.plot(drift.generation, drift.observed, "o-", color="#23364d", ms=3.5)
    axis.plot(drift.generation, drift.likelihood_mean, "--", color="#718096")
    axis.plot(drift.generation, drift.acf_fit_mean, "s-", color="#c4512c", ms=3.5)
    axis.set(title="Balanced-cohort mean birth size", xlabel="Generation", ylabel="Mean birth size")

    axis = axes[1, 1]
    order = ["birth", "q", "alpha", "time"]
    locations, width = np.arange(len(order)), 0.36
    likelihood_ks = marginal.query("model == 'likelihood'").set_index("variable").loc[order].ks_distance
    acf_ks = marginal.query("model == 'birth_acf'").set_index("variable").loc[order].ks_distance
    axis.bar(locations - width / 2, likelihood_ks, width, color="#718096", label="Likelihood fit")
    axis.bar(locations + width / 2, acf_ks, width, color="#c4512c", label="All-parameter ACF fit")
    axis.set_xticks(locations, ["birth", "q", r"$\alpha$", r"$\tau$"])
    axis.set(title="Marginal-distribution trade-off", ylabel="KS distance")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    fig.suptitle("Glu: all parameters calibrated to the birth-size ACF only")
    fig.savefig(OUT / "all_parameter_acf_tradeoff.png", dpi=190)
    fig.savefig(OUT / "all_parameter_acf_tradeoff.pdf")
    plt.close(fig)

    environmental = pd.DataFrame(
        {
            "generation": np.arange(environment_count),
            "likelihood_E": likelihood_params["q"]["posterior_mean_E"],
            "acf_fit_E": acf_params["q"]["posterior_mean_E"],
        }
    )
    environmental.to_csv(OUT / "environmental_effects.csv", index=False)
    fig, axis = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    axis.plot(
        environmental.generation,
        environmental.likelihood_E,
        "o--",
        color="#718096",
        ms=3.2,
        label="Likelihood/posterior reference",
    )
    axis.plot(
        environmental.generation,
        environmental.acf_fit_E,
        "s-",
        color="#c4512c",
        ms=3.2,
        label="Birth-ACF calibration",
    )
    axis.axhline(0.0, color="0.55", lw=0.8)
    axis.set(
        title="Generation-specific environmental effects",
        xlabel="Generation",
        ylabel=r"$E_i$",
    )
    axis.grid(alpha=0.16)
    axis.legend(frameon=False)
    fig.savefig(OUT / "environmental_effects.png", dpi=190)
    fig.savefig(OUT / "environmental_effects.pdf")
    plt.close(fig)
    pd.DataFrame(history1 + history2).to_csv(OUT / "optimizer_history.csv", index=False)
    diagnostics = {
        "objective": f"MSE of mean within-lineage birth-size ACF at lags 1..{args.fit_max_lag}; no likelihood, prior, or marginal penalty",
        "identifiability_constraint": "The fitted E_i values are centered to sum to zero; mu_q absorbs the common level.",
        "parameters_calibrated": GLOBAL_NAMES + [f"E_{i}" for i in range(environment_count)],
        "training": {
            "replicates": args.training_replicates,
            "baseline_mse": baseline_training,
            "selected_refinement_mse": float(min(refinement_scores)),
            "iterations_global": args.iterations,
            "iterations_polish": max(60, args.iterations // 2),
        },
        "validation": metrics,
        "component_log_likelihood_likelihood_fit": likelihood_scores,
        "component_log_likelihood_acf_fit": acf_scores,
        "component_log_likelihood_change": {
            key: acf_scores[key] - likelihood_scores[key] for key in likelihood_scores
        },
        "likelihood_parameters": serializable(likelihood_params),
        "acf_parameters": serializable(acf_params),
        "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps({
        "training": diagnostics["training"],
        "validation": metrics,
        "component_log_likelihood_change": diagnostics["component_log_likelihood_change"],
        "acf_parameters": serializable(acf_params),
        "runtime_minutes": diagnostics["runtime_minutes"],
    }, indent=2), flush=True)
    print(marginal.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
