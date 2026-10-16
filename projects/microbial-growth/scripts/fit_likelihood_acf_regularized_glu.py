#!/usr/bin/env python3
"""Full Glu penalized-MAP fit with birth-size ACF regularization."""

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
from scipy.optimize import minimize


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from autoregressive_residual_model import (  # noqa: E402
    gaussian_copula_ar_log_likelihood,
    lineage_acf,
    profile_hazard_components,
)
from fit_all_parameters_birth_acf_glu import (  # noqa: E402
    GLOBAL_NAMES,
    denormalize,
    marginal_rows,
    normalize,
    parameter_bounds,
    serializable,
    start_vector,
    unpack,
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
OUT = PROJECT_DIR / "outputs/model_comparison/Glu/likelihood_birth_acf_regularized"
SIGMA_INDEX = len(GLOBAL_NAMES)


def regularized_bounds(environment_count):
    base = parameter_bounds(environment_count)
    return np.insert(
        base,
        SIGMA_INDEX,
        [math.log(0.002), math.log(0.10)],
        axis=0,
    )


def unpack_regularized(vector, environment_count):
    base = np.delete(vector, SIGMA_INDEX)
    params = unpack(base, environment_count)
    params["q"]["environment_prior_sd"] = math.exp(vector[SIGMA_INDEX])
    return params


def q_covariance(length, qfit):
    distance = np.abs(np.subtract.outer(np.arange(length), np.arange(length)))
    return (
        qfit["stationary_fast_sd"] ** 2 * qfit["rho_fast"] ** distance
        + qfit["stationary_slow_sd"] ** 2 * qfit["rho_slow"] ** distance
    )


def q_mode_and_log_density(hyper, sequences, environment_count):
    mu, rho_fast, rho_slow, log_fast_sd, log_slow_sd, log_environment_sd = hyper
    qfit = {
        "mean": float(mu),
        "rho_fast": float(rho_fast),
        "rho_slow": float(rho_slow),
        "stationary_fast_sd": math.exp(log_fast_sd),
        "stationary_slow_sd": math.exp(log_slow_sd),
        "environment_prior_sd": math.exp(log_environment_sd),
    }
    precision = np.eye(environment_count) / qfit["environment_prior_sd"] ** 2
    score = np.zeros(environment_count)
    cache = {}
    for values in sequences:
        length = len(values)
        if length not in cache:
            covariance = q_covariance(length, qfit)
            cholesky = np.linalg.cholesky(covariance)
            inverse = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, np.eye(length)))
            logdet = 2.0 * float(np.log(np.diag(cholesky)).sum())
            cache[length] = inverse, logdet
        inverse, _ = cache[length]
        precision[:length, :length] += inverse
        score[:length] += inverse @ (values - mu)
    effects = np.linalg.solve(precision, score)
    qfit["posterior_mean_E"] = effects
    qfit["posterior_covariance_E"] = np.linalg.inv(precision)
    qfit["posterior_sd_E"] = np.sqrt(np.diag(qfit["posterior_covariance_E"]))
    qfit["sigma_fast_innovation"] = qfit["stationary_fast_sd"] * math.sqrt(1.0 - rho_fast**2)
    qfit["sigma_slow_innovation"] = qfit["stationary_slow_sd"] * math.sqrt(1.0 - rho_slow**2)
    return qfit, q_joint_log_density(qfit, sequences)


def q_joint_log_density(qfit, sequences):
    effects = np.asarray(qfit["posterior_mean_E"])
    cache, value = {}, 0.0
    for sequence in sequences:
        length = len(sequence)
        if length not in cache:
            covariance = q_covariance(length, qfit)
            cholesky = np.linalg.cholesky(covariance)
            cache[length] = cholesky
        cholesky = cache[length]
        residual = sequence - qfit["mean"] - effects[:length]
        whitened = np.linalg.solve(cholesky, residual)
        value += -0.5 * (
            length * math.log(2.0 * math.pi)
            + 2.0 * np.log(np.diag(cholesky)).sum()
            + whitened @ whitened
        )
    sigma = qfit["environment_prior_sd"]
    value += float(
        np.sum(
            -0.5
            * (
                math.log(2.0 * math.pi * sigma**2)
                + effects**2 / sigma**2
            )
        )
    )
    return float(value)


def fit_q_pure_map(source_q, sequences, environment_count):
    start = np.asarray(
        [
            source_q["mean"],
            source_q["rho_fast"],
            source_q["rho_slow"],
            math.log(source_q["stationary_fast_sd"]),
            math.log(source_q["stationary_slow_sd"]),
            math.log(source_q["environment_prior_sd"]),
        ]
    )
    bounds = [
        (-0.70, -0.35),
        (-0.65, 0.65),
        (0.35, 0.995),
        (math.log(0.02), math.log(0.18)),
        (math.log(0.002), math.log(0.16)),
        (math.log(0.002), math.log(0.10)),
    ]

    def objective(vector):
        try:
            return -q_mode_and_log_density(vector, sequences, environment_count)[1]
        except np.linalg.LinAlgError:
            return 1e100

    starts = [
        start,
        start + np.asarray([0.0, 0.10, -0.05, 0.0, 0.15, 0.10]),
        start + np.asarray([0.0, -0.10, 0.03, 0.0, -0.15, -0.10]),
    ]
    fits = [
        minimize(
            objective,
            np.clip(value, [x[0] for x in bounds], [x[1] for x in bounds]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1500, "ftol": 1e-12},
        )
        for value in starts
    ]
    fitted = min(fits, key=lambda result: result.fun)
    qfit, value = q_mode_and_log_density(fitted.x, sequences, environment_count)
    return qfit, value, fitted


class LikelihoodData:
    def __init__(self, prefixes, groups, sequences):
        self.sequences = sequences
        self.transition = build_transitions(groups)
        frame = pd.concat(
            [group[["lineage", "generation", "alpha"]] for group in prefixes],
            ignore_index=True,
        )
        self.alpha = frame.alpha.to_numpy(float)
        self.alpha_predecessor = predecessor_indices(
            frame.lineage.to_numpy(), frame.generation.to_numpy(int)
        )
        self.log_birth_transition = np.log(self.transition.birth.to_numpy(float))
        self.transition_alpha = self.transition.alpha.to_numpy(float)
        self.transition_time = self.transition.time.to_numpy(float)
        self.initial_log_birth = np.log(
            [float(group.length_birth.iloc[0]) for group in prefixes]
        )
        self.n_data = (
            sum(map(len, sequences))
            + len(self.alpha)
            + len(self.transition)
            + len(self.initial_log_birth)
        )

    def components(self, params):
        q_value = q_joint_log_density(params["q"], self.sequences)
        alpha = params["alpha"]
        alpha_value = gaussian_copula_ar_log_likelihood(
            self.alpha,
            self.alpha_predecessor,
            "gamma",
            (alpha["shape"], alpha["scale"]),
            alpha["rho"],
        )
        hazard = params["hazard"]
        log_fraction, integral = profile_hazard_components(
            self.log_birth_transition,
            self.transition_alpha * self.transition_time,
            math.log(hazard["u"]),
            hazard["eta"],
            order=48,
        )
        hazard_value = float(
            np.sum(np.log(self.transition_alpha) + math.log(hazard["omega2"]) + log_fraction)
            - hazard["omega2"] * integral.sum()
        )
        initial = params["initial_birth"]
        initial_value = float(
            np.sum(
                -0.5 * math.log(2.0 * math.pi * initial["log_sd"] ** 2)
                - 0.5
                * ((self.initial_log_birth - initial["log_mean"]) / initial["log_sd"]) ** 2
            )
        )
        return {
            "q_joint": q_value,
            "alpha": float(alpha_value),
            "hazard": hazard_value,
            "initial_birth": initial_value,
            "total": float(q_value + alpha_value + hazard_value + initial_value),
        }


def spsa(objective, initial, iterations, seed, label):
    rng = np.random.default_rng(seed)
    x = np.clip(initial.copy(), -1.0, 1.0)
    best_x, best_score = x.copy(), objective(x)
    m, v = np.zeros_like(x), np.zeros_like(x)
    candidates, history = [(best_score, best_x.copy())], []
    for iteration in range(1, iterations + 1):
        delta = rng.choice([-1.0, 1.0], size=len(x))
        radius = 0.075 / (iteration + 8.0) ** 0.101
        plus = np.clip(x + radius * delta, -1.0, 1.0)
        minus = np.clip(x - radius * delta, -1.0, 1.0)
        plus_score, minus_score = objective(plus), objective(minus)
        gradient = (plus_score - minus_score) * delta / (2.0 * radius)
        m = 0.90 * m + 0.10 * gradient
        v = 0.999 * v + 0.001 * gradient**2
        mhat = m / (1.0 - 0.90**iteration)
        vhat = v / (1.0 - 0.999**iteration)
        rate = 0.022 / (1.0 + iteration / 90.0) ** 0.4
        proposal = np.clip(x - rate * mhat / (np.sqrt(vhat) + 1e-9), -1.0, 1.0)
        if iteration % 4 == 0:
            score = objective(proposal)
            candidates.append((score, proposal.copy()))
            if score < best_score:
                best_x, best_score = proposal.copy(), score
            x = proposal if score < 1.20 * best_score else 0.5 * (proposal + best_x)
        else:
            x = proposal
        if iteration % 10 == 0 or iteration == 1:
            history.append(
                {
                    "phase": label,
                    "iteration": iteration,
                    "best_objective": best_score,
                }
            )
            print(f"{label} iteration {iteration:3d}: best objective={best_score:.7g}", flush=True)
    return best_x, best_score, candidates, history


def parameter_table(likelihood, regularized):
    rows = []
    blocks = {
        "hazard": ["u", "omega2", "eta"],
        "alpha": ["mean", "sd", "rho"],
        "q": [
            "mean",
            "rho_fast",
            "rho_slow",
            "stationary_fast_sd",
            "stationary_slow_sd",
            "environment_prior_sd",
        ],
        "initial_birth": ["log_mean", "log_sd"],
    }
    for block, names in blocks.items():
        for name in names:
            old, new = float(likelihood[block][name]), float(regularized[block][name])
            rows.append(
                {
                    "parameter": f"{block}.{name}",
                    "likelihood": old,
                    "regularized": new,
                    "absolute_change": new - old,
                    "relative_change": (new - old) / abs(old) if old else np.nan,
                }
            )
    old_e = np.asarray(likelihood["q"]["posterior_mean_E"])
    new_e = np.asarray(regularized["q"]["posterior_mean_E"])
    rows.extend(
        [
            {
                "parameter": "environment.rms_E",
                "likelihood": float(np.sqrt(np.mean(old_e**2))),
                "regularized": float(np.sqrt(np.mean(new_e**2))),
                "absolute_change": float(np.sqrt(np.mean(new_e**2)) - np.sqrt(np.mean(old_e**2))),
                "relative_change": float(np.sqrt(np.mean(new_e**2)) / np.sqrt(np.mean(old_e**2)) - 1.0),
            },
            {
                "parameter": "environment.path_correlation",
                "likelihood": 1.0,
                "regularized": float(np.corrcoef(old_e, new_e)[0, 1]),
                "absolute_change": float(np.corrcoef(old_e, new_e)[0, 1] - 1.0),
                "relative_change": np.nan,
            },
        ]
    )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acf-weight", type=float, default=1.0)
    parser.add_argument("--fit-max-lag", type=int, default=15)
    parser.add_argument("--training-replicates", type=int, default=8)
    parser.add_argument("--validation-replicates", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--seed", type=int, default=61931)
    args = parser.parse_args()
    started = time.time()
    output = OUT / f"weight_{args.acf_weight:g}"
    output.mkdir(parents=True, exist_ok=True)

    source = json.loads((SOURCE / "fit_results.json").read_text())
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    environment_count = int(lengths.max()) - 1
    q_values, q_lengths = q_matrix(prefixes)
    sequences = [q_values[index, :length] for index, length in enumerate(q_lengths) if length]
    qfit, _, q_optimizer = fit_q_pure_map(source["fit"]["q"], sequences, environment_count)

    baseline = {
        "initial_birth": source["fit"]["initial_birth"],
        "alpha": source["fit"]["alpha"],
        "hazard": source["fit"]["hazard"],
        "q": qfit,
    }
    likelihood = LikelihoodData(prefixes, groups, sequences)
    baseline_components = likelihood.components(baseline)

    bounds = regularized_bounds(environment_count)
    source_start = start_vector(source, environment_count)
    physical_start = np.insert(source_start, SIGMA_INDEX, math.log(qfit["environment_prior_sd"]))
    physical_start[6:11] = [
        qfit["mean"],
        qfit["rho_fast"],
        qfit["rho_slow"],
        math.log(qfit["stationary_fast_sd"]),
        math.log(qfit["stationary_slow_sd"]),
    ]
    physical_start[SIGMA_INDEX + 1 :] = qfit["posterior_mean_E"]
    physical_start = np.clip(physical_start, bounds[:, 0], bounds[:, 1])
    normalized_start = normalize(physical_start, bounds)

    target = lineage_acf(observed_birth, lengths, args.fit_max_lag)[0]
    fit_slice = slice(1, args.fit_max_lag + 1)

    def acf_mse(params, replicates, seed):
        generated = simulate(params, params["q"], lengths, replicates, seed=seed)
        predicted = np.nanmean(
            lineage_acf(generated["birth"], lengths, args.fit_max_lag), axis=0
        )
        return float(np.mean((predicted[fit_slice] - target[fit_slice]) ** 2))

    baseline_acf_mse = acf_mse(baseline, args.training_replicates, args.seed)

    def objective(normalized, replicates=args.training_replicates, seed=args.seed):
        try:
            physical = denormalize(np.clip(normalized, -1.0, 1.0), bounds)
            params = unpack_regularized(physical, environment_count)
            components = likelihood.components(params)
            if not np.isfinite(components["total"]):
                return 1e6
            mse = acf_mse(params, replicates, seed)
            return float(
                (baseline_components["total"] - components["total"]) / likelihood.n_data
                + args.acf_weight * mse / baseline_acf_mse
            )
        except (FloatingPointError, OverflowError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return 1e6

    first, _, candidates1, history1 = spsa(
        objective, normalized_start, args.iterations, args.seed + 1, "global"
    )
    second, _, candidates2, history2 = spsa(
        objective, first, max(60, args.iterations // 2), args.seed + 2, "polish"
    )
    candidates = sorted(
        candidates1 + candidates2 + [(objective(normalized_start), normalized_start)],
        key=lambda item: item[0],
    )[:16]
    refinement_scores = [
        objective(vector, replicates=32, seed=args.seed + 3) for _, vector in candidates
    ]
    chosen_normalized = candidates[int(np.argmin(refinement_scores))][1]
    chosen = unpack_regularized(
        denormalize(chosen_normalized, bounds), environment_count
    )
    chosen_components = likelihood.components(chosen)

    likelihood_generated = simulate(
        baseline, baseline["q"], lengths, args.validation_replicates, seed=args.seed + 10
    )
    regularized_generated = simulate(
        chosen, chosen["q"], lengths, args.validation_replicates, seed=args.seed + 11
    )
    baseline_birth_acf = summarize_acf(
        observed_birth, likelihood_generated["birth"], lengths, args.fit_max_lag
    )
    regularized_birth_acf = summarize_acf(
        observed_birth, regularized_generated["birth"], lengths, args.fit_max_lag
    )
    birth_acf = pd.DataFrame(
        {
            "lag": baseline_birth_acf.lag,
            "observed": baseline_birth_acf.observed,
            "likelihood_mean": baseline_birth_acf.simulated_mean,
            "regularized_mean": regularized_birth_acf.simulated_mean,
            "regularized_lo95": regularized_birth_acf.simulated_lo95,
            "regularized_hi95": regularized_birth_acf.simulated_hi95,
        }
    )
    birth_acf.to_csv(output / "birth_acf_comparison.csv", index=False)

    observed_q_acf = pooled_acf_replicates(q_values, q_lengths, args.fit_max_lag)[0]
    baseline_q_samples = pooled_acf_replicates(
        likelihood_generated["q"], q_lengths, args.fit_max_lag
    )
    regularized_q_samples = pooled_acf_replicates(
        regularized_generated["q"], q_lengths, args.fit_max_lag
    )
    q_acf = pd.DataFrame(
        {
            "lag": np.arange(args.fit_max_lag + 1),
            "observed": observed_q_acf,
            "likelihood_mean": np.nanmean(baseline_q_samples, axis=0),
            "regularized_mean": np.nanmean(regularized_q_samples, axis=0),
        }
    )
    q_acf.to_csv(output / "q_acf_comparison.csv", index=False)

    horizon = min(20, int(lengths.max()) - 1)
    complete = lengths > horizon
    generation = np.arange(horizon + 1)
    observed_mean = observed_birth[complete, : horizon + 1].mean(axis=0)
    baseline_paths = likelihood_generated["birth"][:, complete, : horizon + 1].mean(axis=1)
    regularized_paths = regularized_generated["birth"][:, complete, : horizon + 1].mean(axis=1)
    drift = pd.DataFrame(
        {
            "generation": generation,
            "observed": observed_mean,
            "likelihood_mean": baseline_paths.mean(axis=0),
            "regularized_mean": regularized_paths.mean(axis=0),
        }
    )
    drift.to_csv(output / "mean_birth_drift_comparison.csv", index=False)

    marginals = pd.DataFrame(
        marginal_rows(prefixes, likelihood_generated, lengths, "likelihood")
        + marginal_rows(prefixes, regularized_generated, lengths, "regularized")
    )
    marginals.to_csv(output / "marginal_comparison.csv", index=False)
    parameters = parameter_table(baseline, chosen)
    parameters.to_csv(output / "parameter_comparison.csv", index=False)
    effects = pd.DataFrame(
        {
            "generation": np.arange(environment_count),
            "likelihood_E": baseline["q"]["posterior_mean_E"],
            "regularized_E": chosen["q"]["posterior_mean_E"],
        }
    )
    effects.to_csv(output / "environmental_effects.csv", index=False)

    indices = np.arange(1, args.fit_max_lag + 1)
    metrics = {
        "birth_acf_rmse_likelihood": float(np.sqrt(np.mean((birth_acf.observed.iloc[indices] - birth_acf.likelihood_mean.iloc[indices]) ** 2))),
        "birth_acf_rmse_regularized": float(np.sqrt(np.mean((birth_acf.observed.iloc[indices] - birth_acf.regularized_mean.iloc[indices]) ** 2))),
        "q_acf_rmse_likelihood": float(np.sqrt(np.mean((q_acf.observed.iloc[indices] - q_acf.likelihood_mean.iloc[indices]) ** 2))),
        "q_acf_rmse_regularized": float(np.sqrt(np.mean((q_acf.observed.iloc[indices] - q_acf.regularized_mean.iloc[indices]) ** 2))),
        "mean_birth_rmse_likelihood": float(np.sqrt(np.mean((drift.observed - drift.likelihood_mean) ** 2))),
        "mean_birth_rmse_regularized": float(np.sqrt(np.mean((drift.observed - drift.regularized_mean) ** 2))),
        "generation_20_change_observed": float(drift.observed.iloc[-1] - drift.observed.iloc[0]),
        "generation_20_change_likelihood": float(drift.likelihood_mean.iloc[-1] - drift.likelihood_mean.iloc[0]),
        "generation_20_change_regularized": float(drift.regularized_mean.iloc[-1] - drift.regularized_mean.iloc[0]),
        "log_density_change": chosen_components["total"] - baseline_components["total"],
        "log_density_change_per_observation": (chosen_components["total"] - baseline_components["total"]) / likelihood.n_data,
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(birth_acf.lag, birth_acf.observed, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.plot(birth_acf.lag, birth_acf.likelihood_mean, "--", color="#718096", label="Likelihood")
    axis.fill_between(birth_acf.lag, birth_acf.regularized_lo95, birth_acf.regularized_hi95, color="#c4512c", alpha=0.18)
    axis.plot(birth_acf.lag, birth_acf.regularized_mean, "s-", color="#c4512c", ms=3.5, label="Likelihood + ACF")
    axis.set(title="Birth-size ACF", xlabel="Lag", ylabel="Correlation")
    axis.legend(frameon=False)
    axis = axes[0, 1]
    axis.plot(q_acf.lag, q_acf.observed, "o-", color="#23364d", ms=3.5)
    axis.plot(q_acf.lag, q_acf.likelihood_mean, "--", color="#718096")
    axis.plot(q_acf.lag, q_acf.regularized_mean, "s-", color="#c4512c", ms=3.5)
    axis.set(title=r"Effective-jump ACF ($q$)", xlabel="Lag", ylabel="Correlation")
    axis = axes[1, 0]
    axis.plot(drift.generation, drift.observed, "o-", color="#23364d", ms=3.5)
    axis.plot(drift.generation, drift.likelihood_mean, "--", color="#718096")
    axis.plot(drift.generation, drift.regularized_mean, "s-", color="#c4512c", ms=3.5)
    axis.set(title="Balanced-cohort mean birth size", xlabel="Generation", ylabel="Mean birth size")
    axis = axes[1, 1]
    order = ["birth", "q", "alpha", "time"]
    positions, width = np.arange(4), 0.36
    left = marginals.query("model == 'likelihood'").set_index("variable").loc[order].ks_distance
    right = marginals.query("model == 'regularized'").set_index("variable").loc[order].ks_distance
    axis.bar(positions - width / 2, left, width, color="#718096", label="Likelihood")
    axis.bar(positions + width / 2, right, width, color="#c4512c", label="Likelihood + ACF")
    axis.set_xticks(positions, ["birth", "q", r"$\alpha$", r"$\tau$"])
    axis.set(title="Marginal distributions", ylabel="KS distance")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    fig.suptitle(f"Glu full penalized MAP: normalized birth-ACF weight {args.acf_weight:g}")
    fig.savefig(output / "regularized_fit_tradeoff.png", dpi=190)
    fig.savefig(output / "regularized_fit_tradeoff.pdf")
    plt.close(fig)

    pd.DataFrame(history1 + history2).to_csv(output / "optimizer_history.csv", index=False)
    diagnostics = {
        "objective": "(baseline_log_density-current_log_density)/N_data + acf_weight*(birth_ACF_MSE/baseline_birth_ACF_MSE)",
        "acf_weight": args.acf_weight,
        "n_data": likelihood.n_data,
        "pure_likelihood_q_optimizer": {
            "success": bool(q_optimizer.success),
            "message": str(q_optimizer.message),
        },
        "baseline_components": baseline_components,
        "regularized_components": chosen_components,
        "component_changes": {
            key: chosen_components[key] - baseline_components[key] for key in baseline_components
        },
        "training_baseline_acf_mse": baseline_acf_mse,
        "selected_refinement_objective": float(min(refinement_scores)),
        "validation": metrics,
        "likelihood_parameters": serializable(baseline),
        "regularized_parameters": serializable(chosen),
        "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(json.dumps({
        "component_changes": diagnostics["component_changes"],
        "validation": metrics,
        "runtime_minutes": diagnostics["runtime_minutes"],
    }, indent=2), flush=True)
    print(parameters.sort_values("relative_change", key=lambda value: value.abs(), ascending=False).head(12).to_string(index=False), flush=True)
    print(marginals.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
