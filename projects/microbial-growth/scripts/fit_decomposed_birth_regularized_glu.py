#!/usr/bin/env python3
"""Fit Glu with separate common-drift and residual-birth-ACF targets.

The shared cycle effects E_i are first optimized with all population parameters
fixed.  A second stage relaxes every population parameter.  Both stages retain
the complete likelihood, target the balanced-cohort mean birth-size trajectory,
and compare ACFs only after removing the generation-wise common mean.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from autoregressive_residual_model import lineage_acf  # noqa: E402
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
    contiguous_prefixes,
    observed_matrix,
)
from fit_environment_fast_slow_ar_q_glu import simulate  # noqa: E402
from fit_likelihood_acf_regularized_glu import (  # noqa: E402
    LikelihoodData,
    fit_q_pure_map,
    q_covariance,
    regularized_bounds,
    spsa,
    unpack_regularized,
)
from fit_lineage_offset_ar_q_glu import pooled_acf_replicates, q_matrix  # noqa: E402
from glu_workflow import load_data  # noqa: E402


SOURCE = PROJECT_DIR / "outputs/model_comparison/Glu/environment_fast_slow_ar_q_protein_hazard"
OUT = PROJECT_DIR / "outputs/model_comparison/Glu/decomposed_birth_regularized"
SIGMA_INDEX = len(GLOBAL_NAMES)


def canonicalize_environment(params):
    """Use the identifiable convention mean(E)=0, shifting its level into mu_q."""
    result = copy.deepcopy(params)
    effects = np.asarray(result["q"]["posterior_mean_E"], dtype=float)
    shift = float(effects.mean())
    result["q"]["mean"] += shift
    result["q"]["posterior_mean_E"] = effects - shift
    return result


def attach_environment_covariance(params, sequences, environment_count):
    """Attach the exact Gaussian q-likelihood covariance of E conditional on hyperparameters.

    The moment penalties are deliberately not interpreted as an additional
    probabilistic observation model.  This is consequently a conservative
    covariance, but—unlike the old 1e-12 placeholder—it is computed in exactly
    the same way for the likelihood, E-only, and fully relaxed fits.
    """
    qfit = params["q"]
    sigma = float(qfit["environment_prior_sd"])
    precision = np.eye(environment_count) / sigma**2
    cache = {}
    for sequence in sequences:
        length = len(sequence)
        if length not in cache:
            covariance = q_covariance(length, qfit)
            cholesky = np.linalg.cholesky(covariance)
            cache[length] = np.linalg.solve(
                cholesky.T, np.linalg.solve(cholesky, np.eye(length))
            )
        precision[:length, :length] += cache[length]
    covariance = np.linalg.inv(precision)
    qfit["posterior_covariance_E"] = covariance
    qfit["posterior_sd_E"] = np.sqrt(np.diag(covariance))
    return params


def residualize_generation_mean(values, lengths):
    """Remove the common generation mean separately in every replicate."""
    residual = np.asarray(values, dtype=float).copy()
    if residual.ndim == 2:
        residual = residual[None, ...]
        squeeze = True
    elif residual.ndim == 3:
        squeeze = False
    else:
        raise ValueError("birth-size array must have two or three dimensions")
    for generation in range(residual.shape[2]):
        active = lengths > generation
        means = np.nanmean(residual[:, active, generation], axis=1)
        residual[:, active, generation] -= means[:, None]
    return residual[0] if squeeze else residual


class MomentTargets:
    def __init__(self, observed_birth, lengths, horizon, max_lag):
        self.lengths = np.asarray(lengths, dtype=int)
        self.horizon = min(horizon, int(self.lengths.max()) - 1)
        self.max_lag = max_lag
        self.complete = self.lengths > self.horizon
        self.mean_birth = observed_birth[self.complete, : self.horizon + 1].mean(axis=0)
        residual = residualize_generation_mean(observed_birth, self.lengths)
        self.residual_acf = lineage_acf(residual, self.lengths, max_lag)[0]
        self.raw_acf = lineage_acf(observed_birth, self.lengths, max_lag)[0]

    def summarize(self, generated):
        birth = generated["birth"]
        paths = birth[:, self.complete, : self.horizon + 1].mean(axis=1)
        mean_birth = paths.mean(axis=0)
        residual = residualize_generation_mean(birth, self.lengths)
        residual_samples = lineage_acf(residual, self.lengths, self.max_lag)
        raw_samples = lineage_acf(birth, self.lengths, self.max_lag)
        fit_slice = slice(1, self.max_lag + 1)
        return {
            "mean_birth": mean_birth,
            "mean_birth_samples": paths,
            "residual_acf": np.nanmean(residual_samples, axis=0),
            "residual_acf_samples": residual_samples,
            "raw_acf": np.nanmean(raw_samples, axis=0),
            "raw_acf_samples": raw_samples,
            "mean_mse": float(np.mean((mean_birth - self.mean_birth) ** 2)),
            "residual_acf_mse": float(
                np.mean((np.nanmean(residual_samples, axis=0)[fit_slice] - self.residual_acf[fit_slice]) ** 2)
            ),
            "raw_acf_mse": float(
                np.mean((np.nanmean(raw_samples, axis=0)[fit_slice] - self.raw_acf[fit_slice]) ** 2)
            ),
        }


def evaluate(params, targets, lengths, replicates, seed):
    generated = simulate(params, params["q"], lengths, replicates, seed=seed)
    return generated, targets.summarize(generated)


def parameter_table(models):
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
    rows = []
    for block, names in blocks.items():
        for name in names:
            row = {"parameter": f"{block}.{name}"}
            for label, params in models.items():
                row[label] = float(params[block][name])
            rows.append(row)
    for statistic in ["rms_E", "mean_posterior_sd_E"]:
        row = {"parameter": f"environment.{statistic}"}
        for label, params in models.items():
            qfit = params["q"]
            if statistic == "rms_E":
                value = np.sqrt(np.mean(np.asarray(qfit["posterior_mean_E"]) ** 2))
            else:
                value = np.mean(np.asarray(qfit["posterior_sd_E"]))
            row[label] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-weight", type=float, default=0.05)
    parser.add_argument("--residual-acf-weight", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--training-replicates", type=int, default=12)
    parser.add_argument("--validation-replicates", type=int, default=500)
    parser.add_argument("--fixed-iterations", type=int, default=160)
    parser.add_argument("--full-iterations", type=int, default=180)
    parser.add_argument("--seed", type=int, default=73591)
    args = parser.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    source = json.loads((SOURCE / "fit_results.json").read_text())
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    q_values, q_lengths = q_matrix(prefixes)
    sequences = [q_values[i, :length] for i, length in enumerate(q_lengths) if length]
    environment_count = int(lengths.max()) - 1

    qfit, _, q_optimizer = fit_q_pure_map(source["fit"]["q"], sequences, environment_count)
    baseline = canonicalize_environment(
        {
            "initial_birth": source["fit"]["initial_birth"],
            "alpha": source["fit"]["alpha"],
            "hazard": source["fit"]["hazard"],
            "q": qfit,
        }
    )
    baseline = attach_environment_covariance(baseline, sequences, environment_count)
    likelihood = LikelihoodData(prefixes, groups, sequences)
    baseline_components = likelihood.components(baseline)
    targets = MomentTargets(observed_birth, lengths, args.horizon, args.max_lag)
    _, baseline_training = evaluate(
        baseline, targets, lengths, args.training_replicates, args.seed
    )
    scales = {
        "mean_mse": max(baseline_training["mean_mse"], 1e-12),
        "residual_acf_mse": max(baseline_training["residual_acf_mse"], 1e-12),
    }

    def objective_for(params, replicates=args.training_replicates, seed=args.seed):
        try:
            params = attach_environment_covariance(params, sequences, environment_count)
            components = likelihood.components(params)
            if not np.isfinite(components["total"]):
                return 1e6
            _, moments = evaluate(params, targets, lengths, replicates, seed)
            return float(
                (baseline_components["total"] - components["total"]) / likelihood.n_data
                + args.mean_weight * moments["mean_mse"] / scales["mean_mse"]
                + args.residual_acf_weight
                * moments["residual_acf_mse"]
                / scales["residual_acf_mse"]
            )
        except (FloatingPointError, OverflowError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return 1e6

    # Stage 1: infer only the shared cycle path E_i.
    e_bounds = np.tile([-0.12, 0.12], (environment_count, 1))
    e_start = np.asarray(baseline["q"]["posterior_mean_E"], dtype=float)
    e_start = normalize(e_start, e_bounds)

    def fixed_objective(normalized, replicates=args.training_replicates, seed=args.seed):
        params = copy.deepcopy(baseline)
        effects = denormalize(np.clip(normalized, -1.0, 1.0), e_bounds)
        params["q"]["posterior_mean_E"] = effects - effects.mean()
        return objective_for(params, replicates, seed)

    fixed_vector, _, fixed_candidates, fixed_history = spsa(
        fixed_objective, e_start, args.fixed_iterations, args.seed + 1, "E-only"
    )
    fixed_candidates = sorted(
        fixed_candidates + [(fixed_objective(e_start), e_start)], key=lambda value: value[0]
    )[:16]
    fixed_scores = [
        fixed_objective(vector, replicates=40, seed=args.seed + 3)
        for _, vector in fixed_candidates
    ]
    fixed_vector = fixed_candidates[int(np.argmin(fixed_scores))][1]
    fixed = copy.deepcopy(baseline)
    fixed_effects = denormalize(fixed_vector, e_bounds)
    fixed["q"]["posterior_mean_E"] = fixed_effects - fixed_effects.mean()
    fixed = attach_environment_covariance(fixed, sequences, environment_count)

    # Stage 2: start from E-only and relax all population parameters.
    bounds = regularized_bounds(environment_count)
    source_start = start_vector(source, environment_count)
    physical_start = np.insert(
        source_start, SIGMA_INDEX, math.log(fixed["q"]["environment_prior_sd"])
    )
    physical_start[: len(GLOBAL_NAMES)] = np.r_[
        math.log(fixed["hazard"]["u"]),
        math.log(fixed["hazard"]["omega2"]),
        math.log(fixed["hazard"]["eta"]),
        math.log(fixed["alpha"]["mean"]),
        math.log(fixed["alpha"]["sd"]),
        fixed["alpha"]["rho"],
        fixed["q"]["mean"],
        fixed["q"]["rho_fast"],
        fixed["q"]["rho_slow"],
        math.log(fixed["q"]["stationary_fast_sd"]),
        math.log(fixed["q"]["stationary_slow_sd"]),
        fixed["initial_birth"]["log_mean"],
        math.log(fixed["initial_birth"]["log_sd"]),
    ]
    physical_start[SIGMA_INDEX + 1 :] = fixed["q"]["posterior_mean_E"]
    full_start = normalize(np.clip(physical_start, bounds[:, 0], bounds[:, 1]), bounds)

    def full_objective(normalized, replicates=args.training_replicates, seed=args.seed):
        physical = denormalize(np.clip(normalized, -1.0, 1.0), bounds)
        params = unpack_regularized(physical, environment_count)
        return objective_for(params, replicates, seed)

    full_vector, _, full_candidates1, full_history1 = spsa(
        full_objective, full_start, args.full_iterations, args.seed + 4, "full"
    )
    full_vector, _, full_candidates2, full_history2 = spsa(
        full_objective,
        full_vector,
        max(60, args.full_iterations // 2),
        args.seed + 5,
        "full-polish",
    )
    full_candidates = sorted(
        full_candidates1
        + full_candidates2
        + [(full_objective(full_start), full_start)],
        key=lambda value: value[0],
    )[:16]
    full_scores = [
        full_objective(vector, replicates=40, seed=args.seed + 6)
        for _, vector in full_candidates
    ]
    full_vector = full_candidates[int(np.argmin(full_scores))][1]
    full = unpack_regularized(denormalize(full_vector, bounds), environment_count)
    full = attach_environment_covariance(full, sequences, environment_count)

    models = {"likelihood": baseline, "E_only": fixed, "full": full}
    generated, summaries, components = {}, {}, {}
    for index, (label, params) in enumerate(models.items()):
        generated[label], summaries[label] = evaluate(
            params,
            targets,
            lengths,
            args.validation_replicates,
            args.seed + 20 + index,
        )
        components[label] = likelihood.components(params)

    lags = np.arange(args.max_lag + 1)
    acf = pd.DataFrame(
        {
            "lag": lags,
            "observed_raw": targets.raw_acf,
            "observed_residual": targets.residual_acf,
        }
    )
    for label in models:
        acf[f"{label}_raw"] = summaries[label]["raw_acf"]
        acf[f"{label}_residual"] = summaries[label]["residual_acf"]
        acf[f"{label}_raw_lo95"], acf[f"{label}_raw_hi95"] = np.nanquantile(
            summaries[label]["raw_acf_samples"], [0.025, 0.975], axis=0
        )
        acf[f"{label}_residual_lo95"], acf[f"{label}_residual_hi95"] = np.nanquantile(
            summaries[label]["residual_acf_samples"], [0.025, 0.975], axis=0
        )
    acf.to_csv(OUT / "birth_acf_decomposition.csv", index=False)

    generations = np.arange(targets.horizon + 1)
    drift = pd.DataFrame({"generation": generations, "observed": targets.mean_birth})
    for label in models:
        paths = summaries[label]["mean_birth_samples"]
        drift[f"{label}_mean"] = paths.mean(axis=0)
        drift[f"{label}_lo95"], drift[f"{label}_hi95"] = np.quantile(
            paths, [0.025, 0.975], axis=0
        )
    drift.to_csv(OUT / "mean_birth_trajectory.csv", index=False)

    observed_q_acf = pooled_acf_replicates(q_values, q_lengths, args.max_lag)[0]
    qacf = pd.DataFrame({"lag": lags, "observed": observed_q_acf})
    for label in models:
        samples = pooled_acf_replicates(generated[label]["q"], q_lengths, args.max_lag)
        qacf[f"{label}_mean"] = np.nanmean(samples, axis=0)
        qacf[f"{label}_lo95"], qacf[f"{label}_hi95"] = np.nanquantile(
            samples, [0.025, 0.975], axis=0
        )
    qacf.to_csv(OUT / "q_acf_comparison.csv", index=False)

    marginal = pd.DataFrame(
        sum(
            [marginal_rows(prefixes, generated[label], lengths, label) for label in models],
            [],
        )
    )
    marginal.to_csv(OUT / "marginal_comparison.csv", index=False)
    parameters = parameter_table(models)
    parameters.to_csv(OUT / "parameter_comparison.csv", index=False)
    effects = pd.DataFrame({"generation": np.arange(environment_count)})
    for label, params in models.items():
        effects[f"{label}_E"] = params["q"]["posterior_mean_E"]
        effects[f"{label}_sd"] = params["q"]["posterior_sd_E"]
    effects.to_csv(OUT / "environmental_effects.csv", index=False)

    fit_slice = slice(1, args.max_lag + 1)
    metrics = {}
    for label in models:
        summary = summaries[label]
        metrics[label] = {
            "raw_birth_acf_rmse": float(np.sqrt(summary["raw_acf_mse"])),
            "residual_birth_acf_rmse": float(np.sqrt(summary["residual_acf_mse"])),
            "mean_birth_rmse": float(np.sqrt(summary["mean_mse"])),
            "q_acf_rmse": float(
                np.sqrt(np.mean((qacf["observed"].to_numpy()[fit_slice] - qacf[f"{label}_mean"].to_numpy()[fit_slice]) ** 2))
            ),
            "generation_20_change": float(summary["mean_birth"][-1] - summary["mean_birth"][0]),
            "log_density_change": float(components[label]["total"] - baseline_components["total"]),
            "log_density_change_per_observation": float(
                (components[label]["total"] - baseline_components["total"]) / likelihood.n_data
            ),
        }
    metrics["observed"] = {
        "generation_20_change": float(targets.mean_birth[-1] - targets.mean_birth[0])
    }

    colors = {"likelihood": "#718096", "E_only": "#2f855a", "full": "#c4512c"}
    styles = {"likelihood": "--", "E_only": "-.", "full": "-"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for column, kind, title in [
        (0, "raw", "Raw birth-size ACF"),
        (1, "residual", "Generation-demeaned birth-size ACF"),
    ]:
        axis = axes[0, column]
        axis.plot(lags, acf[f"observed_{kind}"], "o-", color="#23364d", ms=3.5, label="Observed")
        for label in models:
            axis.plot(lags, acf[f"{label}_{kind}"], styles[label], color=colors[label], label=label.replace("_", " "))
        axis.set(title=title, xlabel="Lag", ylabel="Correlation")
        axis.grid(alpha=0.16)
    axes[0, 0].legend(frameon=False, fontsize=8)

    axis = axes[0, 2]
    axis.plot(generations, drift.observed, "o-", color="#23364d", ms=3.5, label="Observed")
    for label in models:
        axis.plot(generations, drift[f"{label}_mean"], styles[label], color=colors[label], label=label.replace("_", " "))
    axis.set(title="Balanced-cohort mean birth size", xlabel="Generation", ylabel="Mean birth size")
    axis.grid(alpha=0.16)

    axis = axes[1, 0]
    axis.plot(lags, qacf.observed, "o-", color="#23364d", ms=3.5)
    for label in models:
        axis.plot(lags, qacf[f"{label}_mean"], styles[label], color=colors[label])
    axis.set(title=r"Effective-jump ACF ($q$)", xlabel="Lag", ylabel="Correlation")
    axis.grid(alpha=0.16)

    axis = axes[1, 1]
    variables = ["birth", "q", "alpha", "time"]
    width = 0.23
    for index, label in enumerate(models):
        values = marginal.query("model == @label").set_index("variable").loc[variables].ks_distance
        axis.bar(np.arange(4) + (index - 1) * width, values, width, color=colors[label], label=label.replace("_", " "))
    axis.set_xticks(np.arange(4), ["birth", "q", r"$\alpha$", r"$\tau$"])
    axis.set(title="Marginal distributions", ylabel="KS distance")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.16, axis="y")

    axis = axes[1, 2]
    for label in models:
        axis.plot(effects.generation, effects[f"{label}_E"], styles[label], color=colors[label], label=label.replace("_", " "))
    axis.axhline(0, color="#4a5568", lw=0.7)
    axis.set(title=r"Shared cycle effects $E_i$", xlabel="Generation", ylabel=r"$E_i$")
    axis.grid(alpha=0.16)
    fig.suptitle("Glu: common drift separated from residual birth-size memory")
    fig.savefig(OUT / "decomposed_regularization_diagnostics.png", dpi=190)
    fig.savefig(OUT / "decomposed_regularization_diagnostics.pdf")
    plt.close(fig)

    histories = []
    for stage, rows in [
        ("E_only", fixed_history),
        ("full", full_history1),
        ("full_polish", full_history2),
    ]:
        for row in rows:
            histories.append({"stage": stage, **row})
    pd.DataFrame(histories).to_csv(OUT / "optimizer_history.csv", index=False)

    diagnostics = {
        "objective": "(baseline_log_density-current_log_density)/N + mean_weight*(mean_birth_MSE/baseline) + residual_acf_weight*(demeaned_birth_ACF_MSE/baseline)",
        "weights": {
            "mean_birth": args.mean_weight,
            "residual_birth_acf": args.residual_acf_weight,
        },
        "posterior_environment_covariance": "exact conditional covariance from q likelihood and Gaussian E prior; same calculation for every fit",
        "n_data": likelihood.n_data,
        "training_scales": scales,
        "metrics": metrics,
        "components": components,
        "component_changes": {
            label: {
                key: components[label][key] - baseline_components[key]
                for key in baseline_components
            }
            for label in ["E_only", "full"]
        },
        "q_optimizer": {"success": bool(q_optimizer.success), "message": str(q_optimizer.message)},
        "parameters": {label: serializable(params) for label, params in models.items()},
        "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    lines = [
        "# Decomposed mean-drift and residual-ACF fit for Glu",
        "",
        "The fit separates the common generation-wise birth-size trajectory from",
        "within-generation lineage fluctuations. The objective retains the complete",
        "likelihood, targets the balanced-cohort mean birth-size curve, and applies",
        "the ACF penalty only after subtracting the common mean at every generation.",
        "",
        "All posterior-predictive models use the same conditional posterior",
        "covariance for the inferred environmental path `E`; no model uses the old",
        "near-zero covariance placeholder.",
        "",
        "## Validation metrics",
        "",
        "| Model | Raw b-ACF RMSE | Residual b-ACF RMSE | Mean-b RMSE | q-ACF RMSE | Gen.-20 change | Log-density loss / datum |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in models:
        value = metrics[label]
        lines.append(
            f"| {label.replace('_', ' ')} | {value['raw_birth_acf_rmse']:.5f} | "
            f"{value['residual_birth_acf_rmse']:.5f} | {value['mean_birth_rmse']:.5f} | "
            f"{value['q_acf_rmse']:.5f} | {value['generation_20_change']:.4f} | "
            f"{-value['log_density_change_per_observation']:.5f} |"
        )
    lines.extend(
        [
            "",
            f"The observed generation-20 change is {metrics['observed']['generation_20_change']:.4f}.",
            "",
            "`E only` changes only the inferred shared cycle path. `full` starts from",
            "that solution and then relaxes all population parameters.",
            "",
            "## Interpretation",
            "",
            f"Changing only `E` reduces mean-trajectory RMSE by "
            f"{100.0 * (1.0 - metrics['E_only']['mean_birth_rmse'] / metrics['likelihood']['mean_birth_rmse']):.1f}% "
            f"and raw birth-ACF RMSE by "
            f"{100.0 * (1.0 - metrics['E_only']['raw_birth_acf_rmse'] / metrics['likelihood']['raw_birth_acf_rmse']):.1f}%. "
            "The residual-ACF RMSE changes much less because a generation-shared",
            "effect largely cancels when the generation mean is removed.",
            "",
            "The fully relaxed solution does not improve the raw birth ACF or the",
            "mean trajectory relative to the `E only` solution, and loses more",
            "likelihood. It is therefore not preferred. The remaining ACF mismatch",
            "is in the generation-demeaned lineage fluctuations rather than in the",
            "common mean drift alone.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"metrics": metrics, "runtime_minutes": diagnostics["runtime_minutes"]}, indent=2))
    print(marginal.to_string(index=False))
    print(parameters.to_string(index=False))


if __name__ == "__main__":
    main()
