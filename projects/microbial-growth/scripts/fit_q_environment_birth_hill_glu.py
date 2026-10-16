#!/usr/bin/env python3
"""Fit Glu with individual q effects and a four-parameter p-and-b Hill hazard."""

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
from scipy.optimize import differential_evolution, minimize


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from autoregressive_residual_model import (  # noqa: E402
    lineage_acf,
    profile_birth_dependent_hazard_log_likelihood,
)
from fit_all_parameters_birth_acf_glu import marginal_rows  # noqa: E402
from fit_autoregressive_residual_glu import contiguous_prefixes, observed_matrix  # noqa: E402
from fit_decomposed_birth_regularized_glu import residualize_generation_mean  # noqa: E402
from fit_environment_lineage_offset_ar_q_glu import acf_summary, fit_q  # noqa: E402
from fit_individual_environment_lineage_effects_glu import (  # noqa: E402
    joint_effect_posterior,
    posterior_summary_tables,
    serializable,
    simulate,
)
from fit_individual_q_alpha_environment_effects_glu import trajectory  # noqa: E402
from fit_lineage_offset_ar_q_glu import q_matrix  # noqa: E402
from glu_workflow import load_data  # noqa: E402


BASE = PROJECT_DIR / "outputs/model_comparison/Glu"
SOURCE = BASE / "environment_fast_slow_ar_q_protein_hazard"
OUT = BASE / "q_environment_birth_hill_protein_hazard"


def hazard_arrays(prefixes):
    frames = [group.iloc[:-1] for group in prefixes if len(group) > 1]
    transition = pd.concat(frames, ignore_index=True)
    return (
        np.log(transition.length_birth.to_numpy(float)),
        transition.alpha.to_numpy(float),
        transition.time.to_numpy(float),
    )


def fit_birth_dependent_hazard(prefixes, source_hazard, seed=54127):
    log_birth, alpha, time_values = hazard_arrays(prefixes)
    log_b_star = float(log_birth.mean())
    bounds = [
        (math.log(0.05), math.log(10.0)),
        (math.log(0.1), math.log(50.0)),
        (-15.0, 15.0),
    ]

    def objective(theta, kappa_free=True, order=48):
        kappa = float(theta[2]) if kappa_free else 0.0
        value, _ = profile_birth_dependent_hazard_log_likelihood(
            log_birth,
            alpha,
            time_values,
            float(theta[0]),
            float(theta[1]),
            kappa,
            log_b_star,
            order=order,
        )
        return -value

    initial = np.asarray(
        [math.log(source_hazard["u"]), math.log(source_hazard["eta"]), 0.0]
    )
    global_fit = differential_evolution(
        objective,
        bounds,
        seed=seed,
        popsize=14,
        maxiter=100,
        polish=False,
        updating="immediate",
    )
    candidates = [global_fit.x, initial]
    local = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-13},
        )
        for start in candidates
    ]
    fitted = min(local, key=lambda item: item.fun)
    extended_ll, omega2 = profile_birth_dependent_hazard_log_likelihood(
        log_birth,
        alpha,
        time_values,
        float(fitted.x[0]),
        float(fitted.x[1]),
        float(fitted.x[2]),
        log_b_star,
        order=96,
    )
    extended = {
        "u": math.exp(float(fitted.x[0])),
        "omega2": omega2,
        "eta": math.exp(float(fitted.x[1])),
        "kappa": float(fitted.x[2]),
        "b_star": math.exp(log_b_star),
        "log_likelihood": extended_ll,
        "success": bool(fitted.success),
        "message": str(fitted.message),
    }

    nested_bounds = bounds[:2]

    def nested_objective(theta):
        return objective(np.r_[theta, 0.0], kappa_free=False)

    nested_starts = [initial[:2], fitted.x[:2]]
    nested_fits = [
        minimize(
            nested_objective,
            start,
            method="L-BFGS-B",
            bounds=nested_bounds,
            options={"maxiter": 2000, "ftol": 1e-13},
        )
        for start in nested_starts
    ]
    nested_fit = min(nested_fits, key=lambda item: item.fun)
    nested_ll, nested_omega = profile_birth_dependent_hazard_log_likelihood(
        log_birth,
        alpha,
        time_values,
        float(nested_fit.x[0]),
        float(nested_fit.x[1]),
        0.0,
        log_b_star,
        order=96,
    )
    nested = {
        "u": math.exp(float(nested_fit.x[0])),
        "omega2": nested_omega,
        "eta": math.exp(float(nested_fit.x[1])),
        "b_star": math.exp(log_b_star),
        "log_likelihood": nested_ll,
        "success": bool(nested_fit.success),
        "message": str(nested_fit.message),
    }
    comparison = {
        "n_transitions": len(log_birth),
        "twice_log_likelihood_gain": 2.0 * (extended_ll - nested_ll),
        "delta_bic_extended_minus_nested": math.log(len(log_birth))
        - 2.0 * (extended_ll - nested_ll),
    }
    return extended, nested, comparison


def time_acf(prefixes, generated, q_lengths, max_lag):
    observed = np.full((len(prefixes), int(q_lengths.max())), np.nan)
    for index, (group, length) in enumerate(zip(prefixes, q_lengths)):
        if length:
            observed[index, :length] = group.time.to_numpy(float)[:length]
    target = lineage_acf(observed, q_lengths, max_lag)[0]
    samples = lineage_acf(generated["time"], q_lengths, max_lag)
    low, high = np.nanquantile(samples, [0.025, 0.975], axis=0)
    return target, np.nanmean(samples, axis=0), low, high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-lag", type=int, default=15)
    parser.add_argument("--seed", type=int, default=54127)
    args = parser.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    source = json.loads((SOURCE / "fit_results.json").read_text())
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed_birth, lengths = observed_matrix(prefixes)
    q_values, q_lengths = q_matrix(prefixes)
    qfit, q_density, q_optimizer = fit_q(
        [q_values[index, :length] for index, length in enumerate(q_lengths) if length],
        q_values.shape[1],
    )
    q_posterior = joint_effect_posterior(q_values, q_lengths, qfit)
    lineage_table, environment_table, _ = posterior_summary_tables(
        prefixes, q_values, q_lengths, q_posterior
    )
    lineage_table.to_csv(OUT / "lineage_effect_posterior.csv", index=False)
    environment_table.to_csv(OUT / "q_environment_effect_posterior.csv", index=False)

    extended, nested, likelihood_comparison = fit_birth_dependent_hazard(
        prefixes, source["fit"]["hazard"], args.seed
    )
    common = {
        "initial_birth": source["fit"]["initial_birth"],
        "alpha": source["fit"]["alpha"],
        "q": qfit,
    }
    initial = np.asarray([float(group.length_birth.iloc[0]) for group in prefixes])
    generated = {
        "protein_hill": simulate(
            {**common, "hazard": nested},
            lengths,
            q_posterior,
            q_lengths,
            args.replicates,
            args.seed + 1,
            conditional=True,
            observed_initial=initial,
        ),
        "birth_hill": simulate(
            {**common, "hazard": extended},
            lengths,
            q_posterior,
            q_lengths,
            args.replicates,
            args.seed + 2,
            conditional=True,
            observed_initial=initial,
        ),
    }

    trajectories, marginal_rows_all = [], []
    birth_acf = pd.DataFrame({"lag": np.arange(args.max_lag + 1)})
    demeaned_acf = pd.DataFrame({"lag": np.arange(args.max_lag + 1)})
    tau_acf = pd.DataFrame({"lag": np.arange(args.max_lag + 1)})
    for label, simulation in generated.items():
        values = trajectory(prefixes, lengths, simulation, args.horizon)
        values["model"] = label
        trajectories.append(values)
        for table, residualized in [(birth_acf, False), (demeaned_acf, True)]:
            target, mean, low, high = acf_summary(
                observed_birth,
                simulation["birth"],
                lengths,
                args.max_lag,
                residualized,
            )
            table["observed"] = target
            table[f"{label}_mean"] = mean
            table[f"{label}_lo95"] = low
            table[f"{label}_hi95"] = high
        target, mean, low, high = time_acf(
            prefixes, simulation, q_lengths, args.max_lag
        )
        tau_acf["observed"] = target
        tau_acf[f"{label}_mean"] = mean
        tau_acf[f"{label}_lo95"] = low
        tau_acf[f"{label}_hi95"] = high
        marginal_rows_all.extend(
            marginal_rows(prefixes, simulation, lengths, label)
        )
    trajectories = pd.concat(trajectories, ignore_index=True)
    marginal = pd.DataFrame(marginal_rows_all)
    trajectories.to_csv(OUT / "balanced_trajectory_comparison.csv", index=False)
    birth_acf.to_csv(OUT / "birth_acf_comparison.csv", index=False)
    demeaned_acf.to_csv(OUT / "demeaned_birth_acf_comparison.csv", index=False)
    tau_acf.to_csv(OUT / "tau_acf_comparison.csv", index=False)
    marginal.to_csv(OUT / "marginal_comparison.csv", index=False)

    metrics = {}
    fit_slice = slice(1, args.max_lag + 1)
    for label in generated:
        metrics[label] = {}
        for variable in ["birth", "alpha", "time", "growth"]:
            values = trajectories.query("model == @label and variable == @variable")
            metrics[label][f"{variable}_trajectory_rmse"] = float(
                np.sqrt(np.mean((values.observed - values.simulated_mean) ** 2))
            )
        for name, table in [
            ("birth_acf", birth_acf),
            ("demeaned_birth_acf", demeaned_acf),
            ("tau_acf", tau_acf),
        ]:
            metrics[label][f"{name}_rmse"] = float(
                np.sqrt(
                    np.mean(
                        (table.observed.to_numpy()[fit_slice] - table[f"{label}_mean"].to_numpy()[fit_slice]) ** 2
                    )
                )
            )
        lookup = marginal.query("model == @label").set_index("variable")
        for variable in ["birth", "q", "alpha", "time"]:
            metrics[label][f"{variable}_ks"] = float(
                lookup.loc[variable, "ks_distance"]
            )

    colors = {"protein_hill": "#718096", "birth_hill": "#c4512c"}
    labels = {"protein_hill": r"$p$-only Hill", "birth_hill": r"$p+b$ Hill"}
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.1), constrained_layout=True)
    for axis, variable, title, ylabel in [
        (axes[0, 0], "birth", "Balanced-cohort mean birth size", "Birth size"),
        (axes[0, 1], "growth", r"Mean growth increment $\alpha\tau$", r"$\alpha\tau$"),
        (axes[0, 2], "time", r"Mean division time $\tau$", r"$\tau$"),
    ]:
        for label in generated:
            values = trajectories.query("model == @label and variable == @variable")
            axis.fill_between(values.generation, values.simulated_lo95, values.simulated_hi95, color=colors[label], alpha=0.09)
            axis.plot(values.generation, values.simulated_mean, "--" if label == "protein_hill" else "-", color=colors[label], label=labels[label])
        observed = trajectories.query("model == 'protein_hill' and variable == @variable")
        axis.plot(observed.generation, observed.observed, "o-", color="#23364d", ms=3, label="Observed")
        axis.set(title=title, xlabel="Generation", ylabel=ylabel)
    axes[0, 0].legend(frameon=False, fontsize=8)

    for axis, table, title in [
        (axes[1, 0], birth_acf, "Raw birth-size ACF"),
        (axes[1, 1], demeaned_acf, "Generation-demeaned birth-size ACF"),
        (axes[1, 2], tau_acf, "Division-time ACF"),
    ]:
        axis.plot(table.lag, table.observed, "o-", color="#23364d", ms=3)
        for label in generated:
            axis.fill_between(table.lag, table[f"{label}_lo95"], table[f"{label}_hi95"], color=colors[label], alpha=0.09)
            axis.plot(table.lag, table[f"{label}_mean"], "--" if label == "protein_hill" else "-", color=colors[label])
        axis.set(title=title, xlabel="Lag", ylabel="Correlation")
    for axis in axes.ravel():
        axis.grid(alpha=0.16)
    fig.suptitle(r"Glu: inferred $E_i$ in q with explicit birth-size Hill feedback")
    fig.savefig(OUT / "birth_hill_diagnostics.png", dpi=190)
    fig.savefig(OUT / "birth_hill_diagnostics.pdf")
    plt.close(fig)

    result = {
        "model": {
            "q": "q_li=A_l+E_i+r_li; every A_l and E_i inferred",
            "alpha": "stationary Gamma marginal with Gaussian-copula AR(1); no G_i process",
            "hazard": "h(t|b,alpha)=alpha*omega2*sigmoid(eta*log(b*(exp(alpha*t)-1)/u)+kappa*log(b/b_star))",
            "nested_model": "kappa=0",
        },
        "fit": {
            "birth_dependent_hazard": serializable(extended),
            "nested_protein_hazard": serializable(nested),
            "likelihood_comparison": serializable(likelihood_comparison),
            "q_log_joint_density": float(q_density),
            "q_optimizer_success": bool(q_optimizer.success),
        },
        "metrics": serializable(metrics),
        "run_config": vars(args),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    (OUT / "fit_results.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Glu q-environment model with explicit birth-size Hill feedback",
        "",
        r"The q process remains $q_{\ell i}=A_\ell+E_i+r_{\ell i}$, with every",
        r"$A_\ell$ and $E_i$ inferred. The alpha-environment $G_i$ is removed.",
        "The timing hazard is",
        "",
        r"$$h(t\mid b,\alpha)=\alpha\omega_2\,\operatorname{sigmoid}\!\left[",
        r"\eta\log\frac{b(e^{\alpha t}-1)}{u}+\kappa\log\frac{b}{b_\star}\right].$$",
        "",
        r"The nested protein-only model has $\kappa=0$. The fixed $b_\star$ is the",
        "geometric mean birth size in the fitted transition panel.",
        "",
        "## Hazard fit",
        "",
        f"- u: {extended['u']:.6f}",
        f"- omega2: {extended['omega2']:.6f}",
        f"- eta: {extended['eta']:.6f}",
        f"- kappa: {extended['kappa']:.6f}",
        f"- b_star: {extended['b_star']:.6f}",
        f"- Twice log-likelihood gain over kappa=0: {likelihood_comparison['twice_log_likelihood_gain']:.3f}",
        f"- Delta BIC (extended minus nested): {likelihood_comparison['delta_bic_extended_minus_nested']:.3f}",
        "",
        "## Posterior checks",
        "",
        "| Metric | Protein-only Hill | Explicit-b Hill |",
        "|---|---:|---:|",
    ]
    for key, title in [
        ("birth_trajectory_rmse", "Mean-b trajectory RMSE"),
        ("growth_trajectory_rmse", "Alpha-tau trajectory RMSE"),
        ("time_trajectory_rmse", "Tau trajectory RMSE"),
        ("birth_acf_rmse", "Raw b-ACF RMSE"),
        ("demeaned_birth_acf_rmse", "Demeaned b-ACF RMSE"),
        ("tau_acf_rmse", "Tau-ACF RMSE"),
        ("birth_ks", "Birth KS"),
        ("time_ks", "Tau KS"),
    ]:
        lines.append(
            f"| {title} | {metrics['protein_hill'][key]:.5f} | {metrics['birth_hill'][key]:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The fitted explicit-birth coefficient is kappa={extended['kappa']:.5f},",
            "essentially zero. The likelihood-ratio statistic is only",
            f"{likelihood_comparison['twice_log_likelihood_gain']:.4f}, and adding kappa",
            f"raises BIC by {likelihood_comparison['delta_bic_extended_minus_nested']:.3f}.",
            "The posterior checks are correspondingly indistinguishable from the",
            "protein-only hazard. The protein variable p=b(exp(alpha*t)-1) already",
            "contains birth size; these data do not support an additional static",
            "birth-size exponent. More useful timing flexibility must therefore be",
            "time/cycle dependent or affect another part of the hazard.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
