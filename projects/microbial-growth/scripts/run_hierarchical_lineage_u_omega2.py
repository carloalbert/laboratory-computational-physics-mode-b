#!/usr/bin/env python3
"""Fit Glu with both u_l and omega2_l varying persistently by lineage."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import emcee
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from hierarchical_lineage_u_endpoint_residual import (  # noqa: E402
    hermite_rule, prepare_lineage_transitions
)
from hierarchical_lineage_u_omega2_endpoint_residual import (  # noqa: E402
    PARAMETER_NAMES, log_marginal_division_likelihood,
    sample_conditional_lineage_effects,
)
from run_hierarchical_lineage_u import (  # noqa: E402
    NUISANCE_NAMES, lineage_groups, simulate_groups, summarize_chain,
)
from run_loglogistic_endpoint_residual import load_endpoint_data, mean_acf  # noqa: E402


MODEL_NAME = "hierarchical_lineage_u_omega2_endpoint_residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walkers", type=int, default=24)
    parser.add_argument("--burn", type=int, default=300)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--acf-replicates", type=int, default=300)
    parser.add_argument("--acf-max-lag", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4237)
    return parser.parse_args()


def bounds(data: dict[str, np.ndarray], pivot: float) -> list[tuple[float, float]]:
    u_max = max(20.0, 5 * float(np.percentile(data["added_size"], 99)))
    return [(-pivot + 1e-5, u_max), (0.05, 30), (1e-6, 1 - 1e-6),
            (1e-5, 1.25), (1e-5, 1.25)]


def valid(params: np.ndarray, limits: list[tuple[float, float]], pivot: float) -> bool:
    return (
        len(params) == 5
        and all(low < value < high for value, (low, high) in zip(params, limits))
        and params[0] + params[2] * pivot > 1e-5
    )


def aligned_effects(params, prepared, groups, pivot, nodes, weights, rng):
    u_transition, omega_transition = sample_conditional_lineage_effects(
        params, prepared, pivot, nodes, weights, rng
    )
    labels = np.asarray(prepared["lineage_labels"])
    lookup = {label: (u, omega) for label, u, omega in zip(labels, u_transition, omega_transition)}
    u_bar = params[0] + params[2] * pivot
    result_u, result_omega = np.empty(len(groups)), np.empty(len(groups))
    for index, group in enumerate(groups):
        label = group.lineage.iloc[0]
        if label in lookup:
            result_u[index], result_omega[index] = lookup[label]
        else:
            result_u[index] = u_bar * np.exp(params[3] * rng.normal())
            result_omega[index] = params[1] * np.exp(params[4] * rng.normal())
    return result_u, result_omega


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = PROJECT_DIR / "outputs" / "model_comparison" / "Glu" / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    real, transition, data, _ = load_endpoint_data("Glu")
    pivot = float(np.mean(data["transition_birth"]))
    prepared = prepare_lineage_transitions(
        transition.length_birth.to_numpy(float), transition.added_size.to_numpy(float),
        transition.lineage.to_numpy(),
    )
    nodes, weights = hermite_rule(args.quadrature_order)
    common_dir = PROJECT_DIR / "outputs" / "model_comparison" / "Glu" / "loglogistic_endpoint_residual"
    common = pd.read_csv(common_dir / "flat_chain.csv")
    common_logp = pd.read_csv(common_dir / "flat_log_probability.csv").iloc[:, 0].to_numpy(float)
    center_common = common.iloc[int(np.argmax(common_logp))][["p_star", "omega2", "eta"]].to_numpy(float)
    nuisance_chain = common[NUISANCE_NAMES].to_numpy(float)
    limits = bounds(data, pivot)

    def log_prob(params):
        if not valid(params, limits, pivot):
            return -np.inf
        return log_marginal_division_likelihood(params, prepared, pivot, nodes, weights)

    def objective(params):
        value = log_prob(params)
        return -value if np.isfinite(value) else 1e100

    starts = [
        np.r_[center_common, su, so]
        for su, so in [(0.055, 0.075), (0.10, 0.10), (0.20, 0.15)]
    ]
    fits = [minimize(objective, start, method="Nelder-Mead",
                     options={"maxiter": 3500, "xatol": 2e-6, "fatol": 2e-4})
            for start in starts]
    mle_fit = min(fits, key=lambda item: item.fun)
    mle = mle_fit.x
    if not valid(mle, limits, pivot):
        raise RuntimeError(f"Invalid MLE {mle}")
    print("MLE", dict(zip(PARAMETER_NAMES, mle)), "logL", -mle_fit.fun)

    rng = np.random.default_rng(args.seed)
    scales = np.maximum(np.abs(mle) * 0.015, [0.002, 0.03, 0.001, 0.002, 0.002])
    walkers = []
    while len(walkers) < args.walkers:
        proposal = mle + rng.normal(size=5) * scales
        if np.isfinite(log_prob(proposal)):
            walkers.append(proposal)
    sampler = emcee.EnsembleSampler(args.walkers, 5, log_prob, moves=[
        (emcee.moves.StretchMove(), 0.55), (emcee.moves.DEMove(), 0.45)
    ])
    state = sampler.run_mcmc(np.asarray(walkers), args.burn, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, args.steps, progress=True)
    chain = sampler.get_chain()
    flat = sampler.get_chain(thin=args.thin, flat=True)
    flat_logp = sampler.get_log_prob(thin=args.thin, flat=True)
    summary = summarize_chain(flat, flat_logp, PARAMETER_NAMES)
    pd.DataFrame(flat, columns=PARAMETER_NAMES).to_csv(output_dir / "hierarchy_flat_chain.csv", index=False)
    pd.DataFrame({"log_probability": flat_logp}).to_csv(output_dir / "flat_log_probability.csv", index=False)
    summary.to_csv(output_dir / "hierarchy_posterior_summary.csv", index=False)

    groups = lineage_groups(real)
    max_lag = min(args.acf_max_lag, max(len(group) for group in groups) - 1)
    observed = mean_acf(real, "length_birth", max_lag)
    samples = np.full((args.acf_replicates, max_lag + 1), np.nan)
    rng = np.random.default_rng(args.seed + 20_000)
    for replicate in range(args.acf_replicates):
        params = flat[rng.integers(len(flat))]
        nuisance = nuisance_chain[rng.integers(len(nuisance_chain))]
        lineage_u, lineage_omega = aligned_effects(
            params, prepared, groups, pivot, nodes, weights, rng
        )
        simulated = simulate_groups(
            groups, lineage_u, lineage_omega, nuisance, float(params[2]), rng, True
        )
        samples[replicate] = mean_acf(simulated, "length_birth", max_lag)
    median = np.nanmedian(samples, axis=0)
    lower, upper = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    contributing = [sum(len(group) >= 3 and len(group) > lag for group in groups)
                    for lag in range(max_lag + 1)]
    frame = pd.DataFrame({"lag": np.arange(max_lag + 1),
                          "contributing_lineages": contributing,
                          "observed_birth_acf": observed,
                          "replay_median_birth_acf": median,
                          "replay_q2.5_birth_acf": lower,
                          "replay_q97.5_birth_acf": upper})
    frame.to_csv(output_dir / "posterior_replay_birth_acf.csv", index=False)

    reference = {}
    for name, folder in [
        ("lineage_u", "hierarchical_lineage_u_endpoint_residual"),
        ("lineage_omega2", "hierarchical_lineage_omega2_endpoint_residual"),
    ]:
        table = pd.read_csv(PROJECT_DIR / "outputs" / "model_comparison" / "Glu" / folder / "posterior_acf_comparison.csv")
        reference[name] = table.replay_median_length_birth.to_numpy(float)[:max_lag + 1]
    common_curve = pd.read_csv(common_dir / "acf_measured_inputs_validation.csv")
    common_column = "measured_inputs_median_length_birth"
    reference["common"] = common_curve[common_column].to_numpy(float)[:max_lag + 1]

    lags = np.arange(max_lag + 1)
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.plot(lags, observed, color="black", lw=2.2, label="Observed")
    axis.plot(lags, reference["common"], color="0.55", lw=1.4, label="Common parameters")
    axis.plot(lags, reference["lineage_u"], color="tab:orange", lw=1.4, label=r"Lineage $u_\ell$")
    axis.plot(lags, reference["lineage_omega2"], color="tab:green", lw=1.4, label=r"Lineage $\omega_{2,\ell}$")
    axis.fill_between(lags, lower, upper, color="tab:blue", alpha=0.18)
    axis.plot(lags, median, color="tab:blue", lw=2.0, label=r"Lineage $u_\ell$ + $\omega_{2,\ell}$")
    axis.axhline(0, color="0.65", lw=0.8)
    axis.set(xlabel="Lag (generations)", ylabel="Mean lineage birth-size ACF", xlim=(0, max_lag))
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_replay_birth_acf_comparison.png", dpi=200)
    fig.savefig(output_dir / "posterior_replay_birth_acf_comparison.pdf")
    plt.close(fig)

    valid_lags = np.isfinite(observed[1:])
    metrics = {name: float(np.mean(np.abs(curve[1:][valid_lags] - observed[1:][valid_lags])))
               for name, curve in reference.items()}
    metrics["lineage_u_omega2"] = float(np.mean(np.abs(median[1:][valid_lags] - observed[1:][valid_lags])))
    metrics["covered_lags"] = int(np.sum((observed[1:] >= lower[1:]) & (observed[1:] <= upper[1:])))
    median_params = summary.set_index("parameter").loc[PARAMETER_NAMES, "median"].to_numpy(float)
    quadrature_checks = {}
    for order in [8, 12, 16, 24]:
        check_nodes, check_weights = hermite_rule(order)
        quadrature_checks[str(order)] = log_marginal_division_likelihood(
            median_params, prepared, pivot, check_nodes, check_weights
        )
    try:
        autocorr = dict(zip(PARAMETER_NAMES, map(float, sampler.get_autocorr_time(tol=0))))
    except Exception as exc:
        autocorr = {"error": str(exc)}
    u_diagnostics = json.load(open(
        PROJECT_DIR / "outputs" / "model_comparison" / "Glu"
        / "hierarchical_lineage_u_endpoint_residual" / "diagnostics.json"
    ))
    u_only_loglike = float(u_diagnostics["mle"]["log_marginal_division_likelihood"])
    refined_joint_loglike = float(quadrature_checks["24"])
    delta_bic_vs_u = float(
        np.log(len(transition)) - 2 * (refined_joint_loglike - u_only_loglike)
    )
    diagnostics = {
        "model": MODEL_NAME, "sugar": "Glu", "independent_lineage_effects": True,
        "quadrature_order_per_dimension": args.quadrature_order,
        "mle": dict(zip(PARAMETER_NAMES, map(float, mle))),
        "mle_log_marginal_division_likelihood": float(-mle_fit.fun),
        "posterior": summary.set_index("parameter").to_dict(orient="index"),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": autocorr,
        "quadrature_log_likelihood_check_at_posterior_median": quadrature_checks,
        "delta_bic_vs_lineage_u": delta_bic_vs_u,
        "acf_mae_lags_1_to_30": metrics,
        "runtime_minutes": (time.time() - started) / 60,
        "run_config": vars(args),
    }
    with (output_dir / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(summary.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
