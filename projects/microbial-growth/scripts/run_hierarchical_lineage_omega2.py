#!/usr/bin/env python3
"""Fit Glu with a persistent log-logistic shape ``omega2_l`` per lineage."""

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

try:
    import corner
except ImportError:
    corner = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from hierarchical_lineage_omega2_endpoint_residual import (  # noqa: E402
    HIERARCHICAL_OMEGA_PARAMETER_NAMES,
    common_omega2_log_likelihood,
    conditional_lineage_omega2_weights,
    draw_observed_lineage_omega2,
    log_marginal_division_likelihood,
)
from hierarchical_lineage_u_endpoint_residual import (  # noqa: E402
    hermite_rule,
    population_scale,
    prepare_lineage_transitions,
)
from loglogistic_endpoint_residual import RESIDUAL_DF  # noqa: E402
from run_hierarchical_lineage_u import (  # noqa: E402
    NUISANCE_NAMES,
    lineage_groups,
    run_posterior_checks,
    summarize_chain,
    write_common_u_comparison,
)
from run_loglogistic_endpoint_residual import load_endpoint_data  # noqa: E402


MODEL_NAME = "hierarchical_lineage_omega2_endpoint_residual"
ALL_PARAMETER_NAMES = HIERARCHICAL_OMEGA_PARAMETER_NAMES + NUISANCE_NAMES
CORNER_LABELS = [
    r"$p_\star$",
    r"$\bar\omega_2$",
    r"$\eta$",
    r"$\sigma_{\log\omega_2}$",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sugar", default="Glu", choices=["Glu"])
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--burn", type=int, default=600)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--thin", type=int, default=3)
    parser.add_argument("--quadrature-order", type=int, default=24)
    parser.add_argument("--acf-replicates", type=int, default=300)
    parser.add_argument("--marginal-replicates", type=int, default=100)
    parser.add_argument("--acf-max-lag", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4217)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def hierarchy_bounds(
    data: dict[str, np.ndarray], birth_pivot: float
) -> dict[str, tuple[float, float]]:
    added = data["added_size"]
    u_bounds = (1e-5, max(20.0, 5.0 * float(np.percentile(added, 99))))
    eta_bounds = (1e-6, 1.0 - 1e-6)
    return {
        "p_star": (
            u_bounds[0] - eta_bounds[1] * birth_pivot,
            u_bounds[1] - eta_bounds[0] * birth_pivot,
        ),
        "u": u_bounds,
        "omega2": (0.05, 30.0),
        "eta": eta_bounds,
        "sigma_log_omega2": (1e-5, 1.25),
    }


def valid_hierarchy(
    params: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    birth_pivot: float,
) -> bool:
    if len(params) != 4:
        return False
    for value, name in zip(params, HIERARCHICAL_OMEGA_PARAMETER_NAMES):
        low, high = bounds[name]
        if not (low < value < high):
            return False
    u = population_scale(params, birth_pivot)
    return bounds["u"][0] < u < bounds["u"][1]


def fit_hierarchy_mle(
    common_center: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    bounds: dict[str, tuple[float, float]],
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    scipy_bounds = [
        bounds[name] for name in HIERARCHICAL_OMEGA_PARAMETER_NAMES
    ]

    def objective(params: np.ndarray) -> float:
        if not valid_hierarchy(params, bounds, birth_pivot):
            return 1e100
        value = log_marginal_division_likelihood(
            params, prepared, birth_pivot, z_nodes, log_weights
        )
        return -value if np.isfinite(value) else 1e100

    fits = []
    for sigma_start in [0.025, 0.08, 0.16, 0.30, 0.55]:
        start = np.r_[common_center[:3], sigma_start]
        rough = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 2e-7, "fatol": 2e-5},
        )
        polished = minimize(
            objective,
            rough.x,
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7},
        )
        fits.append(polished if polished.fun <= rough.fun else rough)
    best = min(fits, key=lambda item: item.fun)
    if not valid_hierarchy(best.x, bounds, birth_pivot):
        raise RuntimeError(f"Invalid hierarchical optimum: {best.x}")
    return np.asarray(best.x), {
        "success": bool(best.success),
        "message": str(best.message),
        "log_marginal_division_likelihood": float(-best.fun),
        "starts": len(fits),
    }


def initialize_walkers(
    center: np.ndarray,
    n_walkers: int,
    bounds: dict[str, tuple[float, float]],
    birth_pivot: float,
    log_probability,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = np.maximum(
        np.abs(center) * 0.018, [2e-3, 0.03, 2e-3, 2e-3]
    )
    walkers = []
    for _ in range(200_000):
        proposal = center + rng.normal(0.0, scales)
        if valid_hierarchy(proposal, bounds, birth_pivot) and np.isfinite(
            log_probability(proposal)
        ):
            walkers.append(proposal)
            if len(walkers) == n_walkers:
                return np.asarray(walkers)
    raise RuntimeError("Unable to initialize omega2 hierarchy walkers")


def lineage_omega2_summary(
    output_dir: Path,
    real: pd.DataFrame,
    prepared: dict[str, np.ndarray | int],
    hierarchy_chain: np.ndarray,
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    groups = lineage_groups(real)
    n_draws = min(1000, len(hierarchy_chain))
    indices = rng.choice(len(hierarchy_chain), n_draws, replace=False)
    samples = np.empty((n_draws, len(groups)))
    for row, index in enumerate(indices):
        samples[row] = draw_observed_lineage_omega2(
            hierarchy_chain[index],
            prepared,
            groups,
            birth_pivot,
            z_nodes,
            log_weights,
            rng,
        )
    labels = np.asarray([group.lineage.iloc[0] for group in groups])
    lengths = np.asarray([len(group) for group in groups])
    mean_birth = np.asarray([group.length_birth.mean() for group in groups])
    sd_birth = np.asarray(
        [group.length_birth.std(ddof=1) if len(group) > 1 else np.nan for group in groups]
    )
    summary = pd.DataFrame(
        {
            "lineage": labels,
            "n_observations": lengths,
            "mean_birth_length": mean_birth,
            "sd_birth_length": sd_birth,
            "omega2_lineage_mean": np.mean(samples, axis=0),
            "omega2_lineage_median": np.median(samples, axis=0),
            "omega2_lineage_q2.5": np.percentile(samples, 2.5, axis=0),
            "omega2_lineage_q97.5": np.percentile(samples, 97.5, axis=0),
        }
    )
    summary.to_csv(output_dir / "lineage_omega2_posterior.csv", index=False)
    valid = (lengths >= 3) & np.isfinite(sd_birth)
    corr_mean = float(
        np.corrcoef(
            summary.loc[valid, "mean_birth_length"],
            summary.loc[valid, "omega2_lineage_median"],
        )[0, 1]
    )
    corr_sd = float(
        np.corrcoef(
            summary.loc[valid, "sd_birth_length"],
            summary.loc[valid, "omega2_lineage_median"],
        )[0, 1]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(
        summary.mean_birth_length,
        summary.omega2_lineage_median,
        s=18,
        alpha=0.65,
    )
    axes[0].set_xlabel("Observed lineage mean birth length")
    axes[0].set_ylabel(r"Posterior median $\omega_{2,\ell}$")
    axes[0].set_title(f"Mean size (r = {corr_mean:.3f})")
    axes[1].scatter(
        summary.sd_birth_length,
        summary.omega2_lineage_median,
        s=18,
        alpha=0.65,
    )
    axes[1].set_xlabel("Observed lineage birth-length SD")
    axes[1].set_ylabel(r"Posterior median $\omega_{2,\ell}$")
    axes[1].set_title(f"Within-lineage spread (r = {corr_sd:.3f})")
    fig.suptitle("Glu inferred lineage-specific hazard shapes")
    fig.tight_layout()
    fig.savefig(output_dir / "lineage_omega2_relationships.png", dpi=190)
    plt.close(fig)
    return summary, {
        "pearson_omega2_median_vs_mean_birth": corr_mean,
        "pearson_omega2_median_vs_birth_sd": corr_sd,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        PROJECT_DIR / "outputs" / "model_comparison" / args.sugar / MODEL_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    real, transition, data, _ = load_endpoint_data(args.sugar)
    birth_pivot = float(np.mean(data["transition_birth"]))
    prepared = prepare_lineage_transitions(
        transition.length_birth.to_numpy(float),
        transition.added_size.to_numpy(float),
        transition.lineage.to_numpy(),
    )
    z_nodes, log_weights = hermite_rule(args.quadrature_order)
    common_dir = (
        PROJECT_DIR / "outputs" / "model_comparison" / args.sugar
        / "loglogistic_endpoint_residual"
    )
    common_chain_frame = pd.read_csv(common_dir / "flat_chain.csv")
    common_logp = pd.read_csv(
        common_dir / "flat_log_probability.csv"
    ).iloc[:, 0].to_numpy(float)
    common_map = common_chain_frame.iloc[int(np.argmax(common_logp))]
    common_center = common_map[["p_star", "omega2", "eta"]].to_numpy(float)
    nuisance_chain = common_chain_frame[NUISANCE_NAMES].to_numpy(float)
    bounds = hierarchy_bounds(data, birth_pivot)

    mle, mle_details = fit_hierarchy_mle(
        common_center, prepared, birth_pivot, bounds, z_nodes, log_weights
    )
    print("Hierarchical omega2 MLE", dict(zip(HIERARCHICAL_OMEGA_PARAMETER_NAMES, mle)))
    print("Marginal division log likelihood", mle_details["log_marginal_division_likelihood"])

    def log_probability(params: np.ndarray) -> float:
        if not valid_hierarchy(params, bounds, birth_pivot):
            return -np.inf
        return log_marginal_division_likelihood(
            params, prepared, birth_pivot, z_nodes, log_weights
        )

    n_walkers = max(args.walkers, 10)
    initial = initialize_walkers(
        mle, n_walkers, bounds, birth_pivot, log_probability, args.seed
    )
    sampler = emcee.EnsembleSampler(
        n_walkers,
        4,
        log_probability,
        moves=[
            (emcee.moves.StretchMove(), 0.5),
            (emcee.moves.DEMove(), 0.35),
            (emcee.moves.DESnookerMove(), 0.15),
        ],
    )
    print(f"Burn-in: {args.burn} steps with {n_walkers} walkers")
    state = sampler.run_mcmc(initial, args.burn, progress=True)
    sampler.reset()
    print(f"Production: {args.steps} steps")
    sampler.run_mcmc(state, args.steps, progress=True)
    chain = sampler.get_chain()
    flat_chain = sampler.get_chain(thin=args.thin, flat=True)
    flat_logp = sampler.get_log_prob(thin=args.thin, flat=True)

    hierarchy_summary = summarize_chain(
        flat_chain, flat_logp, HIERARCHICAL_OMEGA_PARAMETER_NAMES
    )
    rng = np.random.default_rng(args.seed + 10_000)
    nuisance_indices = rng.integers(0, len(nuisance_chain), len(flat_chain))
    combined_chain = np.column_stack(
        [flat_chain, nuisance_chain[nuisance_indices]]
    )
    common_summary = pd.read_csv(common_dir / "posterior_summary.csv")
    nuisance_summary = common_summary[
        common_summary.parameter.isin(NUISANCE_NAMES)
    ].set_index("parameter").loc[NUISANCE_NAMES].reset_index()
    combined_summary = pd.concat(
        [hierarchy_summary, nuisance_summary], ignore_index=True
    )
    u_samples = flat_chain[:, 0] + flat_chain[:, 2] * birth_pivot
    u_summary = pd.DataFrame(
        [{
            "parameter": "u",
            "mode": u_samples[int(np.argmax(flat_logp))],
            "mean": np.mean(u_samples),
            "median": np.median(u_samples),
            "ci_2.5": np.percentile(u_samples, 2.5),
            "ci_97.5": np.percentile(u_samples, 97.5),
            "std": np.std(u_samples, ddof=1),
        }]
    )
    hierarchy_summary.to_csv(
        output_dir / "hierarchy_posterior_summary.csv", index=False
    )
    combined_summary.to_csv(output_dir / "posterior_summary.csv", index=False)
    u_summary.to_csv(output_dir / "derived_global_scale_summary.csv", index=False)
    pd.DataFrame(
        flat_chain, columns=HIERARCHICAL_OMEGA_PARAMETER_NAMES
    ).to_csv(output_dir / "hierarchy_flat_chain.csv", index=False)
    pd.DataFrame(combined_chain, columns=ALL_PARAMETER_NAMES).to_csv(
        output_dir / "flat_chain.csv", index=False
    )
    pd.DataFrame({"log_probability": flat_logp}).to_csv(
        output_dir / "flat_log_probability.csv", index=False
    )
    transition.to_csv(output_dir / "reconstructed_transitions.csv", index=False)

    median_params = hierarchy_summary.set_index("parameter").loc[
        HIERARCHICAL_OMEGA_PARAMETER_NAMES, "median"
    ].to_numpy(float)
    quadrature_checks = {}
    for order in [16, 24, 32, 48, 64, 80]:
        check_nodes, check_weights = hermite_rule(order)
        quadrature_checks[str(order)] = log_marginal_division_likelihood(
            median_params,
            prepared,
            birth_pivot,
            check_nodes,
            check_weights,
        )

    acf_frame, marginal_summary, acf_metrics = run_posterior_checks(
        args,
        output_dir,
        real,
        transition,
        prepared,
        flat_chain,
        nuisance_chain,
        birth_pivot,
        z_nodes,
        log_weights,
        draw_lineage_effects=draw_observed_lineage_omega2,
        effect_kind="omega2",
        model_name=MODEL_NAME,
        model_display=r"lineage-specific $\omega_{2,\ell}$",
    )
    comparison = write_common_u_comparison(
        output_dir, args.sugar, acf_frame, marginal_summary
    )
    lineage_summary, lineage_metrics = lineage_omega2_summary(
        output_dir,
        real,
        prepared,
        flat_chain,
        birth_pivot,
        z_nodes,
        log_weights,
        args.seed + 40_000,
    )

    if corner is not None:
        figure = corner.corner(
            flat_chain,
            labels=CORNER_LABELS,
            truths=median_params,
            show_titles=True,
            title_fmt=".3g",
            quantiles=[0.16, 0.5, 0.84],
            max_n_ticks=3,
        )
        figure.suptitle("Glu: hierarchical lineage-shape posterior", y=1.0)
        figure.savefig(
            output_dir / "corner_plot.png", dpi=170, bbox_inches="tight"
        )
        plt.close(figure)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, index, name in zip(
        axes.flat, range(4), HIERARCHICAL_OMEGA_PARAMETER_NAMES
    ):
        axis.plot(chain[:, :, index], alpha=0.35, lw=0.5)
        axis.set_ylabel(name)
    axes[-1, 0].set_xlabel("Production step")
    axes[-1, 1].set_xlabel("Production step")
    fig.tight_layout()
    fig.savefig(output_dir / "trace_plots.png", dpi=180)
    plt.close(fig)

    common_params_at_zero = np.r_[common_center, 0.0]
    common_division_loglike = common_omega2_log_likelihood(
        common_params_at_zero, prepared, birth_pivot
    )
    try:
        autocorr = dict(
            zip(
                HIERARCHICAL_OMEGA_PARAMETER_NAMES,
                map(float, sampler.get_autocorr_time(tol=0)),
            )
        )
    except Exception as exc:
        autocorr = {"error": str(exc)}
    diagnostics = {
        "sugar": args.sugar,
        "model": MODEL_NAME,
        "lineage_shape_distribution": (
            "log(omega2_l) ~ Normal(log(omega2_bar), sigma_log_omega2^2)"
        ),
        "global_scale_relation": "u = p_star + eta * birth_pivot",
        "birth_pivot": birth_pivot,
        "quadrature_order": args.quadrature_order,
        "quadrature_log_likelihood_check": quadrature_checks,
        "n_observations": int(len(real)),
        "n_transitions": int(len(transition)),
        "n_observed_lineages": int(real.lineage.nunique()),
        "n_transition_lineages": int(prepared["n_lineages"]),
        "mle": mle_details,
        "common_omega2_map_division_log_likelihood": common_division_loglike,
        "marginal_log_likelihood_gain_at_mle": float(
            mle_details["log_marginal_division_likelihood"]
            - common_division_loglike
        ),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": autocorr,
        "acf_metrics": acf_metrics,
        "lineage_metrics": lineage_metrics,
        "u_posterior": u_summary.iloc[0].to_dict(),
        "sigma_log_omega2_posterior": hierarchy_summary.set_index(
            "parameter"
        ).loc["sigma_log_omega2"].to_dict(),
        "run_config": vars(args) | {"output_dir": str(output_dir)},
        "runtime_minutes": (time.time() - started) / 60.0,
        "n_lineage_summary_rows": int(len(lineage_summary)),
        "common_u_comparison_rows": int(len(comparison)),
        "marginal_summary": marginal_summary.set_index("variable").to_dict(
            orient="index"
        ),
        "acf_table_rows": int(len(acf_frame)),
        "residual_df": RESIDUAL_DF,
    }
    with (output_dir / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2, default=float)

    print("\nHierarchy posterior")
    print(hierarchy_summary.to_string(index=False))
    print("\nGlobal size scale")
    print(u_summary.to_string(index=False))
    print("\nMarginal summary")
    print(marginal_summary.to_string(index=False))
    print("\nACF metrics")
    print(json.dumps(acf_metrics, indent=2))
    print("Saved to", output_dir)


if __name__ == "__main__":
    main()
