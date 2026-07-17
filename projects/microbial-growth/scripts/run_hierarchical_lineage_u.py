#!/usr/bin/env python3
"""Fit the Glu endpoint-residual model with one persistent scale per lineage.

Only the division block changes relative to ``run_loglogistic_endpoint_residual``:

    log(u_l) ~ Normal(log(u_bar), sigma_log_u**2).

The lineage effects are integrated by Gauss-Hermite quadrature.  The growth,
division-fraction and endpoint-residual blocks factorize from the division
block, so their already-sampled Glu posterior is reused exactly rather than
needlessly resampled.  Posterior diagnostics condition on each observed
lineage's inferred ``u_l`` and preserve its measured initial size and length.
"""

from __future__ import annotations

import argparse
import json
import math
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
from scipy.stats import ks_2samp

try:
    import corner
except ImportError:
    corner = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from hierarchical_lineage_u_endpoint_residual import (  # noqa: E402
    HIERARCHICAL_PARAMETER_NAMES,
    common_u_log_likelihood,
    hermite_rule,
    log_marginal_division_likelihood,
    population_scale,
    prepare_lineage_transitions,
    sample_conditional_lineage_scales,
    sample_population_lineage_scales,
)
from loglogistic_endpoint_residual import (  # noqa: E402
    EPS,
    RESIDUAL_DF,
    sample_added_size,
    sample_valid_residual,
)
from run_loglogistic_endpoint_residual import (  # noqa: E402
    acf_1d,
    load_endpoint_data,
    mean_acf,
    metric_row,
)
from weibull_adder_sizer import beta_ab, gamma_shape_scale  # noqa: E402


MODEL_NAME = "hierarchical_lineage_u_endpoint_residual"
NUISANCE_NAMES = [
    "mu_alpha",
    "sigma_alpha",
    "mu_f",
    "sigma_f",
    "mu_epsilon",
    "sigma_epsilon",
]
ALL_PARAMETER_NAMES = HIERARCHICAL_PARAMETER_NAMES + NUISANCE_NAMES
CORNER_LABELS = [r"$p_\star$", r"$\omega_2$", r"$\eta$", r"$\sigma_{\log u}$"]


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
    parser.add_argument("--seed", type=int, default=4207)
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
        "u_bar": u_bounds,
        "omega2": (0.05, 30.0),
        "eta": eta_bounds,
        "sigma_log_u": (1e-5, 1.25),
    }


def valid_hierarchy(
    params: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    birth_pivot: float,
) -> bool:
    if len(params) != 4:
        return False
    for value, name in zip(params, HIERARCHICAL_PARAMETER_NAMES):
        low, high = bounds[name]
        if not (low < value < high):
            return False
    u_bar = population_scale(params, birth_pivot)
    return bounds["u_bar"][0] < u_bar < bounds["u_bar"][1]


def fit_hierarchy_mle(
    common_center: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    birth_pivot: float,
    bounds: dict[str, tuple[float, float]],
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    scipy_bounds = [bounds[name] for name in HIERARCHICAL_PARAMETER_NAMES]

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
        fit = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 2e-7, "fatol": 2e-5},
        )
        polished = minimize(
            objective,
            fit.x,
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7},
        )
        fits.append(polished if polished.fun <= fit.fun else fit)
    best = min(fits, key=lambda item: item.fun)
    if not valid_hierarchy(best.x, bounds, birth_pivot):
        raise RuntimeError(f"Invalid hierarchical optimum: {best.x}")
    return np.asarray(best.x), {
        "success": bool(best.success),
        "message": str(best.message),
        "log_marginal_division_likelihood": float(-best.fun),
        "starts": 5,
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
    scales = np.maximum(np.abs(center) * 0.018, [2e-3, 0.03, 2e-3, 2e-3])
    walkers = []
    for _ in range(200_000):
        proposal = center + rng.normal(0.0, scales)
        if valid_hierarchy(proposal, bounds, birth_pivot) and np.isfinite(
            log_probability(proposal)
        ):
            walkers.append(proposal)
            if len(walkers) == n_walkers:
                return np.asarray(walkers)
    raise RuntimeError("Unable to initialize hierarchical MCMC walkers")


def summarize_chain(
    chain: np.ndarray, log_probability: np.ndarray, names: list[str]
) -> pd.DataFrame:
    mode = chain[int(np.argmax(log_probability))]
    rows = []
    for index, name in enumerate(names):
        values = chain[:, index]
        rows.append(
            {
                "parameter": name,
                "mode": mode[index],
                "mean": np.mean(values),
                "median": np.median(values),
                "ci_2.5": np.percentile(values, 2.5),
                "ci_97.5": np.percentile(values, 97.5),
                "std": np.std(values, ddof=1),
            }
        )
    return pd.DataFrame(rows)


def lineage_groups(frame: pd.DataFrame) -> list[pd.DataFrame]:
    return [
        group.sort_values("generation").reset_index(drop=True)
        for _, group in frame.groupby("lineage", sort=False)
    ]


def draw_scales_for_observed_groups(
    hierarchy_params: np.ndarray,
    prepared: dict[str, np.ndarray | int],
    groups: list[pd.DataFrame],
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    transition_scales = sample_conditional_lineage_scales(
        hierarchy_params,
        prepared,
        birth_pivot,
        z_nodes,
        log_weights,
        rng,
    )
    labels = np.asarray(prepared["lineage_labels"])
    lookup = {label: scale for label, scale in zip(labels, transition_scales)}
    u_bar = population_scale(hierarchy_params, birth_pivot)
    sigma = float(hierarchy_params[3])
    result = np.empty(len(groups), dtype=float)
    for index, group in enumerate(groups):
        label = group.lineage.iloc[0]
        result[index] = lookup.get(
            label, sample_population_lineage_scales(u_bar, sigma, 1, rng)[0]
        )
    return result


def simulate_groups(
    groups: list[pd.DataFrame],
    lineage_u: np.ndarray,
    lineage_omega2: np.ndarray,
    nuisance: np.ndarray,
    eta: float,
    rng: np.random.Generator,
    replay_inputs: bool,
) -> pd.DataFrame:
    mu_alpha, sigma_alpha, mu_f, sigma_f, mu_epsilon, sigma_epsilon = map(
        float, nuisance
    )
    gamma_shape, gamma_scale = gamma_shape_scale(mu_alpha, sigma_alpha)
    beta_a, beta_b = beta_ab(mu_f, sigma_f)
    if gamma_shape is None or beta_a is None:
        raise ValueError("Invalid nuisance parameters")
    rows = []
    for lineage_index, (group, u_lineage, omega2_lineage) in enumerate(
        zip(groups, lineage_u, lineage_omega2)
    ):
        n_observations = len(group)
        birth = float(group.length_birth.iloc[0])
        current_fraction = (
            float(group.f.iloc[0])
            if replay_inputs
            else float(rng.beta(beta_a, beta_b))
        )
        for generation in range(n_observations):
            alpha = (
                float(group.alpha.iloc[generation])
                if replay_inputs
                else float(rng.gamma(gamma_shape, gamma_scale))
            )
            p = sample_added_size(
                birth, u_lineage, omega2_lineage, eta, rng
            )
            final = birth + p
            log_growth = math.log1p(p / birth)
            epsilon = sample_valid_residual(
                log_growth, mu_epsilon, sigma_epsilon, rng
            )
            division_time = (log_growth - epsilon) / alpha
            if generation + 1 < n_observations:
                next_fraction = (
                    float(group.f.iloc[generation + 1])
                    if replay_inputs
                    else float(rng.beta(beta_a, beta_b))
                )
            else:
                next_fraction = current_fraction
            rows.append(
                {
                    "lineage": lineage_index,
                    "generation": generation,
                    "time": division_time,
                    "alpha": alpha,
                    "f": current_fraction,
                    "next_f": next_fraction,
                    "length_birth": birth,
                    "length_final": final,
                    "added_size": p,
                    "epsilon": epsilon,
                    "u_lineage": u_lineage,
                    "omega2_lineage": omega2_lineage,
                }
            )
            if generation + 1 < n_observations:
                birth = max(next_fraction * final, EPS)
                current_fraction = next_fraction
    return pd.DataFrame(rows)


def summarize_curves(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmedian(values, axis=0),
        np.nanpercentile(values, 2.5, axis=0),
        np.nanpercentile(values, 97.5, axis=0),
    )


def variable_observations(
    real: pd.DataFrame, transition: pd.DataFrame
) -> dict[str, np.ndarray]:
    return {
        "time": real.time.to_numpy(float),
        "alpha": real.alpha.to_numpy(float),
        "f": real.f.to_numpy(float),
        "length_birth": real.length_birth.to_numpy(float),
        "length_final": transition.length_final.to_numpy(float),
        "added_size": transition.added_size.to_numpy(float),
        "epsilon": transition.epsilon.to_numpy(float),
    }


def run_posterior_checks(
    args: argparse.Namespace,
    output_dir: Path,
    real: pd.DataFrame,
    transition: pd.DataFrame,
    prepared: dict[str, np.ndarray | int],
    hierarchy_chain: np.ndarray,
    nuisance_chain: np.ndarray,
    birth_pivot: float,
    z_nodes: np.ndarray,
    log_weights: np.ndarray,
    draw_lineage_effects=draw_scales_for_observed_groups,
    effect_kind: str = "u",
    model_name: str = MODEL_NAME,
    model_display: str = "lineage-specific division scale",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(args.seed + 20_000)
    groups = lineage_groups(real)
    design = np.asarray([len(group) for group in groups], dtype=int)
    max_lag = min(args.acf_max_lag, int(design.max()) - 1)
    observed_acf = {
        variable: mean_acf(real, variable, max_lag)
        for variable in ["time", "length_birth"]
    }
    acf_samples = {
        mode: {
            variable: np.full((args.acf_replicates, max_lag + 1), np.nan)
            for variable in ["time", "length_birth"]
        }
        for mode in ["iid", "replay"]
    }

    for replicate in range(args.acf_replicates):
        hierarchy = hierarchy_chain[rng.integers(len(hierarchy_chain))]
        nuisance = nuisance_chain[rng.integers(len(nuisance_chain))]
        effects = draw_lineage_effects(
            hierarchy,
            prepared,
            groups,
            birth_pivot,
            z_nodes,
            log_weights,
            rng,
        )
        if effect_kind == "u":
            lineage_u = effects
            lineage_omega2 = np.full(len(groups), float(hierarchy[1]))
        elif effect_kind == "omega2":
            lineage_u = np.full(
                len(groups), population_scale(hierarchy, birth_pivot)
            )
            lineage_omega2 = effects
        else:
            raise ValueError(f"Unknown lineage effect {effect_kind!r}")
        for mode, replay in [("iid", False), ("replay", True)]:
            simulated = simulate_groups(
                groups,
                lineage_u,
                lineage_omega2,
                nuisance,
                float(hierarchy[2]),
                rng,
                replay,
            )
            for variable in ["time", "length_birth"]:
                acf_samples[mode][variable][replicate] = mean_acf(
                    simulated, variable, max_lag
                )

    common_path = (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / "loglogistic_endpoint_residual"
        / "acf_observation_matched.csv"
    )
    common = pd.read_csv(common_path)
    common = common[common.lag.le(max_lag)].reset_index(drop=True)
    replay_common_path = common_path.with_name("acf_measured_inputs_validation.csv")
    replay_common = pd.read_csv(replay_common_path)
    replay_common = replay_common[replay_common.lag.le(max_lag)].reset_index(drop=True)
    lags = np.arange(max_lag + 1)
    acf_table: dict[str, np.ndarray] = {
        "lag": lags,
        "contributing_lineages": np.asarray(
            [np.sum((design >= 3) & (design > lag)) for lag in lags]
        ),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True, sharey=True)
    for row, mode in enumerate(["iid", "replay"]):
        for col, variable in enumerate(["time", "length_birth"]):
            axis = axes[row, col]
            median, lower, upper = summarize_curves(acf_samples[mode][variable])
            observed = observed_acf[variable]
            common_median = (
                common[f"predictive_median_{variable}"].to_numpy(float)
                if mode == "iid"
                else replay_common[f"measured_inputs_median_{variable}"].to_numpy(float)
            )
            acf_table[f"observed_{variable}"] = observed
            acf_table[f"{mode}_median_{variable}"] = median
            acf_table[f"{mode}_q2.5_{variable}"] = lower
            acf_table[f"{mode}_q97.5_{variable}"] = upper
            acf_table[f"common_u_{mode}_median_{variable}"] = common_median
            axis.fill_between(
                lags, lower, upper, color="tab:orange", alpha=0.25,
                label="Hierarchical 95% band",
            )
            axis.plot(lags, median, color="tab:orange", marker="o", ms=3,
                      label="Hierarchical median")
            axis.plot(lags, common_median, color="0.4", ls="--",
                      label="Common-u median")
            axis.plot(lags, observed, color="tab:blue", marker="o", ms=3,
                      lw=2, label="Glu data")
            axis.axhline(0.0, color="black", lw=0.7)
            axis.set_title(
                ("IID fitted inputs: " if mode == "iid" else "Measured input replay: ")
                + ("division time" if variable == "time" else "birth length")
            )
            axis.set_xlabel("Generation lag")
            axis.set_ylabel("Mean sample ACF")
            axis.set_xticks(np.arange(0, max_lag + 1, 5))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=False
    )
    fig.suptitle(f"Glu: {model_display} posterior ACF check", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "posterior_acf_comparison.png", dpi=190)
    fig.savefig(output_dir / "posterior_acf_comparison.pdf")
    plt.close(fig)
    acf_frame = pd.DataFrame(acf_table)
    acf_frame.to_csv(output_dir / "posterior_acf_comparison.csv", index=False)

    observed_values = variable_observations(real, transition)
    variables = list(observed_values)
    bins = {}
    for variable, values in observed_values.items():
        low, high = np.percentile(values[np.isfinite(values)], [0.25, 99.75])
        bins[variable] = np.linspace(low, high, 41)
    density_samples = {
        variable: np.full((args.marginal_replicates, 40), np.nan)
        for variable in variables
    }
    metric_rows = []
    saved_simulation = None
    for replicate in range(args.marginal_replicates):
        hierarchy = hierarchy_chain[rng.integers(len(hierarchy_chain))]
        nuisance = nuisance_chain[rng.integers(len(nuisance_chain))]
        effects = draw_lineage_effects(
            hierarchy, prepared, groups, birth_pivot, z_nodes, log_weights, rng
        )
        if effect_kind == "u":
            lineage_u = effects
            lineage_omega2 = np.full(len(groups), float(hierarchy[1]))
        else:
            lineage_u = np.full(
                len(groups), population_scale(hierarchy, birth_pivot)
            )
            lineage_omega2 = effects
        simulated = simulate_groups(
            groups, lineage_u, lineage_omega2, nuisance,
            float(hierarchy[2]), rng, False
        )
        if saved_simulation is None:
            saved_simulation = simulated.copy()
        for variable in variables:
            sim_values = simulated[variable].to_numpy(float)
            density_samples[variable][replicate] = np.histogram(
                sim_values, bins=bins[variable], density=True
            )[0]
            row = metric_row(
                model_name, variable, observed_values[variable], sim_values
            )
            row["replicate"] = replicate
            metric_rows.append(row)
    assert saved_simulation is not None
    saved_simulation.to_csv(
        output_dir / "posterior_predictive_simulation.csv", index=False
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "posterior_predictive_metrics.csv", index=False)
    metric_summary = (
        metrics.groupby("variable")
        .agg(
            mean_ratio_median=("mean_ratio", "median"),
            mean_ratio_q2_5=("mean_ratio", lambda x: np.percentile(x, 2.5)),
            mean_ratio_q97_5=("mean_ratio", lambda x: np.percentile(x, 97.5)),
            sd_ratio_median=("sd_ratio", "median"),
            sd_ratio_q2_5=("sd_ratio", lambda x: np.percentile(x, 2.5)),
            sd_ratio_q97_5=("sd_ratio", lambda x: np.percentile(x, 97.5)),
            ks_median=("ks_statistic", "median"),
            ks_q2_5=("ks_statistic", lambda x: np.percentile(x, 2.5)),
            ks_q97_5=("ks_statistic", lambda x: np.percentile(x, 97.5)),
        )
        .reset_index()
    )
    metric_summary.to_csv(
        output_dir / "posterior_predictive_summary.csv", index=False
    )

    labels_map = {
        "time": "Division time", "alpha": "Growth rate", "f": "Division fraction",
        "length_birth": "Birth length", "length_final": "Final length",
        "added_size": "Added length", "epsilon": r"Endpoint residual $\epsilon$",
    }
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, variable in zip(axes.flat, variables):
        edges = bins[variable]
        centers = 0.5 * (edges[:-1] + edges[1:])
        observed_density = np.histogram(
            observed_values[variable], bins=edges, density=True
        )[0]
        median, lower, upper = summarize_curves(density_samples[variable])
        axis.fill_between(centers, lower, upper, color="tab:orange", alpha=0.25,
                          label="Hierarchical 95% band")
        axis.plot(centers, median, color="tab:orange", lw=2,
                  label="Hierarchical median")
        axis.step(centers, observed_density, where="mid", color="tab:blue", lw=2,
                  label="Glu data")
        axis.set_xlabel(labels_map[variable])
        axis.set_ylabel("Density")
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Glu: {model_display} posterior marginals")
    fig.tight_layout()
    fig.savefig(output_dir / "posterior_predictive_marginals.png", dpi=190)
    fig.savefig(output_dir / "posterior_predictive_marginals.pdf")
    plt.close(fig)

    acf_metrics = {}
    for mode in ["iid", "replay"]:
        for variable in ["time", "length_birth"]:
            median = summarize_curves(acf_samples[mode][variable])[0]
            valid = np.isfinite(median[1:]) & np.isfinite(observed_acf[variable][1:])
            acf_metrics[f"{mode}_{variable}_mae_lags_1_{max_lag}"] = float(
                np.mean(np.abs(median[1:][valid] - observed_acf[variable][1:][valid]))
            )
            acf_metrics[f"{mode}_{variable}_lag1_observed"] = float(observed_acf[variable][1])
            acf_metrics[f"{mode}_{variable}_lag1_median"] = float(median[1])
            lower = summarize_curves(acf_samples[mode][variable])[1]
            upper = summarize_curves(acf_samples[mode][variable])[2]
            acf_metrics[f"{mode}_{variable}_covered_lags"] = int(
                np.sum(
                    (observed_acf[variable][1:] >= lower[1:])
                    & (observed_acf[variable][1:] <= upper[1:])
                )
            )
    return acf_frame, metric_summary, acf_metrics


def write_common_u_comparison(
    output_dir: Path,
    sugar: str,
    acf_frame: pd.DataFrame,
    marginal_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Write like-for-like scalar comparisons with the saved common-u fit."""
    common_dir = (
        PROJECT_DIR / "outputs" / "model_comparison" / sugar
        / "loglogistic_endpoint_residual"
    )
    common_marginal = pd.read_csv(
        common_dir / "replicated_predictive_summary.csv"
    ).set_index("metric")
    hierarchy_marginal = marginal_summary.set_index("variable")
    rows: list[dict[str, object]] = []
    for variable in hierarchy_marginal.index:
        old_sd = float(common_marginal.loc[f"{variable}_sd_ratio", "median"])
        new_sd = float(hierarchy_marginal.loc[variable, "sd_ratio_median"])
        rows.append(
            {
                "diagnostic": "marginal_sd_ratio",
                "variable": variable,
                "common_u": old_sd,
                "hierarchical_u": new_sd,
                "improvement": abs(old_sd - 1.0) - abs(new_sd - 1.0),
                "improvement_definition": "positive means closer to one",
            }
        )
        old_ks = float(common_marginal.loc[f"{variable}_ks", "median"])
        new_ks = float(hierarchy_marginal.loc[variable, "ks_median"])
        rows.append(
            {
                "diagnostic": "marginal_ks",
                "variable": variable,
                "common_u": old_ks,
                "hierarchical_u": new_ks,
                "improvement": old_ks - new_ks,
                "improvement_definition": "positive means smaller KS distance",
            }
        )
    for mode in ["iid", "replay"]:
        for variable in ["time", "length_birth"]:
            observed = acf_frame[f"observed_{variable}"].to_numpy(float)[1:]
            common = acf_frame[
                f"common_u_{mode}_median_{variable}"
            ].to_numpy(float)[1:]
            hierarchy = acf_frame[
                f"{mode}_median_{variable}"
            ].to_numpy(float)[1:]
            common_mae = float(np.nanmean(np.abs(observed - common)))
            hierarchy_mae = float(np.nanmean(np.abs(observed - hierarchy)))
            rows.append(
                {
                    "diagnostic": f"{mode}_acf_mae_lags_1_{len(observed)}",
                    "variable": variable,
                    "common_u": common_mae,
                    "hierarchical_u": hierarchy_mae,
                    "improvement": common_mae - hierarchy_mae,
                    "improvement_definition": "positive means smaller ACF MAE",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "common_u_comparison.csv", index=False)
    return result


def lineage_scale_summary(
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
        samples[row] = draw_scales_for_observed_groups(
            hierarchy_chain[index], prepared, groups, birth_pivot,
            z_nodes, log_weights, rng
        )
    mean_birth = np.asarray([group.length_birth.mean() for group in groups])
    labels = np.asarray([group.lineage.iloc[0] for group in groups])
    lengths = np.asarray([len(group) for group in groups])
    summary = pd.DataFrame(
        {
            "lineage": labels,
            "n_observations": lengths,
            "mean_birth_length": mean_birth,
            "u_lineage_mean": np.mean(samples, axis=0),
            "u_lineage_median": np.median(samples, axis=0),
            "u_lineage_q2.5": np.percentile(samples, 2.5, axis=0),
            "u_lineage_q97.5": np.percentile(samples, 97.5, axis=0),
        }
    )
    summary.to_csv(output_dir / "lineage_scale_posterior.csv", index=False)
    valid = lengths >= 3
    correlation = float(
        np.corrcoef(
            summary.loc[valid, "mean_birth_length"],
            summary.loc[valid, "u_lineage_median"],
        )[0, 1]
    )
    fig, axis = plt.subplots(figsize=(6.7, 5.2))
    axis.scatter(
        summary.mean_birth_length, summary.u_lineage_median,
        s=18, alpha=0.65, color="tab:blue"
    )
    axis.set_xlabel("Observed lineage mean birth length")
    axis.set_ylabel(r"Posterior median $u_\ell$")
    axis.set_title(f"Glu lineage scales (Pearson r = {correlation:.3f})")
    fig.tight_layout()
    fig.savefig(output_dir / "lineage_scale_vs_mean_birth.png", dpi=190)
    plt.close(fig)
    return summary, {"pearson_u_median_vs_mean_birth": correlation}


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
    common_logp = pd.read_csv(common_dir / "flat_log_probability.csv").iloc[:, 0].to_numpy(float)
    common_map = common_chain_frame.iloc[int(np.argmax(common_logp))]
    common_center = common_map[["p_star", "omega2", "eta"]].to_numpy(float)
    nuisance_chain = common_chain_frame[NUISANCE_NAMES].to_numpy(float)
    bounds = hierarchy_bounds(data, birth_pivot)

    mle, mle_details = fit_hierarchy_mle(
        common_center, prepared, birth_pivot, bounds, z_nodes, log_weights
    )
    print("Hierarchical MLE", dict(zip(HIERARCHICAL_PARAMETER_NAMES, mle)))
    print("Marginal division log likelihood", mle_details["log_marginal_division_likelihood"])

    def log_probability(params: np.ndarray) -> float:
        if not valid_hierarchy(params, bounds, birth_pivot):
            return -np.inf
        return log_marginal_division_likelihood(
            params, prepared, birth_pivot, z_nodes, log_weights
        )

    n_walkers = max(args.walkers, 2 * len(HIERARCHICAL_PARAMETER_NAMES) + 2)
    initial = initialize_walkers(
        mle, n_walkers, bounds, birth_pivot, log_probability, args.seed
    )
    sampler = emcee.EnsembleSampler(
        n_walkers,
        len(HIERARCHICAL_PARAMETER_NAMES),
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
        flat_chain, flat_logp, HIERARCHICAL_PARAMETER_NAMES
    )
    rng = np.random.default_rng(args.seed + 10_000)
    nuisance_indices = rng.integers(0, len(nuisance_chain), len(flat_chain))
    combined_chain = np.column_stack([flat_chain, nuisance_chain[nuisance_indices]])
    common_summary = pd.read_csv(common_dir / "posterior_summary.csv")
    nuisance_summary = common_summary[
        common_summary.parameter.isin(NUISANCE_NAMES)
    ].set_index("parameter").loc[NUISANCE_NAMES].reset_index()
    combined_summary = pd.concat(
        [hierarchy_summary, nuisance_summary], ignore_index=True
    )
    u_bar_samples = flat_chain[:, 0] + flat_chain[:, 2] * birth_pivot
    u_bar_summary = pd.DataFrame(
        [{
            "parameter": "u_bar",
            "mode": u_bar_samples[int(np.argmax(flat_logp))],
            "mean": np.mean(u_bar_samples),
            "median": np.median(u_bar_samples),
            "ci_2.5": np.percentile(u_bar_samples, 2.5),
            "ci_97.5": np.percentile(u_bar_samples, 97.5),
            "std": np.std(u_bar_samples, ddof=1),
        }]
    )
    hierarchy_summary.to_csv(output_dir / "hierarchy_posterior_summary.csv", index=False)
    combined_summary.to_csv(output_dir / "posterior_summary.csv", index=False)
    u_bar_summary.to_csv(output_dir / "derived_population_scale_summary.csv", index=False)
    pd.DataFrame(flat_chain, columns=HIERARCHICAL_PARAMETER_NAMES).to_csv(
        output_dir / "hierarchy_flat_chain.csv", index=False
    )
    pd.DataFrame(combined_chain, columns=ALL_PARAMETER_NAMES).to_csv(
        output_dir / "flat_chain.csv", index=False
    )
    pd.DataFrame({"log_probability": flat_logp}).to_csv(
        output_dir / "flat_log_probability.csv", index=False
    )
    transition.to_csv(output_dir / "reconstructed_transitions.csv", index=False)

    quadrature_checks = {}
    for order in [24, 32, 48, 64, 80]:
        check_nodes, check_weights = hermite_rule(order)
        quadrature_checks[str(order)] = log_marginal_division_likelihood(
            hierarchy_summary.set_index("parameter").loc[
                HIERARCHICAL_PARAMETER_NAMES, "median"
            ].to_numpy(float),
            prepared,
            birth_pivot,
            check_nodes,
            check_weights,
        )

    acf_frame, marginal_summary, acf_metrics = run_posterior_checks(
        args, output_dir, real, transition, prepared, flat_chain, nuisance_chain,
        birth_pivot, z_nodes, log_weights
    )
    common_u_comparison = write_common_u_comparison(
        output_dir, args.sugar, acf_frame, marginal_summary
    )
    lineage_summary, lineage_metrics = lineage_scale_summary(
        output_dir, real, prepared, flat_chain, birth_pivot,
        z_nodes, log_weights, args.seed + 40_000
    )

    if corner is not None:
        figure = corner.corner(
            flat_chain, labels=CORNER_LABELS,
            truths=hierarchy_summary.set_index("parameter").loc[
                HIERARCHICAL_PARAMETER_NAMES, "median"
            ].to_numpy(float),
            show_titles=True, title_fmt=".3g", quantiles=[0.16, 0.5, 0.84],
            max_n_ticks=3,
        )
        figure.suptitle("Glu: hierarchical lineage-scale posterior", y=1.0)
        figure.savefig(output_dir / "corner_plot.png", dpi=170, bbox_inches="tight")
        plt.close(figure)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, index, name in zip(axes.flat, range(4), HIERARCHICAL_PARAMETER_NAMES):
        axis.plot(chain[:, :, index], alpha=0.35, lw=0.5)
        axis.set_ylabel(name)
    axes[-1, 0].set_xlabel("Production step")
    axes[-1, 1].set_xlabel("Production step")
    fig.tight_layout()
    fig.savefig(output_dir / "trace_plots.png", dpi=180)
    plt.close(fig)

    common_params_at_zero = np.r_[common_center, 0.0]
    common_division_loglike = common_u_log_likelihood(
        common_params_at_zero, prepared, birth_pivot
    )
    try:
        autocorr = dict(
            zip(
                HIERARCHICAL_PARAMETER_NAMES,
                map(float, sampler.get_autocorr_time(tol=0)),
            )
        )
    except Exception as exc:
        autocorr = {"error": str(exc)}
    diagnostics = {
        "sugar": args.sugar,
        "model": MODEL_NAME,
        "lineage_scale_distribution": "log(u_l) ~ Normal(log(u_bar), sigma_log_u^2)",
        "u_bar_relation": "u_bar = p_star + eta * birth_pivot",
        "birth_pivot": birth_pivot,
        "quadrature_order": args.quadrature_order,
        "quadrature_log_likelihood_check": quadrature_checks,
        "n_observations": int(len(real)),
        "n_transitions": int(len(transition)),
        "n_observed_lineages": int(real.lineage.nunique()),
        "n_transition_lineages": int(prepared["n_lineages"]),
        "mle": mle_details,
        "common_u_map_division_log_likelihood": common_division_loglike,
        "marginal_log_likelihood_gain_at_mle": float(
            mle_details["log_marginal_division_likelihood"] - common_division_loglike
        ),
        "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "autocorr_time": autocorr,
        "acf_metrics": acf_metrics,
        "lineage_metrics": lineage_metrics,
        "u_bar_posterior": u_bar_summary.iloc[0].to_dict(),
        "sigma_log_u_posterior": hierarchy_summary.set_index("parameter").loc[
            "sigma_log_u"
        ].to_dict(),
        "run_config": vars(args) | {"output_dir": str(output_dir)},
        "runtime_minutes": (time.time() - started) / 60.0,
        "n_lineage_scale_summary_rows": int(len(lineage_summary)),
        "marginal_summary": marginal_summary.set_index("variable").to_dict(orient="index"),
        "common_u_comparison_rows": int(len(common_u_comparison)),
        "acf_table_rows": int(len(acf_frame)),
        "residual_df": RESIDUAL_DF,
    }
    with (output_dir / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2, default=float)

    print("\nHierarchy posterior")
    print(hierarchy_summary.to_string(index=False))
    print("\nPopulation scale")
    print(u_bar_summary.to_string(index=False))
    print("\nMarginal summary")
    print(marginal_summary.to_string(index=False))
    print("\nACF metrics")
    print(json.dumps(acf_metrics, indent=2))
    print("Saved to", output_dir)


if __name__ == "__main__":
    main()
