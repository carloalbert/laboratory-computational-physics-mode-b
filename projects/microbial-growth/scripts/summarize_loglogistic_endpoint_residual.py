#!/usr/bin/env python3
"""Aggregate the completed endpoint-residual fits across all sugars."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_DIR / "outputs" / "model_comparison"
SUMMARY_DIR = PROJECT_DIR / "outputs" / "summary"
SUGARS = ["Ace", "DMa", "Glu", "Gly", "Lac", "Pyr", "Rha", "Suc", "Xyl"]
MODEL_DIRNAME = "loglogistic_endpoint_residual"
EXPECTED_FILES = {
    "acf_comparison.png",
    "acf_observation_matched.csv",
    "acf_observation_matched.png",
    "added_size_survival.png",
    "comparison_metrics.csv",
    "corner_plot.png",
    "derived_physical_posterior_summary.csv",
    "diagnostics.json",
    "endpoint_residual_diagnostics.png",
    "flat_chain.csv",
    "flat_log_probability.csv",
    "posterior_predictive_comparison.png",
    "posterior_predictive_simulation.csv",
    "posterior_summary.csv",
    "reconstructed_transitions.csv",
    "replicated_predictive_metrics.csv",
    "replicated_predictive_summary.csv",
    "trace_plots.png",
}


def model_dir(sugar: str) -> Path:
    return MODEL_ROOT / sugar / MODEL_DIRNAME


def require_complete_bundle(sugar: str) -> None:
    directory = model_dir(sugar)
    found = {path.name for path in directory.iterdir() if path.is_file()}
    missing = sorted(EXPECTED_FILES - found)
    if missing:
        raise FileNotFoundError(f"{sugar}: missing output files: {missing}")


def interval_contains(value: float, lower: float, upper: float) -> bool:
    return bool(lower <= value <= upper)


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    posterior_pieces = []
    predictive_pieces = []
    diagnostics_rows = []
    common_config = None

    for sugar in SUGARS:
        require_complete_bundle(sugar)
        directory = model_dir(sugar)
        with (directory / "diagnostics.json").open() as handle:
            diagnostics = json.load(handle)

        config = diagnostics["run_config"]
        if common_config is None:
            common_config = config
        elif config != common_config:
            raise ValueError(f"{sugar}: run configuration differs from Ace")

        posterior = pd.read_csv(directory / "posterior_summary.csv")
        posterior.insert(0, "sugar", sugar)
        posterior.insert(2, "parameter_type", "sampled")
        derived = pd.read_csv(directory / "derived_physical_posterior_summary.csv")
        derived.insert(0, "sugar", sugar)
        derived.insert(2, "parameter_type", "derived")
        posterior_pieces.extend([posterior, derived])

        predictive = pd.read_csv(directory / "replicated_predictive_summary.csv")
        predictive.insert(0, "sugar", sugar)
        predictive_pieces.append(predictive)

        tau = diagnostics["autocorr_time"]
        matched = diagnostics["observation_matched_acf_lag1"]
        time_acf = matched["time"]
        birth_acf = matched["length_birth"]
        diagnostics_rows.append(
            {
                "sugar": sugar,
                "n_observations": diagnostics["n_complete_observations"],
                "n_transitions": diagnostics["n_consecutive_transitions"],
                "n_transition_lineages": diagnostics["n_transition_lineages"],
                "birth_pivot": diagnostics["parameterization"]["birth_pivot"],
                "mean_acceptance_fraction": diagnostics["mean_acceptance_fraction"],
                "tau_min": min(tau.values()),
                "tau_max": max(tau.values()),
                "max_sampled_log_probability": diagnostics[
                    "max_sampled_log_probability"
                ],
                "runtime_minutes": diagnostics["runtime_minutes"],
                "corr_p_star_eta": diagnostics["parameterization"][
                    "posterior_corr_p_star_eta"
                ],
                "corr_derived_u_eta": diagnostics["parameterization"][
                    "posterior_corr_derived_u_eta"
                ],
                "time_acf_observed": time_acf["observed"],
                "time_acf_predictive_median": time_acf["predictive_median"],
                "time_acf_predictive_q2.5": time_acf["predictive_q2.5"],
                "time_acf_predictive_q97.5": time_acf["predictive_q97.5"],
                "time_acf_covered": interval_contains(
                    time_acf["observed"],
                    time_acf["predictive_q2.5"],
                    time_acf["predictive_q97.5"],
                ),
                "birth_acf_observed": birth_acf["observed"],
                "birth_acf_predictive_median": birth_acf["predictive_median"],
                "birth_acf_predictive_q2.5": birth_acf["predictive_q2.5"],
                "birth_acf_predictive_q97.5": birth_acf["predictive_q97.5"],
                "birth_acf_covered": interval_contains(
                    birth_acf["observed"],
                    birth_acf["predictive_q2.5"],
                    birth_acf["predictive_q97.5"],
                ),
            }
        )

    return (
        pd.concat(posterior_pieces, ignore_index=True),
        pd.concat(predictive_pieces, ignore_index=True),
        pd.DataFrame(diagnostics_rows),
        common_config,
    )


def make_parameter_plot(posterior: pd.DataFrame) -> None:
    parameters = [
        "p_star",
        "u",
        "omega2",
        "eta",
        "mu_alpha",
        "sigma_alpha",
        "mu_f",
        "sigma_f",
        "mu_epsilon",
        "sigma_epsilon",
    ]
    labels = [
        r"$p_\star$",
        r"derived $u$",
        r"$\omega_2$",
        r"$\eta$",
        r"$\mu_\alpha$",
        r"$\sigma_\alpha$",
        r"$\mu_f$",
        r"$\sigma_f$",
        r"$\mu_\epsilon$",
        r"$\sigma_\epsilon$",
    ]
    x = np.arange(len(SUGARS))
    fig, axes = plt.subplots(5, 2, figsize=(12, 15), sharex=True)
    for axis, parameter, label in zip(axes.flat, parameters, labels):
        values = (
            posterior[posterior.parameter.eq(parameter)]
            .set_index("sugar")
            .loc[SUGARS]
        )
        median = values["median"].to_numpy(float)
        lower = values["ci_2.5"].to_numpy(float)
        upper = values["ci_97.5"].to_numpy(float)
        axis.errorbar(
            x,
            median,
            yerr=np.vstack([median - lower, upper - median]),
            fmt="o",
            color="tab:blue",
            capsize=3,
        )
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[-1]:
        axis.set_xticks(x, SUGARS, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(SUMMARY_DIR / "sugar_parameter_scatter.png", dpi=180)
    plt.close(fig)


def make_acf_plot(diagnostics: pd.DataFrame) -> None:
    x = np.arange(len(SUGARS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for axis, prefix, title in [
        (axes[0], "time", "Division-time ACF, lag 1"),
        (axes[1], "birth", "Birth-length ACF, lag 1"),
    ]:
        observed = diagnostics[f"{prefix}_acf_observed"].to_numpy(float)
        median = diagnostics[f"{prefix}_acf_predictive_median"].to_numpy(float)
        lower = diagnostics[f"{prefix}_acf_predictive_q2.5"].to_numpy(float)
        upper = diagnostics[f"{prefix}_acf_predictive_q97.5"].to_numpy(float)
        axis.vlines(x, lower, upper, color="tab:orange", lw=5, alpha=0.45)
        axis.scatter(x, median, color="tab:orange", marker="_", s=150,
                     label="Matched-length predictive median and 95% interval")
        axis.scatter(x, observed, color="tab:blue", zorder=3, label="Observed")
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_title(title)
        axis.set_ylabel("Mean sample ACF")
        axis.set_xticks(x, SUGARS, rotation=45, ha="right")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(SUMMARY_DIR / "all_sugars_acf_lag1.png", dpi=180)
    plt.close(fig)


def write_acf_pairing_comparison() -> None:
    rows = []
    for sugar in SUGARS:
        directory = model_dir(sugar)
        randomized = pd.read_csv(
            directory / "acf_observation_matched_randomized_b0.csv"
        )
        paired = pd.read_csv(directory / "acf_observation_matched.csv")
        for variable in ["time", "length_birth"]:
            old = randomized.loc[randomized.lag.eq(1)].iloc[0]
            new = paired.loc[paired.lag.eq(1)].iloc[0]
            old_median = float(old[f"predictive_median_{variable}"])
            new_median = float(new[f"predictive_median_{variable}"])
            rows.append(
                {
                    "sugar": sugar,
                    "variable": variable,
                    "observed_lag1": float(new[f"observed_{variable}"]),
                    "randomized_b0_predictive_median": old_median,
                    "exact_pair_predictive_median": new_median,
                    "median_change_exact_minus_randomized": (
                        new_median - old_median
                    ),
                    "randomized_b0_q2.5": float(
                        old[f"predictive_q2.5_{variable}"]
                    ),
                    "randomized_b0_q97.5": float(
                        old[f"predictive_q97.5_{variable}"]
                    ),
                    "exact_pair_q2.5": float(
                        new[f"predictive_q2.5_{variable}"]
                    ),
                    "exact_pair_q97.5": float(
                        new[f"predictive_q97.5_{variable}"]
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(
        SUMMARY_DIR / "all_sugars_acf_pairing_comparison.csv", index=False
    )


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    posterior, predictive, diagnostics, run_config = collect()
    posterior.to_csv(SUMMARY_DIR / "all_sugars_posterior_summary.csv", index=False)
    predictive.to_csv(SUMMARY_DIR / "all_sugars_predictive_summary.csv", index=False)
    diagnostics.to_csv(SUMMARY_DIR / "all_sugars_diagnostics.csv", index=False)

    modes = posterior.pivot(index="sugar", columns="parameter", values="mode")
    modes = modes.reindex(SUGARS).reset_index()
    modes.to_csv(SUMMARY_DIR / "all_sugars_parameter_modes.csv", index=False)

    with (SUMMARY_DIR / "run_config.json").open("w") as handle:
        json.dump(
            {
                "model": "loglogistic_mixed_adder_sizer_with_endpoint_residual",
                "sampled_parameters": [
                    "p_star",
                    "omega2",
                    "eta",
                    "mu_alpha",
                    "sigma_alpha",
                    "mu_f",
                    "sigma_f",
                    "mu_epsilon",
                    "sigma_epsilon",
                ],
                "derived_parameters": ["u"],
                "sugars": SUGARS,
                "single_sugar_run_config": run_config,
                "expected_files_per_sugar": sorted(EXPECTED_FILES),
            },
            handle,
            indent=2,
        )

    make_parameter_plot(posterior)
    make_acf_plot(diagnostics)
    write_acf_pairing_comparison()
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
