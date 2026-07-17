#!/usr/bin/env python3
"""Compare common, lineage-u and lineage-omega2 Glu fits."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE = PROJECT_DIR / "outputs" / "model_comparison" / "Glu"
COMMON = BASE / "loglogistic_endpoint_residual"
LINEAGE_U = BASE / "hierarchical_lineage_u_endpoint_residual"
LINEAGE_OMEGA = BASE / "hierarchical_lineage_omega2_endpoint_residual"


def marginal_value(directory: Path, variable: str, column: str) -> float:
    table = pd.read_csv(directory / "posterior_predictive_summary.csv").set_index(
        "variable"
    )
    return float(table.loc[variable, column])


def main() -> None:
    with (COMMON / "diagnostics.json").open() as handle:
        common_diag = json.load(handle)
    with (LINEAGE_U / "diagnostics.json").open() as handle:
        u_diag = json.load(handle)
    with (LINEAGE_OMEGA / "diagnostics.json").open() as handle:
        omega_diag = json.load(handle)

    common_pred = pd.read_csv(COMMON / "replicated_predictive_summary.csv").set_index(
        "metric"
    )
    common_posterior = pd.read_csv(COMMON / "posterior_summary.csv").set_index(
        "parameter"
    )
    common_iid = pd.read_csv(COMMON / "acf_observation_matched.csv")
    common_replay = pd.read_csv(COMMON / "acf_measured_inputs_validation.csv")
    observed_birth_acf = common_iid.observed_length_birth.to_numpy(float)[1:]
    common_iid_mae = float(
        np.mean(
            np.abs(
                observed_birth_acf
                - common_iid.predictive_median_length_birth.to_numpy(float)[1:]
            )
        )
    )
    common_replay_mae = float(
        np.mean(
            np.abs(
                observed_birth_acf
                - common_replay.measured_inputs_median_length_birth.to_numpy(float)[1:]
            )
        )
    )
    n_transitions = int(common_diag["n_consecutive_transitions"])
    rows = [
        {
            "model": "common",
            "log_likelihood_gain": 0.0,
            "delta_bic_vs_common": 0.0,
            "heterogeneity_parameter": "none",
            "heterogeneity_median": 0.0,
            "heterogeneity_ci_2.5": 0.0,
            "heterogeneity_ci_97.5": 0.0,
            "population_cv": 0.0,
            "u_median": common_diag["parameterization"]["derived_physical_u"]["median"],
            "omega2_median": float(common_posterior.loc["omega2", "median"]),
            "eta_median": float(common_posterior.loc["eta", "median"]),
            "birth_ks": float(common_pred.loc["length_birth_ks", "median"]),
            "birth_sd_ratio": float(
                common_pred.loc["length_birth_sd_ratio", "median"]
            ),
            "iid_birth_acf_mae": common_iid_mae,
            "replay_birth_acf_mae": common_replay_mae,
        }
    ]
    for name, directory, diagnostics, sigma_name in [
        ("lineage_u", LINEAGE_U, u_diag, "sigma_log_u"),
        (
            "lineage_omega2",
            LINEAGE_OMEGA,
            omega_diag,
            "sigma_log_omega2",
        ),
    ]:
        posterior = pd.read_csv(directory / "hierarchy_posterior_summary.csv").set_index(
            "parameter"
        )
        sigma = posterior.loc[sigma_name]
        gain = float(diagnostics["marginal_log_likelihood_gain_at_mle"])
        u_summary_name = (
            "derived_population_scale_summary.csv"
            if name == "lineage_u"
            else "derived_global_scale_summary.csv"
        )
        u_median = float(pd.read_csv(directory / u_summary_name).iloc[0]["median"])
        rows.append(
            {
                "model": name,
                "log_likelihood_gain": gain,
                "delta_bic_vs_common": math.log(n_transitions) - 2.0 * gain,
                "heterogeneity_parameter": sigma_name,
                "heterogeneity_median": float(sigma["median"]),
                "heterogeneity_ci_2.5": float(sigma["ci_2.5"]),
                "heterogeneity_ci_97.5": float(sigma["ci_97.5"]),
                "population_cv": math.sqrt(
                    math.exp(float(sigma["median"]) ** 2) - 1.0
                ),
                "u_median": u_median,
                "omega2_median": float(posterior.loc["omega2", "median"]),
                "eta_median": float(posterior.loc["eta", "median"]),
                "birth_ks": marginal_value(directory, "length_birth", "ks_median"),
                "birth_sd_ratio": marginal_value(
                    directory, "length_birth", "sd_ratio_median"
                ),
                "iid_birth_acf_mae": float(
                    diagnostics["acf_metrics"][
                        "iid_length_birth_mae_lags_1_30"
                    ]
                ),
                "replay_birth_acf_mae": float(
                    diagnostics["acf_metrics"][
                        "replay_length_birth_mae_lags_1_30"
                    ]
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(BASE / "lineage_hierarchy_comparison.csv", index=False)

    u_acf = pd.read_csv(LINEAGE_U / "posterior_acf_comparison.csv")
    omega_acf = pd.read_csv(LINEAGE_OMEGA / "posterior_acf_comparison.csv")
    lags = common_iid.lag.to_numpy(int)
    observed = common_iid.observed_length_birth.to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharex=True, sharey=True)
    panels = [
        (
            axes[0],
            "IID fitted inputs",
            common_iid.predictive_median_length_birth.to_numpy(float),
            u_acf.iid_median_length_birth.to_numpy(float),
            omega_acf.iid_median_length_birth.to_numpy(float),
        ),
        (
            axes[1],
            "Measured input replay",
            common_replay.measured_inputs_median_length_birth.to_numpy(float),
            u_acf.replay_median_length_birth.to_numpy(float),
            omega_acf.replay_median_length_birth.to_numpy(float),
        ),
    ]
    for axis, title, common_curve, u_curve, omega_curve in panels:
        axis.plot(lags, observed, color="tab:blue", marker="o", ms=3, lw=2,
                  label="Glu data")
        axis.plot(lags, common_curve, color="0.35", ls="--", lw=1.8,
                  label="Common parameters")
        axis.plot(lags, u_curve, color="tab:orange", lw=1.8,
                  label=r"Lineage-specific $u_\ell$")
        axis.plot(lags, omega_curve, color="tab:green", lw=1.8,
                  label=r"Lineage-specific $\omega_{2,\ell}$")
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_title(title)
        axis.set_xlabel("Generation lag")
        axis.set_ylabel("Mean sample birth-length ACF")
        axis.set_xticks(np.arange(0, 31, 5))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
        ncol=4, frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(BASE / "lineage_hierarchy_birth_acf_comparison.png", dpi=190)
    fig.savefig(BASE / "lineage_hierarchy_birth_acf_comparison.pdf")
    plt.close(fig)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
