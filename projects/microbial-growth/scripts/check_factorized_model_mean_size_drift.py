#!/usr/bin/env python3
"""Check mean birth-size drift in the factorized q/alpha + hazard model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_autoregressive_residual_glu import contiguous_prefixes, observed_matrix  # noqa: E402
from fit_fast_slow_ar_q_glu import simulate  # noqa: E402
from glu_workflow import load_data  # noqa: E402


SOURCE = PROJECT_DIR / "outputs/model_comparison/Glu/fast_slow_ar_q_protein_hazard"
OUT = SOURCE / "mean_size_drift"


def intervals(values):
    return (
        np.mean(values, axis=0),
        np.quantile(values, 0.025, axis=0),
        np.quantile(values, 0.975, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=20)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    params = json.loads((SOURCE / "fit_results.json").read_text())["fit"]
    _, groups = load_data()
    prefixes = contiguous_prefixes(groups)
    observed, lengths = observed_matrix(prefixes)
    horizon = min(args.horizon, int(lengths.max()) - 1)
    generated = simulate(params, lengths, args.replicates, seed=482931)["birth"]

    balanced = lengths > horizon
    if not balanced.any():
        raise ValueError("No lineages span the requested horizon")
    generation = np.arange(horizon + 1)
    observed_balanced = observed[balanced, : horizon + 1].mean(axis=0)
    simulated_balanced = generated[:, balanced, : horizon + 1].mean(axis=1)
    sim_mean, sim_low, sim_high = intervals(simulated_balanced)

    observed_log_balanced = np.log(observed[balanced, : horizon + 1]).mean(axis=0)
    simulated_log_balanced = np.log(generated[:, balanced, : horizon + 1]).mean(axis=1)
    sim_log_mean, sim_log_low, sim_log_high = intervals(simulated_log_balanced)

    rows = []
    for index in generation:
        rows.append(
            {
                "generation": index,
                "n_balanced_lineages": int(balanced.sum()),
                "observed_mean_birth": observed_balanced[index],
                "simulated_mean_birth": sim_mean[index],
                "simulated_lo95_birth": sim_low[index],
                "simulated_hi95_birth": sim_high[index],
                "observed_change_from_generation_0": observed_balanced[index]
                - observed_balanced[0],
                "simulated_change_from_generation_0": sim_mean[index] - sim_mean[0],
                "observed_mean_log_birth": observed_log_balanced[index],
                "simulated_mean_log_birth": sim_log_mean[index],
                "simulated_lo95_log_birth": sim_log_low[index],
                "simulated_hi95_log_birth": sim_log_high[index],
                "observed_log_change_from_generation_0": observed_log_balanced[index]
                - observed_log_balanced[0],
                "simulated_log_change_from_generation_0": sim_log_mean[index]
                - sim_log_mean[0],
            }
        )
    balanced_table = pd.DataFrame(rows)
    balanced_table.to_csv(OUT / "balanced_generation_means.csv", index=False)

    # Also calculate generation-specific paired changes using every lineage
    # available at that generation. This diagnoses any survivor-composition effect.
    paired_rows = []
    for index in generation:
        available = lengths > index
        observed_change = observed[available, index] - observed[available, 0]
        simulated_change = (
            generated[:, available, index] - generated[:, available, 0]
        ).mean(axis=1)
        observed_log_change = (
            np.log(observed[available, index]) - np.log(observed[available, 0])
        )
        simulated_log_change = (
            np.log(generated[:, available, index])
            - np.log(generated[:, available, 0])
        ).mean(axis=1)
        paired_rows.append(
            {
                "generation": index,
                "n_lineages": int(available.sum()),
                "observed_mean_paired_change": observed_change.mean(),
                "simulated_mean_paired_change": simulated_change.mean(),
                "simulated_lo95_paired_change": np.quantile(simulated_change, 0.025),
                "simulated_hi95_paired_change": np.quantile(simulated_change, 0.975),
                "observed_mean_paired_log_change": observed_log_change.mean(),
                "simulated_mean_paired_log_change": simulated_log_change.mean(),
                "simulated_lo95_paired_log_change": np.quantile(
                    simulated_log_change, 0.025
                ),
                "simulated_hi95_paired_log_change": np.quantile(
                    simulated_log_change, 0.975
                ),
            }
        )
    paired_table = pd.DataFrame(paired_rows)
    paired_table.to_csv(OUT / "paired_generation_changes.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axis = axes[0]
    axis.fill_between(generation, sim_low, sim_high, color="#c4512c", alpha=0.2)
    axis.plot(generation, sim_mean, "s-", color="#c4512c", ms=3.5, label="Factorized model")
    axis.plot(generation, observed_balanced, "o-", color="#23364d", ms=3.5, label="Observed")
    axis.set_title(f"Balanced cohort ({balanced.sum()} lineages)")
    axis.set_ylabel("Mean size at birth")
    axis.legend(frameon=False)

    axis = axes[1]
    axis.fill_between(
        paired_table.generation,
        paired_table.simulated_lo95_paired_change,
        paired_table.simulated_hi95_paired_change,
        color="#c4512c",
        alpha=0.2,
    )
    axis.plot(
        paired_table.generation,
        paired_table.simulated_mean_paired_change,
        "s-",
        color="#c4512c",
        ms=3.5,
        label="Factorized model",
    )
    axis.plot(
        paired_table.generation,
        paired_table.observed_mean_paired_change,
        "o-",
        color="#23364d",
        ms=3.5,
        label="Observed",
    )
    axis.axhline(0.0, color="0.6", lw=0.8)
    axis.set_title("Paired change from each lineage's generation 0")
    axis.set_ylabel(r"Mean $b_i-b_0$")
    for axis in axes:
        axis.set_xlabel("Generation")
        axis.grid(alpha=0.16)
    fig.suptitle("Glu: mean birth-size drift in the factorized q/alpha + hazard model")
    fig.savefig(OUT / "mean_size_drift.png", dpi=190)
    fig.savefig(OUT / "mean_size_drift.pdf")
    plt.close(fig)

    observed_slope = np.polyfit(generation, observed_balanced, 1)[0]
    simulated_slopes = np.polyfit(generation, simulated_balanced.T, 1)[0]
    summary = {
        "model": "factorized fast+slow q, Gamma-copula AR(1) alpha, protein Hill hazard for tau",
        "uncertainty": "fixed fitted parameters; bands are process-predictive, not parameter-posterior intervals",
        "replicates": args.replicates,
        "balanced_lineages": int(balanced.sum()),
        "horizon": horizon,
        "generation_0": {
            "observed_mean_birth": float(observed_balanced[0]),
            "simulated_mean_birth": float(sim_mean[0]),
            "simulated_lo95": float(sim_low[0]),
            "simulated_hi95": float(sim_high[0]),
        },
        "generation_horizon": {
            "observed_mean_birth": float(observed_balanced[-1]),
            "simulated_mean_birth": float(sim_mean[-1]),
            "simulated_lo95": float(sim_low[-1]),
            "simulated_hi95": float(sim_high[-1]),
            "observed_change": float(observed_balanced[-1] - observed_balanced[0]),
            "simulated_change": float(sim_mean[-1] - sim_mean[0]),
        },
        "linear_slope_birth_per_generation": {
            "observed": float(observed_slope),
            "simulated_mean": float(simulated_slopes.mean()),
            "simulated_lo95": float(np.quantile(simulated_slopes, 0.025)),
            "simulated_hi95": float(np.quantile(simulated_slopes, 0.975)),
        },
        "rmse_generation_mean_birth": float(
            np.sqrt(np.mean((observed_balanced - sim_mean) ** 2))
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
