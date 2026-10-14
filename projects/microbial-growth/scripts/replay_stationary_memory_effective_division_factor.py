#!/usr/bin/env python3
"""Replay the fitted Glu stationary-memory model with effective factors.

The effective daughter factor absorbs the observed endpoint residual,

    f_eff[n+1] = b[n+1] / (b[n] * exp(alpha[n] * time[n])).

The division-time likelihood and its fitted parameters are unchanged.  This
script only changes the observation-matched size recursion used for the ACF
check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_stationary_memory_time_hazard import (  # noqa: E402
    PARAMETER_NAMES,
    acf_1d,
    load_data,
    make_plot,
    mean_acf,
    replay_once,
)


SOURCE_DIR = (
    PROJECT_DIR
    / "outputs"
    / "model_comparison"
    / "Glu"
    / "stationary_memory_time_hazard_shared"
)
MODEL_NAME = "stationary_memory_time_hazard_shared_effective_f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def add_effective_factors(
    groups: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame], np.ndarray, float]:
    updated = []
    all_effective = []
    maximum_relative_error = 0.0
    for original in groups:
        group = original.copy()
        birth = group["length_birth"].to_numpy(float)
        alpha = group["alpha"].to_numpy(float)
        division_time = group["time"].to_numpy(float)
        effective = np.full(len(group), np.nan)
        if len(group) > 1:
            effective[1:] = birth[1:] / (
                birth[:-1] * np.exp(alpha[:-1] * division_time[:-1])
            )
            reconstructed = (
                effective[1:]
                * birth[:-1]
                * np.exp(alpha[:-1] * division_time[:-1])
            )
            relative_error = np.max(
                np.abs(reconstructed - birth[1:]) / birth[1:]
            )
            maximum_relative_error = max(maximum_relative_error, float(relative_error))
            all_effective.extend(effective[1:])
        group["f_effective"] = effective
        updated.append(group)
    return updated, np.asarray(all_effective), maximum_relative_error


def lineage_lag_summary(
    groups: list[pd.DataFrame], max_lag: int = 3
) -> list[np.ndarray]:
    return [
        np.asarray(
            [
                acf_1d(group["length_birth"].to_numpy(float), max_lag)[lag]
                for group in groups
                if len(group) > lag
            ],
            dtype=float,
        )
        for lag in range(1, max_lag + 1)
    ]


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir or (
        PROJECT_DIR / "outputs" / "model_comparison" / "Glu" / MODEL_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _, groups = load_data()
    nonunit_generation_gaps = int(
        sum(
            np.sum(np.diff(group["generation"].to_numpy(float)) != 1.0)
            for group in groups
        )
    )
    groups, effective, reconstruction_error = add_effective_factors(groups)

    fitted = pd.read_csv(SOURCE_DIR / "mle_parameters.csv").set_index("parameter")
    physical = fitted.loc[PARAMETER_NAMES, "mle"].to_numpy(float)
    pd.DataFrame({"parameter": PARAMETER_NAMES, "mle": physical}).to_csv(
        output_dir / "mle_parameters.csv", index=False
    )

    max_lag = min(args.max_lag, max(len(group) for group in groups) - 1)
    observed_time = mean_acf(groups, "time", max_lag)
    observed_birth = mean_acf(groups, "length_birth", max_lag)
    predictive_time = np.full((args.replicates, max_lag + 1), np.nan)
    predictive_birth = np.full_like(predictive_time, np.nan)
    predictive_memory = np.full_like(predictive_time, np.nan)
    rng = np.random.default_rng(args.seed)
    for replicate in range(args.replicates):
        (
            predictive_time[replicate],
            predictive_birth[replicate],
            predictive_memory[replicate],
        ) = replay_once(
            groups,
            physical,
            max_lag,
            rng,
            division_factor_column="f_effective",
        )
        if (replicate + 1) % 25 == 0:
            print(f"effective-factor replay {replicate + 1}/{args.replicates}", flush=True)

    table = pd.DataFrame({"lag": np.arange(max_lag + 1)})
    for name, observed, predictive in (
        ("time", observed_time, predictive_time),
        ("length_birth", observed_birth, predictive_birth),
        ("memory", np.full(max_lag + 1, np.nan), predictive_memory),
    ):
        table[f"observed_{name}"] = observed
        table[f"predictive_median_{name}"] = np.nanmedian(predictive, axis=0)
        table[f"predictive_q2.5_{name}"] = np.nanpercentile(predictive, 2.5, axis=0)
        table[f"predictive_q97.5_{name}"] = np.nanpercentile(predictive, 97.5, axis=0)
    table.to_csv(output_dir / "effective_f_replay_acf.csv", index=False)
    make_plot(table, output_dir / "effective_f_acf_comparison.png")

    old_table = pd.read_csv(SOURCE_DIR / "posterior_replay_acf.csv")
    comparison_rows = []
    metrics = {}
    for name in ("time", "length_birth"):
        observed = table[f"observed_{name}"].to_numpy(float)
        median = table[f"predictive_median_{name}"].to_numpy(float)
        lower = table[f"predictive_q2.5_{name}"].to_numpy(float)
        upper = table[f"predictive_q97.5_{name}"].to_numpy(float)
        old_median = old_table[f"predictive_median_{name}"].to_numpy(float)
        metrics[name] = {
            "mae_lags_1_to_max_effective_f": float(
                np.nanmean(np.abs(observed[1:] - median[1:]))
            ),
            "mae_lags_1_to_max_recorded_f": float(
                np.nanmean(np.abs(observed[1:] - old_median[1:]))
            ),
            "covered_lags_effective_f": int(
                np.sum((observed[1:] >= lower[1:]) & (observed[1:] <= upper[1:]))
            ),
        }
        for lag in range(1, 4):
            comparison_rows.append(
                {
                    "variable": name,
                    "lag": lag,
                    "observed": observed[lag],
                    "effective_f_predictive_median": median[lag],
                    "recorded_f_predictive_median": old_median[lag],
                    "effective_f_q2.5": lower[lag],
                    "effective_f_q97.5": upper[lag],
                }
            )
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "effective_vs_recorded_f_lag1_to_3.csv", index=False
    )

    observed_lineage_acfs = lineage_lag_summary(groups)
    diagnostics = {
        "model": MODEL_NAME,
        "sugar": "Glu",
        "size_update": "b_(n+1)=f_eff_(n+1)*b_n*exp(alpha_n*t_n)",
        "effective_factor": "f_eff_(n+1)=b_data_(n+1)/(b_data_n*exp(alpha_data_n*t_data_n))",
        "hazard_fit": "unchanged MLE from stationary_memory_time_hazard_shared",
        "mle": dict(zip(PARAMETER_NAMES, map(float, physical))),
        "n_effective_factors": int(len(effective)),
        "n_effective_factors_spanning_nonunit_generation_gaps": nonunit_generation_gaps,
        "effective_factor_summary": {
            "mean": float(np.mean(effective)),
            "sd": float(np.std(effective, ddof=1)),
            "q2.5": float(np.percentile(effective, 2.5)),
            "median": float(np.median(effective)),
            "q97.5": float(np.percentile(effective, 97.5)),
            "minimum": float(np.min(effective)),
            "maximum": float(np.max(effective)),
            "fraction_above_one": float(np.mean(effective > 1.0)),
        },
        "maximum_relative_observed_reconstruction_error": reconstruction_error,
        "acf_metrics": metrics,
        "observed_lineage_birth_acf_mean_lags_1_to_3": [
            float(np.nanmean(values)) for values in observed_lineage_acfs
        ],
        "run_config": vars(args) | {"output_dir": str(output_dir)},
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    with (output_dir / "diagnostics.json").open("w") as stream:
        json.dump(diagnostics, stream, indent=2)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
