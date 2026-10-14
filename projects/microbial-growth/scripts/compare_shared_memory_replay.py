#!/usr/bin/env python3
"""Compare the fitted shared-memory replay with its beta=0 nested model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_decaying_memory_time_hazard import load_data, replay_once  # noqa: E402


OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "model_comparison"
    / "Glu"
    / "decaying_memory_time_hazard_shared"
)


def main() -> None:
    diagnostics = json.loads((OUTPUT_DIR / "diagnostics.json").read_text())
    acf = pd.read_csv(OUTPUT_DIR / "posterior_replay_acf.csv")
    _, groups = load_data("Glu")
    base = diagnostics["no_memory_mle"]
    physical = np.asarray(
        [base["u"], base["omega2"], base["eta"], 10.0, 0.0], dtype=float
    )
    replicates = diagnostics["run_config"]["acf_replicates"]
    max_lag = diagnostics["run_config"]["acf_max_lag"]
    rng = np.random.default_rng(diagnostics["run_config"]["seed"] + 991)
    time = np.full((replicates, max_lag + 1), np.nan)
    birth = np.full_like(time, np.nan)
    drift = np.full(replicates, np.nan)
    for replicate in range(replicates):
        time[replicate], birth[replicate], drift[replicate], _ = replay_once(
            groups, physical, max_lag, rng
        )
    for name, values in (("time", time), ("length_birth", birth)):
        acf[f"no_memory_median_{name}"] = np.nanmedian(values, axis=0)
        acf[f"no_memory_q2.5_{name}"] = np.nanpercentile(values, 2.5, axis=0)
        acf[f"no_memory_q97.5_{name}"] = np.nanpercentile(values, 97.5, axis=0)
    acf.to_csv(OUTPUT_DIR / "memory_vs_no_memory_replay_acf.csv", index=False)

    comparison = {
        "no_memory_division_time_drift": {
            "median": float(np.median(drift)),
            "q2.5": float(np.percentile(drift, 2.5)),
            "q97.5": float(np.percentile(drift, 97.5)),
        },
        "acf_mae_lags_1_to_30": {},
    }
    for name in ("time", "length_birth"):
        observed = acf[f"observed_{name}"].to_numpy(float)[1:]
        memory = acf[f"predictive_median_{name}"].to_numpy(float)[1:]
        no_memory = acf[f"no_memory_median_{name}"].to_numpy(float)[1:]
        comparison["acf_mae_lags_1_to_30"][name] = {
            "memory": float(np.nanmean(np.abs(memory - observed))),
            "no_memory": float(np.nanmean(np.abs(no_memory - observed))),
        }
    with (OUTPUT_DIR / "memory_vs_no_memory_replay_summary.json").open("w") as stream:
        json.dump(comparison, stream, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
