#!/usr/bin/env python3
"""Recompute matched ACF diagnostics with exact observed (b0, length) pairs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_loglogistic_endpoint_residual import (  # noqa: E402
    load_endpoint_data,
    make_observation_matched_acf_plot,
)


SUGARS = ["Ace", "DMa", "Glu", "Gly", "Lac", "Pyr", "Rha", "Suc", "Xyl"]
MODEL_NAME = "loglogistic_endpoint_residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sugar", required=True, choices=SUGARS)
    parser.add_argument("--replicates", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / MODEL_NAME
    )
    table_path = directory / "acf_observation_matched.csv"
    legacy_path = directory / "acf_observation_matched_randomized_b0.csv"
    if not legacy_path.exists():
        shutil.copyfile(table_path, legacy_path)

    real, _, data, _ = load_endpoint_data(args.sugar)
    flat_chain = pd.read_csv(directory / "flat_chain.csv").to_numpy(float)
    long_simulation = pd.read_csv(
        directory / "posterior_predictive_simulation.csv"
    )
    birth_pivot = float(data["transition_birth"].mean())

    matched = make_observation_matched_acf_plot(
        directory,
        args.sugar,
        real,
        flat_chain,
        long_simulation,
        birth_pivot,
        max_lag=30,
        seed=42 + 20_000,
        n_replicates=args.replicates,
    )

    diagnostics_path = directory / "diagnostics.json"
    with diagnostics_path.open() as handle:
        diagnostics = json.load(handle)
    diagnostics["observation_matched_acf_lag1"] = {
        variable: {
            "observed": float(matched.loc[1, f"observed_{column}"]),
            "predictive_median": float(
                matched.loc[1, f"predictive_median_{column}"]
            ),
            "predictive_q2.5": float(
                matched.loc[1, f"predictive_q2.5_{column}"]
            ),
            "predictive_q97.5": float(
                matched.loc[1, f"predictive_q97.5_{column}"]
            ),
        }
        for variable, column in [
            ("time", "time"),
            ("length_birth", "length_birth"),
        ]
    }
    diagnostics["observation_matched_design"] = {
        "posterior_replicates": args.replicates,
        "preserves_initial_size_lineage_length_pairing": True,
        "description": (
            "Each simulated lineage uses its observed (b0, lineage length) pair."
        ),
        "legacy_randomized_b0_table": legacy_path.name,
    }
    with diagnostics_path.open("w") as handle:
        json.dump(diagnostics, handle, indent=2)

    print(
        f"{args.sugar}: exact-pair matched ACF updated; "
        f"birth lag1 median={matched.loc[1, 'predictive_median_length_birth']:.6f}"
    )


if __name__ == "__main__":
    main()
