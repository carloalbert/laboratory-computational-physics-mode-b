#!/usr/bin/env python3
"""Profile the Glu decaying-memory likelihood over the memory time scale."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_decaying_memory_time_hazard import (  # noqa: E402
    Likelihood,
    TRANSFORMED_BOUNDS,
    load_data,
    unpack,
)


OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "model_comparison"
    / "Glu"
    / "decaying_memory_time_hazard_shared"
)


def polish_at_fixed_tau(
    likelihood: Likelihood, initial: np.ndarray, tau_m: float
) -> tuple[np.ndarray, float]:
    point = initial.copy()
    point[3] = math.log(tau_m)
    low, high = TRANSFORMED_BOUNDS.T
    # Work with log u_eff(0) = log u + beta.  This removes the strongest
    # coordinate-wise ridge between the asymptotic u and transient amplitude.
    working = np.asarray(
        [point[0] + point[4], point[1], point[2], point[4]], dtype=float
    )

    def to_original(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                values[0] - values[3],
                values[1],
                values[2],
                math.log(tau_m),
                values[3],
            ]
        )

    def objective(values: np.ndarray) -> float:
        original = to_original(values)
        if not (
            low[0] < original[0] < high[0]
            and low[4] < original[4] < high[4]
        ):
            return -np.inf
        return likelihood(original)

    working_low = np.asarray([low[0] + low[4], low[1], low[2], low[4]])
    working_high = np.asarray([high[0] + high[4], high[1], high[2], high[4]])
    base_step = np.asarray([0.04, 0.04, 0.04, 0.08])
    score = objective(working)
    for refinement in range(8):
        step = base_step * 0.5**refinement
        for _ in range(50):
            improved = False
            for dimension in range(4):
                candidates = []
                for direction in (-1.0, 1.0):
                    candidate = working.copy()
                    candidate[dimension] = np.clip(
                        candidate[dimension] + direction * step[dimension],
                        working_low[dimension] + 1e-10,
                        working_high[dimension] - 1e-10,
                    )
                    candidates.append((objective(candidate), candidate, direction))
                candidate_score, candidate, direction = max(
                    candidates, key=lambda item: item[0]
                )
                if candidate_score <= score:
                    continue
                working = candidate
                score = float(candidate_score)
                improved = True
                for _ in range(100):
                    candidate = working.copy()
                    candidate[dimension] = np.clip(
                        candidate[dimension] + direction * step[dimension],
                        working_low[dimension] + 1e-10,
                        working_high[dimension] - 1e-10,
                    )
                    if candidate[dimension] == working[dimension]:
                        break
                    candidate_score = objective(candidate)
                    if candidate_score <= score:
                        break
                    working = candidate
                    score = float(candidate_score)
            if not improved:
                break
    return to_original(working), score


def profile_branch(
    likelihood: Likelihood, initial: np.ndarray, tau_values: list[float]
) -> list[dict[str, float]]:
    rows = []
    point = initial.copy()
    for tau_m in tau_values:
        point, score = polish_at_fixed_tau(likelihood, point, tau_m)
        u, omega2, eta, _, beta = unpack(point)
        rows.append(
            {
                "tau_m": tau_m,
                "profile_log_likelihood": score,
                "u": u,
                "omega2": omega2,
                "eta": eta,
                "beta": beta,
                "u_effective_at_lineage_start": u * math.exp(beta),
                "change_in_log_u_effective_over_20_hours": beta
                * (math.exp(-20.0 / tau_m) - 1.0),
            }
        )
        print(tau_m, score, flush=True)
    return rows


def main() -> None:
    diagnostics = json.loads((OUTPUT_DIR / "diagnostics.json").read_text())
    mle = diagnostics["mle"]
    initial = np.asarray(
        [
            math.log(mle["u"]),
            math.log(mle["omega2"]),
            math.log(mle["eta"]),
            math.log(mle["tau_m"]),
            mle["beta"],
        ]
    )
    real, _ = load_data("Glu")
    likelihood = Likelihood(real)
    rows = profile_branch(
        likelihood, initial, [40.0, 20.0, 10.0, 5.0, 2.0]
    )
    rows.extend(
        profile_branch(
            likelihood, initial, [50.0, 100.0, 200.0, 400.0]
        )
    )
    table = pd.DataFrame(rows).sort_values("tau_m")
    maximum = float(table["profile_log_likelihood"].max())
    table["delta_log_likelihood"] = table["profile_log_likelihood"] - maximum
    table.to_csv(OUTPUT_DIR / "tau_m_profile_likelihood.csv", index=False)


if __name__ == "__main__":
    main()
