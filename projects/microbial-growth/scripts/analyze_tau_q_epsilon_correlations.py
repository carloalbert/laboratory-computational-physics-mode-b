#!/usr/bin/env python3
"""Check contemporaneous and lagged Glu correlations of tau with q and epsilon.

Definitions for transition i in lineage l:

    q_i = log(b_(i+1)/b_i) - alpha_i*tau_i
    epsilon_i = q_i - log(f_i)

where f_i is the actual division fraction recorded on the daughter row i+1.
The primary sample is the fixed balanced 172-lineage, 20-transition Glu cohort.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_empirical_detrended_hazard import cohort  # noqa: E402
from glu_workflow import load_config, load_data  # noqa: E402


OUT = PROJECT_DIR / "outputs" / "summary" / "Glu_tau_q_epsilon_correlations"


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return float(np.corrcoef(x, y)[0, 1])


def balanced_arrays() -> dict[str, np.ndarray]:
    _, groups = load_data()
    horizon = int(load_config()["horizon"])
    selected = [
        group.iloc[: horizon + 1]
        for group in groups
        if len(group) > horizon
        and np.array_equal(
            group.generation.to_numpy()[: horizon + 1],
            np.arange(horizon + 1),
        )
    ]
    data = cohort(groups, horizon=horizon)
    actual_f = np.stack(
        [group.f.to_numpy(float)[1 : horizon + 1] for group in selected]
    )
    q = np.diff(data["x"], axis=1) - data["g"]
    epsilon = q - np.log(actual_f)
    return {**data, "f": actual_f, "q": q, "epsilon": epsilon}


def transform_pair(
    x: np.ndarray, y: np.ndarray, adjustment: str
) -> tuple[np.ndarray, np.ndarray]:
    if adjustment == "pooled":
        return x.ravel(), y.ravel()
    if adjustment == "within_lineage":
        return (
            (x - x.mean(axis=1, keepdims=True)).ravel(),
            (y - y.mean(axis=1, keepdims=True)).ravel(),
        )
    if adjustment == "within_generation":
        return (
            (x - x.mean(axis=0, keepdims=True)).ravel(),
            (y - y.mean(axis=0, keepdims=True)).ravel(),
        )
    if adjustment == "two_way":
        x_centered = (
            x
            - x.mean(axis=1, keepdims=True)
            - x.mean(axis=0, keepdims=True)
            + x.mean()
        )
        y_centered = (
            y
            - y.mean(axis=1, keepdims=True)
            - y.mean(axis=0, keepdims=True)
            + y.mean()
        )
        return x_centered.ravel(), y_centered.ravel()
    raise ValueError(adjustment)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    adjustment: str,
    replicates: int = 5000,
    seed: int = 7621,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    values = np.empty(replicates)
    for j in range(replicates):
        indices = rng.integers(0, n, n)
        xb, yb = transform_pair(x[indices], y[indices], adjustment)
        values[j] = pearson(xb, yb)
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def balanced_correlations(data: dict[str, np.ndarray]) -> pd.DataFrame:
    tau = data["t"]
    variables = {
        "q": data["q"],
        "epsilon": data["epsilon"],
        "log_f": np.log(data["f"]),
        "alpha_tau": data["g"],
    }
    rows = []
    for adjustment in ["pooled", "within_lineage", "within_generation", "two_way"]:
        for number, (name, values) in enumerate(variables.items()):
            x, y = transform_pair(tau, values, adjustment)
            lo, hi = bootstrap_ci(
                tau,
                values,
                adjustment,
                seed=7621 + 100 * number + 1000 * len(adjustment),
            )
            rows.append(
                {
                    "sample": "balanced",
                    "adjustment": adjustment,
                    "x": "tau",
                    "y": name,
                    "n_transitions": len(x),
                    "pearson_r": pearson(x, y),
                    "lineage_bootstrap_lo95": lo,
                    "lineage_bootstrap_hi95": hi,
                    "spearman_rho": float(spearmanr(x, y).statistic),
                }
            )
    lineage_count, generation_count = tau.shape
    generation = np.tile(np.arange(generation_count), lineage_count)
    lineage = np.repeat(np.arange(lineage_count), generation_count)
    design = np.column_stack(
        [
            np.ones(tau.size),
            data["x"][:, :-1].ravel(),
            np.log(data["a"]).ravel(),
            np.eye(generation_count)[generation][:, 1:],
            np.eye(lineage_count)[lineage][:, 1:],
        ]
    )

    def residual(values: np.ndarray) -> np.ndarray:
        flat = values.ravel()
        return (flat - design @ np.linalg.lstsq(design, flat, rcond=None)[0]).reshape(
            tau.shape
        )

    tau_residual = residual(tau)
    for number, (name, values) in enumerate(variables.items()):
        value_residual = residual(values)
        lo, hi = bootstrap_ci(
            tau_residual,
            value_residual,
            "pooled",
            seed=17621 + 100 * number,
        )
        rows.append(
            {
                "sample": "balanced",
                "adjustment": "lineage_generation_birth_alpha",
                "x": "tau",
                "y": name,
                "n_transitions": tau.size,
                "pearson_r": pearson(tau_residual, value_residual),
                "lineage_bootstrap_lo95": lo,
                "lineage_bootstrap_hi95": hi,
                "spearman_rho": float(
                    spearmanr(tau_residual.ravel(), value_residual.ravel()).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def all_available_correlations() -> pd.DataFrame:
    frame, _ = load_data()
    grouped = frame.groupby("lineage", sort=False)
    transition = frame.copy()
    transition["next_generation"] = grouped.generation.shift(-1)
    transition["next_birth"] = grouped.length_birth.shift(-1)
    transition["actual_f"] = grouped.f.shift(-1)
    transition = transition.loc[
        transition.next_generation.eq(transition.generation + 1)
    ].dropna(subset=["next_birth", "actual_f"])
    transition["q"] = (
        np.log(transition.next_birth / transition.length_birth)
        - transition.alpha * transition.time
    )
    transition["epsilon"] = transition.q - np.log(transition.actual_f)
    transition["log_f"] = np.log(transition.actual_f)
    transition["alpha_tau"] = transition.alpha * transition.time
    rows = []
    for name in ["q", "epsilon", "log_f", "alpha_tau"]:
        rows.append(
            {
                "sample": "all_available",
                "adjustment": "pooled",
                "x": "tau",
                "y": name,
                "n_transitions": len(transition),
                "pearson_r": pearson(transition.time, transition[name]),
                "lineage_bootstrap_lo95": np.nan,
                "lineage_bootstrap_hi95": np.nan,
                "spearman_rho": float(
                    spearmanr(transition.time, transition[name]).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def lagged_correlations(data: dict[str, np.ndarray]) -> pd.DataFrame:
    tau = data["t"]
    rows = []
    for name in ["q", "epsilon", "log_f"]:
        values = data[name] if name != "log_f" else np.log(data["f"])
        for lag in [-2, -1, 0, 1, 2]:
            if lag < 0:
                x, y = tau[:, -lag:], values[:, :lag]
            elif lag > 0:
                x, y = tau[:, :-lag], values[:, lag:]
            else:
                x, y = tau, values
            rows.append(
                {
                    "x": "tau_i",
                    "y": f"{name}_i_plus_lag",
                    "lag": lag,
                    "n_pairs": x.size,
                    "pearson_r": pearson(x, y),
                    "spearman_rho": float(spearmanr(x.ravel(), y.ravel()).statistic),
                }
            )
    return pd.DataFrame(rows)


def trimmed_correlations(data: dict[str, np.ndarray]) -> pd.DataFrame:
    tau = data["t"].ravel()
    variables = {
        "q": data["q"].ravel(),
        "epsilon": data["epsilon"].ravel(),
        "log_f": np.log(data["f"]).ravel(),
    }
    rows = []
    for tail_percent in [0.0, 0.5, 1.0, 2.5, 5.0]:
        for name, values in variables.items():
            low, high = np.percentile(
                values, [tail_percent, 100.0 - tail_percent]
            )
            retained = (values >= low) & (values <= high)
            rows.append(
                {
                    "variable": name,
                    "trim_percent_each_tail": tail_percent,
                    "n_pairs": int(retained.sum()),
                    "pearson_r": pearson(tau[retained], values[retained]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = balanced_arrays()
    correlations = pd.concat(
        [balanced_correlations(data), all_available_correlations()],
        ignore_index=True,
    )
    correlations.to_csv(OUT / "correlations.csv", index=False)
    lagged = lagged_correlations(data)
    lagged.to_csv(OUT / "lagged_correlations.csv", index=False)
    trimmed = trimmed_correlations(data)
    trimmed.to_csv(OUT / "trimmed_correlations.csv", index=False)

    key = correlations.loc[
        correlations["sample"].eq("balanced")
        & correlations["adjustment"].isin(
            ["pooled", "two_way", "lineage_generation_birth_alpha"]
        )
        & correlations["y"].isin(["q", "epsilon", "log_f"])
    ]
    tau = data["t"].ravel()
    q = data["q"].ravel()
    epsilon = data["epsilon"].ravel()
    log_f = np.log(data["f"]).ravel()

    def covariance(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((x - x.mean()) * (y - y.mean())))

    covariance_tau_q = covariance(tau, q)
    covariance_tau_log_f = covariance(tau, log_f)
    covariance_tau_epsilon = covariance(tau, epsilon)
    report = {
        "definitions": {
            "q": "log(b_next/b)-alpha*tau = log(actual daughter-row f)+epsilon",
            "epsilon": "q-log(actual daughter-row f)",
        },
        "cohort": {
            "lineages": int(len(data["ids"])),
            "transitions_per_lineage": int(data["t"].shape[1]),
            "total_transitions": int(data["t"].size),
        },
        "key_correlations": key.to_dict(orient="records"),
        "pooled_covariance_decomposition": {
            "cov_tau_q": covariance_tau_q,
            "cov_tau_log_f": covariance_tau_log_f,
            "cov_tau_epsilon": covariance_tau_epsilon,
            "identity_error": float(
                covariance_tau_q
                - covariance_tau_log_f
                - covariance_tau_epsilon
            ),
            "fraction_of_tau_q_covariance_from_log_f": float(
                covariance_tau_log_f / covariance_tau_q
            ),
        },
        "one_percent_trimmed_each_tail": trimmed.loc[
            trimmed.trim_percent_each_tail.eq(1.0)
        ].to_dict(orient="records"),
        "interpretation_note": (
            "q and epsilon are reconstructed using alpha*tau, so their "
            "same-cycle correlations with tau are descriptive and partly "
            "algebraic; lagged correlations help distinguish persistence."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(correlations.to_string(index=False))
    print("\nLagged correlations\n", lagged.to_string(index=False))


if __name__ == "__main__":
    main()
