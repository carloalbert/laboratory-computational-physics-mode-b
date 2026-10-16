#!/usr/bin/env python3
"""Decompose Glu birth-size drift into measured transition components.

For complete transitions, using the actual daughter-row division fraction,

    log b_(i+1) - log b_i = log f_i + alpha_i*tau_i + epsilon_i.

The growth contribution is further decomposed generation by generation as

    mean(alpha*tau) = mean(alpha)*mean(tau) + cov(alpha, tau).

The primary calculation uses the fixed balanced Glu cohort from the retained
workflow, so changing lineage composition cannot create an apparent trend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from fit_empirical_detrended_hazard import cohort  # noqa: E402
from glu_workflow import load_config, load_data  # noqa: E402


OUT = PROJECT_DIR / "outputs" / "summary" / "Glu_birth_size_drift_components"


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
    epsilon = np.diff(data["x"], axis=1) - np.log(actual_f) - data["g"]
    return {
        **data,
        "b": np.exp(data["x"]),
        "f": actual_f,
        "epsilon": epsilon,
    }


def generation_summary(data: dict[str, np.ndarray]) -> pd.DataFrame:
    x, b = data["x"], data["b"]
    alpha, tau, growth = data["a"], data["t"], data["g"]
    log_f, epsilon = np.log(data["f"]), data["epsilon"]
    mean_alpha = alpha.mean(axis=0)
    mean_tau = tau.mean(axis=0)
    independent_growth = mean_alpha * mean_tau
    covariance = growth.mean(axis=0) - independent_growth
    delta_x = np.diff(x, axis=1).mean(axis=0)
    rows = pd.DataFrame(
        {
            "generation": np.arange(growth.shape[1]),
            "mean_birth": b[:, :-1].mean(axis=0),
            "mean_log_birth": x[:, :-1].mean(axis=0),
            "mean_log_f": log_f.mean(axis=0),
            "mean_epsilon": epsilon.mean(axis=0),
            "mean_alpha": mean_alpha,
            "mean_tau": mean_tau,
            "mean_alpha_tau": growth.mean(axis=0),
            "mean_alpha_times_mean_tau": independent_growth,
            "cov_alpha_tau": covariance,
            "corr_alpha_tau": [
                np.corrcoef(alpha[:, i], tau[:, i])[0, 1]
                for i in range(growth.shape[1])
            ],
            "mean_delta_log_birth": delta_x,
        }
    )
    terminal = pd.DataFrame(
        {
            "generation": [growth.shape[1]],
            "mean_birth": [b[:, -1].mean()],
            "mean_log_birth": [x[:, -1].mean()],
        }
    )
    return pd.concat([rows, terminal], ignore_index=True)


def cumulative_components(
    data: dict[str, np.ndarray], indices: np.ndarray | None = None
) -> dict[str, float]:
    if indices is None:
        indices = np.arange(len(data["ids"]))
    alpha = data["a"][indices]
    tau = data["t"][indices]
    growth = alpha * tau
    product = alpha.mean(axis=0) * tau.mean(axis=0)
    result = {
        "log_f": float(np.log(data["f"][indices]).mean(axis=0).sum()),
        "epsilon": float(data["epsilon"][indices].mean(axis=0).sum()),
        "mean_alpha_times_mean_tau": float(product.sum()),
        "cov_alpha_tau": float((growth.mean(axis=0) - product).sum()),
    }
    result["net"] = float(sum(result.values()))
    return result


def bootstrap_components(
    data: dict[str, np.ndarray], replicates: int = 5000, seed: int = 90217
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(data["ids"])
    rows = []
    for _ in range(replicates):
        rows.append(cumulative_components(data, rng.integers(0, n, n)))
    return pd.DataFrame(rows)


def early_late_summary(data: dict[str, np.ndarray], width: int = 5) -> dict:
    alpha, tau = data["a"], data["t"]
    log_f, epsilon = np.log(data["f"]), data["epsilon"]
    early = np.arange(width)
    late = np.arange(alpha.shape[1] - width, alpha.shape[1])

    def block(indices: np.ndarray) -> dict[str, float]:
        a = float(alpha[:, indices].mean())
        t = float(tau[:, indices].mean())
        at = float((alpha[:, indices] * tau[:, indices]).mean())
        cov = at - a * t
        return {
            "mean_log_f": float(log_f[:, indices].mean()),
            "mean_epsilon": float(epsilon[:, indices].mean()),
            "mean_alpha": a,
            "mean_tau": t,
            "mean_alpha_tau": at,
            "mean_alpha_times_mean_tau": a * t,
            "cov_alpha_tau": cov,
            "corr_alpha_tau": float(
                np.corrcoef(alpha[:, indices].ravel(), tau[:, indices].ravel())[0, 1]
            ),
            "mean_delta_log_birth": float(
                (
                    log_f[:, indices]
                    + epsilon[:, indices]
                    + alpha[:, indices] * tau[:, indices]
                ).mean()
            ),
        }

    first, last = block(early), block(late)
    delta_alpha_product = 0.5 * (last["mean_alpha"] - first["mean_alpha"]) * (
        last["mean_tau"] + first["mean_tau"]
    )
    delta_tau_product = 0.5 * (last["mean_tau"] - first["mean_tau"]) * (
        last["mean_alpha"] + first["mean_alpha"]
    )
    return {
        "definition": f"pooled first {width} versus final {width} transitions",
        "early": first,
        "late": last,
        "late_minus_early": {key: last[key] - first[key] for key in first},
        "shapley_change_in_mean_alpha_times_mean_tau": {
            "alpha": float(delta_alpha_product),
            "tau": float(delta_tau_product),
            "sum": float(delta_alpha_product + delta_tau_product),
        },
    }


def bootstrap_early_late(
    data: dict[str, np.ndarray],
    width: int = 5,
    replicates: int = 5000,
    seed: int = 90218,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(data["ids"])
    rows = []
    for _ in range(replicates):
        indices = rng.integers(0, n, n)
        sampled = {
            key: value[indices]
            if isinstance(value, np.ndarray) and value.shape[0] == n
            else value
            for key, value in data.items()
        }
        result = early_late_summary(sampled, width=width)
        row = result["late_minus_early"].copy()
        row.update(
            {
                f"shapley_{key}": value
                for key, value in result[
                    "shapley_change_in_mean_alpha_times_mean_tau"
                ].items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_plot(summary: pd.DataFrame, data: dict[str, np.ndarray]) -> None:
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), constrained_layout=True)
    generation = np.arange(data["x"].shape[1])
    mean_birth = data["b"].mean(axis=0)
    geometric_birth = np.exp(data["x"].mean(axis=0))

    ax = axes[0, 0]
    ax.plot(generation, mean_birth, marker="o", ms=3, label="Arithmetic mean")
    ax.plot(generation, geometric_birth, marker="s", ms=3, label="Geometric mean")
    ax.set(xlabel="Generation", ylabel="Birth size", title="Observed birth-size drift")
    ax.legend(frameon=False)

    transition = summary.iloc[:-1]
    ax = axes[0, 1]
    ax.axhline(0, color="0.65", lw=1)
    for column, label in [
        ("mean_log_f", r"$E[\log f]$"),
        ("mean_epsilon", r"$E[\epsilon]$"),
        ("mean_alpha_times_mean_tau", r"$E[\alpha]E[\tau]$"),
        ("cov_alpha_tau", r"$\mathrm{Cov}(\alpha,\tau)$"),
        ("mean_delta_log_birth", r"$E[\Delta\log b]$"),
    ]:
        ax.plot(transition.generation, transition[column], marker="o", ms=2.5, label=label)
    ax.set(
        xlabel="Starting generation",
        ylabel="Mean log-size contribution per cycle",
        title="Exact additive decomposition",
    )
    ax.legend(frameon=False, ncol=2, fontsize=9)

    ax = axes[1, 0]
    relative_alpha = transition.mean_alpha / transition.mean_alpha.iloc[0]
    relative_tau = transition.mean_tau / transition.mean_tau.iloc[0]
    ax.axhline(1, color="0.65", lw=1)
    ax.plot(transition.generation, relative_alpha, marker="o", ms=3, label=r"Mean $\alpha$")
    ax.plot(transition.generation, relative_tau, marker="s", ms=3, label=r"Mean $\tau$")
    ax.set(
        xlabel="Starting generation",
        ylabel="Mean relative to generation 0",
        title="Growth rates and division times",
    )
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.axhline(0, color="0.65", lw=1)
    ax.plot(
        transition.generation,
        transition.corr_alpha_tau,
        marker="o",
        ms=3,
        label=r"$\mathrm{Corr}(\alpha,\tau)$",
    )
    ax.set(
        xlabel="Starting generation",
        ylabel="Correlation",
        title=r"Within-generation $\alpha$–$\tau$ association",
    )
    ax.set_ylim(-1, 1)

    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.suptitle("Glu balanced cohort: source of birth-size drift", fontsize=15)
    fig.savefig(OUT / "birth_size_drift_decomposition.png", dpi=180)
    fig.savefig(OUT / "birth_size_drift_decomposition.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = balanced_arrays()
    raw, _ = load_data()
    raw_generation = (
        raw.loc[raw.generation.between(0, data["g"].shape[1])]
        .groupby("generation", as_index=False)
        .length_birth.agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_birth", "count": "n_lineages"})
    )
    raw_generation.to_csv(OUT / "all_available_lineages_birth_mean.csv", index=False)
    summary = generation_summary(data)
    summary.to_csv(OUT / "generation_decomposition.csv", index=False)
    bootstrap = bootstrap_components(data)
    early_late = early_late_summary(data)
    early_late_bootstrap = bootstrap_early_late(data)
    point = cumulative_components(data)
    component_rows = []
    for key, value in point.items():
        component_rows.append(
            {
                "component": key,
                "cumulative_log_contribution": value,
                "mean_per_cycle": value / data["g"].shape[1],
                "bootstrap_lo95": float(bootstrap[key].quantile(0.025)),
                "bootstrap_hi95": float(bootstrap[key].quantile(0.975)),
            }
        )
    pd.DataFrame(component_rows).to_csv(
        OUT / "cumulative_component_summary.csv", index=False
    )
    early_late_rows = []
    early_late_point = {
        **early_late["late_minus_early"],
        **{
            f"shapley_{key}": value
            for key, value in early_late[
                "shapley_change_in_mean_alpha_times_mean_tau"
            ].items()
        },
    }
    for key, value in early_late_point.items():
        early_late_rows.append(
            {
                "quantity": key,
                "late_minus_early": value,
                "bootstrap_lo95": float(early_late_bootstrap[key].quantile(0.025)),
                "bootstrap_hi95": float(early_late_bootstrap[key].quantile(0.975)),
            }
        )
    pd.DataFrame(early_late_rows).to_csv(
        OUT / "early_late_component_summary.csv", index=False
    )

    reconstructed = (
        np.log(data["f"]) + data["g"] + data["epsilon"]
    )
    report = {
        "cohort": {
            "sugar": "Glu",
            "lineages": int(len(data["ids"])),
            "transitions_per_lineage": int(data["g"].shape[1]),
            "reason": "fixed balanced panel prevents lineage-dropout composition drift",
        },
        "identity": "delta log b = log f + alpha*tau + epsilon",
        "actual_fraction_alignment": "transition i uses the f recorded on daughter row i+1",
        "maximum_transition_identity_error": float(
            np.max(np.abs(np.diff(data["x"], axis=1) - reconstructed))
        ),
        "mean_birth": {
            "generation_0": float(data["b"][:, 0].mean()),
            "generation_20": float(data["b"][:, -1].mean()),
            "change": float(data["b"][:, -1].mean() - data["b"][:, 0].mean()),
            "relative_change": float(
                data["b"][:, -1].mean() / data["b"][:, 0].mean() - 1
            ),
        },
        "lineage_composition_check": {
            "all_available_generation_0_lineages": int(
                raw_generation.loc[raw_generation.generation.eq(0), "n_lineages"].iloc[0]
            ),
            "all_available_generation_20_lineages": int(
                raw_generation.loc[
                    raw_generation.generation.eq(data["g"].shape[1]), "n_lineages"
                ].iloc[0]
            ),
            "all_available_relative_mean_birth_change": float(
                raw_generation.loc[
                    raw_generation.generation.eq(data["g"].shape[1]), "mean_birth"
                ].iloc[0]
                / raw_generation.loc[
                    raw_generation.generation.eq(0), "mean_birth"
                ].iloc[0]
                - 1
            ),
            "balanced_relative_mean_birth_change": float(
                data["b"][:, -1].mean() / data["b"][:, 0].mean() - 1
            ),
        },
        "mean_log_birth": {
            "generation_0": float(data["x"][:, 0].mean()),
            "generation_20": float(data["x"][:, -1].mean()),
            "change": float(data["x"][:, -1].mean() - data["x"][:, 0].mean()),
        },
        "cumulative_log_contributions": point,
        "early_late": {
            **early_late,
            "lineage_bootstrap_ci95": {
                key: [
                    float(early_late_bootstrap[key].quantile(0.025)),
                    float(early_late_bootstrap[key].quantile(0.975)),
                ]
                for key in early_late_bootstrap
            },
        },
        "bootstrap": {
            "replicates": int(len(bootstrap)),
            "unit": "lineage",
            "seed": 90217,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    make_plot(summary, data)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
