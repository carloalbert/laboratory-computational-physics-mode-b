#!/usr/bin/env python3
"""Replay Glu alpha and f at the alpha-scaled model's critical limit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import t as student_t


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from optimize_alpha_scaled_acf_measured_inputs import (  # noqa: E402
    PARAMETER_NAMES,
    build_design,
    curve_metrics,
    make_randomness,
    simulate_one_replay,
    simulate_replays,
    summarize_curve,
)
from alpha_scaled_hill_time_hazard import RESIDUAL_DF  # noqa: E402
from validate_acf_with_measured_inputs import (  # noqa: E402
    load_sugar,
    observed_acf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--seed", type=int, default=921_700)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def mean_log_replayed_fraction(real: pd.DataFrame) -> float:
    values = []
    for _, group in real.groupby("lineage", sort=False):
        if len(group) < 3:
            continue
        ordered = group.sort_values("generation")
        # Row n+1 supplies the factor producing b_(n+1).
        values.extend(ordered.f.to_numpy(float)[1:])
    return float(np.mean(np.log(np.asarray(values))))


def expected_truncated_residual(
    omega2: float, location: float, scale: float
) -> float:
    """E[epsilon] when s~Exponential(omega2/2), epsilon>-s."""
    rate = omega2 / 2.0

    def integrand(growth_time: float) -> float:
        standardized_lower = (-growth_time - location) / scale
        survival = student_t.sf(standardized_lower, RESIDUAL_DF)
        first_tail_moment = (
            (RESIDUAL_DF + standardized_lower**2)
            / (RESIDUAL_DF - 1.0)
            * student_t.pdf(standardized_lower, RESIDUAL_DF)
        )
        conditional_mean = (
            location + scale * first_tail_moment / survival
        )
        return rate * math.exp(-rate * growth_time) * conditional_mean

    return float(quad(integrand, 0.0, np.inf, epsabs=1e-11)[0])


def critical_omega2(
    mean_log_fraction: float, location: float, scale: float
) -> float:
    """Solve E[log(b[n+1]/b[n])]=0 in the eta->0 timer limit."""

    def drift(omega2: float) -> float:
        return (
            2.0 / omega2
            + expected_truncated_residual(omega2, location, scale)
            + mean_log_fraction
        )

    return float(brentq(drift, 0.2, 20.0))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.eta < 1.0:
        raise ValueError("This critical replay expects 0 < eta < 1")
    # The literal 1/log(eta) is negative.  The positive hazard scale is the
    # intended -1/log(eta).
    u = -1.0 / math.log(args.eta)
    output_dir = args.output_dir or (
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / "Glu"
        / "alpha_scaled_hill_time_hazard"
        / "critical_replay_eta_0p1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    real, _ = load_sugar("Glu")
    design = build_design(real)
    max_lag = min(args.max_lag, int(design.lengths.max()) - 1)
    posterior = pd.read_csv(
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / "Glu"
        / "alpha_scaled_hill_time_hazard"
        / "posterior_summary.csv"
    ).set_index("parameter")
    fitted_params = posterior.loc[PARAMETER_NAMES, "median"].to_numpy(float)
    residual_params = posterior.loc[
        ["mu_epsilon", "sigma_epsilon"], "median"
    ].to_numpy(float)
    mean_log_fraction = mean_log_replayed_fraction(real)
    omega2 = critical_omega2(mean_log_fraction, *residual_params)
    critical_params = np.asarray([u, omega2, args.eta], dtype=float)

    observed = {
        variable: observed_acf(real, variable, max_lag)
        for variable in ["time", "length_birth"]
    }
    if not all(np.isclose(values[0], 1.0) for values in observed.values()):
        raise RuntimeError("Observed ACFs are not normalized at lag zero")
    randomness = make_randomness(args.replicates, design, args.seed)
    critical = simulate_replays(
        critical_params, residual_params, design, randomness, max_lag
    )
    fitted = simulate_replays(
        fitted_params, residual_params, design, randomness, max_lag
    )
    if not all(
        np.allclose(matrix[:, 0], 1.0)
        for replay in [critical, fitted]
        for matrix in replay.values()
    ):
        raise RuntimeError("Simulated ACFs are not normalized at lag zero")

    lags = np.arange(max_lag + 1)
    table: dict[str, np.ndarray] = {"lag": lags}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    fig.patch.set_facecolor("white")
    metrics = []
    for axis, (variable, title) in zip(
        axes,
        [
            ("time", "Division-time ACF"),
            ("length_birth", "Birth-length ACF"),
        ],
    ):
        axis.set_facecolor("white")
        fitted_median, _, _ = summarize_curve(fitted[variable])
        median, lower, upper = summarize_curve(critical[variable])
        table[f"observed_{variable}"] = observed[variable]
        table[f"critical_median_{variable}"] = median
        table[f"critical_q2.5_{variable}"] = lower
        table[f"critical_q97.5_{variable}"] = upper
        table[f"likelihood_median_{variable}"] = fitted_median
        axis.fill_between(
            lags,
            lower,
            upper,
            color="tab:green",
            alpha=0.24,
            label="Critical replay 95% band",
        )
        axis.plot(
            lags,
            median,
            color="tab:green",
            marker="o",
            ms=3,
            lw=1.6,
            label="Critical replay median",
        )
        axis.plot(
            lags,
            fitted_median,
            color="tab:orange",
            ls="--",
            lw=1.5,
            label="Likelihood-fit replay median",
        )
        axis.plot(
            lags,
            observed[variable],
            color="tab:blue",
            marker="o",
            ms=3,
            lw=2,
            label="Glu data",
        )
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_title(title)
        axis.set_xlabel("Generation lag")
        axis.set_ylabel("Mean sample ACF")
        axis.set_xticks(np.arange(0, max_lag + 1, 5))
        axis.grid(axis="y", alpha=0.13)
        metrics.append(
            {
                "variable": variable,
                **curve_metrics(
                    observed[variable], critical[variable], 15, max_lag
                ),
            }
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Glu: measured-input replay at the critical timer point\n"
        + rf"$u=-1/\ln(\eta)={u:.4f}$, "
        + rf"$\omega_{{2,c}}={omega2:.4f}$, $\eta={args.eta:g}$",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.76))
    fig.savefig(
        output_dir / "acf_critical_replay.png",
        dpi=180,
        facecolor="white",
        transparent=False,
    )
    fig.savefig(
        output_dir / "acf_critical_replay.pdf",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)
    pd.DataFrame(table).to_csv(
        output_dir / "acf_critical_replay.csv", index=False
    )
    pd.DataFrame(metrics).to_csv(
        output_dir / "acf_critical_replay_metrics.csv", index=False
    )

    _, _, values = simulate_one_replay(
        critical_params,
        residual_params,
        design,
        randomness.exponential_threshold[0],
        randomness.residual_uniform[0],
        max_lag,
        return_values=True,
    )
    assert values is not None
    diagnostics = {
        "interpretation_of_requested_u": (
            "used u=-1/log(eta)>0 because literal 1/log(eta) is negative"
        ),
        "parameters": dict(zip(PARAMETER_NAMES, critical_params)),
        "critical_point_definition": (
            "zero mean log-size drift in eta->0 timer limit, using replayed "
            "Glu f values and the fitted truncated endpoint residual"
        ),
        "mean_log_replayed_f": mean_log_fraction,
        "fixed_residual_parameters": {
            "mu_epsilon": residual_params[0],
            "sigma_epsilon": residual_params[1],
        },
        "replicates": args.replicates,
        "lineages": int(len(design.lengths)),
        "observations": int(np.sum(design.lengths)),
        "one_replay_marginals": {
            variable: {
                "mean": float(np.mean(values[variable])),
                "sd": float(np.std(values[variable], ddof=1)),
            }
            for variable in ["time", "length_birth"]
        },
        "acf_metrics": metrics,
    }
    with (output_dir / "diagnostics.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
