#!/usr/bin/env python3
"""Fit the Glu p-and-b-dependent hazard and replay with effective factors."""

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
sys.path[:0] = [str(PROJECT_DIR / "src"), str(PROJECT_DIR / "scripts")]

from birth_size_dependent_time_hazard import (  # noqa: E402
    hill_fraction,
    log_time_density,
    sample_growth_time,
)
from replay_stationary_memory_effective_division_factor import (  # noqa: E402
    add_effective_factors,
)
from run_stationary_memory_time_hazard import (  # noqa: E402
    acf_1d,
    load_data,
    make_plot,
    mean_acf,
)


MODEL_NAME = "birth_size_dependent_time_hazard_effective_f"
PARAMETER_NAMES = ["u", "omega2", "eta", "kappa"]
BOUNDS = np.asarray(
    [
        [math.log(0.1), math.log(8.0)],
        [math.log(1.0), math.log(25.0)],
        [math.log(0.5), math.log(20.0)],
        [-12.0, 12.0],
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-population", type=int, default=40)
    parser.add_argument("--optimizer-iterations", type=int, default=90)
    parser.add_argument("--replicates", type=int, default=250)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def unpack(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray([*np.exp(values[:3]), values[3]], dtype=float)


class Likelihood:
    def __init__(self, data: pd.DataFrame, b_star: float, order: int = 24):
        self.time = data["time"].to_numpy(float)
        self.birth = data["length_birth"].to_numpy(float)
        self.alpha = data["alpha"].to_numpy(float)
        self.growth_time = self.alpha * self.time
        self.log_alpha = np.log(self.alpha)
        self.log_birth_ratio = np.log(self.birth / b_star)
        self.event_log_added = np.log(
            np.maximum(
                self.birth * np.expm1(np.clip(self.growth_time, 0.0, 700.0)),
                np.finfo(float).tiny,
            )
        )
        nodes, self.weights = np.polynomial.legendre.leggauss(order)
        integration_time = 0.5 * self.growth_time[:, None] * (nodes + 1.0)
        self.integration_log_added = np.log(
            np.maximum(
                self.birth[:, None]
                * np.expm1(np.clip(integration_time, 0.0, 700.0)),
                np.finfo(float).tiny,
            )
        )
        self.integration_scale = 0.5 * self.growth_time

    def __call__(self, transformed: np.ndarray) -> float:
        transformed = np.asarray(transformed, dtype=float)
        if np.any(transformed <= BOUNDS[:, 0]) or np.any(transformed >= BOUNDS[:, 1]):
            return -np.inf
        log_u, log_omega2, log_eta, kappa = map(float, transformed)
        omega2 = math.exp(log_omega2)
        eta = math.exp(log_eta)
        integration_log_odds = (
            eta * (self.integration_log_added - log_u)
            + kappa * self.log_birth_ratio[:, None]
        )
        cumulative = omega2 * self.integration_scale * np.sum(
            self.weights * hill_fraction(integration_log_odds), axis=1
        )
        event_log_odds = (
            eta * (self.event_log_added - log_u)
            + kappa * self.log_birth_ratio
        )
        values = (
            self.log_alpha
            + log_omega2
            - np.logaddexp(0.0, -event_log_odds)
            - cumulative
        )
        return float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf


def differential_evolution_maximize(
    objective,
    rng: np.random.Generator,
    population_size: int,
    iterations: int,
    seeds: list[np.ndarray],
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    dimensions = BOUNDS.shape[0]
    low, high = BOUNDS.T
    population = rng.uniform(low, high, size=(population_size, dimensions))
    for index, seed in enumerate(seeds[:population_size]):
        population[index] = np.clip(seed, low + 1e-9, high - 1e-9)
    scores = np.asarray([objective(candidate) for candidate in population])
    history = []
    for iteration in range(iterations):
        for index in range(population_size):
            choices = np.delete(np.arange(population_size), index)
            first, second, third = rng.choice(choices, size=3, replace=False)
            mutant = population[first] + 0.72 * (population[second] - population[third])
            mutant = np.clip(mutant, low + 1e-9, high - 1e-9)
            crossover = rng.uniform(size=dimensions) < 0.82
            crossover[rng.integers(dimensions)] = True
            trial = np.where(crossover, mutant, population[index])
            score = objective(trial)
            if score > scores[index]:
                population[index], scores[index] = trial, score
        best = float(np.max(scores))
        history.append({"iteration": iteration + 1, "best_log_likelihood": best})
        if (iteration + 1) % 15 == 0:
            print(f"optimizer iteration {iteration + 1}: {best:.6f}", flush=True)
    best_index = int(np.argmax(scores))
    return population[best_index].copy(), float(scores[best_index]), history


def pattern_polish(objective, initial: np.ndarray) -> tuple[np.ndarray, float]:
    point = np.asarray(initial, dtype=float).copy()
    low, high = BOUNDS.T
    step = np.asarray([0.05, 0.04, 0.04, 0.06])
    score = objective(point)
    while float(np.max(step)) > 1e-5:
        improved = False
        for dimension in range(len(point)):
            candidates = []
            for direction in (-1.0, 1.0):
                candidate = point.copy()
                candidate[dimension] = np.clip(
                    candidate[dimension] + direction * step[dimension],
                    low[dimension] + 1e-9,
                    high[dimension] - 1e-9,
                )
                candidates.append(candidate)
            candidate_scores = [objective(candidate) for candidate in candidates]
            best = int(np.argmax(candidate_scores))
            if candidate_scores[best] > score:
                point, score = candidates[best], float(candidate_scores[best])
                improved = True
        if not improved:
            step *= 0.5
    return point, float(score)


def replay_once(
    groups: list[pd.DataFrame],
    physical: np.ndarray,
    b_star: float,
    max_lag: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    u, omega2, eta, kappa = map(float, physical)
    time_acfs = []
    birth_acfs = []
    for group in groups:
        birth = float(group["length_birth"].iloc[0])
        times = np.empty(len(group))
        births = np.empty(len(group))
        for generation in range(len(group)):
            alpha = float(group["alpha"].iloc[generation])
            growth_time = sample_growth_time(
                birth, u, omega2, eta, kappa, b_star, rng
            )
            times[generation] = growth_time / alpha
            births[generation] = birth
            final = birth * math.exp(min(growth_time, 700.0))
            if generation + 1 < len(group):
                birth = max(
                    float(group["f_effective"].iloc[generation + 1]) * final,
                    1e-12,
                )
        time_acfs.append(acf_1d(times, max_lag))
        birth_acfs.append(acf_1d(births, max_lag))
    return (
        np.nanmean(np.asarray(time_acfs), axis=0),
        np.nanmean(np.asarray(birth_acfs), axis=0),
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir or (
        PROJECT_DIR / "outputs" / "model_comparison" / "Glu" / MODEL_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    real, groups = load_data()
    groups, effective_factors, reconstruction_error = add_effective_factors(groups)
    b_star = float(np.exp(np.mean(np.log(real["length_birth"].to_numpy(float)))))
    likelihood = Likelihood(real, b_star, order=24)
    rng = np.random.default_rng(args.seed)

    old_diagnostics = json.loads(
        (
            PROJECT_DIR
            / "outputs"
            / "model_comparison"
            / "Glu"
            / "stationary_memory_time_hazard_shared"
            / "diagnostics.json"
        ).read_text()
    )
    old = old_diagnostics["mle"]
    base = np.log([old["u"], old["omega2"], old["eta"]])
    seeds = [np.r_[base, kappa] for kappa in (-8, -4, -2, -1, 0, 1, 2, 4, 8)]
    mle, mle_log_likelihood, history = differential_evolution_maximize(
        likelihood,
        rng,
        args.optimizer_population,
        args.optimizer_iterations,
        seeds,
    )
    mle, mle_log_likelihood = pattern_polish(likelihood, mle)
    physical = unpack(mle)
    pd.DataFrame(history).to_csv(output_dir / "optimizer_history.csv", index=False)
    pd.DataFrame({"parameter": PARAMETER_NAMES, "mle": physical}).to_csv(
        output_dir / "mle_parameters.csv", index=False
    )
    print("MLE", dict(zip(PARAMETER_NAMES, physical)), mle_log_likelihood, flush=True)

    baseline = np.r_[base, 0.0]
    baseline, baseline_log_likelihood = pattern_polish(likelihood, baseline)
    baseline[3] = 0.0
    # Re-polish only the three nested parameters while holding kappa exactly zero.
    nested = baseline[:3].copy()
    nested_step = np.asarray([0.04, 0.03, 0.03])
    nested_score = likelihood(np.r_[nested, 0.0])
    while float(np.max(nested_step)) > 1e-5:
        improved = False
        for dimension in range(3):
            for direction in (-1.0, 1.0):
                candidate = nested.copy()
                candidate[dimension] += direction * nested_step[dimension]
                score = likelihood(np.r_[candidate, 0.0])
                if score > nested_score:
                    nested, nested_score = candidate, float(score)
                    improved = True
        if not improved:
            nested_step *= 0.5
    baseline_log_likelihood = nested_score

    max_lag = min(args.max_lag, max(len(group) for group in groups) - 1)
    observed_time = mean_acf(groups, "time", max_lag)
    observed_birth = mean_acf(groups, "length_birth", max_lag)
    predictive_time = np.full((args.replicates, max_lag + 1), np.nan)
    predictive_birth = np.full_like(predictive_time, np.nan)
    for replicate in range(args.replicates):
        predictive_time[replicate], predictive_birth[replicate] = replay_once(
            groups, physical, b_star, max_lag, rng
        )
        if (replicate + 1) % 25 == 0:
            print(f"replay {replicate + 1}/{args.replicates}", flush=True)

    table = pd.DataFrame({"lag": np.arange(max_lag + 1)})
    for name, observed, predictive in (
        ("time", observed_time, predictive_time),
        ("length_birth", observed_birth, predictive_birth),
    ):
        table[f"observed_{name}"] = observed
        table[f"predictive_median_{name}"] = np.nanmedian(predictive, axis=0)
        table[f"predictive_q2.5_{name}"] = np.nanpercentile(predictive, 2.5, axis=0)
        table[f"predictive_q97.5_{name}"] = np.nanpercentile(predictive, 97.5, axis=0)
    table.to_csv(output_dir / "effective_f_replay_acf.csv", index=False)
    make_plot(
        table,
        output_dir / "acf_comparison.png",
        model_label="p-and-b hazard",
    )

    metrics = {}
    for name in ("time", "length_birth"):
        observed = table[f"observed_{name}"].to_numpy(float)
        median = table[f"predictive_median_{name}"].to_numpy(float)
        lower = table[f"predictive_q2.5_{name}"].to_numpy(float)
        upper = table[f"predictive_q97.5_{name}"].to_numpy(float)
        metrics[name] = {
            "mae_lags_1_to_max": float(np.nanmean(np.abs(observed[1:] - median[1:]))),
            "covered_lags": int(np.sum((observed[1:] >= lower[1:]) & (observed[1:] <= upper[1:]))),
            "observed_lags_1_to_3": list(map(float, observed[1:4])),
            "predictive_median_lags_1_to_3": list(map(float, median[1:4])),
        }
    diagnostics = {
        "model": MODEL_NAME,
        "sugar": "Glu",
        "hazard": "alpha*omega2*sigmoid(eta*log(p/u)+kappa*log(b/b_star))",
        "p": "b*(exp(alpha*a)-1)",
        "b_star": b_star,
        "size_update": "b_(n+1)=f_eff_(n+1)*b_n*exp(alpha_n*t_n)",
        "parameter_scope": "u, omega2, eta, kappa shared by all Glu lineages",
        "mle": dict(zip(PARAMETER_NAMES, map(float, physical))),
        "mle_log_likelihood": float(mle_log_likelihood),
        "nested_kappa_zero_log_likelihood": float(baseline_log_likelihood),
        "log_likelihood_improvement": float(mle_log_likelihood - baseline_log_likelihood),
        "delta_bic_model_minus_nested": float(
            math.log(len(real)) - 2.0 * (mle_log_likelihood - baseline_log_likelihood)
        ),
        "n_observations": len(real),
        "n_lineages": len(groups),
        "effective_factor_summary": {
            "mean": float(np.mean(effective_factors)),
            "sd": float(np.std(effective_factors, ddof=1)),
            "median": float(np.median(effective_factors)),
        },
        "maximum_relative_observed_reconstruction_error": reconstruction_error,
        "acf_metrics": metrics,
        "run_config": vars(args) | {"output_dir": str(output_dir)},
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    with (output_dir / "diagnostics.json").open("w") as stream:
        json.dump(diagnostics, stream, indent=2)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
