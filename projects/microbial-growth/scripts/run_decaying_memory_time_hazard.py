#!/usr/bin/env python3
"""Fit the shared-parameter decaying-memory time hazard to Glu.

The five Glu-specific parameters are u, omega2, eta, tau_m, and beta.  There
are no lineage-specific fitted parameters and no ACF term in the likelihood.
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
sys.path.insert(0, str(PROJECT_DIR / "src"))

from decaying_memory_time_hazard import (  # noqa: E402
    log_time_density,
    sample_growth_time,
)


MODEL_NAME = "decaying_memory_time_hazard_shared"
PARAMETER_NAMES = ["u", "omega2", "eta", "tau_m", "beta"]
TRANSFORMED_BOUNDS = np.asarray(
    [
        [math.log(1e-4), math.log(50.0)],
        [math.log(0.1), math.log(30.0)],
        [math.log(1e-3), math.log(20.0)],
        [math.log(0.1), math.log(500.0)],
        [-6.0, 6.0],
    ],
    dtype=float,
)
BETA_PRIOR_SD = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sugar", default="Glu", choices=["Glu"])
    parser.add_argument("--optimizer-population", type=int, default=52)
    parser.add_argument("--optimizer-iterations", type=int, default=140)
    parser.add_argument("--walkers", type=int, default=24)
    parser.add_argument("--burn", type=int, default=600)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--thin", type=int, default=5)
    parser.add_argument("--acf-replicates", type=int, default=300)
    parser.add_argument("--acf-max-lag", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_data(sugar: str) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    path = PROJECT_DIR / "data" / "processed" / "processed_input_data.csv"
    columns = [
        "source",
        "lineage",
        "generation",
        "time",
        "alpha",
        "f",
        "length_birth",
    ]
    data = pd.read_csv(path, usecols=columns)
    for column in columns[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=columns)
    data = data.loc[
        data["source"].eq(sugar)
        & data["time"].gt(0)
        & data["alpha"].gt(0)
        & data["f"].between(0, 1, inclusive="neither")
        & data["length_birth"].gt(0)
    ].sort_values(["lineage", "generation"])
    generation_zero = data.loc[data["generation"].eq(0), "lineage"].unique()
    data = data.loc[data["lineage"].isin(generation_zero)].copy()
    data["lineage_time_start"] = (
        data.groupby("lineage", sort=False)["time"].cumsum() - data["time"]
    )
    groups = [
        group.sort_values("generation").reset_index(drop=True)
        for _, group in data.groupby("lineage", sort=False)
    ]
    return data.reset_index(drop=True), groups


def unpack(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray(
        [math.exp(values[0]), math.exp(values[1]), math.exp(values[2]), math.exp(values[3]), values[4]]
    )


def in_bounds(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float)
    return bool(
        np.all(values > TRANSFORMED_BOUNDS[:, 0])
        and np.all(values < TRANSFORMED_BOUNDS[:, 1])
    )


class Likelihood:
    def __init__(self, real: pd.DataFrame, order: int = 24):
        self.time = real["time"].to_numpy(float)
        self.birth = real["length_birth"].to_numpy(float)
        self.alpha = real["alpha"].to_numpy(float)
        self.start = real["lineage_time_start"].to_numpy(float)
        self.order = order
        self.growth_time = self.alpha * self.time
        self.log_alpha = np.log(self.alpha)
        self.event_absolute_time = self.start + self.time
        self.event_log_added = np.log(
            np.maximum(
                self.birth * np.expm1(np.clip(self.growth_time, 0.0, 700.0)),
                np.finfo(float).tiny,
            )
        )
        nodes, self.weights = np.polynomial.legendre.leggauss(order)
        integration_growth_time = 0.5 * self.growth_time[:, None] * (
            nodes + 1.0
        )
        self.integration_absolute_time = (
            self.start[:, None] + integration_growth_time / self.alpha[:, None]
        )
        self.integration_log_added = np.log(
            np.maximum(
                self.birth[:, None]
                * np.expm1(np.clip(integration_growth_time, 0.0, 700.0)),
                np.finfo(float).tiny,
            )
        )
        self.integration_scale = 0.5 * self.growth_time

    @staticmethod
    def _fraction(values: np.ndarray) -> np.ndarray:
        positive = values >= 0.0
        result = np.empty_like(values)
        result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponential = np.exp(values[~positive])
        result[~positive] = exponential / (1.0 + exponential)
        return result

    def __call__(self, transformed: np.ndarray) -> float:
        if not in_bounds(transformed):
            return -np.inf
        log_u, log_omega2, log_eta, log_tau_m, beta = map(float, transformed)
        omega2 = math.exp(log_omega2)
        eta = math.exp(log_eta)
        tau_m = math.exp(log_tau_m)
        integration_memory = np.exp(-self.integration_absolute_time / tau_m)
        integration_log_ratio = eta * (
            self.integration_log_added - log_u - beta * integration_memory
        )
        fraction = self._fraction(integration_log_ratio)
        cumulative = omega2 * self.integration_scale * np.sum(
            self.weights * fraction, axis=1
        )
        event_memory = np.exp(-self.event_absolute_time / tau_m)
        event_log_ratio = eta * (
            self.event_log_added - log_u - beta * event_memory
        )
        event_log_fraction = -np.logaddexp(0.0, -event_log_ratio)
        values = self.log_alpha + log_omega2 + event_log_fraction - cumulative
        return float(np.sum(values)) if np.all(np.isfinite(values)) else -np.inf

    def posterior(self, transformed: np.ndarray) -> float:
        likelihood = self(transformed)
        if not np.isfinite(likelihood):
            return -np.inf
        beta = float(transformed[4])
        return likelihood - 0.5 * (beta / BETA_PRIOR_SD) ** 2


def differential_evolution_maximize(
    objective,
    rng: np.random.Generator,
    population_size: int,
    iterations: int,
    seeds: list[np.ndarray],
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    dimensions = TRANSFORMED_BOUNDS.shape[0]
    low, high = TRANSFORMED_BOUNDS.T
    population = rng.uniform(low, high, size=(population_size, dimensions))
    for index, seed in enumerate(seeds[:population_size]):
        population[index] = np.clip(seed, low + 1e-10, high - 1e-10)
    scores = np.asarray([objective(candidate) for candidate in population])
    history: list[dict[str, float]] = []
    for iteration in range(iterations):
        for index in range(population_size):
            choices = np.delete(np.arange(population_size), index)
            first, second, third = rng.choice(choices, size=3, replace=False)
            mutant = population[first] + 0.72 * (
                population[second] - population[third]
            )
            mutant = np.clip(mutant, low + 1e-10, high - 1e-10)
            crossover = rng.uniform(size=dimensions) < 0.82
            crossover[rng.integers(dimensions)] = True
            trial = np.where(crossover, mutant, population[index])
            score = objective(trial)
            if score > scores[index]:
                population[index] = trial
                scores[index] = score
        best = float(np.max(scores))
        history.append({"iteration": iteration + 1, "best_log_value": best})
        if (iteration + 1) % 20 == 0:
            print(f"optimizer iteration {iteration + 1}: {best:.6f}", flush=True)
    best_index = int(np.argmax(scores))
    return population[best_index].copy(), float(scores[best_index]), history


def pattern_polish(
    objective,
    initial: np.ndarray,
    initial_step: np.ndarray | None = None,
    tolerance: float = 2e-5,
) -> tuple[np.ndarray, float]:
    point = np.asarray(initial, dtype=float).copy()
    low, high = TRANSFORMED_BOUNDS.T
    step = (
        np.asarray(initial_step, dtype=float).copy()
        if initial_step is not None
        else np.asarray([0.08, 0.05, 0.05, 0.08, 0.05])
    )
    score = objective(point)
    sweeps = 0
    while float(np.max(step)) > tolerance and sweeps < 500:
        sweeps += 1
        improved = False
        for dimension in range(len(point)):
            candidates = []
            for direction in (-1.0, 1.0):
                candidate = point.copy()
                candidate[dimension] = np.clip(
                    candidate[dimension] + direction * step[dimension],
                    low[dimension] + 1e-10,
                    high[dimension] - 1e-10,
                )
                candidates.append(candidate)
            candidate_scores = [objective(candidate) for candidate in candidates]
            best = int(np.argmax(candidate_scores))
            if candidate_scores[best] > score:
                point = candidates[best]
                score = float(candidate_scores[best])
                improved = True
                direction = -1.0 if best == 0 else 1.0
                for _ in range(100):
                    candidate = point.copy()
                    candidate[dimension] = np.clip(
                        candidate[dimension] + direction * step[dimension],
                        low[dimension] + 1e-10,
                        high[dimension] - 1e-10,
                    )
                    if candidate[dimension] == point[dimension]:
                        break
                    candidate_score = objective(candidate)
                    if candidate_score <= score:
                        break
                    point = candidate
                    score = float(candidate_score)
        if not improved:
            step *= 0.5
    return point, float(score)


def numerical_covariance(negative_log_probability, point: np.ndarray) -> np.ndarray:
    dimensions = len(point)
    step = np.asarray([1.5e-3, 1.5e-3, 1.5e-3, 2.0e-3, 1.5e-3])
    hessian = np.empty((dimensions, dimensions))
    center = negative_log_probability(point)
    for i in range(dimensions):
        direction_i = np.zeros(dimensions)
        direction_i[i] = step[i]
        hessian[i, i] = (
            negative_log_probability(point + direction_i)
            - 2.0 * center
            + negative_log_probability(point - direction_i)
        ) / step[i] ** 2
        for j in range(i):
            direction_j = np.zeros(dimensions)
            direction_j[j] = step[j]
            value = (
                negative_log_probability(point + direction_i + direction_j)
                - negative_log_probability(point + direction_i - direction_j)
                - negative_log_probability(point - direction_i + direction_j)
                + negative_log_probability(point - direction_i - direction_j)
            ) / (4.0 * step[i] * step[j])
            hessian[i, j] = hessian[j, i] = value
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    eigenvalues = np.clip(eigenvalues, 1e-3, 1e9)
    covariance = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
    covariance_values, covariance_vectors = np.linalg.eigh(covariance)
    covariance_values = np.clip(covariance_values, 1e-8, 0.75)
    return (covariance_vectors * covariance_values) @ covariance_vectors.T


def initialize_ensemble(
    center: np.ndarray,
    covariance: np.ndarray,
    walkers: int,
    log_probability,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    low, high = TRANSFORMED_BOUNDS.T
    cholesky = np.linalg.cholesky(covariance + 1e-12 * np.eye(len(center)))
    ensemble: list[np.ndarray] = []
    probabilities: list[float] = []
    attempts = 0
    scale = 0.7
    while len(ensemble) < walkers and attempts < 100_000:
        attempts += 1
        candidate = center + scale * cholesky @ rng.normal(size=len(center))
        if np.all(candidate > low) and np.all(candidate < high):
            probability = log_probability(candidate)
            if np.isfinite(probability):
                ensemble.append(candidate)
                probabilities.append(probability)
    if len(ensemble) != walkers:
        raise RuntimeError("Unable to initialize posterior ensemble")
    return np.asarray(ensemble), np.asarray(probabilities)


def run_stretch_sampler(
    log_probability,
    center: np.ndarray,
    covariance: np.ndarray,
    walkers: int,
    burn: int,
    steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    if walkers < 2 * len(center):
        raise ValueError("The stretch sampler needs at least 2 * ndim walkers")
    ensemble, probabilities = initialize_ensemble(
        center, covariance, walkers, log_probability, rng
    )
    chain = np.empty((steps, walkers, len(center)))
    log_chain = np.empty((steps, walkers))
    accepted = 0
    proposed = 0
    total_steps = burn + steps
    stretch_scale = 2.0
    for iteration in range(total_steps):
        for index in range(walkers):
            partner = int(rng.integers(walkers - 1))
            if partner >= index:
                partner += 1
            uniform = rng.uniform()
            stretch = ((stretch_scale - 1.0) * uniform + 1.0) ** 2 / stretch_scale
            candidate = ensemble[partner] + stretch * (
                ensemble[index] - ensemble[partner]
            )
            candidate_probability = log_probability(candidate)
            proposed += 1
            log_acceptance = (
                (len(center) - 1) * math.log(stretch)
                + candidate_probability
                - probabilities[index]
            )
            if math.log(rng.uniform()) < log_acceptance:
                ensemble[index] = candidate
                probabilities[index] = candidate_probability
                accepted += 1
        if iteration >= burn:
            production_index = iteration - burn
            chain[production_index] = ensemble
            log_chain[production_index] = probabilities
        if (iteration + 1) % 400 == 0:
            print(
                f"sampler iteration {iteration + 1}/{total_steps}; "
                f"acceptance {accepted / proposed:.3f}",
                flush=True,
            )
    return chain, log_chain, accepted / proposed


def split_rhat(chain: np.ndarray) -> np.ndarray:
    steps, walkers, dimensions = chain.shape
    half = steps // 2
    split = np.concatenate((chain[:half], chain[-half:]), axis=1)
    chain_means = np.mean(split, axis=0)
    chain_variances = np.var(split, axis=0, ddof=1)
    within = np.mean(chain_variances, axis=0)
    between = half * np.var(chain_means, axis=0, ddof=1)
    variance = (half - 1) / half * within + between / half
    return np.sqrt(variance / within)


def acf_1d(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(max_lag + 1, np.nan)
    if len(values) < 3:
        return result
    centered = values - np.mean(values)
    denominator = np.sum(centered**2)
    if denominator <= 0.0:
        return result
    result[0] = 1.0
    for lag in range(1, min(max_lag + 1, len(values))):
        result[lag] = np.sum(centered[:-lag] * centered[lag:]) / denominator
    return result


def mean_acf_from_groups(groups: list[pd.DataFrame], column: str, max_lag: int) -> np.ndarray:
    pieces = [
        acf_1d(group[column].to_numpy(float), max_lag)
        for group in groups
        if len(group) >= 3
    ]
    array = np.asarray(pieces)
    counts = np.sum(np.isfinite(array), axis=0)
    return np.divide(
        np.nansum(array, axis=0),
        counts,
        out=np.full(max_lag + 1, np.nan),
        where=counts > 0,
    )


def replay_once(
    groups: list[pd.DataFrame],
    physical: np.ndarray,
    max_lag: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    u, omega2, eta, tau_m, beta = map(float, physical)
    time_acfs: list[np.ndarray] = []
    birth_acfs: list[np.ndarray] = []
    drift: list[float] = []
    lag_values = [[], [], []]
    for group in groups:
        birth = float(group["length_birth"].iloc[0])
        lineage_time = 0.0
        times = np.empty(len(group))
        births = np.empty(len(group))
        for generation in range(len(group)):
            alpha = float(group["alpha"].iloc[generation])
            growth_time = sample_growth_time(
                birth,
                alpha,
                lineage_time,
                u,
                omega2,
                eta,
                tau_m,
                beta,
                rng,
            )
            division_time = growth_time / alpha
            times[generation] = division_time
            births[generation] = birth
            final = birth * math.exp(min(growth_time, 700.0))
            lineage_time += division_time
            if generation + 1 < len(group):
                birth = max(float(group["f"].iloc[generation + 1]) * final, 1e-12)
        if len(group) >= 3:
            time_acfs.append(acf_1d(times, max_lag))
            birth_acf = acf_1d(births, max_lag)
            birth_acfs.append(birth_acf)
            for lag in range(1, 4):
                if lag < len(birth_acf) and np.isfinite(birth_acf[lag]):
                    lag_values[lag - 1].append(birth_acf[lag])
        if len(group) >= 6:
            third = max(2, len(group) // 3)
            drift.append(float(np.mean(times[-third:]) - np.mean(times[:third])))
    time_array = np.asarray(time_acfs)
    birth_array = np.asarray(birth_acfs)
    time_mean = np.nanmean(time_array, axis=0)
    birth_mean = np.nanmean(birth_array, axis=0)
    lag_summary = np.asarray(
        [
            [np.mean(values), np.median(values), np.std(values, ddof=1)]
            for values in lag_values
        ]
    )
    return time_mean, birth_mean, float(np.mean(drift)), lag_summary


def observed_drift(groups: list[pd.DataFrame]) -> float:
    differences = []
    for group in groups:
        if len(group) < 6:
            continue
        values = group["time"].to_numpy(float)
        third = max(2, len(group) // 3)
        differences.append(np.mean(values[-third:]) - np.mean(values[:third]))
    return float(np.mean(differences))


def summarize_chain(physical_chain: np.ndarray, mode: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(PARAMETER_NAMES):
        values = physical_chain[:, index]
        rows.append(
            {
                "parameter": name,
                "mode": float(mode[index]),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci_2.5": float(np.percentile(values, 2.5)),
                "ci_97.5": float(np.percentile(values, 97.5)),
                "std": float(np.std(values, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir or (
        PROJECT_DIR / "outputs" / "model_comparison" / args.sugar / MODEL_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    real, groups = load_data(args.sugar)
    likelihood = Likelihood(real)
    rng = np.random.default_rng(args.seed)

    existing_summary = pd.read_csv(
        PROJECT_DIR
        / "outputs"
        / "model_comparison"
        / args.sugar
        / "alpha_scaled_hill_time_hazard"
        / "posterior_summary.csv"
    ).set_index("parameter")
    base = np.log(existing_summary.loc[["u", "omega2", "eta"], "median"].to_numpy(float))
    seeds = []
    for tau_m in (2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
        for beta in (-1.0, 0.25, 0.75, 1.5, 3.0):
            seeds.append(np.r_[base, math.log(tau_m), beta])

    print("Fitting five-parameter likelihood", flush=True)
    mle, mle_ll, history = differential_evolution_maximize(
        likelihood,
        rng,
        args.optimizer_population,
        args.optimizer_iterations,
        seeds,
    )
    mle, mle_ll = pattern_polish(likelihood, mle)
    pd.DataFrame(history).to_csv(output_dir / "optimizer_history.csv", index=False)
    print("MLE", dict(zip(PARAMETER_NAMES, unpack(mle))), mle_ll, flush=True)

    def baseline_objective(values: np.ndarray) -> float:
        complete = np.r_[values, math.log(10.0), 0.0]
        return likelihood(complete)

    baseline_three, baseline_ll = pattern_polish(
        baseline_objective,
        base,
        initial_step=np.asarray([0.04, 0.03, 0.03]),
    )
    print("No-memory log likelihood", baseline_ll, flush=True)

    map_point, map_log_probability = pattern_polish(likelihood.posterior, mle)
    covariance = numerical_covariance(
        lambda values: -likelihood.posterior(values), map_point
    )
    print("Sampling posterior", flush=True)
    transformed_chain, log_chain, acceptance = run_stretch_sampler(
        likelihood.posterior,
        map_point,
        covariance,
        args.walkers,
        args.burn,
        args.steps,
        rng,
    )
    rhat = split_rhat(transformed_chain)
    retained_transformed = transformed_chain[:: args.thin].reshape(-1, 5)
    retained_log_probability = log_chain[:: args.thin].reshape(-1)
    physical_chain = np.column_stack(
        (
            np.exp(retained_transformed[:, :4]),
            retained_transformed[:, 4],
        )
    )
    mode_index = int(np.argmax(retained_log_probability))
    posterior_mode = physical_chain[mode_index]
    summary = summarize_chain(physical_chain, posterior_mode)
    summary.to_csv(output_dir / "posterior_summary.csv", index=False)
    pd.DataFrame(physical_chain, columns=PARAMETER_NAMES).to_csv(
        output_dir / "flat_chain.csv", index=False
    )
    pd.DataFrame(retained_log_probability, columns=["log_probability"]).to_csv(
        output_dir / "flat_log_probability.csv", index=False
    )
    u_draw = physical_chain[:, 0]
    tau_draw = physical_chain[:, 3]
    beta_draw = physical_chain[:, 4]
    derived_values = np.column_stack(
        [
            u_draw * np.exp(beta_draw),
            u_draw * np.exp(beta_draw * np.exp(-10.0 / tau_draw)),
            u_draw * np.exp(beta_draw * np.exp(-20.0 / tau_draw)),
            beta_draw * (np.exp(-20.0 / tau_draw) - 1.0),
        ]
    )
    derived_names = [
        "u_effective_0h",
        "u_effective_10h",
        "u_effective_20h",
        "delta_log_u_0_to_20h",
    ]
    retained_steps = transformed_chain[:: args.thin].shape[0]
    derived_rhat = split_rhat(
        derived_values.reshape(retained_steps, args.walkers, len(derived_names))
    )
    derived_rows = []
    for index, name in enumerate(derived_names):
        values = derived_values[:, index]
        derived_rows.append(
            {
                "quantity": name,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci_2.5": float(np.percentile(values, 2.5)),
                "ci_97.5": float(np.percentile(values, 97.5)),
                "split_rhat": float(derived_rhat[index]),
            }
        )
    pd.DataFrame(derived_rows).to_csv(
        output_dir / "derived_transient_posterior_summary.csv", index=False
    )

    max_lag = min(args.acf_max_lag, max(len(group) for group in groups) - 1)
    observed_time_acf = mean_acf_from_groups(groups, "time", max_lag)
    observed_birth_acf = mean_acf_from_groups(groups, "length_birth", max_lag)
    predictive_time = np.full((args.acf_replicates, max_lag + 1), np.nan)
    predictive_birth = np.full_like(predictive_time, np.nan)
    predictive_drift = np.full(args.acf_replicates, np.nan)
    predictive_lag_summary = np.full((args.acf_replicates, 3, 3), np.nan)
    draw_indices = rng.integers(0, len(physical_chain), size=args.acf_replicates)
    print("Running observation-matched posterior replay", flush=True)
    for replicate, draw_index in enumerate(draw_indices):
        (
            predictive_time[replicate],
            predictive_birth[replicate],
            predictive_drift[replicate],
            predictive_lag_summary[replicate],
        ) = replay_once(
            groups,
            physical_chain[draw_index],
            max_lag,
            rng,
        )
        if (replicate + 1) % 50 == 0:
            print(f"replay {replicate + 1}/{args.acf_replicates}", flush=True)

    lags = np.arange(max_lag + 1)
    acf_table = pd.DataFrame(
        {
            "lag": lags,
            "observed_time": observed_time_acf,
            "predictive_median_time": np.nanmedian(predictive_time, axis=0),
            "predictive_q2.5_time": np.nanpercentile(predictive_time, 2.5, axis=0),
            "predictive_q97.5_time": np.nanpercentile(predictive_time, 97.5, axis=0),
            "observed_length_birth": observed_birth_acf,
            "predictive_median_length_birth": np.nanmedian(predictive_birth, axis=0),
            "predictive_q2.5_length_birth": np.nanpercentile(predictive_birth, 2.5, axis=0),
            "predictive_q97.5_length_birth": np.nanpercentile(predictive_birth, 97.5, axis=0),
        }
    )
    acf_table.to_csv(output_dir / "posterior_replay_acf.csv", index=False)

    observed_lag_summary = []
    observed_birth_pieces = [
        acf_1d(group["length_birth"].to_numpy(float), 3)
        for group in groups
        if len(group) >= 3
    ]
    observed_birth_pieces = np.asarray(observed_birth_pieces)
    for lag in range(1, 4):
        values = observed_birth_pieces[:, lag]
        values = values[np.isfinite(values)]
        observed_lag_summary.append(
            [np.mean(values), np.median(values), np.std(values, ddof=1)]
        )
    lag_rows = []
    for lag in range(1, 4):
        for statistic_index, statistic in enumerate(("mean", "median", "sd")):
            values = predictive_lag_summary[:, lag - 1, statistic_index]
            lag_rows.append(
                {
                    "lag": lag,
                    "statistic": statistic,
                    "observed": observed_lag_summary[lag - 1][statistic_index],
                    "predictive_median": np.nanmedian(values),
                    "predictive_q2.5": np.nanpercentile(values, 2.5),
                    "predictive_q97.5": np.nanpercentile(values, 97.5),
                }
            )
    pd.DataFrame(lag_rows).to_csv(
        output_dir / "lineage_birth_acf_lag1_to_3_summary.csv", index=False
    )

    map_density_24 = log_time_density(
        likelihood.time,
        likelihood.birth,
        likelihood.alpha,
        likelihood.start,
        *unpack(map_point),
        order=24,
    )
    map_density_48 = log_time_density(
        likelihood.time,
        likelihood.birth,
        likelihood.alpha,
        likelihood.start,
        *unpack(map_point),
        order=48,
    )
    n_observations = len(real)
    memory_bic = 5 * math.log(n_observations) - 2.0 * mle_ll
    baseline_bic = 3 * math.log(n_observations) - 2.0 * baseline_ll
    acf_metrics = {}
    for name, observed, predictive in (
        ("time", observed_time_acf, predictive_time),
        ("length_birth", observed_birth_acf, predictive_birth),
    ):
        median = np.nanmedian(predictive, axis=0)
        lower = np.nanpercentile(predictive, 2.5, axis=0)
        upper = np.nanpercentile(predictive, 97.5, axis=0)
        acf_metrics[name] = {
            "mae_lags_1_to_max": float(np.nanmean(np.abs(median[1:] - observed[1:]))),
            "covered_lags_1_to_max": int(
                np.sum((observed[1:] >= lower[1:]) & (observed[1:] <= upper[1:]))
            ),
            "lag1_observed": float(observed[1]),
            "lag1_predictive_median": float(median[1]),
            "lag2_observed": float(observed[2]),
            "lag2_predictive_median": float(median[2]),
            "lag3_observed": float(observed[3]),
            "lag3_predictive_median": float(median[3]),
        }

    diagnostics = {
        "model": MODEL_NAME,
        "sugar": args.sugar,
        "hazard": (
            "h(a)=alpha*omega2*(p(a)/u_eff(T+a))^eta/"
            "(1+(p(a)/u_eff(T+a))^eta)"
        ),
        "memory": "m(T)=exp(-T/tau_m); u_eff(T)=u*exp(beta*m(T))",
        "parameter_scope": "all five parameters shared by all Glu lineages",
        "likelihood": "division-time event likelihood only; no ACF penalty",
        "prior": {
            "positive_parameters": "uniform in log within recorded bounds",
            "beta": f"Normal(0,{BETA_PRIOR_SD}^2), truncated to [-6,6]",
        },
        "n_observations": n_observations,
        "n_lineages": len(groups),
        "transformed_bounds": {
            name: list(map(float, bounds))
            for name, bounds in zip(PARAMETER_NAMES, TRANSFORMED_BOUNDS)
        },
        "mle": dict(zip(PARAMETER_NAMES, map(float, unpack(mle)))),
        "mle_log_likelihood": float(mle_ll),
        "no_memory_mle": dict(
            zip(["u", "omega2", "eta"], map(float, np.exp(baseline_three)))
        ),
        "no_memory_log_likelihood": float(baseline_ll),
        "log_likelihood_improvement": float(mle_ll - baseline_ll),
        "bic_memory": float(memory_bic),
        "bic_no_memory": float(baseline_bic),
        "delta_bic_memory_minus_no_memory": float(memory_bic - baseline_bic),
        "posterior_sampler": {
            "algorithm": "Goodman-Weare stretch ensemble",
            "mean_acceptance_fraction": float(acceptance),
            "split_rhat": dict(zip(PARAMETER_NAMES, map(float, rhat))),
            "retained_samples": int(len(physical_chain)),
        },
        "quadrature_validation_at_map": {
            "log_likelihood_order_24": float(np.sum(map_density_24)),
            "log_likelihood_order_48": float(np.sum(map_density_48)),
            "difference_order_48_minus_24": float(
                np.sum(map_density_48) - np.sum(map_density_24)
            ),
            "max_absolute_per_observation_difference": float(
                np.max(np.abs(map_density_48 - map_density_24))
            ),
        },
        "acf_metrics": acf_metrics,
        "division_time_drift": {
            "observed_late_minus_early_hours": observed_drift(groups),
            "predictive_median": float(np.median(predictive_drift)),
            "predictive_q2.5": float(np.percentile(predictive_drift, 2.5)),
            "predictive_q97.5": float(np.percentile(predictive_drift, 97.5)),
        },
        "run_config": vars(args) | {"output_dir": str(output_dir)},
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    with (output_dir / "diagnostics.json").open("w") as stream:
        json.dump(diagnostics, stream, indent=2)
    with (output_dir / "run_config.json").open("w") as stream:
        json.dump(diagnostics["run_config"], stream, indent=2)

    print(summary.to_string(index=False), flush=True)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
