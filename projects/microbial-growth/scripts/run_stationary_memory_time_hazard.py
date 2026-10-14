#!/usr/bin/env python3
"""Fit and replay the shared stationary-memory division model for Glu."""

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

from stationary_memory_time_hazard import (  # noqa: E402
    log_time_density_and_threshold,
    sample_cycle,
    update_memory,
)


MODEL_NAME = "stationary_memory_time_hazard_shared"
PARAMETER_NAMES = ["u", "omega2", "eta", "tau_m", "beta"]
BOUNDS = np.asarray(
    [
        [math.log(0.1), math.log(8.0)],
        [math.log(1.0), math.log(25.0)],
        [math.log(0.5), math.log(20.0)],
        [math.log(0.2), math.log(500.0)],
        [-4.0, 4.0],
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-population", type=int, default=36)
    parser.add_argument("--optimizer-iterations", type=int, default=60)
    parser.add_argument("--initial-memory-quadrature", type=int, default=5)
    parser.add_argument("--hazard-quadrature", type=int, default=16)
    parser.add_argument("--acf-replicates", type=int, default=200)
    parser.add_argument("--acf-max-lag", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse saved MLE and replay table and only finalize diagnostics.",
    )
    return parser.parse_args()


def load_data() -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    path = PROJECT_DIR / "data" / "processed" / "processed_input_data.csv"
    columns = ["source", "lineage", "generation", "time", "alpha", "f", "length_birth"]
    data = pd.read_csv(path, usecols=columns)
    for column in columns[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=columns)
    data = data.loc[
        data["source"].eq("Glu")
        & data["time"].gt(0.0)
        & data["alpha"].gt(0.0)
        & data["f"].between(0.0, 1.0, inclusive="neither")
        & data["length_birth"].gt(0.0)
    ].sort_values(["lineage", "generation"])
    generation_zero = data.loc[data["generation"].eq(0), "lineage"].unique()
    data = data.loc[data["lineage"].isin(generation_zero)].copy()
    groups = [
        group.sort_values("generation").reset_index(drop=True)
        for _, group in data.groupby("lineage", sort=False)
    ]
    return data.reset_index(drop=True), groups


def unpack(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray([*np.exp(values[:4]), values[4]], dtype=float)


def logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


class SequentialLikelihood:
    """Likelihood marginalized over stationary N(0,1) initial memory."""

    def __init__(self, groups: list[pd.DataFrame], memory_order: int, hazard_order: int):
        nodes, weights = np.polynomial.hermite.hermgauss(memory_order)
        self.initial_memory = math.sqrt(2.0) * nodes
        self.log_initial_weights = np.log(weights / math.sqrt(math.pi))
        self.lengths = np.asarray([len(group) for group in groups], dtype=int)
        self.max_length = int(np.max(self.lengths))
        shape = (len(groups), self.max_length)
        self.times = np.full(shape, np.nan)
        self.births = np.full(shape, np.nan)
        self.alphas = np.full(shape, np.nan)
        for lineage, group in enumerate(groups):
            length = len(group)
            self.times[lineage, :length] = group["time"].to_numpy(float)
            self.births[lineage, :length] = group["length_birth"].to_numpy(float)
            self.alphas[lineage, :length] = group["alpha"].to_numpy(float)
        self.hazard_order = hazard_order

    def __call__(self, transformed: np.ndarray) -> float:
        transformed = np.asarray(transformed, dtype=float)
        if np.any(transformed <= BOUNDS[:, 0]) or np.any(transformed >= BOUNDS[:, 1]):
            return -np.inf
        u, omega2, eta, tau_m, beta = unpack(transformed)
        n_lineages = len(self.lengths)
        n_branches = len(self.initial_memory)
        memory = np.broadcast_to(
            self.initial_memory, (n_lineages, n_branches)
        ).copy()
        branch_log_probability = np.broadcast_to(
            self.log_initial_weights, (n_lineages, n_branches)
        ).copy()
        for generation in range(self.max_length):
            active = self.lengths > generation
            if not np.any(active):
                break
            division_time = self.times[active, generation, None]
            birth = self.births[active, generation, None]
            alpha = self.alphas[active, generation, None]
            active_memory = memory[active]
            log_density, threshold = log_time_density_and_threshold(
                division_time,
                birth,
                alpha,
                active_memory,
                u,
                omega2,
                eta,
                tau_m,
                beta,
                order=self.hazard_order,
            )
            branch_log_probability[active] += log_density
            memory[active] = update_memory(
                active_memory, division_time, threshold, tau_m
            )
        maxima = np.max(branch_log_probability, axis=1)
        if not np.all(np.isfinite(maxima)):
            return -np.inf
        lineage_values = maxima + np.log(
            np.sum(np.exp(branch_log_probability - maxima[:, None]), axis=1)
        )
        return float(np.sum(lineage_values))


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
            mutant = population[first] + 0.7 * (population[second] - population[third])
            mutant = np.clip(mutant, low + 1e-9, high - 1e-9)
            crossover = rng.uniform(size=dimensions) < 0.82
            crossover[rng.integers(dimensions)] = True
            trial = np.where(crossover, mutant, population[index])
            score = objective(trial)
            if score > scores[index]:
                population[index] = trial
                scores[index] = score
        best = float(np.max(scores))
        history.append({"iteration": iteration + 1, "best_log_likelihood": best})
        if (iteration + 1) % 10 == 0:
            print(f"optimizer iteration {iteration + 1}: {best:.6f}", flush=True)
    best_index = int(np.argmax(scores))
    return population[best_index].copy(), float(scores[best_index]), history


def pattern_polish(objective, initial: np.ndarray) -> tuple[np.ndarray, float]:
    point = np.asarray(initial, dtype=float).copy()
    low, high = BOUNDS.T
    step = np.asarray([0.05, 0.04, 0.04, 0.08, 0.05])
    score = objective(point)
    while float(np.max(step)) > 2e-5:
        improved = False
        for dimension in range(len(point)):
            for direction in (-1.0, 1.0):
                candidate = point.copy()
                candidate[dimension] = np.clip(
                    candidate[dimension] + direction * step[dimension],
                    low[dimension] + 1e-9,
                    high[dimension] - 1e-9,
                )
                candidate_score = objective(candidate)
                if candidate_score > score:
                    point, score = candidate, float(candidate_score)
                    improved = True
        if not improved:
            step *= 0.5
    return point, float(score)


def acf_1d(values: np.ndarray, max_lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(max_lag + 1, np.nan)
    centered = values - np.mean(values)
    denominator = float(np.sum(centered**2))
    if len(values) < 3 or denominator <= 0.0:
        return result
    result[0] = 1.0
    for lag in range(1, min(max_lag + 1, len(values))):
        result[lag] = np.sum(centered[:-lag] * centered[lag:]) / denominator
    return result


def mean_acf(groups: list[pd.DataFrame], column: str, max_lag: int) -> np.ndarray:
    array = np.asarray([acf_1d(group[column].to_numpy(float), max_lag) for group in groups])
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
    division_factor_column: str = "f",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, omega2, eta, tau_m, beta = map(float, physical)
    time_acfs = []
    birth_acfs = []
    memory_acfs = []
    for group in groups:
        birth = float(group["length_birth"].iloc[0])
        memory = float(rng.normal())
        times = np.empty(len(group))
        births = np.empty(len(group))
        memories = np.empty(len(group))
        for generation in range(len(group)):
            alpha = float(group["alpha"].iloc[generation])
            times[generation] = np.nan
            births[generation] = birth
            memories[generation] = memory
            growth_time, _, memory_next = sample_cycle(
                birth,
                alpha,
                memory,
                u,
                omega2,
                eta,
                tau_m,
                beta,
                rng,
            )
            times[generation] = growth_time / alpha
            final = birth * math.exp(min(growth_time, 700.0))
            memory = memory_next
            if generation + 1 < len(group):
                birth = max(
                    float(group[division_factor_column].iloc[generation + 1])
                    * final,
                    1e-12,
                )
        time_acfs.append(acf_1d(times, max_lag))
        birth_acfs.append(acf_1d(births, max_lag))
        memory_acfs.append(acf_1d(memories, max_lag))
    return (
        np.nanmean(np.asarray(time_acfs), axis=0),
        np.nanmean(np.asarray(birth_acfs), axis=0),
        np.nanmean(np.asarray(memory_acfs), axis=0),
    )


def make_plot(
    table: pd.DataFrame,
    output_path: Path,
    model_label: str = "stationary-memory model",
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        from PIL import Image, ImageDraw

        width, height = 1100, 450
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        margin_left, margin_top, panel_width, panel_height = 64, 42, 480, 340
        y_min, y_max = -0.22, 1.04
        max_lag = float(table["lag"].max())

        def point(panel: int, lag: float, value: float) -> tuple[float, float]:
            left = margin_left + panel * (panel_width + 60)
            x = left + lag / max_lag * panel_width
            y = margin_top + (y_max - value) / (y_max - y_min) * panel_height
            return x, y

        for panel, (variable, title) in enumerate(
            (("time", "Inter-division time"), ("length_birth", "Birth size"))
        ):
            left = margin_left + panel * (panel_width + 60)
            right = left + panel_width
            bottom = margin_top + panel_height
            for value in (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                y = point(panel, 0.0, value)[1]
                draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
                draw.text((left - 38, y - 7), f"{value:.1f}", fill="black")
            for lag in (0, 5, 10, 15, 20, 25, 30):
                if lag <= max_lag:
                    x = point(panel, lag, 0.0)[0]
                    draw.line((x, bottom, x, bottom + 5), fill="black", width=1)
                    draw.text((x - 7, bottom + 9), str(lag), fill="black")
            draw.rectangle((left, margin_top, right, bottom), outline="black", width=1)
            draw.text((left + panel_width / 2 - 55, 15), title, fill="black")
            draw.text((left + panel_width / 2 - 45, bottom + 38), "lag (generations)", fill="black")
            lags = table["lag"].to_numpy(float)
            lower = table[f"predictive_q2.5_{variable}"].to_numpy(float)
            upper = table[f"predictive_q97.5_{variable}"].to_numpy(float)
            median = table[f"predictive_median_{variable}"].to_numpy(float)
            observed = table[f"observed_{variable}"].to_numpy(float)
            band = [point(panel, lag, value) for lag, value in zip(lags, lower)]
            band += [point(panel, lag, value) for lag, value in zip(lags[::-1], upper[::-1])]
            draw.polygon(band, fill=(255, 229, 204))
            draw.line(
                [point(panel, lag, value) for lag, value in zip(lags, median)],
                fill=(221, 112, 28),
                width=3,
            )
            draw.line(
                [point(panel, lag, value) for lag, value in zip(lags, observed)],
                fill=(35, 95, 165),
                width=3,
            )
        draw.text(
            (440, 420),
            f"blue: Glu data    orange: {model_label} (95% replay band)",
            fill="black",
        )
        image.save(output_path)
        print("matplotlib unavailable; wrote ACF PNG with Pillow", flush=True)
        return True

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for axis, variable, title in zip(
        axes, ("time", "length_birth"), ("Inter-division time", "Birth size")
    ):
        axis.axhline(0.0, color="0.75", linewidth=0.8)
        axis.plot(table["lag"], table[f"observed_{variable}"], "o-", label="Glu data")
        axis.plot(
            table["lag"],
            table[f"predictive_median_{variable}"],
            "o-",
            label=model_label,
        )
        axis.fill_between(
            table["lag"],
            table[f"predictive_q2.5_{variable}"],
            table[f"predictive_q97.5_{variable}"],
            alpha=0.22,
        )
        axis.set(title=title, xlabel="lag (generations)", ylabel="mean lineage ACF")
        axis.set_xlim(0, int(table["lag"].max()))
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return True


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir or (
        PROJECT_DIR / "outputs" / "model_comparison" / "Glu" / MODEL_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    real, groups = load_data()
    likelihood = SequentialLikelihood(
        groups, args.initial_memory_quadrature, args.hazard_quadrature
    )
    rng = np.random.default_rng(args.seed)

    previous = json.loads(
        (
            PROJECT_DIR
            / "outputs"
            / "model_comparison"
            / "Glu"
            / "decaying_memory_time_hazard_shared"
            / "diagnostics.json"
        ).read_text()
    )
    base_values = previous["no_memory_mle"]
    base = np.log([base_values["u"], base_values["omega2"], base_values["eta"]])
    seeds = [
        np.r_[base, math.log(tau_m), beta]
        for tau_m in (2.0, 5.0, 10.0, 25.0, 60.0, 150.0)
        for beta in (-1.5, -0.5, 0.25, 0.75, 1.5)
    ]

    print(f"Loaded {len(real)} observations in {len(groups)} Glu lineages", flush=True)
    if args.reuse_existing and (output_dir / "mle_parameters.csv").exists():
        saved = pd.read_csv(output_dir / "mle_parameters.csv").set_index("parameter")
        physical = saved.loc[PARAMETER_NAMES, "mle"].to_numpy(float)
        mle = np.r_[np.log(physical[:4]), physical[4]]
        mle_log_likelihood = likelihood(mle)
        print("Reusing saved MLE and replay", flush=True)
    else:
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
        pd.DataFrame(
            {"parameter": PARAMETER_NAMES, "mle": physical}
        ).to_csv(output_dir / "mle_parameters.csv", index=False)
    print("MLE", dict(zip(PARAMETER_NAMES, physical)), mle_log_likelihood, flush=True)

    baseline_point = np.r_[base, math.log(10.0), 0.0]
    baseline_log_likelihood = likelihood(baseline_point)
    validation_likelihood = SequentialLikelihood(groups, 7, 24)
    validation_mle_log_likelihood = validation_likelihood(mle)
    validation_baseline_log_likelihood = validation_likelihood(baseline_point)
    max_lag = min(args.acf_max_lag, max(len(group) for group in groups) - 1)
    replay_path = output_dir / "posterior_replay_acf.csv"
    if args.reuse_existing and replay_path.exists():
        table = pd.read_csv(replay_path)
    else:
        observed_time = mean_acf(groups, "time", max_lag)
        observed_birth = mean_acf(groups, "length_birth", max_lag)
        predictive_time = np.full((args.acf_replicates, max_lag + 1), np.nan)
        predictive_birth = np.full_like(predictive_time, np.nan)
        predictive_memory = np.full_like(predictive_time, np.nan)
        for replicate in range(args.acf_replicates):
            predictive_time[replicate], predictive_birth[replicate], predictive_memory[replicate] = replay_once(
                groups, physical, max_lag, rng
            )
            if (replicate + 1) % 25 == 0:
                print(f"replay {replicate + 1}/{args.acf_replicates}", flush=True)

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
        table.to_csv(replay_path, index=False)
    plot_written = make_plot(table, output_dir / "acf_comparison.png")

    mean_division_time = float(real["time"].mean())
    rho_mean_time = math.exp(-mean_division_time / physical[3])
    metrics = {}
    for name in ("time", "length_birth"):
        observed = table[f"observed_{name}"].to_numpy(float)
        median = table[f"predictive_median_{name}"].to_numpy(float)
        lower = table[f"predictive_q2.5_{name}"].to_numpy(float)
        upper = table[f"predictive_q97.5_{name}"].to_numpy(float)
        metrics[name] = {
            "mae_lags_1_to_max": float(np.nanmean(np.abs(observed[1:] - median[1:]))),
            "covered_lags": int(np.sum((observed[1:] >= lower[1:]) & (observed[1:] <= upper[1:]))),
            "lag1_observed": float(observed[1]),
            "lag1_predictive_median": float(median[1]),
            "lag2_observed": float(observed[2]),
            "lag2_predictive_median": float(median[2]),
            "lag3_observed": float(observed[3]),
            "lag3_predictive_median": float(median[3]),
        }
    diagnostics = {
        "model": MODEL_NAME,
        "sugar": "Glu",
        "n_observations": len(real),
        "n_lineages": len(groups),
        "hazard": "h(a)=alpha*omega2*(p/u_eff)^eta/(1+(p/u_eff)^eta)",
        "memory_between_divisions": "m_n(a)=m_n*exp(-a/tau_m)",
        "hazard_coupling": "u_eff(a)=u*exp(beta*m_n(a))",
        "division_update": "m_(n+1)=rho_n*m_n+sqrt(1-rho_n^2)*Phi^-1(1-exp(-E_n))",
        "rho_n": "exp(-d_n/tau_m)",
        "initial_memory": "m_0~N(0,1), marginalized by Gauss-Hermite quadrature",
        "parameter_scope": "u, omega2, eta, tau_m, beta shared by all Glu lineages",
        "mle": dict(zip(PARAMETER_NAMES, map(float, physical))),
        "mle_log_likelihood": float(mle_log_likelihood),
        "no_memory_log_likelihood_at_previous_mle": float(baseline_log_likelihood),
        "log_likelihood_improvement": float(mle_log_likelihood - baseline_log_likelihood),
        "delta_bic_memory_minus_no_memory": float(
            2.0 * math.log(len(real))
            - 2.0 * (mle_log_likelihood - baseline_log_likelihood)
        ),
        "quadrature_validation": {
            "initial_memory_order": 7,
            "hazard_order": 24,
            "mle_log_likelihood": float(validation_mle_log_likelihood),
            "no_memory_log_likelihood": float(validation_baseline_log_likelihood),
        },
        "mean_division_time_hours": mean_division_time,
        "rho_at_mean_division_time": rho_mean_time,
        "slow_correlation_length_generations": float(-1.0 / math.log(rho_mean_time)),
        "acf_plot_written": plot_written,
        "acf_metrics": metrics,
        "run_config": vars(args) | {"output_dir": str(output_dir)},
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    with (output_dir / "diagnostics.json").open("w") as stream:
        json.dump(diagnostics, stream, indent=2)
    with (output_dir / "run_config.json").open("w") as stream:
        json.dump(diagnostics["run_config"], stream, indent=2)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
