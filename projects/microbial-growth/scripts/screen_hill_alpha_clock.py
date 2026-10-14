#!/usr/bin/env python3
"""Exploratory Glu Hill-hazard fits with a free growth-rate clock exponent.

Fits all observations, so these results are not a held-out model comparison.
Conditional replay resets at missing-generation gaps, but uses original
lineage centering and weighting for ACFs, omitting pairs across gaps.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_birth_size_dependent_time_hazard import Likelihood, hill_fraction
from screen_parsimonious_joint_hazards import FastReplay, METRICS, OUT, load_data, segments


OUTPUT_DIR = OUT / "alpha_clock"


class ClockLikelihood:
    """Profile omega analytically for each (log u, log eta, q, kappa)."""

    def __init__(self, data, bstar, astar, order=24):
        self.base = Likelihood(data, bstar, order)
        self.loga = np.log(self.base.alpha / astar)
        self.n = len(data)

    def __call__(self, point, return_omega=False):
        log_u, log_eta, q = point[:3]
        kappa = point[3] if len(point) > 3 else 0.0
        eta = np.exp(log_eta)
        base = self.base
        odds = (
            eta * (base.integration_log_added - log_u)
            + kappa * base.log_birth_ratio[:, None]
        )
        integral = base.integration_scale * np.sum(
            base.weights * hill_fraction(odds), axis=1
        )
        alpha_multiplier = np.exp(q * self.loga)
        omega = self.n / np.sum(alpha_multiplier * integral)
        event = (
            eta * (base.event_log_added - log_u)
            + kappa * base.log_birth_ratio
        )
        score = float(np.sum(
            base.log_alpha + np.log(omega) + q * self.loga
            - np.logaddexp(0.0, -event)
        ) - self.n)
        return (score, float(omega)) if return_omega else score


def fit(likelihood, free_kappa, seed):
    bounds = np.array([
        [np.log(0.15), np.log(6.0)],
        [np.log(0.5), np.log(50.0)],
        [-3.0, 3.0],
    ] + ([[-15.0, 15.0]] if free_kappa else []))
    rng = np.random.default_rng(seed)
    dimensions = len(bounds)
    low, high = bounds.T
    population = rng.uniform(low, high, (36, dimensions))
    population[0] = [np.log(1.449), np.log(8.32), 0.0] + (
        [-0.155] if free_kappa else []
    )
    scores = np.array([likelihood(x) for x in population])
    history = []
    for iteration in range(75):
        for index in range(len(population)):
            first, second, third = rng.choice(
                np.delete(np.arange(len(population)), index), 3, replace=False
            )
            mutant = np.clip(
                population[first] + 0.7 * (population[second] - population[third]),
                low, high,
            )
            cross = rng.random(dimensions) < 0.85
            cross[rng.integers(dimensions)] = True
            candidate = np.where(cross, mutant, population[index])
            score = likelihood(candidate)
            if score > scores[index]:
                population[index], scores[index] = candidate, score
        history.append(float(np.max(scores)))
    point = population[scores.argmax()].copy()
    score = likelihood(point)
    step = np.array([0.05, 0.08, 0.1] + ([0.15] if free_kappa else []))
    for _ in range(2500):
        improved = False
        for dimension in range(dimensions):
            for sign in (-1.0, 1.0):
                candidate = point.copy()
                candidate[dimension] = np.clip(
                    candidate[dimension] + sign * step[dimension],
                    low[dimension], high[dimension],
                )
                value = likelihood(candidate)
                if value > score:
                    point, score, improved = candidate, value, True
        if not improved:
            step *= 0.5
        if step.max() < 1e-5:
            break
    score, omega = likelihood(point, return_omega=True)
    physical = {
        "u": float(np.exp(point[0])), "eta": float(np.exp(point[1])),
        "omega2": omega, "q": float(point[2]),
        "kappa": float(point[3]) if free_kappa else 0.0,
        "log_likelihood": score,
    }
    return physical, history


def exact_replay(replay, physical, seed=223):
    """Thinning in growth time with cell-specific dominating rate."""
    rng = np.random.default_rng(seed)
    shape = (replay.R, *replay.mask.shape)
    births = np.zeros(shape)
    times = np.zeros(shape)
    current = np.broadcast_to(replay.birth0, shape[:2]).copy()
    for generation in range(replay.mask.shape[1]):
        active = replay.mask[:, generation]
        birth = current[:, active]
        alpha = replay.alpha[active, generation][None, :]
        rate = physical["omega2"] * (alpha / replay.astar) ** physical["q"]
        rate = np.broadcast_to(rate, birth.shape)
        growth_time = np.zeros_like(birth)
        pending = np.ones_like(birth, dtype=bool)
        birth_term = physical["kappa"] * np.log(birth / replay.bstar)
        for _ in range(100_000):
            if not pending.any():
                break
            growth_time[pending] += rng.exponential(size=pending.sum()) / rate[pending]
            odds = physical["eta"] * (
                np.log(birth[pending])
                + np.log(np.expm1(np.minimum(growth_time[pending], 700.0)))
                - np.log(physical["u"])
            ) + birth_term[pending]
            accepted = rng.random(pending.sum()) < np.exp(-np.logaddexp(0.0, -odds))
            pending[pending] = ~accepted
        else:
            raise RuntimeError("Thinning failed to finish")
        births[:, active, generation] = birth
        times[:, active, generation] = growth_time / alpha
        if generation + 1 < replay.mask.shape[1]:
            current[:, active] = (
                birth * np.exp(growth_time) * replay.f[active, generation + 1]
            )
    return births, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--reuse-fit", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, original = load_data()
    groups = segments(original)
    bstar = float(np.exp(np.log(data.length_birth).mean()))
    astar = float(np.exp(np.log(data.alpha).mean()))
    likelihood = ClockLikelihood(data, bstar, astar)
    if args.reuse_fit:
        result = json.loads((args.output_dir / "fit_results.json").read_text())
    else:
        result = {
            "hazard": "alpha*omega2*(alpha/alpha_star)^q*sigmoid(eta*log(p/u)+kappa*log(b/b_star))",
            "inference": "All-data exploratory maximum likelihood; not held out",
            "replay_design": "Reset at gaps; original-lineage ACF centering and weighting; pairs across gaps omitted; recorded alpha and effective factors replayed",
            "b_star": bstar, "alpha_star": astar,
            "n_observations": len(data), "n_lineages": len(original),
            "n_contiguous_segments": len(groups), "fits": {},
        }
        # Previously verified nested maximum-likelihood fit, for comparison.
        result["fits"]["hill_baseline_3p"] = {
            "u": 1.44895, "omega2": 11.0223, "eta": 8.3202,
            "q": 0.0, "kappa": 0.0, "log_likelihood": 1263.5704176636,
        }
        for free_kappa in (False, True):
            name = "hill_alpha_clock_5p" if free_kappa else "hill_alpha_clock_4p"
            physical, history = fit(likelihood, free_kappa, 986 if free_kappa else 985)
            result["fits"][name] = physical
            pd.DataFrame({"iteration": np.arange(1, 76), "best_log_likelihood": history}).to_csv(
                args.output_dir / f"{name}_optimizer_history.csv", index=False
            )
            print(name, physical, flush=True)
    replay = FastReplay(groups, bstar, astar, replicates=args.replicates, seed=223)
    observed, observed_births, observed_times = replay.observed()
    likelihood48 = ClockLikelihood(data, bstar, astar, order=48)
    metrics_rows, acf_rows = [], []
    for name, physical in result["fits"].items():
        if physical["q"] != 0.0:
            point = np.array([np.log(physical["u"]), np.log(physical["eta"]), physical["q"]] + (
                [physical["kappa"]] if name.endswith("5p") else []
            ))
            physical["quadrature48_profile_log_likelihood"] = likelihood48(point)
        births, times = exact_replay(replay, physical)
        metrics = replay.statistics(births, times)
        for index, metric in enumerate(METRICS):
            metrics_rows.append({
                "model": name, "metric": metric, "observed": observed[index],
                "predictive_median": np.median(metrics[:, index]),
                "predictive_q2.5": np.percentile(metrics[:, index], 2.5),
                "predictive_q97.5": np.percentile(metrics[:, index], 97.5),
            })
        for variable, values, real in (
            ("length_birth", births, observed_births),
            ("time", times, observed_times),
        ):
            acf, real_acf = replay.acfs(values), replay.acfs(real)[0]
            for lag in range(31):
                acf_rows.append({
                    "model": name, "variable": variable, "lag": lag,
                    "observed": real_acf[lag],
                    "predictive_median": np.median(acf[:, lag]),
                    "predictive_q2.5": np.percentile(acf[:, lag], 2.5),
                    "predictive_q97.5": np.percentile(acf[:, lag], 97.5),
                })
        print("Validated", name, flush=True)
    result["replicates"] = args.replicates
    result["runtime_seconds"] = time.time() - started
    (args.output_dir / "fit_results.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "joint_metrics.csv", index=False)
    pd.DataFrame(acf_rows).to_csv(args.output_dir / "acf_comparison.csv", index=False)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
