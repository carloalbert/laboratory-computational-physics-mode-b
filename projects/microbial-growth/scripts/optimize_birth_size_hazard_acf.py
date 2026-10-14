#!/usr/bin/env python3
"""ACF-only calibration of the four-parameter Glu hazard, without memory.

Common exponential thresholds make training deterministic. A tabulated
integrated hazard accelerates inverse sampling; fresh exact-thinning replays
provide the final validation. Historical retained-row pairing is preserved
for direct comparison, including the 50 gaps in generation indices.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_birth_size_dependent_time_hazard import (
    PROJECT_DIR, PARAMETER_NAMES, BOUNDS, Likelihood, unpack,
    load_data, add_effective_factors, mean_acf, make_plot,
)

OUT = PROJECT_DIR / 'outputs/model_comparison/Glu/birth_size_dependent_time_hazard_acf_only'


class Replay:
    def __init__(self, groups, bstar, replicates=6, seed=7331, fine=False):
        self.lengths = np.array([len(g) for g in groups])
        self.mask = np.arange(self.lengths.max())[None, :] < self.lengths[:, None]
        self.birth0 = np.array([g.length_birth.iloc[0] for g in groups])
        self.alpha = np.ones(self.mask.shape)
        self.f = np.ones(self.mask.shape)
        for i, g in enumerate(groups):
            self.alpha[i, :len(g)] = g.alpha
            self.f[i, 1:len(g)] = g.f_effective.iloc[1:]
        self.bstar = bstar
        self.R = replicates
        self.E = np.random.default_rng(seed).exponential(size=(replicates, *self.mask.shape))
        self.dq = .025 if fine else .075
        self.qgrid = np.arange(-20., 20. + self.dq / 2, self.dq)
        ds = .0025 if fine else .0075
        self.sgrid = np.r_[np.arange(0, 4, ds), np.arange(4, 40.001, ds * 4)]
        self.logp = np.log(np.maximum(np.expm1(self.sgrid), 1e-300))

    def table(self, eta):
        z = eta * (self.logp[None, :] - self.qgrid[:, None])
        fraction = np.exp(-np.logaddexp(0., -z))
        area = .5 * (fraction[:, 1:] + fraction[:, :-1]) * np.diff(self.sgrid)
        return np.pad(np.cumsum(area, axis=1), ((0, 0), (1, 0)))

    def inverse(self, q, e, table):
        if np.any(q < self.qgrid[0]) or np.any(q >= self.qgrid[-1]):
            raise FloatingPointError('outside inverse grid')
        qi = (q - self.qgrid[0]) / self.dq
        i = np.floor(qi).astype(int)
        w = qi - i
        def at(j):
            return (1 - w) * table[i, j] + w * table[i + 1, j]
        lo = np.zeros(q.shape, dtype=int)
        hi = np.full(q.shape, len(self.sgrid) - 1, dtype=int)
        if np.any(at(hi) < e):
            raise FloatingPointError('outside growth-time grid')
        for _ in range(int(np.ceil(np.log2(len(self.sgrid))))):
            mid = (lo + hi) // 2
            below = at(mid) < e
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        a, b = at(lo), at(hi)
        w_s = (e - a) / np.maximum(b - a, 1e-300)
        return self.sgrid[lo] + w_s * (self.sgrid[hi] - self.sgrid[lo])

    def acfs(self, values, lag=30):
        centered = np.where(self.mask[None], values - (
            np.sum(values, axis=2) / self.lengths[None])[:, :, None], 0.)
        den = np.sum(centered**2, axis=2)
        result = np.full((self.R, lag + 1), np.nan)
        result[:, 0] = 1.
        for k in range(1, lag + 1):
            valid = (self.lengths > k) & (self.lengths >= 3)
            numer = np.sum(centered[:, :, :-k] * centered[:, :, k:], axis=2)
            result[:, k] = np.mean(numer[:, valid] / den[:, valid], axis=1)
        return result

    def simulate(self, physical, lag=30):
        u, omega, eta, kappa = physical
        table = self.table(eta)
        shape = (self.R, *self.mask.shape)
        births = np.zeros(shape)
        times = np.zeros(shape)
        current = np.broadcast_to(self.birth0, (self.R, len(self.birth0))).copy()
        for n in range(self.mask.shape[1]):
            active = self.mask[:, n]
            b = current[:, active]
            q = np.log(u / b) - (kappa / eta) * np.log(b / self.bstar)
            s = self.inverse(q, self.E[:, active, n] / omega, table)
            births[:, active, n] = b
            times[:, active, n] = s / self.alpha[active, n]
            if n + 1 < self.mask.shape[1]:
                current[:, active] = b * np.exp(s) * self.f[active, n + 1]
        self.last_marginals = {
            'mean_birth': float(births[:, self.mask].mean()),
            'sd_birth': float(births[:, self.mask].std()),
            'mean_time': float(times[:, self.mask].mean()),
            'sd_time': float(times[:, self.mask].std()),
        }
        return self.acfs(births, lag), self.acfs(times, lag)

    def exact_thinning(self, physical, seed, lag=30, return_times=False):
        """Vectorized exact event sampler, independent of inverse-table training."""
        u, omega, eta, kappa = physical
        rng = np.random.default_rng(seed)
        births = np.zeros((self.R, *self.mask.shape))
        times = np.zeros_like(births)
        current = np.broadcast_to(self.birth0, (self.R, len(self.birth0))).copy()
        for n in range(self.mask.shape[1]):
            active = self.mask[:, n]
            b = current[:, active]
            s = np.zeros_like(b)
            pending = np.ones(b.shape, dtype=bool)
            offset = eta * np.log(b / u) + kappa * np.log(b / self.bstar)
            for _ in range(100000):
                if not pending.any():
                    break
                s[pending] += rng.exponential(1 / omega, size=pending.sum())
                z = offset[pending] + eta * np.log(np.expm1(np.minimum(s[pending], 700)))
                accepted = rng.random(pending.sum()) < np.exp(-np.logaddexp(0., -z))
                pending[pending] = ~accepted
            else:
                raise RuntimeError('thinning did not finish')
            births[:, active, n] = b
            times[:, active, n] = s / self.alpha[active, n]
            if n + 1 < self.mask.shape[1]:
                current[:, active] = b * np.exp(s) * self.f[active, n + 1]
        if return_times:
            return times[:, self.mask]
        return self.acfs(births, lag), self.acfs(times, lag)


def optimize(objective, start, seed, generations=60, popsize=32):
    rng = np.random.default_rng(seed)
    low, high = BOUNDS.T
    pop = rng.uniform(low + 1e-6, high - 1e-6, (popsize, 4))
    pop[0] = start
    for i, kappa in enumerate((-8., -4., -2., 0., 2., 4., 8.), 1):
        pop[i] = start.copy()
        pop[i, 3] = kappa
    pop[8] = np.clip(np.r_[np.log([8., 25., 20.]), 3.72], low+1e-6, high-1e-6)
    scores = np.array([objective(p) for p in pop])
    history = []
    for gen in range(generations):
        for i in range(popsize):
            a, b, c = rng.choice(np.delete(np.arange(popsize), i), 3, replace=False)
            trial = np.clip(pop[a] + .7 * (pop[b] - pop[c]), low + 1e-6, high - 1e-6)
            cross = rng.random(4) < .85
            cross[rng.integers(4)] = True
            trial = np.where(cross, trial, pop[i])
            score = objective(trial)
            if score < scores[i]:
                pop[i], scores[i] = trial, score
        best = int(np.argmin(scores))
        history.append({'generation': gen + 1, 'mse': scores[best]})
        if (gen + 1) % 10 == 0:
            print(f'seed {seed}: generation {gen+1}, MSE {scores[best]:.7g}, parameters {unpack(pop[best])}', flush=True)
    return pop[np.argmin(scores)].copy(), history


def polish(obj, initial):
    x = initial.copy()
    score = obj(x)
    step = np.array([.06, .06, .06, .15])
    for _ in range(160):
        improved = False
        for j in range(4):
            for direction in (-1, 1):
                z = x.copy()
                z[j] += direction * step[j]
                if np.any(z <= BOUNDS[:, 0]) or np.any(z >= BOUNDS[:, 1]):
                    continue
                val = obj(z)
                if val < score:
                    x, score, improved = z, val, True
        if not improved:
            step *= .5
        if max(step) < .002:
            break
    return x, score


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generations', type=int, default=60)
    parser.add_argument('--replicates', type=int, default=200)
    parser.add_argument('--fit-max-lag', type=int, default=30)
    parser.add_argument('--wide-bounds', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reuse-fit', action='store_true')
    args = parser.parse_args()
    if args.wide_bounds:
        BOUNDS[:] = [[math.log(.01), math.log(100.)],
                     [math.log(.15), math.log(200.)],
                     [math.log(.1), math.log(100.)], [-60., 60.]]
    if args.output_dir:
        OUT = args.output_dir
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data, groups = load_data()
    groups, _, _ = add_effective_factors(groups)
    bstar = float(np.exp(np.log(data.length_birth).mean()))
    target = mean_acf(groups, 'length_birth', 30)
    observed_time = mean_acf(groups, 'time', 30)
    source = PROJECT_DIR / 'outputs/model_comparison/Glu/birth_size_dependent_time_hazard_effective_f'
    previous = json.loads((source / 'diagnostics.json').read_text())
    initial = np.array([previous['mle'][n] for n in PARAMETER_NAMES])
    start = np.r_[np.log(initial[:3]), initial[3]]
    def objective(replay):
        def obj(x):
            try:
                birth, _ = replay.simulate(unpack(x))
                value = float(np.mean((np.mean(birth, axis=0)[1:args.fit_max_lag+1] - target[1:args.fit_max_lag+1])**2))
                return value if np.isfinite(value) else 1e6
            except (FloatingPointError, OverflowError):
                return 1e6
        return obj
    train = Replay(groups, bstar, replicates=6, seed=7331)
    obj = objective(train)
    print('likelihood-fit training MSE', obj(start), flush=True)
    refine = Replay(groups, bstar, replicates=24, seed=8342)
    refined_obj = objective(refine)
    if args.reuse_fit:
        physical_saved = pd.read_csv(OUT / 'parameters.csv').set_index('parameter').loc[PARAMETER_NAMES, 'acf_optimum'].to_numpy()
        chosen = np.r_[np.log(physical_saved[:3]), physical_saved[3]]
        score = refined_obj(chosen)
    else:
        candidates, histories = [], []
        for seed in (919, 2029):
            x, history = optimize(obj, start, seed, generations=args.generations)
            candidates.append(x)
            histories.extend([dict(row, seed=seed) for row in history])
        pd.DataFrame(histories).to_csv(OUT / 'optimizer_history.csv', index=False)
        chosen = min(candidates, key=refined_obj)
        chosen, score = polish(refined_obj, chosen)
    physical = unpack(chosen)
    print('ACF optimum', physical, 'refined MSE', score, flush=True)
    pd.DataFrame({'parameter': PARAMETER_NAMES, 'acf_optimum': physical, 'likelihood_fit': initial}).to_csv(OUT / 'parameters.csv', index=False)
    # Fine inverse-table check uses the same thresholds to isolate integration error.
    fine = Replay(groups, bstar, replicates=24, seed=8342, fine=True)
    fine_birth, _ = fine.simulate(physical)
    marginals = fine.last_marginals
    coarse_birth, _ = refine.simulate(physical)
    error = float(np.max(np.abs(fine_birth - coarse_birth)))
    print('fine-grid max ACF difference', error, flush=True)
    # Independent exact thinning: no interpolation or shared training thresholds.
    birth_predictions, time_predictions = [], []
    for i in range(0, args.replicates, 10):
        batch = Replay(groups, bstar, replicates=min(10, args.replicates-i))
        b, t = batch.exact_thinning(physical, 982344 + i)
        birth_predictions.extend(b)
        time_predictions.extend(t)
        if (i + len(b)) % 50 == 0:
            print(f'fresh exact-thinning validation {i+len(b)}/{args.replicates}', flush=True)
    table = pd.DataFrame({'lag': np.arange(31)})
    metrics = {}
    for name, obs, samples in [('length_birth', target, birth_predictions), ('time', observed_time, time_predictions)]:
        samples = np.array(samples)
        med = np.median(samples, axis=0)
        lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
        table[f'observed_{name}'] = obs
        table[f'predictive_median_{name}'] = med
        table[f'predictive_q2.5_{name}'] = lo
        table[f'predictive_q97.5_{name}'] = hi
        metrics[name] = {'mse': float(np.mean((med[1:] - obs[1:])**2)),
                         'mae': float(np.mean(np.abs(med[1:] - obs[1:]))),
                         'covered_lags': int(np.sum((obs[1:] >= lo[1:]) & (obs[1:] <= hi[1:]))),
                         'predicted_lags_1_to_8': med[1:9].tolist(), 'observed_lags_1_to_8': obs[1:9].tolist()}
    old = pd.read_csv(source / 'effective_f_replay_acf.csv')
    table['likelihood_fit_birth'] = old.predictive_median_length_birth
    table.to_csv(OUT / 'acf_comparison.csv', index=False)
    make_plot(table, OUT / 'acf_comparison.png', model_label='ACF-fitted p,b hazard')
    # Use higher-order quadrature for the likelihood cost, excluding it from fitting.
    ll = Likelihood(data, bstar, order=64)
    result = {'parameters': dict(zip(PARAMETER_NAMES, physical.tolist())),
              'b_star': bstar, 'objective': f'MSE of mean-lineage birth-size ACF lags 1..{args.fit_max_lag}; no likelihood term',
              'bounds_transformed': BOUNDS.tolist(), 'training_MSE': score,
              'validation': metrics, 'likelihood_fit_birth_MSE': float(np.mean((old.predictive_median_length_birth.to_numpy()[1:] - target[1:])**2)),
              'likelihood_fit_birth_MAE': float(np.mean(np.abs(old.predictive_median_length_birth.to_numpy()[1:] - target[1:]))),
              'log_likelihood_at_acf_fit': ll(chosen), 'log_likelihood_at_likelihood_fit': ll(start),
              'fine_grid_max_acf_difference': error,
              'acf_fit_marginals_fine_replay': marginals,
              'observed_marginals': {'mean_birth': float(data.length_birth.mean()),
                                     'sd_birth': float(data.length_birth.std(ddof=0)),
                                     'mean_time': float(data.time.mean()),
                                     'sd_time': float(data.time.std(ddof=0))},
              'generation_gaps_bridged_for_historical_comparability': int(sum(np.sum(np.diff(g.generation) != 1) for g in groups)),
              'validation_replicates': args.replicates, 'optimization_seeds': [919, 2029],
              'training_threshold_seed': 7331, 'refinement_threshold_seed': 8342,
              'validation_seed': 982344, 'runtime_minutes': (time.time() - started) / 60}
    (OUT / 'diagnostics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
