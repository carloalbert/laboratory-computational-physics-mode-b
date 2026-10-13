#!/usr/bin/env python3
"""Penalized-likelihood ACF sanity check for the hierarchical Glu time hazard."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.special import expit
from scipy.stats import t as student_t


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from alpha_scaled_hill_time_hazard import EPS, RESIDUAL_DF  # noqa: E402
from run_loglogistic_endpoint_residual import acf_1d, load_endpoint_data, mean_acf  # noqa: E402


NODES, WEIGHTS = np.polynomial.legendre.leggauss(12)
LOG_PHYSICAL_BOUNDS = np.log(np.asarray([[1e-4, 20], [.1, 30], [1e-3, 20]]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-max-lag", type=int, default=15)
    parser.add_argument("--plot-max-lag", type=int, default=30)
    parser.add_argument("--acf-weight", type=float, default=10.0)
    parser.add_argument("--validation-replicates", type=int, default=300)
    parser.add_argument("--maxiter", type=int, default=32)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4289)
    return parser.parse_args()


def cumulative_varying(s, birth, u, omega, eta):
    integration = .5 * s[:, None] * (NODES + 1)
    added = birth[:, None] * np.expm1(np.clip(integration, 0, 80))
    fraction = expit(eta[:, None] * np.log(np.maximum(added, EPS) / u[:, None]))
    return omega * .5 * s * np.sum(WEIGHTS * fraction, axis=1)


def inverse_varying(birth, threshold, u, omega, eta):
    low = np.zeros_like(birth)
    high = np.log1p(u / np.maximum(birth, EPS)) + 2 * (threshold + 2) / omega + 1
    for _ in range(10):
        mask = cumulative_varying(high, birth, u, omega, eta) < threshold
        if not np.any(mask): break
        high[mask] *= 2
    for _ in range(25):
        middle = .5 * (low + high)
        mask = cumulative_varying(middle, birth, u, omega, eta) < threshold
        low = np.where(mask, middle, low); high = np.where(mask, high, middle)
    return .5 * (low + high)


def log_likelihood(groups, effects):
    total = 0.0
    for group, physical in zip(groups, np.exp(effects)):
        u, omega, eta = physical
        s = group.alpha.to_numpy(float) * group.time.to_numpy(float)
        birth = group.length_birth.to_numpy(float)
        added = birth * np.expm1(np.clip(s, 0, 80))
        log_fraction = -np.logaddexp(0, -eta * np.log(np.maximum(added, EPS) / u))
        cumulative = cumulative_varying(
            s, birth, np.full(len(s), u), np.full(len(s), omega), np.full(len(s), eta)
        )
        total += np.sum(np.log(group.alpha.to_numpy(float)) + math.log(omega) + log_fraction - cumulative)
    return float(total)


def simulate(groups, effects, residual, thresholds, uniforms, max_lag):
    n, max_length = thresholds.shape
    births = np.full((n, max_length), np.nan); times = np.full_like(births, np.nan)
    current = np.asarray([g.length_birth.iloc[0] for g in groups], float)
    physical = np.exp(effects)
    mu, sigma = residual
    lengths = np.asarray([len(g) for g in groups])
    for generation in range(max_length):
        active = lengths > generation
        if not np.any(active): break
        s = inverse_varying(current[active], thresholds[active, generation],
                            physical[active, 0], physical[active, 1], physical[active, 2])
        lower_p = student_t.cdf((-s - mu) / sigma, RESIDUAL_DF)
        probability = np.clip(lower_p + uniforms[active, generation] * (1 - lower_p), 1e-12, 1-1e-12)
        epsilon = mu + sigma * student_t.ppf(probability, RESIDUAL_DF)
        final = current[active] * np.exp(np.clip(s + epsilon, -50, 50))
        births[active, generation] = current[active]
        alpha = np.asarray([groups[i].alpha.iloc[generation] for i in np.flatnonzero(active)])
        times[active, generation] = s / alpha
        continuing = lengths > generation + 1
        if np.any(continuing):
            fractions = np.asarray([groups[i].f.iloc[generation + 1] for i in np.flatnonzero(continuing)])
            current[continuing] = fractions * final[np.isin(np.flatnonzero(active), np.flatnonzero(continuing))]
    curves = {}
    for name, array in [("time", times), ("length_birth", births)]:
        pieces = [acf_1d(array[i, :length], max_lag) for i, length in enumerate(lengths)]
        curves[name] = np.nanmean(np.asarray(pieces), axis=0)
    return curves


def transform(values, base, base_mean):
    mean = values[:3]; scale = np.exp(values[3:])
    return mean + (base - base_mean) * scale


def main():
    args = parse_args(); started = time.time()
    output_dir = (
        PROJECT_DIR / "outputs" / "model_comparison" / "Glu"
        / "hierarchical_alpha_scaled_time_hazard_all_parameters"
        / f"acf_penalized_optimization_weight_{args.acf_weight:g}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    real, _, _, _ = load_endpoint_data("Glu")
    groups = [g.sort_values("generation").reset_index(drop=True) for _, g in real.groupby("lineage", sort=False) if len(g) >= 3]
    labels = np.asarray([g.lineage.iloc[0] for g in groups])
    lineage = pd.read_csv(output_dir.parent / "lineage_parameter_posterior.csv").set_index("lineage")
    base = np.log(lineage.loc[labels, ["u_median", "omega2_median", "eta_median"]].to_numpy(float))
    base_mean = np.mean(base, axis=0)
    residual_summary = pd.read_csv(PROJECT_DIR / "outputs" / "model_comparison" / "Glu" / "alpha_scaled_hill_time_hazard" / "posterior_summary.csv").set_index("parameter")
    residual = residual_summary.loc[["mu_epsilon", "sigma_epsilon"], "median"].to_numpy(float)
    max_length = max(len(g) for g in groups); rng = np.random.default_rng(args.seed)
    training_thresholds = rng.exponential(size=(len(groups), max_length)); training_uniforms = rng.uniform(size=(len(groups), max_length))
    observed = {name: mean_acf(real, name, args.plot_max_lag) for name in ["time", "length_birth"]}
    initial = np.r_[base_mean, np.zeros(3)]
    baseline_effects = transform(initial, base, base_mean)
    baseline_ll = log_likelihood(groups, baseline_effects)
    baseline_curves = simulate(groups, baseline_effects, residual, training_thresholds, training_uniforms, args.fit_max_lag)
    lag_slice = slice(1, args.fit_max_lag + 1)
    baseline_mse = sum(np.mean((baseline_curves[name][lag_slice] - observed[name][lag_slice])**2) for name in observed)
    n_obs = sum(len(g) for g in groups)
    history = []
    def components(values):
        effects = transform(values, base, base_mean)
        ll = log_likelihood(groups, effects)
        curves = simulate(groups, effects, residual, training_thresholds, training_uniforms, args.fit_max_lag)
        mse = sum(np.mean((curves[name][lag_slice] - observed[name][lag_slice])**2) for name in observed)
        objective = (baseline_ll - ll) / n_obs + args.acf_weight * mse / baseline_mse
        return objective, ll, mse
    def objective(values):
        try: return components(values)[0]
        except (FloatingPointError, ValueError, OverflowError): return 1e6
    def callback(values, convergence):
        obj,ll,mse=components(values); history.append({"iteration":len(history)+1,"objective":obj,"log_likelihood":ll,"acf_mse":mse,"convergence":convergence})
        return False
    result = differential_evolution(objective,
        bounds=[tuple(row) for row in LOG_PHYSICAL_BOUNDS] + [(math.log(.15),math.log(4))]*3,
        x0=initial, seed=args.seed+1, maxiter=args.maxiter, popsize=args.popsize,
        tol=.004, polish=True, updating="immediate", callback=callback)
    optimized_effects = transform(result.x, base, base_mean)
    optimized_obj, optimized_ll, optimized_mse = components(result.x)
    pd.DataFrame(history).to_csv(output_dir / "optimization_history.csv", index=False)
    pd.DataFrame(np.exp(optimized_effects), columns=["u","omega2","eta"]).assign(lineage=labels).to_csv(output_dir / "optimized_lineage_parameters.csv", index=False)

    validation_thresholds = rng.exponential(size=(args.validation_replicates,len(groups),max_length))
    validation_uniforms = rng.uniform(size=validation_thresholds.shape)
    matrices = {fit:{name:np.full((args.validation_replicates,args.plot_max_lag+1),np.nan) for name in observed} for fit in ["baseline","optimized"]}
    for replicate in range(args.validation_replicates):
        for fit,effects in [("baseline",baseline_effects),("optimized",optimized_effects)]:
            curves=simulate(groups,effects,residual,validation_thresholds[replicate],validation_uniforms[replicate],args.plot_max_lag)
            for name in observed: matrices[fit][name][replicate]=curves[name]
    table={"lag":np.arange(args.plot_max_lag+1)}; metrics=[]
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.8))
    for axis,name,title in zip(axes,observed,["Division-time ACF","Birth-size ACF"]):
        for fit,color,label in [("baseline","tab:orange","Likelihood hierarchy"),("optimized","tab:green","Likelihood + ACF penalty")]:
            med=np.nanmedian(matrices[fit][name],axis=0); lo,hi=np.nanpercentile(matrices[fit][name],[2.5,97.5],axis=0)
            table[f"{fit}_median_{name}"]=med; table[f"{fit}_q2.5_{name}"]=lo; table[f"{fit}_q97.5_{name}"]=hi
            axis.fill_between(table["lag"],lo,hi,color=color,alpha=.13); axis.plot(table["lag"],med,color=color,lw=1.8,label=label)
            metrics.append({"fit":fit,"variable":name,"mae_lags_1_30":float(np.mean(abs(med[1:]-observed[name][1:]))),"rmse_lags_1_15":float(np.sqrt(np.mean((med[1:args.fit_max_lag+1]-observed[name][1:args.fit_max_lag+1])**2)))})
        table[f"observed_{name}"]=observed[name]; axis.plot(table["lag"],observed[name],color="black",lw=2,label="Glu data"); axis.axhline(0,color=".6",lw=.8); axis.set(title=title,xlabel="Lag (generations)",ylabel="Mean lineage ACF")
    axes[0].legend(frameon=False); fig.tight_layout(); fig.savefig(output_dir/"acf_penalized_comparison.png",dpi=200); fig.savefig(output_dir/"acf_penalized_comparison.pdf"); plt.close(fig)
    pd.DataFrame(table).to_csv(output_dir/"acf_penalized_comparison.csv",index=False); pd.DataFrame(metrics).to_csv(output_dir/"acf_penalized_metrics.csv",index=False)
    diagnostics={"purpose":"penalized-likelihood structural sanity check, not posterior inference","parameterization":"six calibration parameters: three population log-medians and three multipliers of fitted lineage log-deviations","objective":"(baseline_logL-logL)/N + acf_weight*(time_ACF_MSE+birth_ACF_MSE)/baseline_ACF_MSE","acf_weight":args.acf_weight,"baseline_log_likelihood":baseline_ll,"optimized_log_likelihood":optimized_ll,"log_likelihood_change":optimized_ll-baseline_ll,"baseline_training_acf_mse":baseline_mse,"optimized_training_acf_mse":optimized_mse,"acf_mse_ratio":optimized_mse/baseline_mse,"optimizer":{"success":bool(result.success),"message":str(result.message),"evaluations":int(result.nfev),"objective":optimized_obj},"optimized_population_median":dict(zip(["u","omega2","eta"],np.exp(result.x[:3]))),"lineage_deviation_multipliers":dict(zip(["u","omega2","eta"],np.exp(result.x[3:]))),"metrics":metrics,"runtime_minutes":(time.time()-started)/60,"run_config":vars(args)}
    with (output_dir/"diagnostics.json").open("w") as handle: json.dump(diagnostics,handle,indent=2)
    print(json.dumps(diagnostics,indent=2))

if __name__ == "__main__": main()
