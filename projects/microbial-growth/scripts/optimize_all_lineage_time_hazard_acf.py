#!/usr/bin/env python3
"""Optimize separate (u, omega2, eta) for every Glu lineage using only birth-ACF MSE."""

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
from scipy.optimize import minimize
from scipy.stats import t as student_t


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from alpha_scaled_hill_time_hazard import EPS, RESIDUAL_DF  # noqa: E402
from optimize_hierarchical_time_hazard_acf_penalty import (  # noqa: E402
    LOG_PHYSICAL_BOUNDS, inverse_varying,
)
from run_loglogistic_endpoint_residual import acf_1d, load_endpoint_data, mean_acf  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-max-lag", type=int, default=30)
    parser.add_argument("--training-replicates", type=int, default=3)
    parser.add_argument("--validation-replicates", type=int, default=300)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--maxiter-per-lineage", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4319)
    return parser.parse_args()


def simulate_lineage_birth_acfs(group, log_params, residual, thresholds, uniforms, max_lag):
    u, omega, eta = np.exp(log_params)
    mu, sigma = residual
    curves = np.full((len(thresholds), max_lag + 1), np.nan)
    for replicate in range(len(thresholds)):
        birth = float(group.length_birth.iloc[0]); births = np.empty(len(group))
        valid = True
        for generation in range(len(group)):
            births[generation] = birth
            s = inverse_varying(
                np.asarray([birth]), np.asarray([thresholds[replicate, generation]]),
                np.asarray([u]), np.asarray([omega]), np.asarray([eta]),
            )[0]
            lower = student_t.cdf((-s - mu) / sigma, RESIDUAL_DF)
            probability = np.clip(lower + uniforms[replicate, generation] * (1 - lower), 1e-12, 1-1e-12)
            epsilon = mu + sigma * student_t.ppf(probability, RESIDUAL_DF)
            final = birth * math.exp(float(np.clip(s + epsilon, -50, 50)))
            if generation + 1 < len(group):
                birth = max(float(group.f.iloc[generation + 1]) * final, EPS)
            if not np.isfinite(birth) or birth > 1e100:
                valid = False
                break
        if valid:
            curves[replicate] = acf_1d(births, max_lag)
    return curves


def main():
    args = parse_args(); started = time.time()
    output_dir = PROJECT_DIR / "outputs" / "model_comparison" / "Glu" / "hierarchical_alpha_scaled_time_hazard_all_parameters" / "all_lineages_birth_acf_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    real, _, _, _ = load_endpoint_data("Glu")
    all_groups = [g.sort_values("generation").reset_index(drop=True) for _,g in real.groupby("lineage",sort=False)]
    groups = [g for g in all_groups if len(g) >= 3]
    labels = np.asarray([g.lineage.iloc[0] for g in groups])
    max_lag = min(args.fit_max_lag, max(len(g) for g in groups)-1)
    observed = mean_acf(real,"length_birth",max_lag)
    residual_summary = pd.read_csv(PROJECT_DIR/"outputs"/"model_comparison"/"Glu"/"alpha_scaled_hill_time_hazard"/"posterior_summary.csv").set_index("parameter")
    residual = residual_summary.loc[["mu_epsilon","sigma_epsilon"],"median"].to_numpy(float)
    resume_path = output_dir/"optimized_lineage_parameters.csv"
    start_path = (
        resume_path if resume_path.exists()
        else output_dir.parent/"acf_penalized_optimization_weight_1000"/"optimized_lineage_parameters.csv"
    )
    start = pd.read_csv(start_path).set_index("lineage")
    params = np.log(start.loc[labels,["u","omega2","eta"]].to_numpy(float))
    max_length=max(len(g) for g in groups); rng=np.random.default_rng(args.seed)
    thresholds=rng.exponential(size=(len(groups),args.training_replicates,max_length))
    uniforms=rng.uniform(size=thresholds.shape)
    curves=[]
    for i,g in enumerate(groups):
        curves.append(simulate_lineage_birth_acfs(g,params[i],residual,thresholds[i,:,:len(g)],uniforms[i,:,:len(g)],max_lag))
    curves=np.asarray(curves)
    counts=np.sum(np.isfinite(curves),axis=0)
    sums=np.nansum(curves,axis=0)
    fit_slice=slice(1,max_lag+1)
    def total_curve(local_sums): return np.nanmean(np.divide(local_sums,counts,out=np.full_like(local_sums,np.nan),where=counts>0),axis=0)
    def mse(local_sums):
        difference=total_curve(local_sums)[fit_slice]-observed[fit_slice]
        return float(np.nanmean(difference**2))
    initial_mse=mse(sums); history=[{"sweep":0,"mse":initial_mse}]
    for sweep in range(1,args.sweeps+1):
        for i in rng.permutation(len(groups)):
            old=curves[i].copy()
            def objective(candidate):
                try:
                    new=simulate_lineage_birth_acfs(groups[i],candidate,residual,thresholds[i,:,:len(groups[i])],uniforms[i,:,:len(groups[i])],max_lag)
                    value=mse(sums-old+new)
                    return value if np.isfinite(value) else 1e6
                except (FloatingPointError,OverflowError,ValueError): return 1e6
            result=minimize(objective,params[i],method="L-BFGS-B",bounds=[tuple(row) for row in LOG_PHYSICAL_BOUNDS],options={"maxiter":args.maxiter_per_lineage,"ftol":2e-10,"gtol":1e-7,"maxls":10})
            try:
                new=simulate_lineage_birth_acfs(groups[i],result.x,residual,thresholds[i,:,:len(groups[i])],uniforms[i,:,:len(groups[i])],max_lag)
                if mse(sums-old+new) <= mse(sums):
                    sums += new-old; curves[i]=new; params[i]=result.x
            except (FloatingPointError, OverflowError, ValueError):
                pass
        history.append({"sweep":sweep,"mse":mse(sums)})
        print(history[-1],flush=True)
    pd.DataFrame(history).to_csv(output_dir/"optimization_history.csv",index=False)
    optimized=pd.DataFrame(np.exp(params),columns=["u","omega2","eta"]);optimized.insert(0,"lineage",labels);optimized.to_csv(output_dir/"optimized_lineage_parameters.csv",index=False)

    validation_thresholds=rng.exponential(size=(len(groups),args.validation_replicates,max_length));validation_uniforms=rng.uniform(size=validation_thresholds.shape)
    validation_curves=[]
    for i,g in enumerate(groups):
        validation_curves.append(simulate_lineage_birth_acfs(g,params[i],residual,validation_thresholds[i,:,:len(g)],validation_uniforms[i,:,:len(g)],max_lag))
    validation=np.asarray(validation_curves)
    aggregate=[]
    for replicate in range(args.validation_replicates):
        aggregate.append(np.nanmean(validation[:,replicate,:],axis=0))
    aggregate=np.asarray(aggregate); median=np.nanmedian(aggregate,axis=0);low,high=np.nanpercentile(aggregate,[2.5,97.5],axis=0)
    training=total_curve(sums)
    table=pd.DataFrame({"lag":np.arange(max_lag+1),"observed":observed,"training_optimized":training,"validation_median":median,"validation_q2.5":low,"validation_q97.5":high});table.to_csv(output_dir/"acf_comparison.csv",index=False)
    fig,axis=plt.subplots(figsize=(8.3,5.1));lags=table.lag.to_numpy();axis.fill_between(lags,low,high,color="tab:green",alpha=.2,label="Validation 95% band");axis.plot(lags,median,color="tab:green",lw=2,label="Validation median");axis.plot(lags,training,color="tab:orange",ls="--",lw=1.6,label="Training realization");axis.plot(lags,observed,color="black",lw=2.2,label="Glu data");axis.axhline(0,color=".6",lw=.8);axis.set(xlabel="Lag (generations)",ylabel="Mean lineage birth-size ACF");axis.legend(frameon=False);fig.tight_layout();fig.savefig(output_dir/"acf_comparison.png",dpi=200);fig.savefig(output_dir/"acf_comparison.pdf");plt.close(fig)
    diagnostics={"purpose":"unrestricted lineage-parameter structural sanity check; objective is birth-ACF MSE only","hazard":"h_l(t)=alpha*omega2_l*(p/u_l)^eta_l/(1+(p/u_l)^eta_l)","optimized_parameters":3*len(groups),"total_glu_lineages":len(all_groups),"acf_contributing_lineages":len(groups),"excluded_short_lineages":len(all_groups)-len(groups),"fit_lags":f"1..{max_lag}","training_replicates":args.training_replicates,"initial_training_mse":initial_mse,"optimized_training_mse":mse(sums),"training_mse_ratio":mse(sums)/initial_mse,"validation_mse":float(np.mean((median[fit_slice]-observed[fit_slice])**2)),"validation_mae":float(np.mean(abs(median[fit_slice]-observed[fit_slice]))),"parameter_summary":{name:{"median":float(np.median(optimized[name])),"q2.5":float(np.percentile(optimized[name],2.5)),"q97.5":float(np.percentile(optimized[name],97.5)),"at_lower_bound":int(np.sum(optimized[name] <= np.exp(LOG_PHYSICAL_BOUNDS[j,0])*1.01)),"at_upper_bound":int(np.sum(optimized[name] >= np.exp(LOG_PHYSICAL_BOUNDS[j,1])*.99))} for j,name in enumerate(["u","omega2","eta"])},"run_config":vars(args),"runtime_minutes":(time.time()-started)/60}
    with (output_dir/"diagnostics.json").open("w") as handle:json.dump(diagnostics,handle,indent=2)
    print(json.dumps(diagnostics,indent=2))

if __name__=="__main__":main()
