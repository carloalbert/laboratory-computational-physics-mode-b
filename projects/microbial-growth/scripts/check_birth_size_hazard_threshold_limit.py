#!/usr/bin/env python3
"""Check the sharp-threshold limit of the p,b hazard after ACF calibration.

Take eta and omega2 large, with c=kappa/eta fixed and log(omega2)/eta -> 0.
Division occurs at p=u*(b/b_star)**(-c). This is a boundary diagnostic,
not a separate stochastic model fit to the division-time likelihood.
"""
import json
import numpy as np
import pandas as pd
from optimize_birth_size_hazard_acf import Replay, load_data, add_effective_factors, mean_acf, PROJECT_DIR


def main():
    data, groups = load_data()
    groups, _, _ = add_effective_factors(groups)
    bstar = float(np.exp(np.log(data.length_birth).mean()))
    design = Replay(groups, bstar, replicates=1)
    target = mean_acf(groups, 'length_birth', 30)

    def simulate(x):
        u, c = np.exp(x[0]), x[1]
        births = np.zeros((1, *design.mask.shape))
        times = np.zeros_like(births)
        b = design.birth0.copy()
        for n in range(design.mask.shape[1]):
            active = design.mask[:, n]
            p = u * (b[active] / bstar)**(-c)
            births[0, active, n] = b[active]
            times[0, active, n] = np.log1p(p/b[active]) / design.alpha[active,n]
            if n+1 < design.mask.shape[1]:
                b[active] = design.f[active,n+1] * (b[active] + p)
        return design.acfs(births)[0], design.acfs(times)[0]

    def objective(x):
        if not (np.log(.01) <= x[0] <= np.log(100) and -1 <= x[1] <= 3):
            return np.inf
        acf, _ = simulate(x)
        return np.mean((acf[1:]-target[1:])**2)

    best, value = None, np.inf
    for logu in np.linspace(np.log(.1),np.log(50),36):
        for c in np.linspace(-.8,2,36):
            x = np.array([logu,c]); score = objective(x)
            if score < value: best,value = x,score
    step = np.array([.08,.04])
    for _ in range(300):
        improved = False
        for j in (0,1):
            for direction in (-1,1):
                x=best.copy();x[j]+=direction*step[j]
                score=objective(x)
                if score<value:best,value,improved=x,score,True
        if not improved:step*=.5
        if step.max()<1e-5:break
    birth,times = simulate(best)
    result = {'u':float(np.exp(best[0])), 'kappa_over_eta':float(best[1]),
              'birth_acf_mse':float(value), 'birth_acf_mae':float(np.mean(abs(birth[1:]-target[1:]))),
              'birth_acf_lags_1_to_8':birth[1:9].tolist(),
              'time_acf_lags_1_to_3':times[1:4].tolist(),
              'definition':'deterministic limit p_div=u*(b/b_star)**(-kappa/eta)'}
    out=PROJECT_DIR/'outputs/model_comparison/Glu/birth_size_dependent_time_hazard_acf_only_wide'
    (out/'threshold_limit.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame({'lag':np.arange(31),'observed_birth':target,'threshold_limit_birth':birth,'threshold_limit_time':times}).to_csv(out/'threshold_limit_acf.csv',index=False)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
