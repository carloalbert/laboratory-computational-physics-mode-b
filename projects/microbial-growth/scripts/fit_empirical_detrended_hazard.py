#!/usr/bin/env python3
"""Exact generation-mean detrending and shifted-support four-parameter Hill fit.

B=exp(log b-mean_i(log b)), G=alpha*t-mean_i(alpha*t).
lambda_i(G|B)=omega sigmoid[eta log(expm1(G+mean_i(g))/u)
                           +(eta+kappa) log B], G>-mean_i(g).
Only the birth predictor is changed by this refit: translating g and its
support has a unit Jacobian and cannot itself alter a likelihood optimum.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from glu_workflow import PROJECT_DIR, load_data, corr, validation_ids, load_config

OUT = PROJECT_DIR/'outputs/model_comparison/Glu/empirical_detrended_hazard'
BOUNDS = np.array([[np.log(.01), np.log(30.)], [np.log(.1), np.log(80.)], [-30., 30.]])


def cohort(groups, horizon=None):
    if horizon is None:
        horizon = load_config()['horizon']
    groups = [g.iloc[:horizon+1] for g in groups if len(g)>horizon
              and np.array_equal(g.generation.to_numpy()[:horizon+1], np.arange(horizon+1))]
    x = np.array([np.log(g.length_birth.to_numpy(float)) for g in groups])
    a = np.array([g.alpha.to_numpy(float)[:-1] for g in groups])
    t = np.array([g.time.to_numpy(float)[:-1] for g in groups])
    growth = a*t; q = np.diff(x, axis=1)-growth
    return dict(ids=np.array([float(g.lineage.iloc[0]) for g in groups]), x=x, a=a, t=t, g=growth, q=q)


def means(data, mask=None):
    if mask is None: mask = np.ones(len(data['ids']), bool)
    return {k: data[k][mask].mean(0) for k in ['x', 'q', 'g']}


def center(data, m):
    return {k: data[k]-m[k] for k in ['x', 'q', 'g']}


class Likelihood:
    def __init__(self, x, g, order=48):
        self.x = np.asarray(x).ravel(); self.g = np.asarray(g).ravel()
        if np.any(self.g<=0): raise ValueError('Uncentered growth must be positive')
        nodes, weights = np.polynomial.legendre.leggauss(order)
        self.logp = np.log(np.expm1(self.g))
        self.integral_logp = np.log(np.expm1(.5*self.g[:, None]*(1+nodes)))
        self.weights = weights; self.n = len(self.g)

    def components(self, z):
        logu, logeta, gamma = z; eta = np.exp(logeta)
        odds = eta*(self.logp-logu)+gamma*self.x
        intodds = eta*(self.integral_logp-logu)+gamma*self.x[:, None]
        integral = .5*self.g*np.sum(self.weights/(1+np.exp(np.clip(-intodds, -700, 700))), axis=1)
        omega = self.n/integral.sum()
        return -np.logaddexp(0., -odds), integral, omega

    def __call__(self, z):
        if np.any(z<BOUNDS[:, 0]) or np.any(z>BOUNDS[:, 1]): return -np.inf
        logfrac, integral, omega = self.components(z)
        return float(np.sum(np.log(omega)+logfrac-omega*integral))

    def scores(self, p):
        z = np.array([np.log(p['u']), np.log(p['eta']), p['eta']+p['kappa']])
        logfrac, integral, _ = self.components(z)
        return np.log(p['omega2'])+logfrac-p['omega2']*integral


def shifted_log_density(G, X, mean_growth, p, order=128):
    """Density with respect to centered growth, including its shifted support."""
    G, X, mean_growth = np.broadcast_arrays(G, X, mean_growth)
    physical_g = G+mean_growth; valid = physical_g>0
    out = np.full(G.shape, -np.inf)
    if valid.any(): out[valid] = Likelihood(X[valid], physical_g[valid], order).scores(p)
    return out


def polish(objective, z):
    z = np.array(z).copy(); step = np.array([.08, .06, .15]); value = objective(z)
    for _ in range(2500):
        improved = False
        for j in range(3):
            for sign in [-1, 1]:
                p = z.copy(); p[j] = np.clip(p[j]+sign*step[j], *BOUNDS[j]); v = objective(p)
                if v>value: z, value, improved = p, v, True
        if not improved: step *= .5
        if step.max()<1e-5: break
    return z, value


def optimize(objective, seed):
    rng = np.random.default_rng(seed); n = 28; low, high = BOUNDS.T
    pop = rng.uniform(low, high, (n, 3))
    pop[:3] = [[np.log(.5), np.log(4), 4], [np.log(1.), np.log(10), 8], [0, 0, 1]]
    scores = np.array([objective(p) for p in pop])
    for _ in range(65):
        for i in range(n):
            a, b, c = rng.choice(np.delete(np.arange(n), i), 3, replace=False)
            mutant = np.clip(pop[a]+.7*(pop[b]-pop[c]), low, high)
            cross = rng.random(3)<.8; cross[rng.integers(3)] = True
            trial = np.where(cross, mutant, pop[i]); val = objective(trial)
            if val>scores[i]: pop[i], scores[i] = trial, val
    return polish(objective, pop[np.argmax(scores)])


def fit(x, g, seed, multistart=True):
    ll = Likelihood(x, g); runs = [optimize(ll, seed)]
    if multistart: runs.append(optimize(ll, seed+1))
    z, _ = max(runs, key=lambda pair: pair[1]); fine = Likelihood(x, g, 128)
    z, value = polish(fine, z); logfrac, integral, omega = fine.components(z)
    p = dict(u=float(np.exp(z[0])), omega2=float(omega), eta=float(np.exp(z[1])), kappa=float(z[2]-np.exp(z[1])))
    check = Likelihood(x, g, 256).scores(p)
    return p, dict(log_likelihood=value, mean_log_density=value/ll.n,
                   optimizer_runs=[float(r[1]) for r in runs], quadrature_128_vs_256_total=float(check.sum()-value),
                   max_event_quadrature_difference=float(np.max(abs(check-fine.scores(p)))),
                   boundary_distances=(np.minimum(z-BOUNDS[:, 0], BOUNDS[:, 1]-z)).tolist())


def sample_g(x, p, rng):
    g = np.zeros_like(x); pending = np.ones(x.shape, bool)
    for _ in range(20000):
        if not pending.any(): return g
        ix = np.flatnonzero(pending); g[ix] += rng.exponential(1/p['omega2'], len(ix))
        z = p['eta']*(np.log(np.expm1(np.minimum(g[ix], 700)))-np.log(p['u']))+(p['eta']+p['kappa'])*x[ix]
        accept = rng.random(len(ix))<1/(1+np.exp(np.clip(-z, -700, 700)))
        pending[ix[accept]] = False
    raise RuntimeError('Thinning did not converge')


def simulate(data, m, p, model, constant, reps=300, seed=9817):
    rng = np.random.default_rng(seed); L, n = data['g'].shape
    x = np.empty((reps, L, n+1)); g = np.empty((reps, L, n))
    x[:, :, 0] = data['x'][:, 0]
    for i in range(n):
        predictor = x[:, :, i]-(m['x'][i] if model=='Detrended' else constant)
        g[:, :, i] = sample_g(predictor.ravel(), p, rng).reshape(reps, L)
        x[:, :, i+1] = x[:, :, i]+data['q'][:, i]+g[:, :, i]
    return x, g


def acf(y, maxlag=10):
    y = y-y.mean(-1, keepdims=True); den = np.sum(y*y, -1)
    return np.stack([np.ones(y.shape[:2])]+[np.sum(y[..., :-k]*y[..., k:], -1)/den for k in range(1, maxlag+1)], -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--replicates', type=int, default=300)
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    _, groups = load_data(); data = cohort(groups); m = means(data); c = center(data, m)
    ids = data['ids']; n = data['g'].shape[1]
    validids = validation_ids()
    valid = np.isin(ids, list(validids)); train = ~valid
    config = dict(n_lineages=len(ids), n_transitions=len(ids)*n, n_train=int(train.sum()), n_validation=int(valid.sum()), horizon=n,
                  model=__doc__, bounds_logu_logeta_gamma=BOUNDS.tolist(),
                  note='All-data empirical means are exact; training means only for validation. Four hazard parameters in addition to nonparametrically estimated mean curves. Balanced first-20-transition cohort selects long uninterrupted records, not all Glu.',
                  balance_error=float(np.max(abs(np.diff(m['x'])-m['q']-m['g']))),
                  centered_identity_error=float(np.max(abs(np.diff(c['x'], axis=1)-c['q']-c['g']))),
                  fraction_centered_growth_nonpositive=float(np.mean(c['g']<=0)), fits={})
    df = pd.DataFrame({'generation': np.arange(n+1), 'mean_log_birth': m['x']})
    df['mean_log_factor'] = np.r_[m['q'], np.nan]; df['mean_growth'] = np.r_[m['g'], np.nan]
    df.to_csv(OUT/'empirical_means.csv', index=False)
    rows = []
    for j, identity in enumerate(ids):
        for i in range(n):
            rows.append(dict(lineage=identity, generation=i, log_birth=data['x'][j,i], log_next_birth=data['x'][j,i+1],
                             log_F=data['q'][j,i], alpha=data['a'][j,i], time=data['t'][j,i], growth=data['g'][j,i],
                             X=c['x'][j,i], X_next=c['x'][j,i+1], Q=c['q'][j,i], G=c['g'][j,i]))
    pd.DataFrame(rows).to_csv(OUT/'detrended_transitions.csv', index=False)
    scores = {}; simulations = {}; acrows = []; metricrows = []; driftrows = []
    for scope, mask in [('train', train), ('all', np.ones(len(ids), bool))]:
        mm = means(data, mask); constant = float(data['x'][mask, :-1].mean())
        config['fits'][scope] = {'means': {k:v.tolist() for k,v in mm.items()}, 'constant_log_birth_reference':constant}
        for name in ['Original', 'Detrended']:
            predictor = data['x'][:, :-1]-(mm['x'][:-1] if name=='Detrended' else constant)
            print('FITTING', scope, name, flush=True)
            p, diag = fit(predictor[mask], data['g'][mask], 4161+(name=='Detrended'), multistart=scope=='all')
            config['fits'][scope][name] = dict(parameters=p, diagnostics=diag)
            print('FITTED', scope, name, p, diag, flush=True)
            if scope=='train': scores[name] = Likelihood(predictor[valid], data['g'][valid], 256).scores(p).reshape(valid.sum(), n)
            else: simulations[name] = simulate(data, m, p, name, constant, args.replicates)
    delta = (scores['Detrended']-scores['Original']).mean(1)
    rng = np.random.default_rng(4481); inds = rng.integers(0, len(delta), (5000, len(delta))); bs = delta[inds].mean(1)
    config['validation_gain_nats_per_transition'] = dict(mean=float(delta.mean()), ci95=np.quantile(bs,[.025,.975]).tolist())
    pd.DataFrame({'lineage':np.repeat(ids[valid],n), 'generation':np.tile(np.arange(n),valid.sum()),
                  **{k:v.ravel() for k,v in scores.items()}}).to_csv(OUT/'validation_scores.csv', index=False)
    runs = {'Observed':(data['x'][None], data['g'][None]), **simulations}
    for name, (x, g) in runs.items():
        X = x-m['x']; G = g-m['g']; t = g/data['a']; b = np.exp(x)
        series = {'X':X, 'G':G, 'birth':b, 'time':t}
        for var, values in series.items():
            ac = acf(values)
            if name=='Observed':
                ix = rng.integers(0,len(ids),(2000,len(ids))); draws = ac[0,ix].mean(1)
            else: draws = ac.mean(1)
            lo, hi = np.quantile(draws,[.025,.975],axis=0); av = ac.mean((0,1))
            acrows.extend(dict(model=name,variable=var,lag=k,mean=av[k],lo=lo[k],hi=hi[k]) for k in range(11))
            for i in range(values.shape[-1]):
                v = values[:,:,i].mean(1); l,h = np.quantile(v,[.025,.975])
                driftrows.append(dict(model=name,variable=var,generation=i,mean=v.mean(),lo=l,hi=h))
        vals = dict(mean_b=b.mean((1,2)),sd_b=b.std((1,2)),mean_t=t.mean((1,2)),sd_t=t.std((1,2)),
                    sd_X=X.std((1,2)),sd_G=G.std((1,2)),mean_G=G.mean((1,2)),
                    corr_X_G=corr(X[:,:,:-1].reshape(len(x),-1),G.reshape(len(x),-1)),
                    corr_b_t=corr(b[:,:,:-1].reshape(len(x),-1),t.reshape(len(x),-1)))
        for key,v in vals.items():
            l,h = np.quantile(v,[.025,.975]); metricrows.append(dict(model=name,metric=key,mean=v.mean(),lo=l,hi=h))
        if name!='Observed': np.savez_compressed(OUT/f'{name}_examples.npz',X=X[:5],G=G[:5],birth=b[:5],time=t[:5],ids=ids)
    pd.DataFrame(acrows).to_csv(OUT/'acf_comparison.csv',index=False)
    pd.DataFrame(metricrows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(driftrows).to_csv(OUT/'mean_replay.csv',index=False)
    (OUT/'fit_results.json').write_text(json.dumps(config,indent=2,allow_nan=False))
    print('DONE',config['validation_gain_nats_per_transition'],flush=True)


if __name__=='__main__': main()
