#!/usr/bin/env python3
"""Shared birth-size drift plus stationary AR(1) residuals (Glu).

Exact Gaussian REML, with a separate fixed intercept per original lineage.
Drift forms are compared by training-lineage grouped CV, not by comparing
REML values with different fixed-effect design matrices. Validation scores
are densities of mean-free contrasts, NOT one-step transition densities.
The unconstrained Gaussian model is deliberately not clipped or truncated.
"""
import argparse
import json
import numpy as np
import pandas as pd
from analyze_acf_heterogeneity_transients import acf, bootstrap
from screen_parsimonious_joint_hazards import PROJECT_DIR, load_data, corr

OUT = PROJECT_DIR/'outputs/model_comparison/Glu/drift_ar1_births'
PHI = np.round(np.arange(-.5, .9951, .0025), 7)
TAUS = [1., 2., 4., 8., 16., 32., 64., 128., 256., 1024.]


def drift_basis(g, tau):
    if tau is None:
        return np.zeros_like(g, dtype=float)
    if np.isinf(tau):
        return np.asarray(g, float)
    return -tau*np.expm1(-np.asarray(g, float)/tau)


def prepare(groups):
    items = []
    for group in groups:
        if len(group) < 6:
            continue
        b = group.length_birth.to_numpy(float)
        g = group.generation.to_numpy(float)-group.generation.iloc[0]
        a = group.alpha.to_numpy(float)
        t = group.time.to_numpy(float)
        adjacent = np.diff(g) == 1
        ix = np.flatnonzero(adjacent)
        f = b[ix+1]/(b[ix]*np.exp(a[ix]*t[ix]))
        items.append(dict(id=float(group.lineage.iloc[0]), g=g, y=b,
                          alpha=a[ix], t=t[ix], ix=ix, F=f))
    return items


def whiten(v, g, phi):
    r = phi**np.diff(g).astype(int)
    return np.r_[v[0], (v[1:]-r*v[:-1])/np.sqrt(1-r*r)]


def sufficient(items, tau, phis=PHI):
    """Whiten by the exact irregular-generation AR(1) correlation matrix."""
    width = max(len(d['y']) for d in items)
    y = np.zeros((len(items), width)); x = np.zeros_like(y)
    mask = np.zeros_like(y, dtype=bool); gaps = np.ones_like(y, dtype=int)
    n = np.array([len(d['y']) for d in items])
    for j, item in enumerate(items):
        y[j, :n[j]] = item['y']; x[j, :n[j]] = drift_basis(item['g'], tau)
        mask[j, :n[j]] = True; gaps[j, 1:n[j]] = np.diff(item['g']).astype(int)
    arrays = {k: [] for k in ['q', 'xy', 'xx', 'log_s', 'logdet']}
    for phi in phis:
        r = phi**gaps[:, 1:]; den = np.sqrt(1-r*r)
        wy = np.column_stack([y[:, 0], (y[:, 1:]-r*y[:, :-1])/den])*mask
        wx = np.column_stack([x[:, 0], (x[:, 1:]-r*x[:, :-1])/den])*mask
        wo = np.column_stack([np.ones(len(items)), (1-r)/den])*mask
        s = np.sum(wo*wo, 1); oy = np.sum(wo*wy, 1); ox = np.sum(wo*wx, 1)
        arrays['q'].append(np.sum(wy*wy, 1)-oy*oy/s)
        arrays['xy'].append(np.sum(wx*wy, 1)-ox*oy/s)
        arrays['xx'].append(np.sum(wx*wx, 1)-ox*ox/s)
        arrays['log_s'].append(np.log(s))
        arrays['logdet'].append(np.sum(np.log1p(-r*r)*mask[:, 1:], 1))
    return {**{k: np.array(v) for k, v in arrays.items()}, 'n': n,
            'phis': np.asarray(phis), 'tau': tau}


def fit(stats, counts=None):
    """Profile the shared drift and stationary variance in exact REML."""
    if counts is None:
        counts = np.ones(len(stats['n']))
    with_drift = stats['tau'] is not None
    q = stats['q']@counts
    xy = stats['xy']@counts; xx = stats['xx']@counts
    beta = xy/xx if with_drift else np.zeros_like(q)
    q = q-beta*xy
    df = float((stats['n']-1)@counts-int(with_drift))
    var = q/df
    logdet = (stats['logdet']+stats['log_s'])@counts
    if with_drift:
        logdet = logdet+np.log(xx)
    objective = -.5*(logdet+df*np.log(var))
    j = int(np.argmax(objective)); phi = float(stats['phis'][j])
    return dict(phi=phi, slope=float(beta[j]), stationary_sd=float(np.sqrt(var[j])),
                innovation_sd=float(np.sqrt(var[j]*(1-phi**2))),
                tau=stats['tau'], grid_index=j, restricted_df=df)


def contrast_scores(stats, p):
    """Orthonormal mean-free contrasts; same nuisance projection in all models.

    No validation-lineage slope or variance is fitted. The offset is removed
    analytically, which is not a forecast of an unseen lineage's birth level.
    """
    j = int(np.argmin(abs(stats['phis']-p['phi'])))
    q = stats['q'][j]-2*p['slope']*stats['xy'][j]+p['slope']**2*stats['xx'][j]
    v = p['stationary_sd']**2
    return -.5*((stats['n']-1)*np.log(2*np.pi*v)+stats['logdet'][j]
                +stats['log_s'][j]-np.log(stats['n'])+q/v)


def offsets(items, p):
    values = []
    for item in items:
        y = item['y']-p['slope']*drift_basis(item['g'], p['tau'])
        one = whiten(np.ones(len(y)), item['g'], p['phi'])
        values.append(float(one@whiten(y, item['g'], p['phi'])/(one@one)))
    return np.array(values)


def simulate(items, p, reps, seed=314159):
    rng = np.random.default_rng(seed); mus = offsets(items, p); series = []
    for mu, item in zip(mus, items):
        g = item['g']; n = len(g); r = np.empty((reps, n))
        r[:, 0] = rng.normal(0, p['stationary_sd'], reps)
        for i in range(1, n):
            phi = p['phi']**int(g[i]-g[i-1])
            r[:, i] = phi*r[:, i-1]+p['stationary_sd']*np.sqrt(1-phi**2)*rng.normal(size=reps)
        series.append(mu+p['slope']*drift_basis(g, p['tau'])+r)
    return series


def metrics(items, series):
    b = np.concatenate(series, axis=1); ts = []; parents = []; children = []
    t_acfs = []; changes = []
    for item, ys in zip(items, series):
        ix = item['ix']; p = ys[:, ix]; c = ys[:, ix+1]
        with np.errstate(invalid='ignore', divide='ignore'):
            t = (np.log(c)-np.log(p)-np.log(item['F']))/item['alpha']
        ts.append(t); parents.append(p); children.append(c)
        t_acfs.append(acf(t, item['g'][ix]))
        third = len(item['y'])//3
        changes.append(ys[:, -third:].mean(1)-ys[:, :third].mean(1))
    t = np.concatenate(ts, axis=1); p = np.concatenate(parents, axis=1); c = np.concatenate(children, axis=1)
    return dict(mean_b=b.mean(1), sd_b=b.std(1), mean_t=np.nanmean(t, 1), sd_t=np.nanstd(t, 1),
                q99_t=np.nanquantile(t, .99, axis=1), nonpositive_birth_fraction=(b<=0).mean(1),
                nonpositive_time_fraction=(t<=0).mean(1), undefined_time_fraction=(~np.isfinite(t)).mean(1),
                corr_b_t=corr(p, t), corr_t_next_b=corr(t, c),
                late_minus_early=np.mean(changes, 0)), np.nanmean(t_acfs, axis=0), t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    _, groups = load_data(); items = prepare(groups); ids = np.array([d['id'] for d in items])
    validids = set(json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/empirical_birth_transitions/fit_results.json').read_text())['validation_ids'])
    train = np.array([v not in validids for v in ids]); valid = ~train
    rng = np.random.default_rng(72813); shuffled = np.flatnonzero(train); rng.shuffle(shuffled)
    folds = np.full(len(items), -1); folds[shuffled] = np.arange(len(shuffled))%5
    forms = {'No_drift': None, 'Linear': np.inf, **{f'Saturating_{tau:g}': tau for tau in TAUS}}
    cache = {}; fits = {}; cvrows = []; cvscores = {}
    for name, tau in forms.items():
        stats = sufficient(items, tau); cache[name] = stats
        fits[name] = {'all': fit(stats), 'train': fit(stats, train.astype(float))}
        scores = np.full(len(items), np.nan)
        for fold in range(5):
            p = fit(stats, ((folds>=0)&(folds!=fold)).astype(float))
            scores[folds==fold] = contrast_scores(stats, p)[folds==fold]
        cvscores[name] = scores
        cvrows.append(dict(model=name, tau=tau, total_score=float(np.nansum(scores)),
                           mean_score=float(np.nanmean(scores))))
    cv = pd.DataFrame(cvrows); cv.to_csv(OUT/'training_grouped_cv.csv', index=False)
    bestsat = cv[cv.model.str.startswith('Saturating')].sort_values('total_score').iloc[-1].model
    names = ['No_drift', 'Linear', bestsat]
    config = dict(n_lineages=len(items), n_births=sum(len(d['y']) for d in items),
                  n_train_lineages=int(train.sum()), n_validation_lineages=int(valid.sum()),
                  best_saturating=bestsat, fits={name: fits[name] for name in names},
                  selection='Five-fold training-lineage CV selects saturation tau; no ACF used in fitting.',
                  score='Gaussian density of mean-free lineage contrasts, using training shared parameters; not one-step predictive density.',
                  simulation='All-data refit; full-record GLS lineage intercepts; stationary initial residual. All births generated, none seeded. Same lengths and generation gaps. Parameter uncertainty not in simulation bands.',
                  time_check='Recorded alpha and derived F_eff fixed on adjacent observed generations. Negative durations retained. Undefined times explicitly counted; nan-aware time summaries exclude only undefined values. Not a complete positive-time physical model.',
                  caveats='Exploratory reused validation split. Shared variance and slope; lineage intercepts are nuisance fitted offsets. Gaussian support permits invalid births/times. Drift is since tracking began, not biological age.')
    scoretable = pd.DataFrame({'lineage': ids[valid]})
    for name in names:
        scoretable[name] = contrast_scores(cache[name], fits[name]['train'])[valid]
    scoretable.to_csv(OUT/'validation_contrast_scores.csv', index=False)
    config['validation_comparisons'] = {}; config['training_cv_comparisons'] = {}
    for a, b in [('Linear', 'No_drift'), (bestsat, 'Linear')]:
        for key, delta in [('validation_comparisons', (scoretable[a]-scoretable[b]).to_numpy()),
                           ('training_cv_comparisons', (cvscores[a]-cvscores[b])[train])]:
            bs = bootstrap(delta[:, None], rng, args.bootstrap)[:, 0]
            config[key][a+' minus '+b] = dict(mean_per_lineage=float(delta.mean()), ci95=np.quantile(bs, [.025, .975]).tolist())
    # Whole-lineage bootstrap; shared structure is refitted, saturation grid
    # selection is NOT repeated. Linear uncertainty is the primary result.
    pdraws = []
    for _ in range(args.bootstrap):
        counts = np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items))
        pdraws.append(fit(cache['Linear'], counts))
    config['linear_bootstrap_ci95'] = {k: np.quantile([p[k] for p in pdraws], [.025, .975]).tolist()
                                         for k in ['slope', 'phi', 'stationary_sd', 'innovation_sd']}
    pd.DataFrame(pdraws).to_csv(OUT/'linear_parameter_bootstrap.csv', index=False)
    print(json.dumps(config, indent=2), flush=True)
    curves = []; rows = []; histrows = []; trendrows = []; residualrows = []
    observed = [d['y'][None, :] for d in items]
    runs = {'Observed': observed, **{name: simulate(items, fits[name]['all'], args.replicates) for name in names}}
    obsacf = np.nanmean(np.stack([acf(y, d['g']) for y, d in zip(observed, items)], 1), 1)[0]
    config['birth_acf_rmse_lags_1_to_10'] = {}
    for name, series in runs.items():
        ac = np.stack([acf(y, d['g']) for y, d in zip(series, items)], 1)
        draw = bootstrap(ac[0], rng, args.bootstrap) if name=='Observed' else np.nanmean(ac, 1)
        mean = np.nanmean(ac, 1).mean(0); lo, hi = np.nanquantile(draw, [.025, .975], axis=0)
        for k in range(11):
            curves.append(dict(model=name, variable='birth', lag=k, mean=mean[k], lo=lo[k], hi=hi[k]))
        if name!='Observed':
            config['birth_acf_rmse_lags_1_to_10'][name] = float(np.sqrt(np.mean((mean[1:]-obsacf[1:])**2)))
        m, tac, times = metrics(items, series)
        for key, vals in m.items():
            l, h = np.nanquantile(vals, [.025, .975]); rows.append(dict(model=name, metric=key, mean=np.nanmean(vals), lo=l, hi=h))
        for k in range(11):
            l, h = np.nanquantile(tac[:, k], [.025, .975]); curves.append(dict(model=name, variable='time', lag=k, mean=np.nanmean(tac[:, k]), lo=l, hi=h))
        for var, values, edges in [('birth', np.concatenate(series, 1), np.linspace(0, 6, 81)),
                                    ('time', times, np.linspace(-3, 8, 111))]:
            finite = values[np.isfinite(values)]
            density = np.histogram(finite, edges)[0]/len(finite)/np.diff(edges)
            histrows.extend(dict(model=name, variable=var, left=a, right=b, density=d) for a, b, d in zip(edges[:-1], edges[1:], density))
        centered = [y-y.mean(1, keepdims=True) for y in series]
        for g in range(int(max(d['g'].max() for d in items))+1):
            values = [y[:, d['g']==g].mean(1) for d, y in zip(items, centered) if np.any(d['g']==g)]
            if not values:
                continue
            vals = np.array(values)
            if name=='Observed':
                means = bootstrap(vals, rng, args.bootstrap)[:, 0]
            else:
                means = vals.mean(0)
            l, h = np.quantile(means, [.025, .975])
            trendrows.append(dict(model=name, generation=g, n_lineages=len(values), mean=vals.mean(), lo=l, hi=h))
        if name in ['Observed', 'Linear']:
            p = fits['Linear']['all']
            detrended = np.stack([acf(y-p['slope']*d['g'], d['g']) for y, d in zip(series, items)], 1)
            dr = bootstrap(detrended[0], rng, args.bootstrap) if name=='Observed' else np.nanmean(detrended, 1)
            for k in range(11):
                l, h = np.quantile(dr[:, k], [.025, .975]); residualrows.append(dict(model=name, lag=k, mean=np.nanmean(detrended, 1)[:, k].mean(), lo=l, hi=h))
        print('CHECKED', name, flush=True)
    pd.DataFrame(curves).to_csv(OUT/'acf_comparison.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(histrows).to_csv(OUT/'marginal_densities.csv', index=False)
    pd.DataFrame(trendrows).to_csv(OUT/'centered_drift.csv', index=False)
    pd.DataFrame(residualrows).to_csv(OUT/'residual_acf.csv', index=False)
    pd.DataFrame({'lineage': ids, **{name: offsets(items, fits[name]['all']) for name in names}}).to_csv(OUT/'lineage_intercepts.csv', index=False)
    # Standard JSON: null denotes infinite saturation time for the linear form.
    config['linear_tau_convention'] = 'null in Linear fits means infinity (linear); null in No_drift means absent.'
    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj): return None
        return obj
    (OUT/'fit_results.json').write_text(json.dumps(clean(config), indent=2, allow_nan=False))


if __name__=='__main__':
    main()
