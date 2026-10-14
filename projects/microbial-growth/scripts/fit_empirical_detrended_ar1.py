#!/usr/bin/env python3
"""Single AR(1) after exact empirical generation-mean removal.

Balanced 172x21 cohort matches fit_empirical_detrended_hazard. Compare a
zero-offset AR(1) and a shared AR(1) with static lineage intercepts. Exact
Gaussian REML accounts for estimation of the generation means. Simulation
repeats generation-mean estimation and the same lineage-demeaned ACF.
"""
import json
import numpy as np
import pandas as pd
from fit_empirical_detrended_hazard import PROJECT_DIR,load_data,cohort,acf

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/empirical_detrended_ar1'
PHIS=np.round(np.arange(-.5,.9951,.0025),7)


def profile(x,offsets=True,phis=PHIS):
    """REML after projecting out arbitrary generation means.

    Orthogonal lineage contrasts leave L-1 independent records with common
    temporal correlation R. With static lineage intercepts, temporal mean
    contrasts leave n-1 dimensions as well. Constants independent of phi
    are omitted. Do not compare scores across the two fixed-effect designs.
    """
    x=np.asarray(x,float);x=x-x.mean(0);L,n=x.shape;p=np.asarray(phis)
    xx=np.sum(x[:,1:]**2);xp=np.sum(x[:,:-1]**2);cross=np.sum(x[:,1:]*x[:,:-1])
    q=np.sum(x[:,0]**2)+(xx-2*p*cross+p*p*xp)/(1-p*p)
    ld=(n-1)*np.log1p(-p*p)
    if offsets:
        s=(2+(n-2)*(1-p))/(1+p)
        dot=(x[:,0]+x[:,-1]+(1-p[:,None])*x[:,1:-1].sum(1))/(1+p[:,None])
        q-=np.sum(dot*dot,axis=1)/s;ld+=np.log(s/n)
    df=(L-1)*(n-1 if offsets else n);var=q/df
    objective=-.5*((L-1)*ld+df*np.log(var))
    return objective,var,df


def fit(x,offsets=True,phis=PHIS):
    scores,var,df=profile(x,offsets,phis);j=int(np.argmax(scores));p=float(phis[j])
    return dict(phi=p,innovation_sd=float(np.sqrt(var[j]*(1-p*p))),stationary_sd=float(np.sqrt(var[j])),
                correlation_length_generations=float(-1/np.log(p)) if p>0 else None,
                restricted_df=int(df),static_lineage_offsets=offsets)


def intercepts(x,p):
    x=x-x.mean(0);phi=p['phi'];n=x.shape[1]
    if not p['static_lineage_offsets']:return np.zeros(len(x))
    return (x[:,0]+x[:,-1]+(1-phi)*x[:,1:-1].sum(1))/(2+(n-2)*(1-phi))


def simulate(x,p,reps=3000,seed=53419):
    rng=np.random.default_rng(seed);L,n=x.shape
    residual=np.empty((reps,L,n));residual[:,:,0]=rng.normal(0,p['stationary_sd'],(reps,L))
    for i in range(1,n):
        residual[:,:,i]=p['phi']*residual[:,:,i-1]+p['innovation_sd']*rng.normal(size=(reps,L))
    # Add the fitted mean trajectory and static offsets, then repeat exactly
    # the empirical generation-mean subtraction used on the observations.
    raw=x.mean(0)+intercepts(x,p)[None,:,None]+residual
    return raw-raw.mean(1,keepdims=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);_,groups=load_data();data=cohort(groups);x=data['x'];X=x-x.mean(0)
    fits={name:fit(x,off) for name,off in [('Zero_offset_AR1',False),('Offset_AR1',True)]}
    rng=np.random.default_rng(19283);rows=[];bs=[];profile_rows=[]
    for i in range(1000):
        draw=x[rng.integers(0,len(x),len(x))]
        for name,off in [('Zero_offset_AR1',False),('Offset_AR1',True)]:
            bs.append(dict(model=name,replicate=i,**fit(draw,off)))
    boot=pd.DataFrame(bs);boot.to_csv(OUT/'parameter_bootstrap.csv',index=False)
    config=dict(n_lineages=len(x),n_births=int(x.size),record_length=x.shape[1],fits=fits,
                fitting='Exact Gaussian REML; empirical generation means re-estimated; common phi and variance. Static offsets are nuisance constants, not dynamic memory. REML scores across different fixed-effect designs are not compared.',
                simulation='3000 stationary-initialized Gaussian datasets per fitted model. All births generated, no observed seeds. Add fitted means/offsets then subtract each simulation own generation means. Same 21-point lineage-demeaned ACF as data.',
                uncertainty='1000 whole-lineage bootstrap refits including re-estimated empirical mean curves. ACF simulation bands conditional on fitted parameters; pointwise, not simultaneous or fitted-null p-values.',
                scope='Descriptive same balanced 172-lineage first-20-transition cohort as detrended hazard. Not a new held-out test. No timing or division-factor model is fitted.',
                bootstrap_ci95={},acf_rmse_lags_1_to_10={})
    for name,p in fits.items():
        b=boot[boot.model==name]
        config['bootstrap_ci95'][name]={k:np.quantile(b[k],[.025,.975]).tolist() for k in ['phi','innovation_sd','stationary_sd','correlation_length_generations']}
        objective,variance,df=profile(x,p['static_lineage_offsets'])
        profile_rows.extend(dict(model=name,phi=ph,relative_restricted_log_likelihood=v-objective.max(),stationary_variance=vv)
                            for ph,v,vv in zip(PHIS,objective,variance))
    runs={'Observed':X[None],**{name:simulate(x,p) for name,p in fits.items()}}
    obs=acf(X[None]).mean(1)[0]
    for name,y in runs.items():
        a=acf(y);avg=a.mean(1)
        if name=='Observed':
            indices=rng.integers(0,len(x),(2000,len(x)))
            # Unlike the hazard's fixed-mean bands, also re-estimate the
            # empirical generation trajectory in every observed bootstrap.
            xb=x[indices];xb-=xb.mean(1,keepdims=True);draws=acf(xb).mean(1)
        else:draws=avg
        lo,hi=np.quantile(draws,[.025,.975],axis=0)
        rows.extend(dict(model=name,lag=k,mean=avg[:,k].mean(),lo=lo[k],hi=hi[k]) for k in range(11))
        if name!='Observed':
            config['acf_rmse_lags_1_to_10'][name]=float(np.sqrt(np.mean((avg.mean(0)[1:]-obs[1:])**2)))
            config.setdefault('observed_lags_outside_pointwise_simulation_band',{})[name]=np.flatnonzero((obs<lo)|(obs>hi)).tolist()
            np.savez_compressed(OUT/f'{name}_examples.npz',centered_log_birth=y[:5],lineage_ids=data['ids'])
    pd.DataFrame(rows).to_csv(OUT/'acf_comparison.csv',index=False)
    pd.DataFrame(profile_rows).to_csv(OUT/'likelihood_profiles.csv',index=False)
    pd.DataFrame(dict(lineage=data['ids'],**{name:intercepts(x,p) for name,p in fits.items()})).to_csv(OUT/'lineage_intercepts.csv',index=False)
    (OUT/'fit_results.json').write_text(json.dumps(config,indent=2,allow_nan=False))
    print(json.dumps(config,indent=2))


if __name__=='__main__':main()
