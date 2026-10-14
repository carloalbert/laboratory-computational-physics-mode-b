#!/usr/bin/env python3
"""Empirical-Bayes static lineage offsets in the detrended Glu Hill hazard."""
import json
import numpy as np
import pandas as pd
from fit_empirical_detrended_hazard import (PROJECT_DIR,load_data,cohort,means,acf,corr,Likelihood,
                                           OUT as BASE_OUT)
from static_offset_growth_hazard import MarginalLikelihood,pack,unpack,minimize_bfgs,BOUNDS
from glu_workflow import validation_ids

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/static_offset_detrended_hazard'


def fit(X,g,base):
    coarse=MarginalLikelihood(X,g,32,41);runs=[]
    for tau in [.0004,.3,.9,1.8]:
        z,diag=minimize_bfgs(coarse.objective,pack(base,tau));runs.append((z,diag))
    z,_=min(runs,key=lambda pair:pair[1]['objective'])
    fine=MarginalLikelihood(X,g,128,121);z,diag=minimize_bfgs(fine.objective,z)
    p=unpack(z);s,_,nodes,post,_=fine.evaluate(z,posterior=True)
    check=MarginalLikelihood(X,g,256,201).evaluate(z,gradient=False)[0]
    if np.max(abs(check-s))>1e-4:
        fine=MarginalLikelihood(X,g,128,201);z,diag=minimize_bfgs(fine.objective,z);p=unpack(z)
        s,_,nodes,post,_=fine.evaluate(z,posterior=True)
        check=MarginalLikelihood(X,g,256,301).evaluate(z,gradient=False)[0]
    if np.max(abs(check-s))>1e-3:raise RuntimeError('Offset quadrature not converged')
    diag.update(marginal_log_likelihood=float(s.sum()),mean_log_density_per_transition=float(s.sum()/g.size),
                multistart=[dict(start_tau=t,**d) for t,(_,d) in zip([.0004,.3,.9,1.8],runs)],
                quadrature_check_total=float(check.sum()-s.sum()),quadrature_check_max_lineage=float(np.max(abs(check-s))),
                boundary_distances=np.minimum(z-BOUNDS[:,0],BOUNDS[:,1]-z).tolist())
    return p,diag,nodes,post


def sample_g(X,a,p,rng):
    X=np.asarray(X).ravel();a=np.broadcast_to(a,X.shape)
    growth=np.zeros_like(X);pending=np.ones(X.shape,bool)
    for _ in range(20000):
        if not pending.any():return growth
        ix=np.flatnonzero(pending);growth[ix]+=rng.exponential(1/p['omega2'],len(ix))
        odds=p['eta']*(np.log(np.expm1(np.minimum(growth[ix],700)))-np.log(p['u']))+(p['eta']+p['kappa'])*X[ix]+a[ix]
        accept=rng.random(len(ix))<1/(1+np.exp(np.clip(-odds,-700,700)))
        pending[ix[accept]]=False
    raise RuntimeError('Offset hazard thinning did not converge')


def simulate(data,m,p,mode,nodes=None,post=None,reps=500,seed=52713):
    rng=np.random.default_rng(seed);L,n=data['g'].shape
    if mode=='Shared':a=np.zeros((reps,L))
    elif mode=='Population_offsets':a=rng.normal(0,p['tau_a'],(reps,L))
    elif mode=='Posterior_offsets':
        a=np.column_stack([rng.choice(nodes,size=reps,p=weights) for weights in post])
    else:raise ValueError(mode)
    # Draw once per entire lineage, never once per generation.
    x=np.empty((reps,L,n+1));g=np.empty((reps,L,n));x[:,:,0]=data['x'][:,0]
    for i in range(n):
        X=x[:,:,i]-m['x'][i]
        g[:,:,i]=sample_g(X,a.ravel(),p,rng).reshape(reps,L)
        x[:,:,i+1]=x[:,:,i]+data['q'][:,i]+g[:,:,i]
    return x,g,a


def main():
    OUT.mkdir(parents=True,exist_ok=True);_,groups=load_data();data=cohort(groups);ids=data['ids'];n=data['g'].shape[1]
    baseline_path=BASE_OUT/'fit_results.json'
    if not baseline_path.exists():
        raise SystemExit('Shared comparator fit is missing. Run scripts/fit_empirical_detrended_hazard.py first (see project README).')
    baseline=json.loads(baseline_path.read_text())
    validids=validation_ids()
    valid=np.isin(ids,list(validids));train=~valid
    config=dict(n_lineages=len(ids),n_train=int(train.sum()),n_validation=int(valid.sum()),transitions_per_lineage=n,
                hazard='omega2 sigmoid[eta log(B*(exp(G+mean_g_i)-1)/u)+kappa log B+a_l], a_l~Normal(0,tau_a^2), G>-mean_g_i',
                estimation='Maximum marginal likelihood (empirical Bayes); integrate ONE static offset over the product of all cycle densities in a lineage. Five population parameters, empirical mean curves fixed.',
                validation='Same exploratory 134/38 whole-lineage split; means and population parameters trained only on training lineages. Score each held-out sequence by prior integration over one offset, not an in-sample posterior plug-in.',
                simulation='500 conditional replays: fixed observed initial birth, alpha and F_eff; data mean curves retained, never re-estimated on simulations. Population_offsets draws a new prior offset once per lineage. Posterior_offsets uses the full observed lineage to infer offsets and is descriptive only.',
                caveats='Balanced first-20-transition subset of 172 lineages. Empirical means and F_eff encode observed data. Offsets assumed independent of initial births and supplied context paths in prior replay. Not an autonomous joint model or a fresh confirmatory test. Hyperparameters are point estimates, not full Bayesian draws.',
                fits={})
    fitted={};posteriors={};scores={};posteriorrows=[]
    for scope,mask in [('train',train),('all',np.ones(len(ids),bool))]:
        m=means(data,mask);base=baseline['fits'][scope]['Detrended']['parameters']
        X=data['x'][:,:-1]-m['x'][:-1]
        print('FITTING',scope,flush=True)
        p,diag,nodes,post=fit(X[mask],data['g'][mask],base)
        config['fits'][scope]=dict(parameters=p,diagnostics=diag,means={k:v.tolist() for k,v in m.items()},
                                  log_threshold_sd=p['tau_a']/p['eta'])
        fitted[scope]=(p,m,base);posteriors[scope]=(nodes,post)
        print('FITTED',scope,p,'quadrature error',diag['quadrature_check_max_lineage'],flush=True)
        if scope=='train':
            scores['Static_offset']=MarginalLikelihood(X[valid],data['g'][valid],256,201).evaluate(pack(p),gradient=False)[0]
            highcheck=MarginalLikelihood(X[valid],data['g'][valid],256,301).evaluate(pack(p),gradient=False)[0]
            config['heldout_offset_quadrature_max_difference']=float(np.max(abs(highcheck-scores['Static_offset'])))
            if config['heldout_offset_quadrature_max_difference']>1e-3:raise RuntimeError('Held-out quadrature not converged')
            scores['Shared']=Likelihood(X[valid],data['g'][valid],256).scores(base).reshape(valid.sum(),n).sum(1)
            # Forecast the last ten cycles after observing only the first ten:
            # log p(full sequence)-log p(prefix), never a full-record EB offset.
            prefix=MarginalLikelihood(X[valid,:10],data['g'][valid,:10],256,201).evaluate(pack(p),gradient=False)[0]
            shared_prefix=Likelihood(X[valid,:10],data['g'][valid,:10],256).scores(base).reshape(valid.sum(),10).sum(1)
            scores['Static_offset_last10']=scores['Static_offset']-prefix
            scores['Shared_last10']=scores['Shared']-shared_prefix
        else:
            for j,identity in enumerate(ids):
                weights=post[j];mu=float(weights@nodes);sd=float(np.sqrt(weights@((nodes-mu)**2)))
                qs=nodes[np.searchsorted(np.cumsum(weights),[.025,.5,.975])]
                posteriorrows.append(dict(lineage=identity,posterior_mean=mu,posterior_sd=sd,q025=qs[0],median=qs[1],q975=qs[2],
                                          equivalent_log_birth_offset=-mu/(p['eta']+p['kappa']),
                                          observed_mean_X=float(X[j].mean())))
    rng=np.random.default_rng(31415);comparisons={}
    for label,a,b,count in [('Full_sequence','Static_offset','Shared',n),('Last10_given_first10','Static_offset_last10','Shared_last10',10)]:
        delta=(scores[a]-scores[b])/count;draw=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
        comparisons[label]=dict(mean_gain_nats_per_transition=float(delta.mean()),ci95=np.quantile(draw,[.025,.975]).tolist())
    config['heldout_comparisons']=comparisons
    pd.DataFrame(dict(lineage=ids[valid],**scores)).to_csv(OUT/'heldout_lineage_scores.csv',index=False)
    posterior=pd.DataFrame(posteriorrows);posterior.to_csv(OUT/'posterior_offsets.csv',index=False)
    config['posterior_offset_vs_mean_X_correlation']=float(np.corrcoef(posterior.posterior_mean,posterior.observed_mean_X)[0,1])
    metricrows=[];acrows=[];meanrows=[];histrows=[]
    for scope,mask,fit_scope in [('all',np.ones(len(ids),bool),'all'),('validation',valid,'train')]:
        d={k:v[mask] for k,v in data.items()};p,m,base=fitted[fit_scope]
        runs={'Observed':(d['x'][None],d['g'][None],None),
              'Shared':simulate(d,m,base,'Shared'),
              'Population_offsets':simulate(d,m,p,'Population_offsets')}
        if scope=='all':runs['Posterior_offsets']=simulate(d,m,p,'Posterior_offsets',*posteriors['all'])
        for name,(x,g,a) in runs.items():
            X=x-m['x'];G=g-m['g'];b=np.exp(x);t=g/d['a']
            values=dict(mean_b=b.mean((1,2)),sd_b=b.std((1,2)),mean_t=t.mean((1,2)),sd_t=t.std((1,2)),
                        q99_t=np.quantile(t.reshape(len(t),-1),.99,axis=1),sd_X=X.std((1,2)),sd_G=G.std((1,2)),
                        corr_X_G=corr(X[:,:,:-1].reshape(len(x),-1),G.reshape(len(x),-1)),
                        corr_b_t=corr(b[:,:,:-1].reshape(len(x),-1),t.reshape(len(x),-1)))
            for key,v in values.items():
                lo,hi=np.quantile(v,[.025,.975]);metricrows.append(dict(scope=scope,model=name,metric=key,mean=v.mean(),lo=lo,hi=hi))
            for var,y in [('X',X),('G',G),('birth',b),('time',t)]:
                ac=acf(y);draws=ac.mean(1)
                if name=='Observed':draws=ac[0,rng.integers(0,len(d['ids']),(2000,len(d['ids'])))].mean(1)
                lo,hi=np.quantile(draws,[.025,.975],axis=0);avg=ac.mean((0,1))
                acrows.extend(dict(scope=scope,model=name,variable=var,lag=k,mean=avg[k],lo=lo[k],hi=hi[k]) for k in range(11))
                for i in range(y.shape[-1]):
                    vals=y[:,:,i].mean(1);lo,hi=np.quantile(vals,[.025,.975]);meanrows.append(dict(scope=scope,model=name,variable=var,generation=i,mean=vals.mean(),lo=lo,hi=hi))
                edges={'X':np.linspace(-.8,.8,81),'G':np.linspace(-.5,1,76),'birth':np.linspace(0,6,81),'time':np.linspace(0,6,61)}[var]
                den=np.histogram(y,edges)[0]/y.size/np.diff(edges)
                histrows.extend(dict(scope=scope,model=name,variable=var,left=left,right=right,density=v) for left,right,v in zip(edges[:-1],edges[1:],den))
            if name!='Observed':
                err=np.max(abs(np.diff(X,axis=-1)-(d['q']-m['q'])-G))
                config.setdefault('simulation_checks',{}).setdefault(scope,{})[name]=dict(min_time=float(t.min()),max_centered_identity_error=float(err))
                np.savez_compressed(OUT/f'{scope}_{name}_examples.npz',X=X[:5],G=G[:5],birth=b[:5],time=t[:5],offsets=a[:5],ids=d['ids'])
        print('SIMULATED',scope,flush=True)
    for file,rows in [('metrics',metricrows),('acf_comparison',acrows),('mean_replay',meanrows),('marginal_densities',histrows)]:
        pd.DataFrame(rows).to_csv(OUT/f'{file}.csv',index=False)
    (OUT/'fit_results.json').write_text(json.dumps(config,indent=2,allow_nan=False))
    print('DONE',comparisons,flush=True)


if __name__=='__main__':main()
