#!/usr/bin/env python3
"""Audit the low-likelihood ACF fit, including birth/time cross-correlations.

Positive lag k means Corr(b_n,t_{n+k}); negative lag means earlier times.
Original retained-row replay reproduces the historical experiment. A second
gap-reset replay checks sensitivity without refitting the selected parameters.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from fit_conditional_effective_jumps import simulate
from screen_parsimonious_joint_hazards import FastReplay,METRICS,PROJECT_DIR,load_data,segments,add_effective_factors

BASE=PROJECT_DIR/'outputs/model_comparison/Glu'
OUT=BASE/'birth_size_dependent_time_hazard_acf_only_wide/cross_correlation_audit'
LAGS=np.arange(-10,11)


def pearson(x,y):
    xc=x-x.mean(axis=1,keepdims=True);yc=y-y.mean(axis=1,keepdims=True)
    return np.sum(xc*yc,axis=1)/np.sqrt(np.sum(xc*xc,axis=1)*np.sum(yc*yc,axis=1))


def ccf_statistics(rp,births,times,log_birth=False,bootstrap=0):
    b=births[:,rp.mask];t=times[:,rp.mask]
    if log_birth:b=np.log(b)
    bm=np.add.reduceat(b,rp.starts,axis=1)/rp.counts
    tm=np.add.reduceat(t,rp.starts,axis=1)/rp.counts
    bc=b-bm[:,rp.event_lineage];tc=t-tm[:,rp.event_lineage]
    den=np.sqrt(np.add.reduceat(bc*bc,rp.starts,axis=1)*np.add.reduceat(tc*tc,rp.starts,axis=1))
    results={name:np.empty((len(b),len(LAGS))) for name in ['pooled_Pearson','within_lineage_Pearson','mean_lineage_CCF']}
    boot={name:np.empty((bootstrap,len(LAGS))) for name in results} if bootstrap else None
    if bootstrap:
        if len(b)!=1:raise ValueError('Bootstrap is only for the observed dataset')
        rng=np.random.default_rng(76140);indices=rng.integers(0,len(rp.counts),(bootstrap,len(rp.counts)))
    for j,k in enumerate(LAGS):
        if k==0:left=right=np.arange(len(rp.event_lineage))
        else:
            left,right=rp.pairs[abs(k)]
            if k<0:left,right=right,left
        x=b[:,left];y=t[:,right];cx=bc[:,left];cy=tc[:,right]
        results['pooled_Pearson'][:,j]=pearson(x,y)
        results['within_lineage_Pearson'][:,j]=pearson(cx,cy)
        products=np.zeros_like(b);products[:,left]=cx*cy
        numerator=np.add.reduceat(products,rp.starts,axis=1)
        available=np.bincount(rp.event_lineage[left],minlength=len(rp.counts))>0
        valid=available&(rp.counts>=3)
        perlineage=np.divide(numerator,den,out=np.full_like(numerator,np.nan),where=den>0)
        perlineage[:,~valid]=np.nan
        results['mean_lineage_CCF'][:,j]=np.nanmean(perlineage,axis=1)
        if not bootstrap:continue
        boot['mean_lineage_CCF'][:,j]=np.nanmean(perlineage[0,indices],axis=1)
        for name,xx,yy in [('pooled_Pearson',x[0],y[0]),('within_lineage_Pearson',cx[0],cy[0])]:
            label=rp.event_lineage[left]
            sums=np.array([np.bincount(label,weights=w,minlength=len(rp.counts)) for w in [np.ones(len(left)),xx,yy,xx*xx,yy*yy,xx*yy]])
            n,sx,sy,sxx,syy,sxy=sums[:,indices].sum(axis=2)
            boot[name][:,j]=(sxy-sx*sy/n)/np.sqrt((sxx-sx*sx/n)*(syy-sy*sy/n))
    return results,boot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicates',type=int,default=200);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    _,original=load_data()
    acf=json.loads((BASE/'birth_size_dependent_time_hazard_acf_only_wide/diagnostics.json').read_text())
    mle=json.loads((BASE/'birth_size_dependent_time_hazard_effective_f/diagnostics.json').read_text())
    models={'ACF_only':dict(acf['parameters'],q=0.),'Likelihood':dict(mle['mle'],q=0.)}
    astar=float(np.exp(np.mean(np.concatenate([np.log(g.alpha) for g in original]))))
    refs={'b':acf['b_star'],'alpha':astar,'t':1.,'g':.5}
    curves=[];metrics=[];acfs=[];hists=[];cond=[]
    for design,groups in [('historical_rows',add_effective_factors(original)[0]),('gap_reset',segments(original))]:
        rp=FastReplay(groups,refs['b'],refs['alpha'],args.replicates,71711)
        obs,ob,ot=rp.observed()
        metrics.extend({'design':design,'model':'Observed','metric':key,'mean':v} for key,v in zip(METRICS,obs))
        for transform in ['birth','log_birth']:
            values,boot=ccf_statistics(rp,ob,ot,transform=='log_birth',1000)
            for statistic,array in values.items():
                lo,hi=np.quantile(boot[statistic],[.025,.975],axis=0)
                for k,v,a,z in zip(LAGS,array[0],lo,hi):curves.append({'design':design,'model':'Observed','birth_transform':transform,'statistic':statistic,'lag':k,'mean':v,'lo':a,'hi':z})
        for variable,arr in [('birth',ob),('time',ot)]:
            curve=rp.acfs(arr)[0]
            acfs.extend({'design':design,'model':'Observed','variable':variable,'lag':k,'mean':v} for k,v in enumerate(curve))
        for index,(name,parameters) in enumerate(models.items()):
            b,t,_=simulate(rp,parameters,None,refs,seed=962011+10000*index+(100000 if design=='gap_reset' else 0))
            values=rp.statistics(b,t)
            metrics.extend({'design':design,'model':name,'metric':key,'mean':v,'lo':lo,'hi':hi} for key,v,lo,hi in zip(METRICS,values.mean(0),*np.quantile(values,[.025,.975],axis=0)))
            for transform in ['birth','log_birth']:
                stats,_=ccf_statistics(rp,b,t,transform=='log_birth')
                for statistic,array in stats.items():
                    lo,hi=np.quantile(array,[.025,.975],axis=0)
                    for k,v,a,z in zip(LAGS,array.mean(0),lo,hi):curves.append({'design':design,'model':name,'birth_transform':transform,'statistic':statistic,'lag':k,'mean':v,'lo':a,'hi':z})
            for variable,arr in [('birth',b),('time',t)]:
                a=rp.acfs(arr);lo,hi=np.quantile(a,[.025,.975],axis=0)
                acfs.extend({'design':design,'model':name,'variable':variable,'lag':k,'mean':v,'lo':l,'hi':h} for k,v,l,h in zip(range(31),a.mean(0),lo,hi))
            if design=='historical_rows':
                for variable,x,observed,edges in [('birth',b[:,rp.mask],ob[:,rp.mask],np.arange(0,6.01,.1)),('time',t[:,rp.mask],ot[:,rp.mask],np.arange(0,12.01,1/3))]:
                    hist=np.histogram(x,bins=edges)[0]/x.size/np.diff(edges);oh=np.histogram(observed,bins=edges)[0]/observed.size/np.diff(edges)
                    hists.extend({'model':name,'variable':variable,'left':l,'right':r,'density':v,'observed_density':ov} for l,r,v,ov in zip(edges[:-1],edges[1:],hist,oh))
                edges=np.r_[-np.inf,np.quantile(ob[:,rp.mask],np.arange(.1,1,.1)),np.inf]
                for l,r in zip(edges[:-1],edges[1:]):
                    mask=(b[:,rp.mask]>=l)&(b[:,rp.mask]<r);om=(ob[:,rp.mask]>=l)&(ob[:,rp.mask]<r)
                    vals=t[:,rp.mask][mask];ovals=ot[:,rp.mask][om]
                    cond.append({'model':name,'birth_left':l,'birth_right':r,'observed_mean_birth':float(ob[:,rp.mask][om].mean()),'observed_mean_time':float(ovals.mean()),'simulated_mean_time':float(vals.mean()),'observed_sd_time':float(ovals.std()),'simulated_sd_time':float(vals.std())})
            print(design,name,dict(zip(METRICS,values.mean(0).round(5))),flush=True)
    pd.DataFrame(curves).to_csv(OUT/'cross_correlations.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'joint_metrics.csv',index=False)
    pd.DataFrame(acfs).to_csv(OUT/'acf_comparison.csv',index=False)
    pd.DataFrame(hists).to_csv(OUT/'marginal_densities.csv',index=False)
    pd.DataFrame(cond).to_csv(OUT/'conditional_time_by_birth.csv',index=False)
    config={'models':models,'replicates':args.replicates,'refitting':False,
            'lag_convention':'Positive k: birth b_n paired with later time t_(n+k); negative k: birth paired with earlier time.',
            'pooled_Pearson':'Pearson correlation of all within-lineage lag pairs, cycle/pair weighted.',
            'within_lineage_Pearson':'Subtract each full original lineage mean separately from b and t, then compute pooled lag-pair Pearson correlation.',
            'mean_lineage_CCF':'Within original lineage, sum centered lag-pair products / sqrt(full birth sum squares * full time sum squares); average available lineages equally (length >=3).',
            'intervals':'Observed: 1000 whole-lineage bootstrap draws. Simulations: 2.5/97.5 percentiles of 200 datasets, fixed parameters; no fitted-parameter uncertainty.',
            'historical_rows':'All retained rows regarded as adjacent, as in the original ACF-only fit; 50 generation gaps bridged.',
            'gap_reset':'Reset at missing-generation gaps; omit lag pairs spanning gaps while retaining original-lineage centering and weighting.'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
