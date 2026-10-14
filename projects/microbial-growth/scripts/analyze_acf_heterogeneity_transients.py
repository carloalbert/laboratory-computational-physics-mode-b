#!/usr/bin/env python3
"""Separate lineage-rate heterogeneity, aligned drift, and within-lineage memory.

Covariance-model scores are scale-free Gaussian restricted (directional)
likelihoods, not birth-transition log densities. Means/scales are nuisance
lineage quantities in all covariance models. No causal attribution is implied.
"""
import json
import itertools
import numpy as np
import pandas as pd
from screen_parsimonious_joint_hazards import PROJECT_DIR,load_data

OUT=PROJECT_DIR/'outputs/summary/Glu_acf_heterogeneity_transients'
RHO=np.r_[np.arange(0,.951,.025),.975,.99,.995]
LAGS=np.arange(11)


def prepare(groups):
    return [{'id':float(g.lineage.iloc[0]),'g':g.generation.to_numpy(float)-g.generation.iloc[0],
        'raw':g.length_birth.to_numpy(float),'y':np.log(g.length_birth.to_numpy(float))} for g in groups if len(g)>=6]


def covariance_score(items,fast,slow=None,weight=None,return_parameters=False):
    """REML with intercept; profile scale, common dimensional constants omitted.

Equivalent model comparisons result from integrating each lineage mean and
scale against the same flat/log-scale priors. Includes irregular-generation
distances, so missing cycles are marginalized, never treated as adjacent.
"""
    scores=[];parameters=[];cache={}
    for item in items:
        gen=item['g'];n=len(gen);key=tuple(gen)
        if key not in cache:
            d=np.abs(gen[:,None]-gen);cov=fast**d
            if slow is not None:cov=weight*cov+(1-weight)*slow**d
            chol=np.linalg.cholesky(cov);inv=np.linalg.solve(chol.T,np.linalg.solve(chol,np.eye(n)))
            one=inv.sum(0);s=one.sum();logdet=2*np.log(np.diag(chol)).sum()
            cache[key]=(inv,one,s,logdet,chol)
        inv,one,s,logdet,chol=cache[key];y=item['y'];mu=one@y/s;r=y-mu
        q=float(r@inv@r);sigma2=q/(n-1)
        scores.append(-.5*(logdet+np.log(s)+(n-1)*np.log(sigma2)))
        if return_parameters:parameters.append((mu,sigma2,chol))
    return (np.array(scores),parameters) if return_parameters else np.array(scores)


def fit_covariances(items):
    ll=np.array([covariance_score(items,r) for r in RHO]);single=int(np.argmax(ll.sum(1)))
    best=(-np.inf,None)
    for i,j in itertools.combinations(range(len(RHO)),2):
        for pi in np.arange(.1,1,.1):
            value=float(np.logaddexp(np.log(pi)+ll[i],np.log1p(-pi)+ll[j]).sum())
            if value>best[0]:best=(value,(float(RHO[i]),float(RHO[j]),float(pi)))
    hetero=best[1]
    best=(-np.inf,None)
    for f,s,w in itertools.product([0,.15,.3,.45,.6,.75],[.6,.75,.85,.9,.95,.98,.995],[.1,.25,.5,.75,.9]):
        if f>=s:continue
        value=float(covariance_score(items,f,s,w).sum())
        if value>best[0]:best=(value,(f,s,w))
    for df,ds,dw in [(.05,.02,.075),(.02,.008,.025)]:
        f,s,w=best[1]
        for ff,ss,ww in itertools.product(np.clip(f+np.arange(-1,2)*df,0,.97),np.clip(s+np.arange(-1,2)*ds,.01,.999),np.clip(w+np.arange(-1,2)*dw,.025,.975)):
            if ff>=ss:continue
            value=float(covariance_score(items,ff,ss,ww).sum())
            if value>best[0]:best=(value,(float(ff),float(ss),float(ww)))
    return {'Common_AR1':{'rho':float(RHO[single])},'Heterogeneous_AR1':{'fast':hetero[0],'slow':hetero[1],'fraction_fast':hetero[2]},
        'Two_modes_within':{'fast':best[1][0],'slow':best[1][1],'weight_fast':best[1][2]}}


def model_scores(items,name,p):
    if name=='Common_AR1':return covariance_score(items,p['rho'])
    if name=='Two_modes_within':return covariance_score(items,p['fast'],p['slow'],p['weight_fast'])
    return np.logaddexp(np.log(p['fraction_fast'])+covariance_score(items,p['fast']),np.log1p(-p['fraction_fast'])+covariance_score(items,p['slow']))


def acf(y,gen):
    y=y-y.mean(-1,keepdims=True);den=np.sum(y*y,axis=-1);out=np.full((*y.shape[:-1],11),np.nan);out[...,0]=1.
    segment=np.r_[0,np.cumsum(np.diff(gen)!=1)]
    for k in range(1,11):
        i,j=np.where((gen[None,:]-gen[:,None]==k)&(segment[None,:]==segment[:,None]))
        if len(i):out[...,k]=np.sum(y[...,i]*y[...,j],axis=-1)/den
    return out


def summaries(items,series):
    # Each series has shape (replicate, observations). Bootstrap units are
    # whole original lineages, not gap fragments or individual lag pairs.
    curves=[];detrended=[];rows=[];early=[];late=[];halfids=[]
    for item,y in zip(items,series):
        x=item['g'];x=(x-x.mean())/(x.max()-x.min());yc=y-y.mean(1,keepdims=True)
        slope=(yc@x)/(x@x);residual=yc-slope[:,None]*x
        curves.append(acf(y,item['g']));detrended.append(acf(residual,item['g']))
        n=y.shape[1];third=max(n//3,2)
        rows.append(np.column_stack([y[:,-third:].mean(1)-y[:,:third].mean(1),slope,
            1-(residual*residual).sum(1)/(yc*yc).sum(1)]))
        if n>=20:
            h=n//2;early.append(acf(y[:,:h],item['g'][:h])[:,1]);late.append(acf(y[:,-h:],item['g'][-h:])[:,1]);halfids.append(item['id'])
    curves=np.stack(curves,1);detrended=np.stack(detrended,1);rows=np.stack(rows,1)
    early=np.stack(early,1);late=np.stack(late,1);ec=early-early.mean(1,keepdims=True);lc=late-late.mean(1,keepdims=True)
    ratecorr=(ec*lc).sum(1)/np.sqrt((ec*ec).sum(1)*(lc*lc).sum(1))
    metrics={'late_minus_early_mean':rows[:,:,0].mean(1),'fraction_declining':(rows[:,:,0]<0).mean(1),
        'mean_end_to_end_slope':rows[:,:,1].mean(1),'mean_trend_R2':rows[:,:,2].mean(1),
        'early_half_acf1':early.mean(1),'late_half_acf1':late.mean(1),'half_acf1_change':(late-early).mean(1),'half_acf1_correlation':ratecorr}
    return curves,detrended,rows,metrics,(early,late,halfids)


def simulate(items,name,p,reps=500,seed=1281):
    rng=np.random.default_rng(seed)
    if name=='Common_AR1':_,pars=covariance_score(items,p['rho'],return_parameters=True)
    elif name=='Two_modes_within':_,pars=covariance_score(items,p['fast'],p['slow'],p['weight_fast'],True)
    else:
        l1,par1=covariance_score(items,p['fast'],return_parameters=True);l2,par2=covariance_score(items,p['slow'],return_parameters=True)
        post=np.exp(np.log(p['fraction_fast'])+l1-np.logaddexp(np.log(p['fraction_fast'])+l1,np.log1p(-p['fraction_fast'])+l2))
    result=[]
    for j,item in enumerate(items):
        n=len(item['y']);z=rng.normal(size=(reps,n))
        if name!='Heterogeneous_AR1':
            mu,sig,chol=pars[j];y=mu+np.sqrt(sig)*z@chol.T
        else:
            mu,sig,chol=par1[j];y=mu+np.sqrt(sig)*z@chol.T
            use=rng.random(reps)>post[j];mu,sig,chol=par2[j];y[use]=mu+np.sqrt(sig)*z[use]@chol.T
        result.append(y)
    return result


def bootstrap(values,rng,n=3000):
    # Input lineage x statistic; NaN denotes no lag pair in a short lineage.
    indices=rng.integers(0,len(values),(n,len(values)))
    return np.nanmean(values[indices],axis=1)


def aligned_common_trend(items,series):
    # A shared trajectory aligned to the first recorded generation, with
    # lineage intercepts removed. Fit on a grid, descriptive only.
    best=None
    for tau in [.5,1.,2.,3.,5.,8.,12.,20.,40.,80.]:
        xs=[np.exp(-d['g']/tau) for d in items];xs=[x-x.mean() for x in xs]
        ys=[y-y.mean() for y in series];beta=sum(float(x@y) for x,y in zip(xs,ys))/sum(float(x@x) for x in xs)
        score=sum(float(np.sum((y-beta*x)**2)) for x,y in zip(xs,ys))
        if best is None or score<best[0]:best=(score,tau,beta,xs)
    _,tau,beta,xs=best
    return tau,beta,[y-beta*x for y,x in zip(series,xs)]


def main():
    OUT.mkdir(parents=True,exist_ok=True);_,groups=load_data();items=prepare(groups)
    validids=set(json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/empirical_birth_transitions/fit_results.json').read_text())['validation_ids'])
    train=[d for d in items if d['id'] not in validids];valid=[d for d in items if d['id'] in validids]
    fits={}
    for scope,subset in [('train',train),('all',items)]:
        fits[scope]=fit_covariances(subset);print('FITTED',scope,fits[scope],flush=True)
    config={'fits':fits,'n_lineages':len(items),'n_observations':sum(len(d['y']) for d in items),
        'score':'Scale-free restricted Gaussian covariance likelihood on log birth, with separate nuisance intercept and scale in each lineage. Constants shared by models omitted. This scores within-lineage covariance shape, not one-step predictive birth density.',
        'caveats':'Exploratory reused validation split. Stationary two-mode covariance is not a deterministic transient. Detrending can remove stationary slow fluctuations. Parametric checks condition on fitted nuisance parameters and do not include fitting uncertainty.'}
    rng=np.random.default_rng(19129);scores={name:model_scores(valid,name,p) for name,p in fits['train'].items()}
    scoretable=pd.DataFrame({'lineage':[d['id'] for d in valid],**scores});scoretable.to_csv(OUT/'heldout_covariance_scores.csv',index=False)
    config['heldout_comparisons']={}
    for a,b in [('Heterogeneous_AR1','Common_AR1'),('Two_modes_within','Common_AR1'),('Two_modes_within','Heterogeneous_AR1')]:
        delta=scores[a]-scores[b];draw=bootstrap(delta[:,None],rng)[:,0]
        config['heldout_comparisons'][a+' minus '+b]={'n_lineages':len(valid),'mean_gain_per_lineage':float(delta.mean()),'total_gain':float(delta.sum()),'ci95':np.quantile(draw,[.025,.975]).tolist()}
    acrows=[];metricrows=[];diagnostics={}
    for transform in ['raw','log']:
        observed=[d['raw'][None,:] if transform=='raw' else d['y'][None,:] for d in items]
        cv,dc,stats,metrics,halves=summaries(items,observed)
        bootindices=rng.integers(0,len(items),(3000,len(items)))
        for name,arr in [('Within_lineage',cv),('Linearly_detrended',dc)]:
            draws=np.nanmean(arr[0,bootindices],axis=1);lo,hi=np.quantile(draws,[.025,.975],axis=0)
            acrows.extend({'transform':transform,'model':'Observed','processing':name,'lag':k,'mean':v,'lo':l,'hi':h} for k,v,l,h in zip(LAGS,np.nanmean(arr[0],axis=0),lo,hi))
        tau,beta,res=aligned_common_trend(items,[y[0] for y in observed]);common=np.stack([acf(y[None],d['g'])[0] for y,d in zip(res,items)])
        draws=np.nanmean(common[bootindices],axis=1);lo,hi=np.quantile(draws,[.025,.975],axis=0)
        acrows.extend({'transform':transform,'model':'Observed','processing':'Common_exponential_removed','lag':k,'mean':v,'lo':l,'hi':h} for k,v,l,h in zip(LAGS,np.nanmean(common,axis=0),lo,hi))
        diagnostics[transform]={'common_exp_tau_generations':tau,'common_exp_amplitude':beta}
        for key,v in metrics.items():
            if key in ['late_minus_early_mean','mean_end_to_end_slope','mean_trend_R2']:
                col={'late_minus_early_mean':0,'mean_end_to_end_slope':1,'mean_trend_R2':2}[key];draw=stats[0,bootindices,col].mean(1)
            elif key=='fraction_declining':draw=(stats[0,bootindices,0]<0).mean(1)
            else:
                early,late,ids=halves;ind=rng.integers(0,len(ids),(3000,len(ids)));e=early[0,ind];l=late[0,ind]
                if key=='early_half_acf1':draw=e.mean(1)
                elif key=='late_half_acf1':draw=l.mean(1)
                elif key=='half_acf1_change':draw=(l-e).mean(1)
                else:
                    e-=e.mean(1,keepdims=True);l-=l.mean(1,keepdims=True);draw=(e*l).sum(1)/np.sqrt((e*e).sum(1)*(l*l).sum(1))
            lo,hi=np.quantile(draw,[.025,.975]);metricrows.append({'transform':transform,'model':'Observed','metric':key,'mean':float(v[0]),'lo':lo,'hi':hi})
        lineage=pd.DataFrame({'lineage':[d['id'] for d in items],'n':[len(d['y']) for d in items],'late_minus_early':stats[0,:,0],'end_to_end_slope':stats[0,:,1],'trend_R2':stats[0,:,2]})
        lineage.to_csv(OUT/f'{transform}_lineage_statistics.csv',index=False)
        pd.DataFrame({'lineage':halves[2],'early_acf1':halves[0][0],'late_acf1':halves[1][0]}).to_csv(OUT/f'{transform}_half_acfs.csv',index=False)
    # Stationary nulls preserve actual lineage lengths and generation gaps.
    for name,p in fits['all'].items():
        sims=simulate(items,name,p)
        for transform in ['raw','log']:
            ys=[np.exp(y) for y in sims] if transform=='raw' else sims
            cv,dc,stats,metrics,halves=summaries(items,ys)
            for processing,arr in [('Within_lineage',cv),('Linearly_detrended',dc)]:
                perrep=np.nanmean(arr,axis=1);lo,hi=np.quantile(perrep,[.025,.975],axis=0)
                acrows.extend({'transform':transform,'model':name,'processing':processing,'lag':k,'mean':v,'lo':l,'hi':h} for k,v,l,h in zip(LAGS,perrep.mean(0),lo,hi))
            for key,v in metrics.items():
                lo,hi=np.quantile(v,[.025,.975]);metricrows.append({'transform':transform,'model':name,'metric':key,'mean':v.mean(),'lo':lo,'hi':hi})
        print('SIMULATED',name,flush=True)
    config['aligned_common_trend']=diagnostics
    # Robustness of the directional drift. Each condition redefines early/late
    # thirds after filtering; uncertainty resamples entire original lineages.
    robust=[]
    for condition,subset,drop in [('at_least_6',items,0),('at_least_20',[d for d in items if len(d['y'])>=20],0),
        ('gap_free',[d for d in items if np.all(np.diff(d['g'])==1)],0),
        ('drop_first_3',[d for d in items if len(d['y'])>=9],3),('drop_first_5',[d for d in items if len(d['y'])>=11],5)]:
        vals=[]
        for d in subset:
            y=d['raw'][drop:];third=max(len(y)//3,2);a=y[:third].mean();z=y[-third:].mean();vals.append([z-a,(z-a)/a,float(z>a)])
        v=np.array(vals);bs=bootstrap(v,rng);low,high=np.quantile(bs,[.025,.975],axis=0)
        for j,metric in enumerate(['late_minus_early','relative_increase','fraction_increasing']):
            robust.append({'condition':condition,'n_lineages':len(v),'metric':metric,'mean':v[:,j].mean(),'lo':low[j],'hi':high[j]})
    pd.DataFrame(robust).to_csv(OUT/'drift_robustness.csv',index=False)
    # Progress-bin means first centered within lineage: no compositional
    # shift from a changing mixture of lineage-specific average sizes.
    bins=[]
    for d in items:
        y=d['raw'];pos=np.arange(len(y))/(len(y)-1);index=np.minimum((pos*8).astype(int),7)
        bins.append([np.mean((y-y.mean())[index==k]) if np.any(index==k) else np.nan for k in range(8)])
    bins=np.array(bins);draw=bootstrap(bins,rng);low,high=np.quantile(draw,[.025,.975],axis=0)
    pd.DataFrame({'progress':(np.arange(8)+.5)/8,'mean':np.nanmean(bins,axis=0),'lo':low,'hi':high}).to_csv(OUT/'birth_progress_trend.csv',index=False)
    # Shared drift is fitted on training lineages and removed from unseen
    # validation lineages, to separate its effect from per-lineage detrending.
    for scope,subset in [('train',train),('validation',valid)]:
        if scope=='train':tau,beta,_=aligned_common_trend(subset,[d['raw'] for d in subset])
        config.setdefault('crossfit_common_trend',{})[scope]={'tau':tau,'amplitude':beta}
        for processing in ['Before','After']:
            arrays=[]
            for d in subset:
                x=np.exp(-d['g']/tau);y=d['raw'] if processing=='Before' else d['raw']-beta*(x-x.mean())
                arrays.append(acf(y[None],d['g'])[0])
            arrays=np.array(arrays);draw=bootstrap(arrays,rng);low,high=np.quantile(draw,[.025,.975],axis=0)
            for k,v,l,h in zip(LAGS,np.nanmean(arrays,axis=0),low,high):
                acrows.append({'transform':'raw','model':scope+'_shared_trend','processing':processing,'lag':k,'mean':v,'lo':l,'hi':h})
    pd.DataFrame(acrows).to_csv(OUT/'acf_comparisons.csv',index=False);pd.DataFrame(metricrows).to_csv(OUT/'drift_and_half_statistics.csv',index=False)
    (OUT/'diagnostics.json').write_text(json.dumps(config,indent=2));print(json.dumps(config,indent=2),flush=True)


if __name__=='__main__':main()
