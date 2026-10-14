#!/usr/bin/env python3
"""Matched first/second-order empirical birth transitions with fixed alpha/F."""
import argparse
import itertools
import json
import numpy as np
import pandas as pd
from fit_empirical_birth_transitions import (PROJECT_DIR,load_data,segments,FastReplay,corr,
    BirthTransitionKDE,normal_sf,interval,ccf_statistics,LAGS,HB,HA,HF,HG)

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/second_order_birth_transitions'
HOLDER=[.04,.08,.16,.32,np.inf]
GRID=list(itertools.product(HB,HA,HF,HG,HOLDER))
KEYS=['h_birth','h_alpha_current','h_factor','h_growth','h_older_birth']


def dataset(groups):
    rows=[]
    for group in groups:
        for n in range(2,len(group)):
            rows.append({'lineage':group.lineage.iloc[n],'generation':group.generation.iloc[n],
                'birth':group.length_birth.iloc[n],'previous_birth':group.length_birth.iloc[n-1],
                'older_birth':group.length_birth.iloc[n-2],'alpha':group.alpha.iloc[n],
                'previous_alpha':group.alpha.iloc[n-1],'previous_time':group.time.iloc[n-1],
                'F':group.f_effective.iloc[n]})
    tab=pd.DataFrame(rows);tab['g']=np.log(tab.birth/(tab.previous_birth*tab.F));return tab


def make_model(tab,h):
    return BirthTransitionKDE(tab.previous_birth,tab.alpha,tab.F,tab.birth,h[:4],tab.older_birth,h[4])


def cv_scores(train,valid):
    columns=['previous_birth','alpha','F','older_birth'];x=np.log(train[columns].to_numpy());q=np.log(valid[columns].to_numpy())
    g=train.g.to_numpy();gq=valid.g.to_numpy();scores=np.zeros(len(GRID))
    indices={h:i for i,h in enumerate(GRID)}
    for start in range(0,len(valid),128):
        sl=slice(start,start+128);d=(q[sl,None]-x[None])**2;delta=gq[sl,None]-g
        kernels=[np.exp(-.5*(delta/h)**2)/(np.sqrt(2*np.pi)*h*normal_sf(-g/h)) for h in HG]
        for hb,ha,hf,ho in itertools.product(HB,HA,HF,HOLDER):
            logits=-.5*(d[:,:,0]/hb**2+d[:,:,1]/ha**2+d[:,:,2]/hf**2+d[:,:,3]/ho**2)
            logits-=logits.max(1,keepdims=True);w=np.exp(logits);w/=w.sum(1,keepdims=True)
            for hg,kernel in zip(HG,kernels):
                density=np.sum(w*kernel,axis=1)/valid.birth.to_numpy()[sl]
                scores[indices[(hb,ha,hf,hg,ho)]]+=np.log(np.maximum(density,1e-300)).sum()
    return scores


def simulate(groups,kde,replicates,seed=732101):
    rng=np.random.default_rng(seed);maxn=max(map(len,groups));b=np.zeros((replicates,len(groups),maxn));t=np.full_like(b,np.nan)
    outside=0;olderoutside=0;total=0;maxerr=0.
    for j,group in enumerate(groups):
        b[:,j,:2]=group.length_birth.to_numpy()[:2]
        for n in range(2,len(group)):
            previous=b[:,j,n-1];older=b[:,j,n-2];a=group.alpha.iloc[n];f=group.f_effective.iloc[n]
            outside+=int(np.sum((np.log(previous)<kde.logb.min())|(np.log(previous)>kde.logb.max())))
            olderoutside+=int(np.sum((np.log(older)<kde.logolder.min())|(np.log(older)>kde.logolder.max())));total+=len(previous)
            birth,g=kde.sample_fixed_context(previous,a,f,rng,b_older=older);b[:,j,n]=birth
            reconstructed=np.log(birth/(previous*f))/group.alpha.iloc[n-1]
            if np.any(reconstructed<=0):raise AssertionError('Nonpositive reconstructed time')
            t[:,j,n-1]=reconstructed;maxerr=max(maxerr,float(np.max(abs(group.alpha.iloc[n-1]*reconstructed-g))))
        if (j+1)%100==0:print('simulated segments',j+1,'/',len(groups),flush=True)
    return b,t,{'fraction_birth_queries_outside':outside/total,'fraction_older_queries_outside':olderoutside/total,
        'maximum_growth_reconstruction_error':maxerr,'mixture_draws':kde.draws,'exact_dense_fallback_draws':kde.fallback_draws}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replicates',type=int,default=200);parser.add_argument('--reuse-bandwidths',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);_,original=load_data();groups=[g for g in segments(original) if len(g)>=3];tab=dataset(groups)
    prior=json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/empirical_birth_transitions/fit_results.json').read_text());validids=set(prior['validation_ids'])
    train=tab[~tab.lineage.isin(validids)].reset_index(drop=True);valid=tab[tab.lineage.isin(validids)].reset_index(drop=True)
    if args.reuse_bandwidths:
        config=json.loads((OUT/'fit_results.json').read_text());chosen={k:[np.inf if x is None else x for x in h] for k,h in config['bandwidths'].items()}
    else:
        ids=np.sort(train.lineage.unique());np.random.default_rng(72910).shuffle(ids);assignment={v:i%5 for i,v in enumerate(ids)};fold=train.lineage.map(assignment).to_numpy()
        total=np.zeros(len(GRID));rows=[]
        for k in range(5):
            scores=cv_scores(train[fold!=k],train[fold==k]);total+=scores
            rows.extend(dict(zip(KEYS,h),fold=k,n=int(np.sum(fold==k)),sum_log_birth_density=float(s)) for h,s in zip(GRID,scores))
            print('Grouped CV fold',k+1,'complete',flush=True)
        first=[i for i,h in enumerate(GRID) if np.isinf(h[-1])]
        chosen={'First_order':GRID[first[int(np.argmax(total[first]))]],'Second_order':GRID[int(np.argmax(total))]}
        pd.DataFrame(rows).to_csv(OUT/'bandwidth_cv.csv',index=False)
        pd.DataFrame([dict(zip(KEYS,h),mean_log_birth_density=float(s/len(train))) for h,s in zip(GRID,total)]).to_csv(OUT/'bandwidth_cv_summary.csv',index=False)
        config={'model':'Positive-growth conditional KDE p(b_i | b_(i-1), b_(i-2), alpha_i, F_i), with 1/b_i Jacobian.',
            'bandwidths':{k:[None if np.isinf(x) else float(x) for x in h] for k,h in chosen.items()},
            'selection':'Five training-lineage CV folds, birth log density; second-order family includes first-order limit h_older=infinity. First-order control retuned and refitted on exactly the same triples.',
            'design':'Both models fix first TWO observed births of every gap-free segment, recorded alpha_i/F_i and lengths. Segments shorter than 3 excluded. Times reconstructed with alpha_(i-1); first and last times excluded for every model and data. Birth ACF checked both with seeds and using only generated births.',
            'caveat':'Fixed effective F derived from observed births/alpha/time; conditional diagnostic, not autonomous generative validation. Reused validation set is exploratory.',
            'n_train_transitions':len(train),'n_validation_transitions':len(valid),'validation_ids':sorted(validids)}
        (OUT/'fit_results.json').write_text(json.dumps(config,indent=2))
    print('SELECTED',chosen,flush=True)
    densities={name:np.log(np.maximum(make_model(train,h).density(valid.birth,valid.previous_birth,valid.alpha,valid.F,b_older=valid.older_birth),1e-300)) for name,h in chosen.items()}
    gain,lo,hi=interval(densities['Second_order']-densities['First_order'],valid.lineage.to_numpy())
    config['heldout_birth_scores']={k:float(v.mean()) for k,v in densities.items()};config['heldout_gain']={'mean':gain,'lo':lo,'hi':hi}
    pd.DataFrame(dict(lineage=valid.lineage,generation=valid.generation,**densities)).to_csv(OUT/'heldout_scores.csv',index=False)
    rows=[];acfs=[];hists=[];ccfs=[]
    for scope,sub,fitdata in [('validation',[g for g in groups if g.lineage.iloc[0] in validids],train),('all',groups,tab)]:
        brp=FastReplay(sub,2.5,.5,args.replicates,85319)
        parents=[g.iloc[1:-1].reset_index(drop=True) for g in sub];trp=FastReplay(parents,2.5,.5,args.replicates,85319)
        generated=[g.iloc[2:].reset_index(drop=True) for g in sub];grp=FastReplay(generated,2.5,.5,args.replicates,85319)
        _,ob,_=brp.observed();_,_,ot=trp.observed();runs={'Observed':(ob,ot)}
        for name,h in chosen.items():
            model=make_model(fitdata,h);b,t,diag=simulate(sub,model,args.replicates)
            runs[name]=(b,t[:,:,1:-1]);config.setdefault('simulation_diagnostics',{}).setdefault(scope,{})[name]=diag
            np.savez_compressed(OUT/f'{scope}_{name}_examples.npz',birth=b[:5],time=t[:5],birth_mask=brp.mask,alpha=brp.alpha,factor=brp.f)
            np.savez_compressed(OUT/f'{scope}_{name}_fit.npz',previous_birth=fitdata.previous_birth,older_birth=fitdata.older_birth,birth=fitdata.birth,alpha=fitdata.alpha,factor=fitdata.F,bandwidths=h)
        config.setdefault('counts',{})[scope]={'segments':len(sub),'births':int(brp.mask.sum()),'generated_births':int(grp.mask.sum()),'times':int(trp.mask.sum())}
        for name,(b,t) in runs.items():
            parentb=b[:,:,1:-1];childb=b[:,:,2:];bv=b[:,brp.mask];tv=t[:,trp.mask];pb=parentb[:,trp.mask];cb=childb[:,trp.mask]
            av=np.broadcast_to(trp.alpha[None][:,trp.mask],tv.shape)
            values={'mean_b':bv.mean(1),'sd_b':bv.std(1),'mean_t':tv.mean(1),'sd_t':tv.std(1),'q99_t':np.quantile(tv,.99,axis=1),
                'corr_b_t':corr(pb,tv),'corr_t_next_b':corr(tv,cb),'corr_alpha_t':corr(av,tv)}
            for var,arr,rp in [('birth',b,brp),('time',t,trp),('birth_generated_only',b[:,:,2:],grp)]:
                ac=rp.acfs(arr);l,u=np.quantile(ac,[.025,.975],axis=0)
                for lag in [1,2,3,5,10]:values[var+'_acf'+str(lag)]=ac[:,lag]
                acfs.extend({'scope':scope,'model':name,'variable':var,'lag':k,'mean':v,'lo':a if name!='Observed' else np.nan,'hi':z if name!='Observed' else np.nan} for k,(v,a,z) in enumerate(zip(ac.mean(0),l,u)))
                if var=='birth_generated_only':continue
                edges=np.arange(0,6.01,.1) if var=='birth' else np.arange(0,12.01,1/6);x=arr[:,rp.mask];den=np.histogram(x,bins=edges)[0]/x.size/np.diff(edges)
                hists.extend({'scope':scope,'model':name,'variable':var,'left':a,'right':z,'density':v} for a,z,v in zip(edges[:-1],edges[1:],den))
            for key,v in values.items():
                l,u=np.quantile(v,[.025,.975]);rows.append({'scope':scope,'model':name,'metric':key,'mean':v.mean(),'lo':l if name!='Observed' else np.nan,'hi':u if name!='Observed' else np.nan})
            stats,boot=ccf_statistics(trp,parentb,t,False,1000 if name=='Observed' else 0)
            for statistic,v in stats.items():
                l,u=np.quantile(boot[statistic] if name=='Observed' else v,[.025,.975],axis=0)
                ccfs.extend({'scope':scope,'model':name,'statistic':statistic,'lag':k,'mean':value,'lo':a,'hi':z} for k,value,a,z in zip(LAGS,v.mean(0),l,u))
            print('RESULT',scope,name,{k:round(float(v.mean()),5) for k,v in values.items()},flush=True)
    for filename,data in [('joint_metrics.csv',rows),('acf_comparison.csv',acfs),('marginal_densities.csv',hists),('cross_correlations.csv',ccfs)]:pd.DataFrame(data).to_csv(OUT/filename,index=False)
    config['replicates']=args.replicates;(OUT/'fit_results.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
