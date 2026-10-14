#!/usr/bin/env python3
"""Second-order empirical births with an endogenous elapsed-time predictor."""
import argparse
import itertools
import json
import numpy as np
import pandas as pd
from fit_second_order_birth_transitions import (PROJECT_DIR,load_data,segments,FastReplay,corr,BirthTransitionKDE,
    normal_sf,interval,ccf_statistics,LAGS,HB,HA,HF,HG,HOLDER,dataset as second_dataset)

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/elapsed_birth_transitions'
HT=[1.,2.,4.,8.,16.,32.,np.inf]
GRID=list(itertools.product(HB,HA,HF,HG,HOLDER,HT))
KEYS=['h_birth','h_alpha_current','h_factor','h_growth','h_older_birth','h_elapsed_hours']


def dataset(groups):
    tab=second_dataset(groups)
    # One uninterrupted record per original lineage. At outcome birth i,
    # predictor S_(i-1) excludes the duration t_(i-1) to be generated.
    clocks={g.lineage.iloc[0]:np.r_[0.,np.cumsum(g.time.to_numpy())] for g in groups}
    times=[]
    for g in groups:times.extend(clocks[g.lineage.iloc[0]][1:len(g)-1])
    tab['elapsed']=times
    return tab


def make_model(tab,h):
    return BirthTransitionKDE(tab.previous_birth,tab.alpha,tab.F,tab.birth,h[:4],tab.older_birth,h[4],tab.elapsed,h[5])


def cv_scores(train,valid):
    cols=['previous_birth','alpha','F','older_birth'];x=np.log(train[cols].to_numpy());q=np.log(valid[cols].to_numpy())
    g=train.g.to_numpy();gq=valid.g.to_numpy();scores=np.zeros(len(GRID));indices={h:i for i,h in enumerate(GRID)}
    for start in range(0,len(valid),128):
        sl=slice(start,start+128);d=(q[sl,None]-x[None])**2;delta=gq[sl,None]-g
        dt=(valid.elapsed.to_numpy()[sl,None]-train.elapsed.to_numpy()[None])**2
        kernels=[np.exp(-.5*(delta/h)**2)/(np.sqrt(2*np.pi)*h*normal_sf(-g/h)) for h in HG]
        for hb,ha,hf,ho in itertools.product(HB,HA,HF,HOLDER):
            base=-.5*(d[:,:,0]/hb**2+d[:,:,1]/ha**2+d[:,:,2]/hf**2+d[:,:,3]/ho**2)
            for ht in HT:
                logits=base-.5*dt/ht**2;logits-=logits.max(1,keepdims=True);w=np.exp(logits);w/=w.sum(1,keepdims=True)
                for hg,kernel in zip(HG,kernels):
                    density=np.sum(w*kernel,axis=1)/valid.birth.to_numpy()[sl]
                    scores[indices[(hb,ha,hf,hg,ho,ht)]]+=np.log(np.maximum(density,1e-300)).sum()
    return scores


def simulate(groups,kde,replicates,seed=837291,cycle_models=None):
    rng=np.random.default_rng(seed);maxn=max(map(len,groups));b=np.zeros((replicates,len(groups),maxn));t=np.full_like(b,np.nan);elapsed=np.full_like(b,np.nan)
    outside=0;total=0;maxerr=0.;maxclock=0.
    for j,g in enumerate(groups):
        b[:,j,:2]=g.length_birth.to_numpy()[:2];elapsed[:,j,0]=0.
        # First time is determined by the seeded birth pair and fixed F/alpha.
        # It initializes the clock but is excluded from all time diagnostics.
        seedtime=np.log(g.length_birth.iloc[1]/(g.length_birth.iloc[0]*g.f_effective.iloc[1]))/g.alpha.iloc[0]
        elapsed[:,j,1]=seedtime;clock=np.full(replicates,seedtime)
        for n in range(2,len(g)):
            active=kde if cycle_models is None else cycle_models[n-1]
            previous=b[:,j,n-1];older=b[:,j,n-2];a=g.alpha.iloc[n];f=g.f_effective.iloc[n]
            outside+=int(np.sum((clock<active.elapsed.min())|(clock>active.elapsed.max())));total+=len(clock)
            birth,growth=active.sample_fixed_context(previous,a,f,rng,b_older=older,elapsed=clock)
            b[:,j,n]=birth;duration=np.log(birth/(previous*f))/g.alpha.iloc[n-1]
            if np.any(duration<=0):raise AssertionError('Nonpositive reconstructed duration')
            t[:,j,n-1]=duration;clock=clock+duration;elapsed[:,j,n]=clock
            maxerr=max(maxerr,float(np.max(abs(g.alpha.iloc[n-1]*duration-growth))));maxclock=max(maxclock,float(clock.max()))
        if (j+1)%75==0:print('simulated lineages',j+1,'/',len(groups),flush=True)
    used=[kde] if cycle_models is None else list(cycle_models.values())
    return b,t,elapsed,{'elapsed_query_fraction_outside_training_range':outside/total,'max_elapsed_hours':maxclock,
        'maximum_growth_reconstruction_error':maxerr,'mixture_draws':sum(m.draws for m in used),'exact_dense_fallback_draws':sum(m.fallback_draws for m in used)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replicates',type=int,default=200);parser.add_argument('--reuse-bandwidths',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);_,original=load_data()
    gapfree=[g for g in original if len(g)>=3 and np.all(np.diff(g.generation)==1)];groups=segments(gapfree);tab=dataset(groups)
    previous=json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/second_order_birth_transitions/fit_results.json').read_text());validids=set(previous['validation_ids'])
    train=tab[~tab.lineage.isin(validids)].reset_index(drop=True);valid=tab[tab.lineage.isin(validids)].reset_index(drop=True)
    if args.reuse_bandwidths:
        config=json.loads((OUT/'fit_results.json').read_text());chosen={k:[np.inf if x is None else x for x in h] for k,h in config['bandwidths'].items()}
    else:
        ids=np.sort(train.lineage.unique());np.random.default_rng(72910).shuffle(ids);assignment={v:i%5 for i,v in enumerate(ids)};fold=train.lineage.map(assignment).to_numpy()
        total=np.zeros(len(GRID));rows=[]
        for k in range(5):
            values=cv_scores(train[fold!=k],train[fold==k]);total+=values
            rows.extend(dict(zip(KEYS,h),fold=k,n=int(np.sum(fold==k)),sum_log_birth_density=float(s)) for h,s in zip(GRID,values))
            print('Grouped CV fold',k+1,'complete',flush=True)
        null=[i for i,h in enumerate(GRID) if np.isinf(h[-1])]
        chosen={'No_elapsed':GRID[null[int(np.argmax(total[null]))]],'Elapsed':GRID[int(np.argmax(total))]}
        pd.DataFrame(rows).to_csv(OUT/'bandwidth_cv.csv',index=False)
        pd.DataFrame([dict(zip(KEYS,h),mean_log_birth_density=float(s/len(train))) for h,s in zip(GRID,total)]).to_csv(OUT/'bandwidth_cv_summary.csv',index=False)
        config={'model':'Positive-growth KDE p(b_i | b_(i-1),b_(i-2),alpha_i,F_i,S_(i-1)); elapsed S_(i-1) is cumulative time BEFORE cycle i-1, not the time of outcome birth i.',
            'bandwidths':{k:[None if np.isinf(x) else float(x) for x in h] for k,h in chosen.items()},
            'selection':'Five grouped training-lineage CV folds; maximize birth log density. Fully retune all bandwidths for each model on identical gap-free data. Infinite time bandwidth includes the no-elapsed model.',
            'simulation':'Fix alpha/F paths and first two births. Initialize S_1 by reconstructing t_0 from seeded births, then advance S with generated times only. First and last times omitted from comparisons. No time/size clipping.',
            'scope':'Gap-free original lineages only, since missing generation durations leave total elapsed time unidentified. S=0 is the first recorded birth, not absolute biological age. Effective F paths are derived from observed births/alpha/time; conditional diagnostic, not an autonomous generative test.',
            'n_lineages':len(groups),'n_births':sum(map(len,groups)),'n_train_transitions':len(train),'n_validation_transitions':len(valid),'validation_ids':sorted(validids),
            'n_grid_candidates':len(GRID)}
        (OUT/'fit_results.json').write_text(json.dumps(config,indent=2))
    print('SELECTED',chosen,flush=True)
    scores={name:np.log(np.maximum(make_model(train,h).density(valid.birth,valid.previous_birth,valid.alpha,valid.F,b_older=valid.older_birth,elapsed=valid.elapsed),1e-300)) for name,h in chosen.items()}
    gain,lo,hi=interval(scores['Elapsed']-scores['No_elapsed'],valid.lineage.to_numpy())
    config['heldout_birth_scores']={k:float(v.mean()) for k,v in scores.items()};config['heldout_gain']={'mean':gain,'lo':lo,'hi':hi}
    pd.DataFrame(dict(lineage=valid.lineage,generation=valid.generation,**scores)).to_csv(OUT/'heldout_scores.csv',index=False)
    rows=[];acfs=[];hists=[];ccfs=[];trends=[]
    for scope,sub,fitdata in [('validation',[g for g in groups if g.lineage.iloc[0] in validids],train),('all',groups,tab)]:
        brp=FastReplay(sub,2.5,.5,args.replicates,85319);trp=FastReplay([g.iloc[1:-1].reset_index(drop=True) for g in sub],2.5,.5,args.replicates,85319)
        grp=FastReplay([g.iloc[2:].reset_index(drop=True) for g in sub],2.5,.5,args.replicates,85319)
        _,ob,_=brp.observed();_,_,ot=trp.observed();runs={'Observed':(ob,ot)}
        for name,h in chosen.items():
            model=make_model(fitdata,h);b,t,clock,diag=simulate(sub,model,args.replicates);runs[name]=(b,t[:,:,1:-1])
            config.setdefault('simulation_diagnostics',{}).setdefault(scope,{})[name]=diag
            np.savez_compressed(OUT/f'{scope}_{name}_examples.npz',birth=b[:5],time=t[:5],elapsed=clock[:5],birth_mask=brp.mask,alpha=brp.alpha,factor=brp.f)
            np.savez_compressed(OUT/f'{scope}_{name}_fit.npz',previous_birth=fitdata.previous_birth,older_birth=fitdata.older_birth,birth=fitdata.birth,alpha=fitdata.alpha,factor=fitdata.F,elapsed=fitdata.elapsed,bandwidths=h)
        config.setdefault('counts',{})[scope]={'lineages':len(sub),'births':int(brp.mask.sum()),'generated_births':int(grp.mask.sum()),'times':int(trp.mask.sum())}
        for name,(b,t) in runs.items():
            pb=b[:,:,1:-1][:,trp.mask];cb=b[:,:,2:][:,trp.mask];bv=b[:,brp.mask];tv=t[:,trp.mask];av=np.broadcast_to(trp.alpha[None][:,trp.mask],tv.shape)
            values={'mean_b':bv.mean(1),'sd_b':bv.std(1),'mean_t':tv.mean(1),'sd_t':tv.std(1),'q99_t':np.quantile(tv,.99,axis=1),
                'corr_b_t':corr(pb,tv),'corr_t_next_b':corr(tv,cb),'corr_alpha_t':corr(av,tv)}
            changes=[];trendbins=[[] for _ in range(8)]
            for j,g in enumerate(sub):
                n=len(g)
                if n<6:continue
                y=b[:,j,:n];third=n//3;changes.append(y[:,-third:].mean(1)-y[:,:third].mean(1))
                bins=np.minimum((np.arange(n)/(n-1)*8).astype(int),7);center=y-y.mean(1,keepdims=True)
                for k in range(8):
                    if np.any(bins==k):trendbins[k].append(center[:,bins==k].mean(1))
            values['birth_late_minus_early']=np.mean(changes,axis=0)
            for k,v in enumerate(trendbins):
                v=np.mean(v,axis=0);l,u=np.quantile(v,[.025,.975]);trends.append({'scope':scope,'model':name,'progress':(k+.5)/8,'mean':v.mean(),'lo':l,'hi':u})
            for var,arr,rp in [('birth',b,brp),('time',t,trp),('birth_generated_only',b[:,:,2:],grp)]:
                ac=rp.acfs(arr);l,u=np.quantile(ac,[.025,.975],axis=0)
                for lag in [1,2,3,5,10]:values[var+'_acf'+str(lag)]=ac[:,lag]
                acfs.extend({'scope':scope,'model':name,'variable':var,'lag':k,'mean':v,'lo':a if name!='Observed' else np.nan,'hi':z if name!='Observed' else np.nan} for k,(v,a,z) in enumerate(zip(ac.mean(0),l,u)))
                if var=='birth_generated_only':continue
                edges=np.arange(0,6.01,.1) if var=='birth' else np.arange(0,12.01,1/6);x=arr[:,rp.mask];den=np.histogram(x,bins=edges)[0]/x.size/np.diff(edges)
                hists.extend({'scope':scope,'model':name,'variable':var,'left':a,'right':z,'density':v} for a,z,v in zip(edges[:-1],edges[1:],den))
            for key,v in values.items():
                l,u=np.quantile(v,[.025,.975]);rows.append({'scope':scope,'model':name,'metric':key,'mean':v.mean(),'lo':l if name!='Observed' else np.nan,'hi':u if name!='Observed' else np.nan})
            stats,boot=ccf_statistics(trp,b[:,:,1:-1],t,False,1000 if name=='Observed' else 0)
            for statistic,v in stats.items():
                l,u=np.quantile(boot[statistic] if name=='Observed' else v,[.025,.975],axis=0)
                ccfs.extend({'scope':scope,'model':name,'statistic':statistic,'lag':k,'mean':value,'lo':a,'hi':z} for k,value,a,z in zip(LAGS,v.mean(0),l,u))
            print('RESULT',scope,name,{k:round(float(v.mean()),5) for k,v in values.items()},flush=True)
    for filename,data in [('joint_metrics.csv',rows),('acf_comparison.csv',acfs),('marginal_densities.csv',hists),('cross_correlations.csv',ccfs),('birth_progress_trend.csv',trends)]:pd.DataFrame(data).to_csv(OUT/filename,index=False)
    config['replicates']=args.replicates;(OUT/'fit_results.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
