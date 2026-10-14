#!/usr/bin/env python3
"""Independent cycle-specific transition KDEs, retaining the generated clock."""
import argparse
import itertools
import json
import numpy as np
import pandas as pd
from fit_elapsed_birth_transitions import (PROJECT_DIR,load_data,segments,dataset as elapsed_dataset,make_model,simulate,
    FastReplay,corr,normal_sf,interval,ccf_statistics,LAGS,HB,HA,HF,HG,HOLDER,HT,KEYS,OUT as BASE)

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/cycle_specific_birth_transitions'
# Infinite birth bandwidth permits an unconditional same-cycle density when
# there are too few same-cycle training observations to resolve size effects.
HB_CYCLE=HB+[np.inf]
GRID=list(itertools.product(HB_CYCLE,HA,HF,HG,HOLDER,HT))


def dataset(groups):
    tab=elapsed_dataset(groups)
    tab['cycle']=np.concatenate([np.arange(1,len(g)-1) for g in groups])
    return tab


def loo_scores(tab):
    """One transition per original lineage per cycle: LOO is lineage held-out."""
    cols=['previous_birth','alpha','F','older_birth'];x=np.log(tab[cols].to_numpy());d=(x[:,None]-x[None])**2
    g=tab.g.to_numpy();delta=g[:,None]-g;dt=(tab.elapsed.to_numpy()[:,None]-tab.elapsed.to_numpy()[None])**2
    kernels=[np.exp(-.5*(delta/h)**2)/(np.sqrt(2*np.pi)*h*normal_sf(-g/h)) for h in HG]
    scores=np.empty(len(GRID));indices={h:i for i,h in enumerate(GRID)}
    for hb,ha,hf,ho in itertools.product(HB_CYCLE,HA,HF,HOLDER):
        base=-.5*(d[:,:,0]/hb**2+d[:,:,1]/ha**2+d[:,:,2]/hf**2+d[:,:,3]/ho**2)
        np.fill_diagonal(base,-np.inf)
        for ht in HT:
            logits=base-.5*dt/ht**2;logits-=logits.max(1,keepdims=True);w=np.exp(logits);w/=w.sum(1,keepdims=True)
            for hg,kernel in zip(HG,kernels):
                density=np.sum(w*kernel,axis=1)/tab.birth.to_numpy()
                scores[indices[(hb,ha,hf,hg,ho,ht)]]=np.log(np.maximum(density,1e-300)).mean()
    return scores


def select_bandwidth(scores):
    # With two training rows, LOO leaves one kernel center, so conditioning
    # bandwidths are unidentifiable. Prefer maximal smoothing in exact ties,
    # rather than arbitrarily selecting the first/narrowest grid entry.
    tied=np.flatnonzero(scores>=np.max(scores)-1e-10)
    def simplicity(index):
        h=np.array(GRID[index]);pred=h[[0,1,2,4,5]]
        return (int(np.isinf(pred).sum()),float(np.log(pred[np.isfinite(pred)]).sum()),float(h[3]))
    return int(max(tied,key=simplicity))


def evaluate(scope,groups,b,t):
    brp=FastReplay(groups,2.5,.5,len(b),85319);trp=FastReplay([g.iloc[1:-1].reset_index(drop=True) for g in groups],2.5,.5,len(b),85319)
    grp=FastReplay([g.iloc[2:].reset_index(drop=True) for g in groups],2.5,.5,len(b),85319)
    tv=t[:,:,1:-1];pb=b[:,:,1:-1][:,trp.mask];cb=b[:,:,2:][:,trp.mask];bv=b[:,brp.mask];v=tv[:,trp.mask];a=np.broadcast_to(trp.alpha[None][:,trp.mask],v.shape)
    values={'mean_b':bv.mean(1),'sd_b':bv.std(1),'mean_t':v.mean(1),'sd_t':v.std(1),'q99_t':np.quantile(v,.99,axis=1),'corr_b_t':corr(pb,v),'corr_t_next_b':corr(v,cb),'corr_alpha_t':corr(a,v)}
    changes=[];bins=[[] for _ in range(8)];trends=[]
    for j,g in enumerate(groups):
        n=len(g)
        if n<6:continue
        y=b[:,j,:n];third=n//3;changes.append(y[:,-third:].mean(1)-y[:,:third].mean(1));index=np.minimum((np.arange(n)/(n-1)*8).astype(int),7);y=y-y.mean(1,keepdims=True)
        for k in range(8):
            if np.any(index==k):bins[k].append(y[:,index==k].mean(1))
    values['birth_late_minus_early']=np.mean(changes,axis=0)
    for k,vs in enumerate(bins):
        vs=np.mean(vs,axis=0);l,u=np.quantile(vs,[.025,.975]);trends.append({'scope':scope,'model':'Cycle_specific','progress':(k+.5)/8,'mean':vs.mean(),'lo':l,'hi':u})
    acfs=[];hists=[];rows=[];ccfs=[]
    for var,arr,rp in [('birth',b,brp),('time',tv,trp),('birth_generated_only',b[:,:,2:],grp)]:
        ac=rp.acfs(arr);l,u=np.quantile(ac,[.025,.975],axis=0)
        for lag in [1,2,3,5,10]:values[var+'_acf'+str(lag)]=ac[:,lag]
        acfs.extend({'scope':scope,'model':'Cycle_specific','variable':var,'lag':k,'mean':v,'lo':a,'hi':z} for k,(v,a,z) in enumerate(zip(ac.mean(0),l,u)))
        if var=='birth_generated_only':continue
        edges=np.arange(0,6.01,.1) if var=='birth' else np.arange(0,12.01,1/6);x=arr[:,rp.mask];den=np.histogram(x,bins=edges)[0]/x.size/np.diff(edges)
        hists.extend({'scope':scope,'model':'Cycle_specific','variable':var,'left':a,'right':z,'density':v} for a,z,v in zip(edges[:-1],edges[1:],den))
    for key,vs in values.items():
        l,u=np.quantile(vs,[.025,.975]);rows.append({'scope':scope,'model':'Cycle_specific','metric':key,'mean':vs.mean(),'lo':l,'hi':u})
    stats,_=ccf_statistics(trp,b[:,:,1:-1],tv)
    for statistic,v in stats.items():
        l,u=np.quantile(v,[.025,.975],axis=0)
        ccfs.extend({'scope':scope,'model':'Cycle_specific','statistic':statistic,'lag':k,'mean':value,'lo':a,'hi':z} for k,value,a,z in zip(LAGS,v.mean(0),l,u))
    print('RESULT',scope,{k:round(float(v.mean()),5) for k,v in values.items()},flush=True)
    return {'joint_metrics.csv':rows,'acf_comparison.csv':acfs,'marginal_densities.csv':hists,'cross_correlations.csv':ccfs,'birth_progress_trend.csv':trends},brp


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replicates',type=int,default=200);parser.add_argument('--reuse-bandwidths',action='store_true');parser.add_argument('--reuse-cv',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);_,original=load_data();groups=segments([g for g in original if len(g)>=3 and np.all(np.diff(g.generation)==1)]);tab=dataset(groups)
    prior=json.loads((BASE/'fit_results.json').read_text());ids=set(prior['validation_ids']);train=tab[~tab.lineage.isin(ids)].reset_index(drop=True);valid=tab[tab.lineage.isin(ids)].reset_index(drop=True)
    fallback=[np.inf if x is None else x for x in prior['bandwidths']['Elapsed']]
    if args.reuse_bandwidths:
        config=json.loads((OUT/'fit_results.json').read_text());chosen={int(k):[np.inf if x is None else x for x in h] for k,h in config['bandwidths_by_cycle'].items()}
    else:
        chosen={};summary=[];gridrows=[]
        stored=pd.read_csv(OUT/'cycle_bandwidth_cv.csv') if args.reuse_cv else None
        for cycle,g in train.groupby('cycle'):
            g=g.reset_index(drop=True)
            if len(g)>=2:
                scores=loo_scores(g) if stored is None else stored.loc[stored.cycle==cycle,'loo_mean_log_birth_density'].to_numpy()
                index=select_bandwidth(scores);h=GRID[index];score=float(scores[index]);status='independent_lineage_LOO'
                gridrows.extend(dict(zip(KEYS,hh),cycle=int(cycle),loo_mean_log_birth_density=float(s)) for hh,s in zip(GRID,scores))
            else:
                h=fallback;score=None;status='one_training_observation_bandwidth_not_identifiable_fixed_reference_widths'
            chosen[int(cycle)]=h
            summary.append(dict(zip(KEYS,h),cycle=int(cycle),n_train=len(g),n_validation=int(np.sum(valid.cycle==cycle)),n_all=int(np.sum(tab.cycle==cycle)),loo_score=score,status=status))
            if cycle%5==0 or len(g)<3:print('FITTED cycle',cycle,'n=',len(g),'bandwidths=',h,flush=True)
        config={'model':'Separate positive-growth conditional KDE for EACH generating cycle k, with own kernel centers and six bandwidths, using b_(i-1), b_(i-2), alpha_i, F_i, and elapsed S_(i-1). No sharing or smoothing across cycles.',
            'cycle_index':'Zero-based generating cycle k=i-1 since first recorded birth. Cycle 0 connects the two observed seeds and is not fitted. Fitted cycles 1..46 generate births 2..47.',
            'selection':'Independent leave-one-lineage-out CV within EACH cycle on the training split; one row per lineage per cycle. Infinite bandwidths permit dropping conditioning variables. Ties within 1e-10 prefer dropping predictors then wider conditioning kernels. Same selected widths used for all-data refits.',
            'singleton_exception':'Cycle 46 has one training/all-data observation and no validation examples; smoothing widths fixed to reference fit because separate bandwidth estimation is impossible. Only its kernel center is learned. No adjacent-cycle borrowing.',
            'scope':prior['scope'],'simulation':prior['simulation'],'replicates':args.replicates,
            'n_bandwidth_candidates_per_cycle':len(GRID),'bandwidths_by_cycle':{str(k):[None if np.isinf(x) else float(x) for x in h] for k,h in chosen.items()},'validation_ids':sorted(ids)}
        pd.DataFrame(summary).to_csv(OUT/'cycle_parameters.csv',index=False);pd.DataFrame(gridrows).to_csv(OUT/'cycle_bandwidth_cv.csv',index=False)
        (OUT/'fit_results.json').write_text(json.dumps(config,indent=2))
    scores=np.empty(len(valid))
    for k,g in valid.groupby('cycle'):
        model=make_model(train[train.cycle==k],chosen[int(k)])
        scores[g.index]=np.log(np.maximum(model.density(g.birth,g.previous_birth,g.alpha,g.F,b_older=g.older_birth,elapsed=g.elapsed),1e-300))
    baseline=make_model(train,fallback);ref=np.log(np.maximum(baseline.density(valid.birth,valid.previous_birth,valid.alpha,valid.F,b_older=valid.older_birth,elapsed=valid.elapsed),1e-300))
    gain,lo,hi=interval(scores-ref,valid.lineage.to_numpy());config['heldout_gain']={'mean':gain,'lo':lo,'hi':hi};config['heldout_scores']={'Cycle_specific':float(scores.mean()),'Shared_elapsed':float(ref.mean())}
    pd.DataFrame({'lineage':valid.lineage,'cycle':valid.cycle,'Cycle_specific':scores,'Shared_elapsed':ref}).to_csv(OUT/'heldout_scores.csv',index=False)
    results={};cyclechecks=[]
    for scope,sub,fitdata in [('validation',[g for g in groups if g.lineage.iloc[0] in ids],train),('all',groups,tab)]:
        models={int(k):make_model(g,chosen[int(k)]) for k,g in fitdata.groupby('cycle')}
        b,t,clock,diag=simulate(sub,None,args.replicates,seed=382199,cycle_models=models);config.setdefault('simulation_diagnostics',{})[scope]=diag
        rows,brp=evaluate(scope,sub,b,t)
        for filename,rs in rows.items():results.setdefault(filename,[]).extend(rs)
        np.savez_compressed(OUT/f'{scope}_examples.npz',birth=b[:5],time=t[:5],elapsed=clock[:5],birth_mask=brp.mask,alpha=brp.alpha,factor=brp.f)
        fitdata.to_csv(OUT/f'{scope}_kernel_centers.csv',index=False)
        for k in range(1,brp.mask.shape[1]-1):
            keep=brp.mask[:,k+1]
            if not np.any(keep):continue
            obs=np.array([g.length_birth.iloc[k+1] for g in sub if len(g)>k+1]);obt=np.array([g.time.iloc[k] for g in sub if len(g)>k+1])
            for var,x,o in [('birth',b[:,keep,k+1],obs),('time',t[:,keep,k],obt)]:
                means=x.mean(1);sds=x.std(1);l,u=np.quantile(means,[.025,.975])
                cyclechecks.append({'scope':scope,'cycle':k,'variable':var,'n_lineages':len(o),'observed_mean':o.mean(),'observed_sd':o.std(),'simulated_mean':means.mean(),'simulated_sd':sds.mean(),'mean_lo':l,'mean_hi':u})
    for filename,rs in results.items():
        old=pd.read_csv(BASE/filename);old=old[old.model.isin(['Observed','Elapsed'])].copy();old.loc[old.model=='Elapsed','model']='Shared_elapsed'
        pd.concat([old,pd.DataFrame(rs)],ignore_index=True).to_csv(OUT/filename,index=False)
    pd.DataFrame(cyclechecks).to_csv(OUT/'cycle_prediction_checks.csv',index=False)
    config['replicates']=args.replicates;(OUT/'fit_results.json').write_text(json.dumps(config,indent=2));print('HELDOUT',config['heldout_gain'],flush=True)


if __name__=='__main__':main()
