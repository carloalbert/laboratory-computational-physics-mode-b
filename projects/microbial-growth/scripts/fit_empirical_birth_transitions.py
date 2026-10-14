#!/usr/bin/env python3
"""Fit generic birth transitions with recorded alpha/F; reconstruct cycle times."""
import argparse
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from screen_parsimonious_joint_hazards import PROJECT_DIR,load_data,segments,FastReplay,corr
sys.path.insert(0,str(PROJECT_DIR/'src'))
from empirical_birth_transition import BirthTransitionKDE
from empirical_conditional_time_density import normal_sf
from check_likelihood_factorization import transitions,interval
from check_acf_fit_cross_correlations import ccf_statistics,LAGS
from fit_joint_growth_timing import simulate as hill_replay,log_time_density

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/empirical_birth_transitions'
HB=[.02,.04,.08,.16,.32];HA=[.08,.16,.32,np.inf];HF=[.02,.04,.08,np.inf];HG=[.02,.04,.08]
GRID=list(itertools.product(HB,HA,HF,HG))


def dataset(groups):
    tab=transitions(groups)
    tab['F']=tab.birth/(tab.previous_birth*np.exp(tab.previous_alpha*tab.previous_time))
    tab['g']=np.log(tab.birth/(tab.previous_birth*tab.F))
    return tab


def make_model(tab,h):return BirthTransitionKDE(tab.previous_birth,tab.alpha,tab.F,tab.birth,h)


def cv_scores(train,valid):
    x=np.log(train[['previous_birth','alpha','F']].to_numpy());q=np.log(valid[['previous_birth','alpha','F']].to_numpy())
    g=train.g.to_numpy();gq=valid.g.to_numpy();scores=np.zeros(len(GRID))
    for start in range(0,len(valid),128):
        sl=slice(start,start+128);d=(q[sl,None]-x[None])**2;delta=gq[sl,None]-g
        kernels=[np.exp(-.5*(delta/h)**2)/(np.sqrt(2*np.pi)*h*normal_sf(-g/h)) for h in HG]
        index=0
        for hb,ha,hf in itertools.product(HB,HA,HF):
            logits=-.5*(d[:,:,0]/hb**2+d[:,:,1]/ha**2+d[:,:,2]/hf**2)
            logits-=logits.max(1,keepdims=True);w=np.exp(logits);w/=w.sum(1,keepdims=True)
            for kernel in kernels:
                density=np.sum(w*kernel,axis=1)/valid.birth.to_numpy()[sl]
                scores[index]+=np.log(np.maximum(density,1e-300)).sum();index+=1
    return scores


def simulate(groups,kde,replicates,seed=129371):
    rng=np.random.default_rng(seed);maxn=max(map(len,groups));b=np.zeros((replicates,len(groups),maxn));t=np.full_like(b,np.nan)
    outside=0;total=0;maxerr=0.
    for j,group in enumerate(groups):
        b[:,j,0]=group.length_birth.iloc[0]
        for n in range(1,len(group)):
            previous=b[:,j,n-1];a=group.alpha.iloc[n];f=group.f_effective.iloc[n]
            outside+=int(np.sum((np.log(previous)<kde.logb.min())|(np.log(previous)>kde.logb.max())));total+=len(previous)
            birth,g=kde.sample_fixed_context(previous,a,f,rng);b[:,j,n]=birth
            reconstructed=np.log(birth/(previous*f))/group.alpha.iloc[n-1]
            if np.any(reconstructed<=0):raise AssertionError('Nonpositive reconstructed time')
            t[:,j,n-1]=reconstructed
            maxerr=max(maxerr,float(np.max(abs(group.alpha.iloc[n-1]*reconstructed-g))))
        if (j+1)%75==0:print('simulated segments',j+1,'/',len(groups),flush=True)
    return b,t,{'birth_queries_outside_training_range':outside,'total_birth_queries':total,'fraction_outside':outside/total,'maximum_growth_reconstruction_error':maxerr,'mixture_draws':kde.draws,'mixture_proposals':kde.proposals,'exact_dense_fallback_draws':kde.fallback_draws}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replicates',type=int,default=200);parser.add_argument('--reuse-bandwidths',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);data,original=load_data();groups=segments(original);tab=dataset(groups)
    prior=json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/joint_growth_timing/fit_results.json').read_text());validids=set(prior['validation_ids'])
    train=tab[~tab.lineage.isin(validids)].reset_index(drop=True);valid=tab[tab.lineage.isin(validids)].reset_index(drop=True)
    if args.reuse_bandwidths:config=json.loads((OUT/'fit_results.json').read_text());h=[np.inf if v is None else v for v in config['bandwidths']]
    else:
        ids=np.sort(train.lineage.unique());np.random.default_rng(72910).shuffle(ids);assignment={v:i%5 for i,v in enumerate(ids)};fold=train.lineage.map(assignment).to_numpy()
        results=[];total=np.zeros(len(GRID))
        for k in range(5):
            vals=cv_scores(train[fold!=k],train[fold==k]);total+=vals
            results.extend({'fold':k,'h_birth':hb,'h_alpha_current':ha,'h_factor':hf,'h_growth':hg,'n':int(np.sum(fold==k)),'sum_log_birth_density':v} for (hb,ha,hf,hg),v in zip(GRID,vals))
            print('Grouped CV fold',k+1,'complete',flush=True)
        h=GRID[int(np.argmax(total))];pd.DataFrame(results).to_csv(OUT/'bandwidth_cv.csv',index=False)
        pd.DataFrame([dict(zip(['h_birth','h_alpha_current','h_factor','h_growth'],hh),mean_log_birth_density=float(v/len(train))) for hh,v in zip(GRID,total)]).to_csv(OUT/'bandwidth_cv_summary.csv',index=False)
        config={'model':'Conditional Gaussian KDE for positive g_(i-1)=log[b_i/(F_i*b_(i-1))], conditioned on (log b_(i-1),log alpha_i,log F_i); Gaussian response kernels truncated at zero. Birth density includes 1/b_i Jacobian.',
                'indexing':'Use alpha_i as requested in the birth transition. Reconstruct t_(i-1) with alpha_(i-1), not alpha_i. Last time in each segment cannot be reconstructed and is excluded from both observed and simulated time checks.',
                'bandwidths':[None if np.isinf(v) else float(v) for v in h],'selection':'Five grouped training-lineage folds; maximize held-out birth log density. No ACF/marginal target used. All-data refit uses these fixed selected bandwidths.',
                'sampling':'Exact conditional-mixture sampling by rejection with exact dense fallback; positive-truncated response draws. No birth/time clipping.',
                'scope':'Recorded alpha and effective-factor paths and initial births are conditioned on. F_eff was constructed using observed births/times, so this is a conditional statistical diagnostic, not independent evidence of a generative causal mechanism.',
                'n_train_transitions':len(train),'n_validation_transitions':len(valid),'validation_ids':sorted(validids),'replicates':args.replicates}
        (OUT/'fit_results.json').write_text(json.dumps(config,indent=2))
    print('SELECTED bandwidths',h,flush=True)
    fitted=make_model(train,h);density=fitted.density(valid.birth,valid.previous_birth,valid.alpha,valid.F)
    p=prior['fits']['train']['hazard']
    hill=log_time_density(valid.previous_time,valid.previous_birth,valid.previous_alpha,p['u'],p['omega2'],p['eta'],p['kappa'],p['b_star'],order=96)-np.log(valid.previous_alpha)-np.log(valid.birth)
    ll=np.log(np.maximum(density,1e-300));gain,lo,hi=interval(ll-hill,valid.lineage.to_numpy())
    config['heldout_birth_score']={'KDE_mean':float(ll.mean()),'Hill_mean':float(hill.mean()),'gain':gain,'lo':lo,'hi':hi,'n_probabilities_below_1e_12':int(np.sum(density<1e-12))}
    pd.DataFrame({'lineage':valid.lineage,'generation':valid.generation,'KDE_log_birth_density':ll,'Hill_log_birth_density':hill}).to_csv(OUT/'heldout_scores.csv',index=False)
    metrics=[];acfs=[];ccfs=[];hists=[]
    for scope,sub,fitdata,oldscope in [('validation',[g for g in groups if g.lineage.iloc[0] in validids],train,'train'),('all',groups,tab,'all')]:
        old=prior['fits'][oldscope];refs=old['references'];brp=FastReplay(sub,refs['b'],refs['alpha'],args.replicates,85319)
        keep=np.array([len(g)>1 for g in sub]);parents=[g.iloc[:-1].reset_index(drop=True) for g in sub if len(g)>1];trp=FastReplay(parents,refs['b'],refs['alpha'],args.replicates,85319)
        _,ob,unused=brp.observed();_,opb,ot=trp.observed();oa=brp.alpha[None];pa=trp.alpha[None]
        kde=make_model(fitdata,h);eb,et,diagnostics=simulate(sub,kde,args.replicates)
        hb,ha,ht=hill_replay(brp,old['hazard'],old['growth_models']['birth_AR'],refs,True,seed=49513)
        runs={'Observed':(ob,ot),'Hill_reference':(hb,ht[:,keep,:-1]),'Empirical_birth':(eb,et[:,keep,:-1])}
        config.setdefault('simulation_diagnostics',{})[scope]=dict(diagnostics,n_births=int(brp.mask.sum()),n_reconstructed_times=int(trp.mask.sum()))
        for name,(b,t) in runs.items():
            parentb=b[:,keep,:-1];childb=b[:,keep,1:];bv=b[:,brp.mask];tv=t[:,trp.mask];pb=parentb[:,trp.mask];cb=childb[:,trp.mask];av=np.broadcast_to(pa[:,trp.mask],tv.shape)
            values={'mean_b':bv.mean(1),'sd_b':bv.std(1),'mean_t':tv.mean(1),'sd_t':tv.std(1),'q99_t':np.quantile(tv,.99,axis=1),'corr_b_t':corr(pb,tv),'corr_t_next_b':corr(tv,cb),'corr_alpha_t':corr(av,tv),'corr_b_alpha':corr(bv,np.broadcast_to(oa[:,brp.mask],bv.shape))}
            for var,arr,rp in [('birth',b,brp),('time',t,trp)]:
                ac=rp.acfs(arr);l,u=np.quantile(ac,[.025,.975],axis=0)
                for lag in [1,2,3,5,10]:values[var+'_acf'+str(lag)]=ac[:,lag]
                acfs.extend({'scope':scope,'model':name,'variable':var,'lag':k,'mean':v,'lo':a if name!='Observed' else np.nan,'hi':z if name!='Observed' else np.nan} for k,(v,a,z) in enumerate(zip(ac.mean(0),l,u)))
                edges=np.arange(0,6.01,.1) if var=='birth' else np.arange(0,12.01,1/6);x=arr[:,rp.mask];den=np.histogram(x,bins=edges)[0]/x.size/np.diff(edges)
                hists.extend({'scope':scope,'model':name,'variable':var,'left':a,'right':z,'density':v} for a,z,v in zip(edges[:-1],edges[1:],den))
            for key,v in values.items():
                l,u=np.quantile(v,[.025,.975]);metrics.append({'scope':scope,'model':name,'metric':key,'mean':v.mean(),'lo':l if name!='Observed' else np.nan,'hi':u if name!='Observed' else np.nan})
            stats,boot=ccf_statistics(trp,parentb,t,False,1000 if name=='Observed' else 0)
            for statistic,v in stats.items():
                l,u=np.quantile(boot[statistic] if name=='Observed' else v,[.025,.975],axis=0)
                ccfs.extend({'scope':scope,'model':name,'statistic':statistic,'lag':k,'mean':value,'lo':a,'hi':z} for k,value,a,z in zip(LAGS,v.mean(0),l,u))
            print('RESULT',scope,name,{k:round(float(v.mean()),5) for k,v in values.items()},flush=True)
        np.savez_compressed(OUT/f'{scope}_examples.npz',birth=eb[:5],time=et[:5],birth_mask=brp.mask,alpha=brp.alpha,factor=brp.f)
    for filename,rows in [('joint_metrics.csv',metrics),('acf_comparison.csv',acfs),('marginal_densities.csv',hists),('cross_correlations.csv',ccfs)]:pd.DataFrame(rows).to_csv(OUT/filename,index=False)
    config['replicates']=args.replicates;(OUT/'fit_results.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
