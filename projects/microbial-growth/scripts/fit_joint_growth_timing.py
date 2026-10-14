#!/usr/bin/env python3
"""Shared Glu lognormal growth transition plus the original b-dependent hazard.

Condition on each segment's first b/alpha and recorded effective jump factors;
generate all subsequent alphas and all division times. No ACF fitting.
"""
import argparse
import json
import numpy as np
import pandas as pd
from screen_parsimonious_joint_hazards import FastReplay,PROJECT_DIR,load_data,segments,corr
from run_birth_size_dependent_time_hazard import Likelihood,pattern_polish,differential_evolution_maximize,unpack,PARAMETER_NAMES
from birth_size_dependent_time_hazard import log_time_density
from check_likelihood_factorization import transitions,interval
from check_acf_fit_cross_correlations import ccf_statistics,LAGS

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/joint_growth_timing'


def growth_matrix(data,refs,kind='birth_AR'):
    cols=[np.ones(len(data))]
    if kind in ['AR','birth_AR']:cols.append(np.log(data.previous_alpha.to_numpy()/refs['alpha']))
    if kind=='birth_AR':cols.append(np.log(data.birth.to_numpy()/refs['b']))
    return np.column_stack(cols)


def fit_growth(data,refs,kind='birth_AR'):
    x=growth_matrix(data,refs,kind);y=np.log(data.alpha.to_numpy()/refs['alpha'])
    coef=np.linalg.lstsq(x,y,rcond=None)[0];residual=y-x@coef
    sigma=float(np.sqrt(np.mean(residual**2)))
    result={'kind':kind,'coefficients':coef.tolist(),'sigma':sigma,'n_parameters':len(coef)+1}
    result['log_likelihood']=float(growth_density(data,result,refs).sum())
    return result


def growth_density(data,fit,refs):
    mu=growth_matrix(data,refs,fit['kind'])@np.array(fit['coefficients'])
    residual=np.log(data.alpha.to_numpy()/refs['alpha'])-mu
    return -.5*(residual/fit['sigma'])**2-np.log(fit['sigma'])-.5*np.log(2*np.pi)-np.log(data.alpha.to_numpy())


def sample_growth(b,hazard,rng):
    g=np.zeros_like(b);pending=np.ones(b.shape,dtype=bool)
    offset=hazard['kappa']*np.log(b/hazard['b_star'])
    for _ in range(100000):
        if not pending.any():return g
        g[pending]+=rng.exponential(size=pending.sum())/hazard['omega2']
        if np.max(g)>700:raise FloatingPointError('Accumulated growth exceeded numerical range')
        odds=hazard['eta']*(np.log(b[pending])+np.log(np.expm1(g[pending]))-np.log(hazard['u']))+offset[pending]
        accepted=rng.random(pending.sum())<np.exp(-np.logaddexp(0.,-odds))
        pending[pending]=~accepted
    raise RuntimeError('Exact thinning failed to finish')


def simulate(rp,hazard,growth,refs,recorded=False,seed=41299):
    event_rng=np.random.default_rng(seed);alpha_rng=np.random.default_rng(seed+100000)
    shape=(rp.R,*rp.mask.shape);births=np.zeros(shape);times=np.zeros(shape);alphas=np.zeros(shape)
    current=np.broadcast_to(rp.birth0,shape[:2]).copy()
    previous_alpha=np.broadcast_to(rp.alpha[:,0],shape[:2]).copy()
    c,rho,beta=growth['coefficients']
    for n in range(rp.mask.shape[1]):
        active=rp.mask[:,n];b=current[:,active]
        if recorded or n==0:a=np.broadcast_to(rp.alpha[active,n],b.shape)
        else:
            mu=c+rho*np.log(previous_alpha[:,active]/refs['alpha'])+beta*np.log(b/refs['b'])
            a=refs['alpha']*np.exp(mu+growth['sigma']*alpha_rng.normal(size=b.shape))
        g=sample_growth(b,hazard,event_rng)
        births[:,active,n]=b;alphas[:,active,n]=a;times[:,active,n]=g/a
        previous_alpha[:,active]=a
        if n+1==rp.mask.shape[1]:continue
        continuing=rp.mask[:,n+1];local=continuing[active]
        current[:,continuing]=b[:,local]*np.exp(g[:,local])*rp.f[continuing,n+1]
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(a)):
            raise FloatingPointError('Nonfinite simulated state; no clipping permitted')
    return births,alphas,times


def statistics(rp,b,a,t):
    x={key:arr[:,rp.mask] for key,arr in [('b',b),('alpha',a),('t',t)]}
    values={}
    for key,v in x.items():
        values['mean_'+key]=v.mean(1);values['sd_'+key]=v.std(1)
        values['q99_'+key]=np.quantile(v,.99,axis=1)
    for left,right in [('b','t'),('alpha','t'),('b','alpha')]:values['corr_'+left+'_'+right]=corr(x[left],x[right])
    values['corr_loga_logt']=corr(np.log(x['alpha']),np.log(x['t']))
    g=x['alpha']*x['t'];values['mean_g']=g.mean(1);values['sd_g']=g.std(1)
    values['corr_logb_g']=corr(np.log(x['b']),g)
    left,right=rp.pairs[1]
    values['corr_t_next_b']=corr(x['t'][:,left],x['b'][:,right])
    values['corr_alpha_next_alpha']=corr(x['alpha'][:,left],x['alpha'][:,right])
    for key,arr in [('b',b),('alpha',a),('t',t)]:
        ac=rp.acfs(arr)
        for k in [1,2,3,5,10]:values[key+'_acf'+str(k)]=ac[:,k]
    return values


def fit_hazard(data,refs,seed):
    previous=json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/birth_size_dependent_time_hazard_effective_f/diagnostics.json').read_text())['mle']
    start=np.array([np.log(previous['u']),np.log(previous['omega2']),np.log(previous['eta']),previous['kappa']])
    objective=Likelihood(data,refs['b'],order=48)
    point,score,history=differential_evolution_maximize(objective,np.random.default_rng(seed),24,45,[start])
    alternatives=[pattern_polish(objective,x) for x in [start,point]]
    point,score=max(alternatives,key=lambda item:item[1]);params=dict(zip(PARAMETER_NAMES,map(float,unpack(point))))
    params['b_star']=refs['b'];check=Likelihood(data,refs['b'],order=96)(point)
    return params,{'log_likelihood':score,'log_likelihood_order96':check,'quadrature_difference':check-score,'search_history':history}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replicates',type=int,default=200);parser.add_argument('--reuse-fit',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);data,original=load_data();groups=segments(original)
    old=json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/conditional_effective_jumps/fit_results.json').read_text())
    validids=set(old['validation_ids']);train_data=data[~data.lineage.isin(validids)]
    if args.reuse_fit:config=json.loads((OUT/'fit_results.json').read_text())
    else:
        config={'hazard':'alpha*omega2*sigmoid(eta*log(b*(exp(alpha*t)-1)/u)+kappa*log(b/b_star)); q=0, no latent memory',
                'growth':'log(alpha_i/alpha_star) = c + rho*log(alpha_(i-1)/alpha_star) + beta*log(b_i/b_star) + sigma*Z_i; iid standard-normal Z_i',
                'parameter_scope':'Eight parameters shared by all Glu lineages: four timing, four growth. Reference scales fixed from fitting data.',
                'likelihood':'Timing log density for all cycles plus growth-transition log density for consecutive cycles. Initial segment alphas conditioned upon. Parameter blocks separate exactly; separate MLE maximizations equal joint maximization.',
                'simulation':'Recorded segment lengths, initial b and alpha, and effective division factors. Generate all times and every noninitial alpha. Reset at every generation gap. This is conditional on jump histories, not a fully generative jump model.',
                'validation':'Original project held-out lineages; train-only fits used for validation simulation and density scores. All-data refit is descriptive. Validation set has been examined in prior project analyses.',
                'intervals':'Simulation 2.5-97.5 percentiles at fixed parameters; observed CCF intervals use 1000 whole-lineage bootstrap draws. Parameter uncertainty not included.',
                'replicates':args.replicates,'validation_ids':sorted(validids),'fits':{}}
        for scope,cycles in [('train',train_data),('all',data)]:
            refs={'b':float(np.exp(np.log(cycles.length_birth).mean())),'alpha':float(np.exp(np.log(cycles.alpha).mean()))}
            tr=transitions([g for g in original if g.lineage.iloc[0] in set(cycles.lineage)])
            growth={kind:fit_growth(tr,refs,kind) for kind in ['iid','AR','birth_AR']}
            hazard,diagnostics=fit_hazard(cycles,refs,91241+len(cycles))
            config['fits'][scope]={'references':refs,'growth_models':growth,'hazard':hazard,'hazard_fit':diagnostics,'n_cycles':len(cycles),'n_growth_transitions':len(tr)}
            print('FIT',scope,'growth',growth['birth_AR'],'hazard',hazard,flush=True)
        (OUT/'fit_results.json').write_text(json.dumps(config,indent=2))
    # Honest held-out one-step density evaluation, distinct from free replay.
    validtr=transitions([g for g in original if g.lineage.iloc[0] in validids]);fit=config['fits']['train'];refs=fit['references'];densities={}
    for name,model in fit['growth_models'].items():densities[name]=growth_density(validtr,model,refs)
    densityrows=[]
    for name,ll in densities.items():
        gain,lo,hi=interval(ll-densities['iid'],validtr.lineage.to_numpy())
        densityrows.append({'growth_model':name,'mean_log_density_alpha':ll.mean(),'gain_vs_iid':gain,'lo':lo,'hi':hi})
    gain,lo,hi=interval(densities['birth_AR']-densities['AR'],validtr.lineage.to_numpy());config['birth_effect_density_gain_vs_AR']={'gain':gain,'lo':lo,'hi':hi}
    hp=fit['hazard'];tl=log_time_density(validtr.time.to_numpy(),validtr.birth.to_numpy(),validtr.alpha.to_numpy(),hp['u'],hp['omega2'],hp['eta'],hp['kappa'],hp['b_star'],order=96)
    config['heldout_joint_mean_log_density_on_transitions']=float(np.mean(tl+densities['birth_AR']))
    pd.DataFrame(densityrows).to_csv(OUT/'heldout_density_scores.csv',index=False)
    metricrows=[];acfs=[];ccfs=[];hists=[];invariants=[]
    for scope,subset,fitscope in [('validation',[g for g in groups if g.lineage.iloc[0] in validids],'train'),('all',groups,'all')]:
        fit=config['fits'][fitscope];refs=fit['references'];rp=FastReplay(subset,refs['b'],refs['alpha'],args.replicates,34973)
        _,ob,ot=rp.observed();oa=rp.alpha[None].copy();observed={'birth':ob,'alpha':oa,'time':ot}
        runs={'Observed':(ob,oa,ot)}
        for name,recorded in [('Recorded_alpha',True),('Joint_generated',False)]:
            runs[name]=simulate(rp,fit['hazard'],fit['growth_models']['birth_AR'],refs,recorded,seed=77149)
        invariant=float(np.max(abs(runs['Recorded_alpha'][0]-runs['Joint_generated'][0])))
        if invariant!=0:raise AssertionError('q=0 invariant failed: growth generation changed birth paths')
        invariants.append({'scope':scope,'maximum_paired_birth_difference':invariant,'n_segments':len(subset),'n_cycles':int(rp.mask.sum()),'n_seeded_alphas':len(subset)})
        for name,(b,a,t) in runs.items():
            metrics=statistics(rp,b,a,t)
            for key,v in metrics.items():
                lo,hi=np.quantile(v,[.025,.975]);metricrows.append({'scope':scope,'model':name,'metric':key,'mean':v.mean(),'lo':lo if name!='Observed' else np.nan,'hi':hi if name!='Observed' else np.nan})
            arrays={'birth':b,'alpha':a,'time':t}
            for variable,arr in arrays.items():
                ac=rp.acfs(arr);lo,hi=np.quantile(ac,[.025,.975],axis=0)
                acfs.extend({'scope':scope,'model':name,'variable':variable,'lag':k,'mean':v,'lo':l if name!='Observed' else np.nan,'hi':h if name!='Observed' else np.nan} for k,(v,l,h) in enumerate(zip(ac.mean(0),lo,hi)))
                edges={'birth':np.arange(0,6.01,.1),'alpha':np.arange(0,2.01,.04),'time':np.arange(0,12.01,1/6)}[variable]
                vals=arr[:,rp.mask];density=np.histogram(vals,bins=edges)[0]/vals.size/np.diff(edges)
                hists.extend({'scope':scope,'model':name,'variable':variable,'left':l,'right':r,'density':d} for l,r,d in zip(edges[:-1],edges[1:],density))
            for left,right in [('birth','time'),('alpha','time'),('birth','alpha')]:
                stats,boot=ccf_statistics(rp,arrays[left],arrays[right],False,1000 if name=='Observed' else 0)
                for statistic,v in stats.items():
                    lo,hi=np.quantile(boot[statistic] if name=='Observed' else v,[.025,.975],axis=0)
                    ccfs.extend({'scope':scope,'model':name,'left_variable':left,'right_variable':right,'statistic':statistic,'lag':k,'mean':value,'lo':l,'hi':h} for k,value,l,h in zip(LAGS,v.mean(0),lo,hi))
            print('REPLAY',scope,name,{k:round(float(v.mean()),5) for k,v in metrics.items()},flush=True)
        # Compact example trajectories (all repetitions need not be saved).
        np.savez_compressed(OUT/f'{scope}_example_trajectories.npz',birth=runs['Joint_generated'][0][:5],alpha=runs['Joint_generated'][1][:5],time=runs['Joint_generated'][2][:5],mask=rp.mask,recorded_f=rp.f,lineage=np.array([g.lineage.iloc[0] for g in subset]),generation=np.array([g.generation.iloc[0] for g in subset]))
    pd.DataFrame(metricrows).to_csv(OUT/'joint_metrics.csv',index=False);pd.DataFrame(acfs).to_csv(OUT/'acf_comparison.csv',index=False)
    pd.DataFrame(ccfs).to_csv(OUT/'cross_correlations.csv',index=False);pd.DataFrame(hists).to_csv(OUT/'marginal_densities.csv',index=False)
    config['paired_birth_invariance_checks']=invariants;config['replicates']=args.replicates
    (OUT/'fit_results.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
