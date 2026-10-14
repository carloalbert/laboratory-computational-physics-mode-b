#!/usr/bin/env python3
"""Generate Glu timing, AR growth rates, and iid effective division factors.

Joint model: product p(t_i|b_i,alpha_i) p(alpha_i|alpha_(i-1)) p(F_i),
b_i = b_(i-1) exp(alpha_(i-1)t_(i-1)) F_i. Condition on segment starts only.
"""
import argparse
import json
import numpy as np
import pandas as pd
from fit_joint_growth_timing import (PROJECT_DIR,FastReplay,load_data,segments,transitions,
    fit_growth,growth_density,sample_growth,statistics,corr,log_time_density,Likelihood)
from check_acf_fit_cross_correlations import ccf_statistics,LAGS

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/generative_ar_growth_iid_factors'


def fit_factors(groups):
    values=np.concatenate([g.f_effective.to_numpy()[1:] for g in groups if len(g)>1])
    logs=np.log(values);mu=float(logs.mean());sigma=float(logs.std())
    return {'mu_log_F':mu,'sigma_log_F':sigma,'mean_F':float(np.exp(mu+sigma*sigma/2)),
            'sd_F':float(np.sqrt(np.expm1(sigma*sigma)*np.exp(2*mu+sigma*sigma))),
            'n_transitions':len(values),'observed_min':float(values.min()),'observed_max':float(values.max()),
            'log_likelihood':float((-.5*((logs-mu)/sigma)**2-np.log(sigma)-.5*np.log(2*np.pi)-logs).sum())}


def factor_density(f,fit):
    y=np.log(f)
    return -.5*((y-fit['mu_log_F'])/fit['sigma_log_F'])**2-np.log(fit['sigma_log_F'])-.5*np.log(2*np.pi)-y


def simulate(rp,fit,recorded_f=False,seed=53122):
    event_rng=np.random.default_rng(seed);alpha_rng=np.random.default_rng(seed+100000);factor_rng=np.random.default_rng(seed+200000)
    shape=(rp.R,*rp.mask.shape);b=np.zeros(shape);a=np.zeros(shape);t=np.zeros(shape);f=np.full(shape,np.nan)
    current=np.broadcast_to(rp.birth0,shape[:2]).copy();previous_alpha=np.broadcast_to(rp.alpha[:,0],shape[:2]).copy()
    growth=fit['growth'];jump=fit['factor'];refs=fit['references'];c,rho=growth['coefficients']
    for n in range(rp.mask.shape[1]):
        active=rp.mask[:,n];bn=current[:,active]
        if n==0:an=np.broadcast_to(rp.alpha[active,0],bn.shape)
        else:
            mu=c+rho*np.log(previous_alpha[:,active]/refs['alpha'])
            an=refs['alpha']*np.exp(mu+growth['sigma']*alpha_rng.normal(size=bn.shape))
        g=sample_growth(bn,fit['hazard'],event_rng)
        b[:,active,n]=bn;a[:,active,n]=an;t[:,active,n]=g/an;previous_alpha[:,active]=an
        if n+1==rp.mask.shape[1]:continue
        continuing=rp.mask[:,n+1];local=continuing[active];next_shape=(rp.R,int(continuing.sum()))
        fn=np.broadcast_to(rp.f[continuing,n+1],next_shape) if recorded_f else np.exp(jump['mu_log_F']+jump['sigma_log_F']*factor_rng.normal(size=next_shape))
        f[:,continuing,n+1]=fn
        current[:,continuing]=bn[:,local]*np.exp(g[:,local])*fn
        if not np.all(np.isfinite(current)) or not np.all(np.isfinite(an)):raise FloatingPointError('Nonfinite state; no clipping allowed')
    return b,a,t,f


def check_recursion(rp,run):
    b,a,t,f=run;left,right=rp.pairs[1]
    flat=[x[:,rp.mask] for x in run];bb,aa,tt,ff=flat
    predicted=bb[:,left]*np.exp(aa[:,left]*tt[:,left])*ff[:,right]
    return float(np.max(abs(predicted-bb[:,right])/np.maximum(bb[:,right],1e-300)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replicates',type=int,default=200);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);data,original=load_data();groups=segments(original)
    previous_path=PROJECT_DIR/'outputs/model_comparison/Glu/joint_growth_timing/fit_results.json'
    previous=json.loads(previous_path.read_text());validids=set(previous['validation_ids'])
    config={'factorization':'product p(t_i|b_i,alpha_i) p(alpha_i|alpha_(i-1)) p(F_i)',
            'growth':'log(alpha_i/alpha_star)=c+rho*log(alpha_(i-1)/alpha_star)+sigma_alpha*Z_i',
            'factor':'log(F_i)=mu_F+sigma_F*W_i; W_i iid N(0,1), independent of growth and timing innovations',
            'hazard':previous['hazard'],'recursion':'b_i=b_(i-1)*exp(alpha_(i-1)*t_(i-1))*F_i',
            'parameters':'Nine shared Glu parameters: four timing, three growth AR, two iid factor. Fixed reference scales not counted.',
            'fit':'Joint MLE separates into three parameter blocks. Reuse and verify the unchanged timing optimum from joint_growth_timing; refit AR-only growth and iid factor parameters on matching data.',
            'simulation':'Generate every time, every noninitial alpha, every noninitial F and every noninitial b. Condition only on segment initial b/alpha and lengths. Reset after generation gaps. No observed factors enter the Generative simulation.',
            'support':'Effective factors are positive, not forced below one; they absorb observation/growth-reconstruction errors and are not physical partition fractions.',
            'validation':'Training-only fit evaluated on original held-out lineages. All-data refit separately descriptive; validation lineages have been inspected in earlier analyses.',
            'intervals':'Simulations: 2.5-97.5 percentiles of fixed-parameter datasets. Observed CCF: 1000 whole-lineage bootstrap draws. No parameter uncertainty.',
            'replicates':args.replicates,'validation_ids':sorted(validids),'fits':{}}
    for scope,sub in [('train',[g for g in groups if g.lineage.iloc[0] not in validids]),('all',groups)]:
        old=previous['fits'][scope];refs=old['references'];cycles=pd.concat(sub,ignore_index=True)
        tr=transitions(sub);growth=fit_growth(tr,refs,'AR');factors=fit_factors(sub);hp=old['hazard']
        theta=np.r_[np.log([hp['u'],hp['omega2'],hp['eta']]),hp['kappa']]
        score=Likelihood(cycles,refs['b'],order=96)(theta)
        if abs(score-old['hazard_fit']['log_likelihood_order96'])>1e-6:raise AssertionError('Hazard fit/data mismatch')
        # F->b Jacobian: p(b_i|past)=p(F_i)*F_i/b_i. It is parameter-independent.
        logjac=sum(float((np.log(g.f_effective.iloc[1:])-np.log(g.length_birth.iloc[1:])).sum()) for g in sub)
        config['fits'][scope]={'references':refs,'growth':growth,'factor':factors,'hazard':hp,'timing_log_likelihood':score,
                              'joint_log_likelihood_in_F_coordinates':score+growth['log_likelihood']+factors['log_likelihood'],
                              'joint_log_likelihood_in_birth_coordinates':score+growth['log_likelihood']+factors['log_likelihood']+logjac,
                              'n_cycles':len(cycles),'n_segments':len(sub)}
        print('FIT',scope,growth,factors,flush=True)
    valgroups=[g for g in groups if g.lineage.iloc[0] in validids];tr=transitions(valgroups);fit=config['fits']['train'];hp=fit['hazard']
    vals=np.concatenate([g.f_effective.to_numpy()[1:] for g in valgroups if len(g)>1])
    alpha_ll=growth_density(tr,fit['growth'],fit['references']);f_ll=factor_density(vals,fit['factor'])
    t_ll=log_time_density(tr.time.to_numpy(),tr.birth.to_numpy(),tr.alpha.to_numpy(),hp['u'],hp['omega2'],hp['eta'],hp['kappa'],hp['b_star'],order=96)
    jac=np.log(vals)-np.log(tr.birth.to_numpy())
    pd.DataFrame({'lineage':tr.lineage,'generation':tr.generation,'log_density_time':t_ll,'log_density_alpha':alpha_ll,'log_density_F':f_ll,'F_to_birth_log_jacobian':jac,'joint_log_density_birth_coordinates':t_ll+alpha_ll+f_ll+jac}).to_csv(OUT/'heldout_density_scores.csv',index=False)
    config['heldout_mean_joint_log_density_birth_coordinates']=float(np.mean(t_ll+alpha_ll+f_ll+jac))
    metrics=[];acfs=[];ccfs=[];hists=[];checks=[]
    for scope,sub,fitscope in [('validation',valgroups,'train'),('all',groups,'all')]:
        fit=config['fits'][fitscope];refs=fit['references'];rp=FastReplay(sub,refs['b'],refs['alpha'],args.replicates,93413)
        _,ob,ot=rp.observed();oa=rp.alpha[None].copy();of=np.broadcast_to(rp.f[None],ob.shape).copy();of[:,:,0]=np.nan
        observed=(ob,oa,ot,of);runs={'Observed':observed,'Recorded_F':simulate(rp,fit,True),'Generative':simulate(rp,fit,False)}
        included=np.array([len(g)>1 for g in sub]);fgroups=[g.iloc[1:].reset_index(drop=True) for g in sub if len(g)>1]
        frp=FastReplay(fgroups,refs['b'],refs['alpha'],args.replicates,93413)
        for name,run in runs.items():
            b,a,t,f=run;values=statistics(rp,b,a,t)
            fa=f[:,included,1:];fv=fa[:,frp.mask];values['mean_F']=fv.mean(1);values['sd_F']=fv.std(1);values['q99_F']=np.quantile(fv,.99,axis=1)
            left,right=rp.pairs[1];ff=f[:,rp.mask];tt=t[:,rp.mask];aa=a[:,rp.mask]
            values['corr_t_next_F']=corr(tt[:,left],ff[:,right]);values['corr_alpha_next_F']=corr(aa[:,left],ff[:,right])
            values['F_acf1']=frp.acfs(fa)[:,1]
            for key,v in values.items():
                lo,hi=np.quantile(v,[.025,.975]);metrics.append({'scope':scope,'model':name,'metric':key,'mean':v.mean(),'lo':lo if name!='Observed' else np.nan,'hi':hi if name!='Observed' else np.nan})
            arrays={'birth':b,'alpha':a,'time':t,'factor':fa}
            for variable,arr in arrays.items():
                replay=frp if variable=='factor' else rp;ac=replay.acfs(arr);lo,hi=np.quantile(ac,[.025,.975],axis=0)
                acfs.extend({'scope':scope,'model':name,'variable':variable,'lag':k,'mean':v,'lo':l if name!='Observed' else np.nan,'hi':h if name!='Observed' else np.nan} for k,(v,l,h) in enumerate(zip(ac.mean(0),lo,hi)))
                edges={'birth':np.arange(0,6.01,.1),'alpha':np.arange(0,2.01,.04),'time':np.arange(0,12.01,1/6),'factor':np.arange(.2,1.201,.02)}[variable]
                flat=arr[:,replay.mask];density=np.histogram(flat,bins=edges)[0]/flat.size/np.diff(edges)
                hists.extend({'scope':scope,'model':name,'variable':variable,'left':l,'right':r,'density':d} for l,r,d in zip(edges[:-1],edges[1:],density))
            for v1,v2 in [('birth','time'),('alpha','time'),('birth','alpha')]:
                stats,boot=ccf_statistics(rp,arrays[v1],arrays[v2],False,1000 if name=='Observed' else 0)
                for statistic,v in stats.items():
                    lo,hi=np.quantile(boot[statistic] if name=='Observed' else v,[.025,.975],axis=0)
                    ccfs.extend({'scope':scope,'model':name,'left_variable':v1,'right_variable':v2,'statistic':statistic,'lag':k,'mean':value,'lo':l,'hi':h} for k,value,l,h in zip(LAGS,v.mean(0),lo,hi))
            checks.append({'scope':scope,'model':name,'max_relative_recursion_error':check_recursion(rp,run)})
            print('REPLAY',scope,name,{k:round(float(v.mean()),5) for k,v in values.items()},flush=True)
        config.setdefault('control_checks',[]).append({'scope':scope,'max_birth_change_when_F_generated':float(np.max(abs(runs['Recorded_F'][0]-runs['Generative'][0]))),'max_alpha_change_when_F_generated':float(np.max(abs(runs['Recorded_F'][1]-runs['Generative'][1])))})
        b,a,t,f=runs['Generative'];np.savez_compressed(OUT/f'{scope}_example_trajectories.npz',birth=b[:5],alpha=a[:5],time=t[:5],factor=f[:5],mask=rp.mask,lineage=np.array([g.lineage.iloc[0] for g in sub]),first_generation=np.array([g.generation.iloc[0] for g in sub]))
    for filename,rows in [('joint_metrics.csv',metrics),('acf_comparison.csv',acfs),('cross_correlations.csv',ccfs),('marginal_densities.csv',hists),('recursion_checks.csv',checks)]:pd.DataFrame(rows).to_csv(OUT/filename,index=False)
    (OUT/'fit_results.json').write_text(json.dumps(config,indent=2))


if __name__=='__main__':main()
