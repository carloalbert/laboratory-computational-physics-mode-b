#!/usr/bin/env python3
"""Fit and validate stochastic effective division jumps, without physical f.

Joint density factors into event-time hazard and conditional lognormal jump.
Given observed b, alpha, t, their parameter blocks separate exactly in the
conditional likelihood. Generated factors are drawn AFTER the division time.
Gaps reset dynamics; original-lineage ACF centering/weighting is preserved.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from screen_parsimonious_joint_hazards import (
    FastReplay, METRICS, PROJECT_DIR, load_data, segments, bootstrap_errors,
)
from screen_hill_alpha_clock import ClockLikelihood, fit as fit_clock

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/conditional_effective_jumps'
FAMILIES={'iid':[], 'birth':['b'], 'age':['t'], 'growth':['g'],
          'birth_age':['b','t'], 'birth_age_alpha':['b','t','a'],
          'birth_growth_alpha':['b','g','a']}


def transitions(groups):
    rows=[]
    for group in groups:
        for n in range(len(group)-1):
            rows.append({'lineage':int(group.lineage.iloc[n]),
                'generation':int(group.generation.iloc[n]),
                'b':float(group.length_birth.iloc[n]),
                'alpha':float(group.alpha.iloc[n]),'t':float(group.time.iloc[n]),
                'next_b':float(group.length_birth.iloc[n+1]),
                'f_eff':float(group.f_effective.iloc[n+1])})
    return pd.DataFrame(rows)


def predictors(b,alpha,t,refs):
    return {'b':np.log(b/refs['b']), 'a':np.log(alpha/refs['alpha']),
            't':np.log(t/refs['t']), 'g':alpha*t-refs['g']}


def matrix(data,features,refs):
    p=predictors(data.b.to_numpy(),data.alpha.to_numpy(),data.t.to_numpy(),refs)
    return np.column_stack([np.ones(len(data))]+[p[key] for key in features])


def fit_jump(data,features,refs):
    X=matrix(data,features,refs);y=np.log(data.f_eff.to_numpy())
    beta=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@beta
    sigma=float(np.sqrt(np.mean(res*res)))
    ll=-len(y)*(np.log(sigma)+.5*np.log(2*np.pi)+.5)-y.sum()
    nparams=len(beta)+1
    return {'features':features,'coefficients':beta.tolist(),'sigma':sigma,
            'n_params':nparams,'log_likelihood_F':float(ll),
            'BIC_F':float(nparams*np.log(len(y))-2*ll)}


def jump_log_density(data,params,refs,variable='F'):
    mu=matrix(data,params['features'],refs)@np.array(params['coefficients'])
    y=np.log(data.f_eff.to_numpy());sigma=params['sigma']
    ll=-.5*((y-mu)/sigma)**2-np.log(sigma)-.5*np.log(2*np.pi)
    # In next-b coordinates, d log(F)/d(next_b)=1/next_b. The Jacobian is
    # parameter-independent, so separate hazard/jump fitting is joint MLE.
    return ll-np.log(data.next_b.to_numpy() if variable=='next_b' else data.f_eff.to_numpy())


def cross_validation(data,refs):
    ids=np.sort(data.lineage.unique());np.random.default_rng(60107).shuffle(ids)
    folds={key:i%5 for i,key in enumerate(ids)}
    assignment=data.lineage.map(folds).to_numpy();rows=[];summary=[]
    for name,features in FAMILIES.items():
        scores=[]
        for fold in range(5):
            tr=data.loc[assignment!=fold];va=data.loc[assignment==fold]
            p=fit_jump(tr,features,refs)
            score=float(jump_log_density(va,p,refs).mean());scores.append(score)
            rows.append({'model':name,'fold':fold,'n_validation':len(va),'mean_log_density_F':score})
        summary.append({'model':name,'n_params':len(features)+2,'mean_log_density_F':float(np.mean(scores)),
                        'se_across_folds':float(np.std(scores,ddof=1)/np.sqrt(5))})
    summary=pd.DataFrame(summary);best=summary.iloc[summary.mean_log_density_F.argmax()]
    eligible=summary[summary.mean_log_density_F>=best.mean_log_density_F-best.se_across_folds]
    chosen=eligible.sort_values(['n_params','mean_log_density_F'],ascending=[True,False]).iloc[0].model
    return str(chosen),summary,pd.DataFrame(rows)


def simulate(replay,hazard,jump,refs,seed=309172):
    event_rng=np.random.default_rng(seed);jump_rng=np.random.default_rng(seed+100003)
    shape=(replay.R,*replay.mask.shape)
    births=np.zeros(shape);times=np.zeros(shape);factors=np.full(shape,np.nan)
    current=np.broadcast_to(replay.birth0,shape[:2]).copy()
    jump_noise=jump_rng.normal(size=shape)
    if jump is not None and 'empirical_residuals' in jump:
        jump_noise=jump_rng.choice(jump['empirical_residuals'],size=shape)
    elif jump is not None:
        jump_noise*=jump['sigma']
    for n in range(replay.mask.shape[1]):
        active=replay.mask[:,n]
        b=current[:,active];alpha=replay.alpha[active,n][None,:]
        rate=np.broadcast_to(hazard['omega2']*(alpha/replay.astar)**hazard['q'],b.shape)
        growth=np.zeros_like(b);pending=np.ones(b.shape,dtype=bool)
        offset=hazard.get('kappa',0.)*np.log(b/replay.bstar)
        for _ in range(100000):
            if not pending.any():break
            growth[pending]+=event_rng.exponential(size=pending.sum())/rate[pending]
            z=hazard['eta']*(np.log(b[pending])+np.log(np.expm1(np.minimum(growth[pending],700)))-np.log(hazard['u']))+offset[pending]
            accepted=event_rng.random(pending.sum())<np.exp(-np.logaddexp(0.,-z))
            pending[pending]=~accepted
        else:raise RuntimeError('Thinning did not finish')
        t=growth/alpha;births[:,active,n]=b;times[:,active,n]=t
        if n+1==replay.mask.shape[1]:continue
        continuing=replay.mask[:,n+1]
        local=continuing[active]
        if not local.any():continue
        bc=b[:,local];ac=alpha[:,local];tc=t[:,local];gc=growth[:,local]
        if jump is None:
            f=np.broadcast_to(replay.f[continuing,n+1],bc.shape)
        else:
            p=predictors(bc,ac,tc,refs);coef=jump['coefficients']
            mu=np.full_like(bc,coef[0])
            for c,feature in zip(coef[1:],jump['features']):mu+=c*p[feature]
            f=np.exp(mu+jump_noise[:,continuing,n])
        factors[:,continuing,n]=f
        current[:,continuing]=bc*np.exp(gc)*f
        if not np.all(np.isfinite(current)) or np.max(current)>1e8:
            raise FloatingPointError('Unstable size trajectory; do not silently clip')
    return births,times,factors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicates',type=int,default=200)
    parser.add_argument('--reuse-fit',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    data,original=load_data();groups=segments(original)
    split=json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/parsimonious_joint_hazard_screen/fit_results.json').read_text())
    valid_ids=set(split['validation_lineages']);train=[g for g in groups if g.lineage.iloc[0] not in valid_ids]
    valid=[g for g in groups if g.lineage.iloc[0] in valid_ids]
    tt=transitions(train);alltr=transitions(groups);train_data=pd.concat(train,ignore_index=True)
    if args.reuse_fit:result=json.loads((OUT/'fit_results.json').read_text())
    else:
        result={'model':'log F_eff = intercept + selected centered predictors + sigma Z; Z iid standard normal',
                'hazard':'alpha*omega2*(alpha/alpha_star)^q*Hill(p/u,eta); kappa=0',
                'design':'Recorded alpha histories and initial segment sizes; no physical f in fit; original-lineage split and ACF weighting; gap resets. Both hazard and jump fitted on training lineages for held-out check. All-data refit separately labeled descriptive.',
                'selection':'Five original-lineage folds within training set; one-standard-error parsimonious rule using conditional jump log density; no ACF calibration.',
                'n_cycles':len(data),'n_transitions':len(alltr),'n_segments':len(groups),
                'n_train_lineages':len(set(g.lineage.iloc[0] for g in train)),
                'n_validation_lineages':len(valid_ids),'validation_ids':sorted(valid_ids),'fits':{}}
        for scope,cycles,tr in [('train',train_data,tt),('all',data,alltr)]:
            refs={'b':float(np.exp(np.log(cycles.length_birth).mean())),
                  'alpha':float(np.exp(np.log(cycles.alpha).mean())),
                  't':float(np.exp(np.log(tr.t).mean())), 'g':float(np.mean(tr.alpha*tr.t))}
            likelihood=ClockLikelihood(cycles,refs['b'],refs['alpha'])
            hazard,history=fit_clock(likelihood,False,901+len(cycles))
            jumpfits={name:fit_jump(tr,features,refs) for name,features in FAMILIES.items()}
            result['fits'][scope]={'references':refs,'hazard':hazard,'jumps':jumpfits}
            pd.DataFrame({'iteration':np.arange(1,len(history)+1),'hazard_LL':history}).to_csv(OUT/f'{scope}_hazard_optimizer.csv',index=False)
            if scope=='train':
                chosen,cv,cvfold=cross_validation(tr,refs);result['selected_jump']=chosen
                cv.to_csv(OUT/'jump_cross_validation.csv',index=False);cvfold.to_csv(OUT/'jump_cv_folds.csv',index=False)
            print('fit',scope,'hazard',hazard,flush=True)
        (OUT/'fit_results.json').write_text(json.dumps(result,indent=2))
    chosen=result['selected_jump'];print('Selected jump',chosen,flush=True)
    # Held-out conditional density compares all nested jump laws, never selecting
    # with validation ACFs. The two Jacobians are documented separately.
    hold=transitions(valid);trainfit=result['fits']['train'];score_rows=[]
    for name,p in trainfit['jumps'].items():
        score_rows.append({'model':name,'n_jump_params':p['n_params'],
            'validation_n':len(hold),'validation_mean_log_density_F':jump_log_density(hold,p,trainfit['references']).mean(),
            'validation_mean_log_density_next_b':jump_log_density(hold,p,trainfit['references'],'next_b').mean()})
    pd.DataFrame(score_rows).to_csv(OUT/'heldout_jump_scores.csv',index=False)
    metricrows=[];acfrows=[];densities=[];conditional=[];jumpstats=[]
    for scope,gs,fit_scope in [('validation',valid,'train'),('all',groups,'all')]:
        fitted=result['fits'][fit_scope];refs=fitted['references'];hazard=fitted['hazard']
        rp=FastReplay(gs,refs['b'],refs['alpha'],args.replicates,913008)
        obs,ob,ot=rp.observed();ose=bootstrap_errors(gs,refs['b'],refs['alpha'],150)
        metricrows.extend({'scope':scope,'model':'Observed','metric':key,'mean':v,'lo':v-1.96*e,'hi':v+1.96*e} for key,v,e in zip(METRICS,obs,ose))
        for variable,arr in [('birth',ob),('time',ot)]:
            obsacf=rp.acfs(arr)[0]
            acfrows.extend({'scope':scope,'model':'Observed','variable':variable,'lag':k,'mean':v} for k,v in enumerate(obsacf))
        for j,name in enumerate(['fixed_factors']+list(FAMILIES)):
            p=None if name=='fixed_factors' else fitted['jumps'][name]
            b,t,f=simulate(rp,hazard,p,refs,seed=91402+1000*j+(0 if scope=='all' else 100000))
            stats=rp.statistics(b,t)
            metricrows.extend({'scope':scope,'model':name,'metric':key,'mean':v,'lo':lo,'hi':hi} for key,v,lo,hi in zip(METRICS,stats.mean(0),*np.quantile(stats,[.025,.975],axis=0)))
            for variable,arr in [('birth',b),('time',t)]:
                ac=rp.acfs(arr)
                for k in range(31):acfrows.append({'scope':scope,'model':name,'variable':variable,'lag':k,'mean':ac[:,k].mean(),'lo':np.quantile(ac[:,k],.025),'hi':np.quantile(ac[:,k],.975)})
            if scope=='all':
                for variable,v,o,edges in [('birth',b[:,rp.mask],ob[:,rp.mask],np.arange(0,8.01,.1)),('time',t[:,rp.mask],ot[:,rp.mask],np.arange(0,12.01,1/3))]:
                    hist=np.histogram(v,bins=edges)[0]/v.size/np.diff(edges);histobs=np.histogram(o,bins=edges)[0]/o.size/np.diff(edges)
                    densities.extend({'model':name,'variable':variable,'left':left,'right':right,'density':y,'observed_density':yo} for left,right,y,yo in zip(edges[:-1],edges[1:],hist,histobs))
                edges=np.quantile(ob[:,rp.mask],np.linspace(0,1,9));edges[0]=-np.inf;edges[-1]=np.inf
                for left,right in zip(edges[:-1],edges[1:]):
                    mask=(b[:,rp.mask]>=left)&(b[:,rp.mask]<right);om=(ob[:,rp.mask]>=left)&(ob[:,rp.mask]<right)
                    conditional.append({'model':name,'birth_left':left,'birth_right':right,'observed_mean_birth':ob[:,rp.mask][om].mean(),'observed_mean_time':ot[:,rp.mask][om].mean(),'simulated_mean_time':t[:,rp.mask][mask].mean()})
            fs=f[np.isfinite(f)]
            jumpstats.append({'scope':scope,'model':name,'mean_f':fs.mean(),'sd_f':fs.std(),'fraction_above_one':np.mean(fs>1)})
            print(scope,name,dict(zip(METRICS[:12],stats.mean(0)[:12].round(5))),flush=True)
    pd.DataFrame(metricrows).to_csv(OUT/'joint_metrics.csv',index=False)
    pd.DataFrame(acfrows).to_csv(OUT/'acf_comparison.csv',index=False)
    pd.DataFrame(densities).to_csv(OUT/'marginal_densities.csv',index=False)
    pd.DataFrame(conditional).to_csv(OUT/'conditional_time_by_birth.csv',index=False)
    pd.DataFrame(jumpstats).to_csv(OUT/'jump_marginals.csv',index=False)
    result['replicates']=args.replicates
    result['observed_effective_factors']={'min':float(alltr.f_eff.min()),'max':float(alltr.f_eff.max()),'mean':float(alltr.f_eff.mean()),'sd':float(alltr.f_eff.std(ddof=0))}
    (OUT/'fit_results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
