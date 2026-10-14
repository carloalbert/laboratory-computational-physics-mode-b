#!/usr/bin/env python3
"""Check alpha independence and history-free timing on held-out Glu lineages.

Growth-rate factors use Gaussian conditional KDEs in log alpha. Timing checks
use the existing cross-validated nonlinear regressions as a separate diagnostic.
"""
import json
import numpy as np
import pandas as pd
from run_stationary_memory_time_hazard import load_data,PROJECT_DIR
from check_markov_memory import fit_predict,score,interval

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/factorization_check'
HX=[.15,.3,.6,1.2,2.4,np.inf]
HY=[.05,.1,.2,.35,.6]
MODELS={'marginal':[], 'birth':[0], 'previous_alpha':[1],
        'birth_previous_alpha':[0,1], 'birth_previous_state':[0,1,2,3],
        'birth_generation':[0,4], 'birth_previous_state_generation':[0,1,2,3,4]}
CANDIDATES={name:([(hb,hp) for hb in HX for hp in HX] if name=='birth_previous_alpha' else [(h,h) for h in (HX if columns else [np.inf])]) for name,columns in MODELS.items()}


def transitions(groups):
    rows=[]
    for g in groups:
        for j in range(1,len(g)):
            if g.generation.iloc[j]-g.generation.iloc[j-1]!=1:continue
            previous=g.iloc[j-1];current=g.iloc[j]
            rows.append({'lineage':current.lineage,'generation':current.generation,
                         'time':current.time,'alpha':current.alpha,'birth':current.length_birth,
                         'previous_time':previous.time,'previous_alpha':previous.alpha,'previous_birth':previous.length_birth})
    return pd.DataFrame(rows)


def kde_scores(x,y,xq,yq):
    """All bandwidth scores; transformations depend on training rows only."""
    xm=x.mean(0);xs=np.maximum(x.std(0),1e-9);ym=y.mean();ys=y.std()
    xt=(x-xm)/xs;query=(xq-xm)/xs;yt=(y-ym)/ys;yquery=(yq-ym)/ys
    scores={name:np.empty((len(CANDIDATES[name]),len(HY),len(query))) for name in MODELS}
    for start in range(0,len(query),160):
        sl=slice(start,min(start+160,len(query)))
        distances=(query[sl,None,:]-xt[None,:,:])**2
        dy=yquery[sl,None]-yt
        kernels=np.array([np.exp(-.5*(dy/h)**2)/(np.sqrt(2*np.pi)*h*ys) for h in HY])
        for name,columns in MODELS.items():
            dist=distances[:,:,columns].sum(2) if columns else np.zeros(dy.shape)
            for k,(hb,hp) in enumerate(CANDIDATES[name]):
                scaled=distances[:,:,0]/hb**2+distances[:,:,1]/hp**2 if name=='birth_previous_alpha' else dist/hb**2
                logw=-.5*scaled;logw-=logw.max(1,keepdims=True)
                w=np.exp(logw);w/=w.sum(1,keepdims=True)
                density=np.einsum('ij,kij->ki',w,kernels)
                scores[name][k,:,sl]=np.log(np.maximum(density,1e-300))
    return scores


def alpha_density_check(tab,train,test,folds):
    x=np.column_stack([np.log(tab.birth),np.log(tab.previous_alpha),np.log(tab.previous_time),np.log(tab.previous_birth),np.log1p(tab.generation)])
    y=np.log(tab.alpha.to_numpy());xt=x[train];yt=y[train]
    cv={name:np.zeros((len(CANDIDATES[name]),len(HY))) for name in MODELS}
    for k in range(5):
        tr=folds!=k;va=~tr;values=kde_scores(xt[tr],yt[tr],xt[va],yt[va])
        for name in cv:cv[name]+=values[name].sum(2)
        print('Alpha KDE training fold',k+1,'complete',flush=True)
    validation=kde_scores(xt,yt,x[test],y[test]);selected={};predictions={};cvrows=[]
    for name in cv:
        wi,hi=np.unravel_index(np.argmax(cv[name]),cv[name].shape)
        h,hp=CANDIDATES[name][wi]
        selected[name]={'predictor_columns':MODELS[name],'predictor_bandwidth':float(h) if np.isfinite(h) else None,'previous_alpha_bandwidth':float(hp) if np.isfinite(hp) else None,'response_bandwidth':HY[hi],'training_cv_log_density_log_alpha':float(cv[name][wi,hi]/train.sum())}
        # Jacobian converts log-alpha density to alpha density. It cancels in
        # paired comparisons, but ensures the reported factors are p(alpha).
        predictions[name]=validation[name][wi,hi]-y[test]
        for w in range(cv[name].shape[0]):
            for j in range(len(HY)):cvrows.append({'model':name,'h_predictor':CANDIDATES[name][w][0],'h_previous_alpha':CANDIDATES[name][w][1],'h_response':HY[j],'mean_log_density_log_alpha':cv[name][w,j]/train.sum()})
    pd.DataFrame(cvrows).to_csv(OUT/'alpha_density_training_cv.csv',index=False)
    return predictions,selected


def cluster_corr(x,y,ids,center=False):
    unique,inverse=np.unique(ids,return_inverse=True);counts=np.bincount(inverse)
    if center:
        x=x-np.bincount(inverse,weights=x)[inverse]/counts[inverse]
        y=y-np.bincount(inverse,weights=y)[inverse]/counts[inverse]
    summaries=np.array([np.bincount(inverse,weights=v) for v in [np.ones(len(x)),x,y,x*x,y*y,x*y]])
    def corr(s):
        n,sx,sy,sxx,syy,sxy=s
        return (sxy-sx*sy/n)/np.sqrt((sxx-sx*sx/n)*(syy-sy*sy/n))
    rng=np.random.default_rng(28901);ix=rng.integers(0,len(unique),(5000,len(unique)))
    boot=corr(summaries[:,ix].sum(2))
    return float(corr(summaries.sum(1))),*np.quantile(boot,[.025,.975]).tolist()


def relative_gain(a,b,ids):
    unique=np.unique(ids);values=np.array([[a[ids==i].sum(),b[ids==i].sum()] for i in unique])
    rng=np.random.default_rng(70155);ix=rng.integers(0,len(unique),(5000,len(unique)))
    total=values[ix].sum(1);boot=100*(1-total[:,1]/total[:,0])
    return float(100*(1-b.mean()/a.mean())),*np.quantile(boot,[.025,.975]).tolist()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data,groups=load_data();tab=transitions(groups);ids=tab.lineage.to_numpy()
    old=json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/conditional_effective_jumps/fit_results.json').read_text())
    test=np.isin(ids,old['validation_ids']);train=~test
    ti=np.unique(ids[train]);np.random.default_rng(31029).shuffle(ti)
    mapping={i:k%5 for k,i in enumerate(ti)};folds=np.array([mapping[i] for i in ids[train]])
    correlations=[]
    for scope,mask in [('all',np.ones(len(tab),bool)),('validation',test)]:
        sub=tab[mask]
        for label,x,y in [('alpha_with_birth',sub.alpha.to_numpy(),sub.birth.to_numpy()),('alpha_lag1',sub.alpha.to_numpy(),sub.previous_alpha.to_numpy()),('alpha_with_previous_time',sub.alpha.to_numpy(),sub.previous_time.to_numpy())]:
            for center in [False,True]:
                value,lo,hi=cluster_corr(x,y,ids[mask],center)
                correlations.append({'scope':scope,'pair':label,'lineage_demeaned':center,'correlation':value,'lo':lo,'hi':hi})
    alpha,selected=alpha_density_check(tab,train,test,folds)
    comparisons=[];densityrows=[]
    for name,values in alpha.items():
        densityrows.extend({'model':name,'lineage':i,'generation':g,'log_density_alpha':v} for i,g,v in zip(ids[test],tab.generation.to_numpy()[test],values))
    for base,extended in [('marginal','birth'),('marginal','previous_alpha'),('marginal','birth_previous_alpha'),('birth','birth_previous_alpha'),('previous_alpha','birth_previous_alpha'),('birth_previous_alpha','birth_previous_state'),('birth_generation','birth_previous_state_generation')]:
        gain,lo,hi=interval(alpha[extended]-alpha[base],ids[test])
        comparisons.append({'factor':'alpha','baseline':base,'extended':extended,'family':'conditional_KDE','metric':'log_density_gain_nats','gain':gain,'lo':lo,'hi':hi})
    current=np.log(tab[['birth','alpha']].to_numpy());history=np.log(tab[['previous_time','previous_alpha','previous_birth']].to_numpy())
    age=np.log1p(tab.generation.to_numpy())[:,None];y=np.log(tab.time.to_numpy())[:,None]
    designs={'current':(current,2),'current_history':(np.column_stack([current,history]),2),'current_age':(np.column_stack([current,age]),3),'current_age_history':(np.column_stack([current,age,history]),3)}
    timing={};timingchoices={};timepred=[];timedensity=[]
    for name,(x,base_dim) in designs.items():
        timing[name]=fit_predict(x,y,train,test,folds,base_dim);timingchoices[name]={}
        for family,res in timing[name].items():
            timingchoices[name][family]={'parameters':res['choices'],'training_oof_residual_variance':float(res['covariance'][0,0])}
            timepred.extend({'model':name,'family':family,'lineage':i,'generation':g,'observed_log_time':v,'predicted_log_time':p} for i,g,v,p in zip(ids[test],tab.generation.to_numpy()[test],y[test,0],res['prediction'][:,0]))
            ll=score(y[test],res['prediction'],res['covariance'])-y[test,0]
            timedensity.extend({'model':name,'family':family,'lineage':i,'generation':g,'log_density_time':v} for i,g,v in zip(ids[test],tab.generation.to_numpy()[test],ll))
        print('Timing fit',name,'complete',flush=True)
    for base,extended in [('current','current_history'),('current_age','current_age_history')]:
        for family in ['quadratic','nonlinear']:
            a=timing[base][family];b=timing[extended][family]
            gain,lo,hi=relative_gain((y[test,0]-a['prediction'][:,0])**2,(y[test,0]-b['prediction'][:,0])**2,ids[test])
            comparisons.append({'factor':'time','baseline':base,'extended':extended,'family':family,'metric':'log_time_MSE_reduction_percent','gain':gain,'lo':lo,'hi':hi})
            gain,lo,hi=interval(score(y[test],b['prediction'],b['covariance'])-score(y[test],a['prediction'],a['covariance']),ids[test])
            comparisons.append({'factor':'time','baseline':base,'extended':extended,'family':family,'metric':'Gaussian_log_density_gain_nats','gain':gain,'lo':lo,'hi':hi})
    joint=[]
    for family in ['quadratic','nonlinear']:
        base=timing['current'][family];extended=timing['current_history'][family]
        ta=score(y[test],base['prediction'],base['covariance'])-y[test,0]
        tb=score(y[test],extended['prediction'],extended['covariance'])-y[test,0]
        densities={'original_factorization':alpha['marginal']+ta,
                   'dependent_alpha':alpha['birth_previous_alpha']+ta,
                   'dependent_alpha_and_timing_history':alpha['birth_previous_alpha']+tb}
        for name,ll in densities.items():
            gain,lo,hi=interval(ll-densities['original_factorization'],ids[test])
            joint.append({'family':family,'model':name,'mean_joint_log_density':float(ll.mean()),'gain_over_original_nats':gain,'lo':lo,'hi':hi})
    pd.DataFrame(correlations).to_csv(OUT/'observed_correlations.csv',index=False)
    pd.DataFrame(densityrows).to_csv(OUT/'alpha_heldout_log_densities.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'heldout_comparisons.csv',index=False)
    pd.DataFrame(timepred).to_csv(OUT/'time_heldout_predictions.csv',index=False)
    pd.DataFrame(timedensity).to_csv(OUT/'time_heldout_log_densities.csv',index=False)
    pd.DataFrame(joint).to_csv(OUT/'joint_factorization_scores.csv',index=False)
    pd.DataFrame([{'model':name,'mean_log_density_alpha':float(v.mean())} for name,v in alpha.items()]).to_csv(OUT/'alpha_heldout_scores.csv',index=False)
    config={'n_input_cycles':len(data),'n_consecutive_transitions':len(tab),'n_train_cycles':int(train.sum()),'n_validation_cycles':int(test.sum()),'n_train_lineages':int(len(np.unique(ids[train]))),'n_validation_lineages':int(len(np.unique(ids[test]))),
            'validation_ids':list(map(float,np.unique(ids[test]))),'alpha_model':'Gaussian conditional KDE in log alpha; predictors standardized using fitting rows only. Bandwidths selected by five grouped training folds. Jacobian included for density in alpha.',
            'predictor_columns':['log birth_i','log alpha_(i-1)','log time_(i-1)','log birth_(i-1)','log(1+generation_i)'],'predictor_bandwidth_candidates':[None if np.isinf(h) else h for h in HX],'response_bandwidth_candidates':HY,'selected_alpha_models':selected,'time_models':timingchoices,
            'bandwidth_note':'The birth-plus-previous-alpha KDE tunes separate predictor bandwidths, so its candidate set includes both simpler conditional KDEs. Higher-dimensional exploratory KDEs use a common predictor bandwidth and are not reliable tests for absence of incremental dependence.',
            'joint_score':'Growth conditional KDE times working lognormal time density with nonlinear conditional mean and constant residual variance calibrated on training out-of-fold residuals. Both Jacobians included. Not a refit of the mechanistic hazard.',
            'uncertainty':'5000 whole-lineage paired bootstrap samples on validation; pointwise 95% intervals conditional on fitted predictors. Same validation split as earlier project diagnostics; exploratory, not a new untouched test set.',
            'factorization':'Sequential event factor given current birth: p(alpha_i|birth_i,history_i) p(time_i|birth_i,alpha_i,history_i). User factorization omits birth/history from alpha factor and history from time factor.',
            'scope':'Does not fit a birth/partition transition law, identify causality, or distinguish physical memory from observation error or persistent lineage effects. A conditional alpha model does not prove birth causes alpha.',
            'conditioning_note':'An alpha density with separate parameters is a constant with respect to timing-hazard parameters when alpha is treated as observed. Correcting this factor alone cannot change that conditional timing MLE.'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    print(pd.DataFrame(correlations).to_string(index=False),flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)
    print(pd.DataFrame(joint).to_string(index=False),flush=True)


if __name__=='__main__':main()
