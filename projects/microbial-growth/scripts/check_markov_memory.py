#!/usr/bin/env python3
"""Held-out-lineage tests of predictive history beyond the current Glu state.

These are tests of predictive conditional means and a Gaussian residual score,
not a distribution-free proof of the Markov property. All tuning is training-only.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from run_stationary_memory_time_hazard import load_data, PROJECT_DIR

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/markov_memory_check'
WIDTHS=[0.,.5,1.,2.]
RIDGES=np.logspace(-6,1,8)
CENTERS=192
BASIS_SEED=42041


def windows(groups):
    """Four consecutive states: oldest, previous, current, next."""
    values=[];ids=[];gens=[]
    for group in groups:
        z=np.log(group[['time','alpha','length_birth']].to_numpy())
        generation=group.generation.to_numpy()
        for end in range(3,len(group)):
            if np.all(np.diff(generation[end-3:end+1])==1):
                values.append(z[end-3:end+1]);ids.append(group.lineage.iloc[0]);gens.append(generation[end])
    return np.array(values),np.array(ids),np.array(gens)


class Basis:
    def __init__(self,x,width,base_dim):
        self.mean=x.mean(0);self.sd=np.maximum(x.std(0),1e-8);self.width=width;self.base_dim=base_dim
        z=(x-self.mean)/self.sd
        rng=np.random.default_rng(BASIS_SEED)
        self.centers=z[rng.choice(len(z),min(CENTERS,len(z)),replace=False)]
        phi=self.raw(x);self.pm=phi.mean(0);self.ps=np.maximum(phi.std(0),1e-6)
    def raw(self,x):
        z=(x-self.mean)/self.sd
        parts=[z]+[(z[:,i]*z[:,j])[:,None] for i in range(z.shape[1]) for j in range(i,z.shape[1])]
        if self.width:
            # Preserve the current-state nonlinear basis when adding history.
            # Otherwise history could win merely by changing the approximation.
            for dim in sorted(set([self.base_dim,z.shape[1]])):
                q=z[:,:dim];c=self.centers[:,:dim]
                distance=np.maximum((q*q).sum(1)[:,None]+(c*c).sum(1)-2*q@c.T,0)
                parts.append(np.exp(-distance/(2*self.width**2*dim)))
        return np.column_stack(parts)
    def transform(self,x):return (self.raw(x)-self.pm)/self.ps


def ridge_path(x,y,xq):
    ym=y.mean(0);sd=np.maximum(y.std(0),1e-8);yc=(y-ym)/sd
    eig,v=np.linalg.eigh(x.T@x/len(x))
    projected=v.T@(x.T@yc/len(x))
    return np.array([((xq@v)@(projected/(np.maximum(eig,0)[:,None]+r))) *sd+ym for r in RIDGES])


def fit_predict(x,y,train,test,folds,base_dim):
    xt=x[train];yt=y[train];xv=x[test]
    oof=np.empty((len(WIDTHS),len(RIDGES),len(xt),y.shape[1]));pred=np.empty((len(WIDTHS),len(RIDGES),len(xv),y.shape[1]))
    for wi,width in enumerate(WIDTHS):
        for k in range(5):
            tr=folds!=k;va=~tr;b=Basis(xt[tr],width,base_dim)
            oof[wi,:,va,:]=ridge_path(b.transform(xt[tr]),yt[tr],b.transform(xt[va])).transpose(1,0,2)
        b=Basis(xt,width,base_dim);pred[wi]=ridge_path(b.transform(xt),yt,b.transform(xv))
    errors=((oof-yt[None,None])**2).mean(2)
    result={}
    for family,nw in [('quadratic',1),('nonlinear',len(WIDTHS))]:
        choices=[np.unravel_index(np.argmin(errors[:nw,:,j]),(nw,len(RIDGES))) for j in range(y.shape[1])]
        p=np.column_stack([pred[w,r,:,j] for j,(w,r) in enumerate(choices)])
        cv=np.column_stack([oof[w,r,:,j] for j,(w,r) in enumerate(choices)])
        residual=yt-cv;cov=residual.T@residual/len(residual)+np.eye(y.shape[1])*1e-10
        result[family]={'prediction':p,'cv_prediction':cv,'covariance':cov,'choices':[{'width':WIDTHS[w],'ridge':float(RIDGES[r]),'cv_mse':float(errors[w,r,j])} for j,(w,r) in enumerate(choices)]}
    return result


def score(y,p,cov):
    e=y-p
    return -.5*(y.shape[1]*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+np.sum((e@np.linalg.inv(cov))*e,axis=1))


def interval(delta,ids):
    unique=np.unique(ids);sums=np.array([delta[ids==i].sum() for i in unique]);counts=np.array([np.sum(ids==i) for i in unique])
    rng=np.random.default_rng(70155);ix=rng.integers(0,len(unique),(5000,len(unique)))
    boots=sums[ix].sum(1)/counts[ix].sum(1)
    return float(delta.mean()),float(np.quantile(boots,.025)),float(np.quantile(boots,.975))


def state_features(z):
    # Accumulated growth is a deterministic function of the two current logs.
    return np.column_stack([z,np.exp(z[:,0]+z[:,1])])


def main():
    global OUT,CENTERS,BASIS_SEED
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--centers',type=int,default=CENTERS)
    parser.add_argument('--basis-seed',type=int,default=BASIS_SEED)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output_dir;CENTERS=args.centers;BASIS_SEED=args.basis_seed
    OUT.mkdir(parents=True,exist_ok=True)
    data,groups=load_data();z,ids,gens=windows(groups)
    past=json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/conditional_effective_jumps/fit_results.json').read_text())
    test=np.isin(ids,past['validation_ids']);train=~test
    trainids=np.unique(ids[train]);np.random.default_rng(31029).shuffle(trainids)
    mapping={i:k%5 for k,i in enumerate(trainids)};folds=np.array([mapping[i] for i in ids[train]])
    states=[state_features(z[:,k]) for k in range(3)]
    # Predict the jump innovation instead of next log birth, enforcing the known
    # kinematic contribution exactly in every model. Errors equal log-birth errors.
    y=z[:,3].copy();offset=z[:,2,2]+np.exp(z[:,2,0]+z[:,2,1]);y[:,2]-=offset
    designs={
        'state_order1':(states[2],y),
        'state_order2':(np.column_stack([states[2],states[1]]),y),
        'state_order3':(np.column_stack([states[2],states[1],states[0]]),y),
        'timing_no_history':(z[:,3,1:],y[:,:1]),
        'timing_one_previous':(np.column_stack([z[:,3,1:],states[2]]),y[:,:1]),
        'timing_two_previous':(np.column_stack([z[:,3,1:],states[2],states[1]]),y[:,:1]),
        'state_order1_age':(np.column_stack([states[2],np.log1p(gens-1)]),y),
        'state_order2_age':(np.column_stack([states[2],np.log1p(gens-1),states[1]]),y),
        'timing_no_history_age':(np.column_stack([z[:,3,1:],np.log1p(gens)]),y[:,:1]),
        'timing_one_previous_age':(np.column_stack([z[:,3,1:],np.log1p(gens),states[2]]),y[:,:1]),
    }
    results={};tuning={};metrics=[];predictions=[]
    for name,(x,target) in designs.items():
        base_dim=(4 if name.startswith('state') else 2)+int(name.endswith('_age'))
        results[name]=fit_predict(x,target,train,test,folds,base_dim);tuning[name]={}
        for family,result in results[name].items():
            p=result['prediction'];mse=((p-target[test])**2).mean(0);ll=score(target[test],p,result['covariance'])
            tuning[name][family]={'parameters':result['choices'],'residual_covariance':result['covariance'].tolist()}
            for j,m in enumerate(mse):metrics.append({'model':name,'family':family,'outcome':['log_time','log_alpha','log_birth'][j],'mse':m,'rmse':np.sqrt(m),'mean_gaussian_log_score':ll.mean()})
            for j in range(target.shape[1]):
                predictions.extend({'model':name,'family':family,'lineage':i,'generation':g,'outcome':['log_time','log_alpha','log_birth'][j],'observed':v+(o if j==2 else 0),'predicted':w+(o if j==2 else 0)} for i,g,v,w,o in zip(ids[test],gens[test],target[test,j],p[:,j],offset[test]))
        print(name,[(f,((r['prediction']-target[test])**2).mean(0).round(6).tolist()) for f,r in results[name].items()],flush=True)
    comparisons=[]
    pairs=[('state_order1','state_order2'),('state_order1','state_order3'),('state_order2','state_order3'),('timing_no_history','timing_one_previous'),('timing_one_previous','timing_two_previous')]
    pairs.extend([('state_order1_age','state_order2_age'),('timing_no_history_age','timing_one_previous_age')])
    for base,extended in pairs:
        target=designs[base][1][test]
        for family in ['quadratic','nonlinear']:
            a=results[base][family];b=results[extended][family]
            ea=(target-a['prediction'])**2;eb=(target-b['prediction'])**2
            for j in range(target.shape[1]):
                gain,lo,hi=interval(ea[:,j]-eb[:,j],ids[test])
                comparisons.append({'baseline':base,'extended':extended,'family':family,'metric':['log_time_MSE','log_alpha_MSE','log_birth_MSE'][j],'gain':gain,'lo':lo,'hi':hi,'relative_reduction_percent':100*gain/ea[:,j].mean()})
            gain,lo,hi=interval(score(target,b['prediction'],b['covariance'])-score(target,a['prediction'],a['covariance']),ids[test])
            comparisons.append({'baseline':base,'extended':extended,'family':family,'metric':'Gaussian_log_score','gain':gain,'lo':lo,'hi':hi})
    pd.DataFrame(metrics).to_csv(OUT/'heldout_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'heldout_comparisons.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'heldout_predictions.csv',index=False)
    config={'n_input_cycles':len(data),'n_windows':len(z),'n_train_windows':int(train.sum()),'n_test_windows':int(test.sum()),'n_train_lineages':len(np.unique(ids[train])),'n_test_lineages':len(np.unique(ids[test])),
            'heldout_lineages':list(map(float,np.unique(ids[test]))),'definition':'S_i=(log t_i, log alpha_i, log b_i); test predictive gain for S_(i+1) from S_(i-1) or S_(i-2), beyond S_i. Windows require four consecutive generations.',
            'features':'Quadratic polynomial features and optionally Gaussian radial-basis landmarks for the current-state predictors and the extended history predictors. The current-state basis is retained in augmented models. Growth alpha*t supplied as deterministic current-state feature. Each predictor and basis standardized on fitting data only.',
            'landmarks_per_block':CENTERS,'landmark_seed':BASIS_SEED,
            'target_birth':'log b_(i+1) minus [log b_i+alpha_i*t_i]; known contribution added back. MSE equals next-log-birth MSE.',
            'selection':'Five grouped folds of training lineages select bandwidth/ridge per outcome by CV MSE. Validation lineages never used to choose parameters. Nonlinear family includes quadratic-only option.',
            'intervals':'5000 paired whole-validation-lineage bootstrap resamples; 95% pointwise intervals, no multiplicity correction. Conditional on fitted predictors.',
            'log_score':'Joint Gaussian residual score on log outcomes, with full constant residual covariance estimated from training out-of-fold predictions. A working model, not a nonparametric conditional-independence test.',
            'age_control':'Models ending in _age condition additionally on log(1+generation) at the conditioning cycle. This allows common generation-dependent drift, but not arbitrary lineage-specific nonstationarity.',
            'limitations':'Predictive mean tests can miss higher-moment dependence. Observational measurement noise, time drift and fixed lineage heterogeneity can all produce apparent memory. Reuses existing validation split, previously examined in other project analyses.',
            'widths':WIDTHS,'ridges':RIDGES.tolist(),'tuning':tuning}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=='__main__':main()
