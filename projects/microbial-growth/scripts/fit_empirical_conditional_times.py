#!/usr/bin/env python3
"""Grouped-CV empirical Glu f(t|b,alpha), scored at recording resolution."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_DIR/'src'))
from empirical_conditional_time_density import ConditionalTimeKDE,normal_interval,normal_sf
from birth_size_dependent_time_hazard import cumulative_hazard_growth_time
from run_stationary_memory_time_hazard import load_data

OUT=PROJECT_DIR/'outputs/model_comparison/Glu/empirical_conditional_time_density'
DELTA=1/6
HB=[.03,.06,.12,.24,np.inf]
HA=[.06,.12,.24,.48,np.inf]
HG=[.02,.04,.08,.16]


def fit_cv(data):
    ids=np.sort(data.lineage.unique());np.random.default_rng(14706).shuffle(ids)
    assignment={i:k%5 for k,i in enumerate(ids)}
    fold=data.lineage.map(assignment).to_numpy();records=[]
    for k in range(5):
        tr=data[fold!=k];va=data[fold==k]
        bt=np.log(tr.length_birth.to_numpy());at=np.log(tr.alpha.to_numpy())
        gt=(tr.alpha*tr.time).to_numpy()
        bq=np.log(va.length_birth.to_numpy());aq=np.log(va.alpha.to_numpy())
        a=va.alpha.to_numpy();t=va.time.to_numpy()
        db=(bq[:,None]-bt)**2;da=(aq[:,None]-at)**2
        # Probability mass in one-frame bins around recorded times. This is a
        # scoring convention, not a claim about latent timestamp-error physics.
        kernel=[]
        for hg in HG:
            lo=(a[:,None]*np.maximum(t[:,None]-DELTA/2,0)-gt)/hg
            hi=(a[:,None]*(t[:,None]+DELTA/2)-gt)/hg
            kernel.append(normal_interval(lo,hi)/normal_sf(-gt/hg))
        for hb in HB:
            for ha in HA:
                z=-.5*(db/hb**2+da/ha**2);z-=z.max(axis=1,keepdims=True)
                w=np.exp(z);w/=w.sum(axis=1,keepdims=True)
                for hg,ker in zip(HG,kernel):
                    prob=np.sum(w*ker,axis=1)
                    score=np.log(np.maximum(prob,1e-300))
                    records.append({'fold':k,'h_birth':hb,'h_alpha':ha,'h_growth':hg,'n':len(va),'sum_log_bin_probability':float(score.sum()),'mean_log_bin_probability':float(score.mean()),'n_below_1e_12':int(np.sum(prob<1e-12))})
        print('CV fold',k+1,'finished',flush=True)
    tab=pd.DataFrame(records);tab.to_csv(OUT/'bandwidth_cv_folds.csv',index=False)
    summary=tab.groupby(['h_birth','h_alpha','h_growth'],as_index=False).agg(total_log_score=('sum_log_bin_probability','sum'),n=('n','sum'),se_folds=('mean_log_bin_probability',lambda v:np.std(v,ddof=1)/np.sqrt(len(v))))
    summary['mean_log_bin_probability']=summary.total_log_score/summary.n
    summary.to_csv(OUT/'bandwidth_cv_summary.csv',index=False)
    choices={}
    for name,mask in [('b_alpha',np.isfinite(summary.h_birth)&np.isfinite(summary.h_alpha)),
        ('b_only',np.isfinite(summary.h_birth)&np.isinf(summary.h_alpha)),
        ('alpha_only',np.isinf(summary.h_birth)&np.isfinite(summary.h_alpha)),
        ('neither',np.isinf(summary.h_birth)&np.isinf(summary.h_alpha))]:
        sub=summary[mask];row=sub.loc[sub.mean_log_bin_probability.idxmax()]
        choices[name]={key:float(row[key]) for key in ['h_birth','h_alpha','h_growth','mean_log_bin_probability']}
    return choices


def model(data,choice):
    hb=np.inf if choice['h_birth'] is None else choice['h_birth']
    ha=np.inf if choice['h_alpha'] is None else choice['h_alpha']
    return ConditionalTimeKDE(data.length_birth.to_numpy(),data.alpha.to_numpy(),data.time.to_numpy(),hb,ha,choice['h_growth'])


def hill_prob(lo,hi,data,fit):
    refs=fit['references'];p=fit['hazard'];b=data.length_birth.to_numpy();a=data.alpha.to_numpy()
    scale=(a/refs['alpha'])**p['q']
    def H(t):return scale*cumulative_hazard_growth_time(a*t,b,p['u'],p['omega2'],p['eta'],0,refs['b'],order=48)
    hl=H(np.maximum(lo,0));hu=H(hi)
    return np.exp(-hl)*(-np.expm1(-(hu-hl)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-bandwidths',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    data,_=load_data();past=json.loads((PROJECT_DIR/'outputs/model_comparison/Glu/conditional_effective_jumps/fit_results.json').read_text())
    valid_ids=set(past['validation_ids']);train=data[~data.lineage.isin(valid_ids)].reset_index(drop=True);valid=data[data.lineage.isin(valid_ids)].reset_index(drop=True)
    if args.reuse_bandwidths:result=json.loads((OUT/'fit_results.json').read_text());choices=result['choices']
    else:
        choices=fit_cv(train)
        result={'description':'Conditional Gaussian KDE in growth exponent g=alpha*t, with kernels truncated at g=0; predictor kernels in log birth and log alpha. No lineage effects or history inputs.',
                'choices':choices,'recording_bin_width_hours':DELTA,'n_all':len(data),'n_train':len(train),'n_validation':len(valid),
                'n_train_lineages':int(train.lineage.nunique()),'n_validation_lineages':len(valid_ids),
                'validation_lineages':sorted(valid_ids),'bandwidth_selection':'5 grouped folds of training lineages, highest mean log interval probability; 75 held-out lineages not used to choose bandwidths.',
                'normal_cdf_approximation':'A&S approximation, absolute error below 8e-8; probabilities floored at 1e-300 for log scoring.'}
        (OUT/'fit_results.json').write_text(json.dumps(result,indent=2))
    # JSON null denotes an omitted predictor/infinite predictor bandwidth.
    for choice in choices.values():
        for key in ['h_birth','h_alpha']:
            if choice[key] is not None and not np.isfinite(choice[key]):choice[key]=None
    scores=[];pred=[];pitrows=[];probabilities={}
    a=valid.alpha.to_numpy();b=valid.length_birth.to_numpy();t=valid.time.to_numpy();lo=np.maximum(t-DELTA/2,0);hi=t+DELTA/2
    for name,choice in choices.items():
        kde=model(train,choice);prob=kde.probability(lo,hi,b,a);probabilities[name]=prob
        cdflo=kde.cdf(lo,b,a);cdfhi=cdflo+prob
        pit=cdflo+np.random.default_rng(61316).random(len(t))*prob
        scores.append({'model':name,'mean_log_bin_probability':float(np.log(np.maximum(prob,1e-300)).mean()),'mean_log_bin_average_density':float(np.log(np.maximum(prob,1e-300)/DELTA).mean()),'n_below_1e_12':int(np.sum(prob<1e-12))})
        ess=kde.effective_sample_size(b,a)
        if name=='b_alpha':
            consecutive=(valid.lineage.to_numpy()[1:]==valid.lineage.to_numpy()[:-1])&(np.diff(valid.generation.to_numpy())==1)
            result['validation_diagnostics']={'mean_randomized_pit':float(pit.mean()),'variance_randomized_pit':float(pit.var()),
                'adjacent_randomized_pit_correlation':float(np.corrcoef(pit[1:][consecutive],pit[:-1][consecutive])[0,1]),
                'kernel_effective_counts_quantiles_0_5_50_95_100':np.quantile(ess,[0,.05,.5,.95,1]).tolist(),
                'n_kernel_effective_count_below_30':int(np.sum(ess<30))}
        for i in range(len(t)):
            pred.append({'model':name,'lineage':int(valid.lineage.iloc[i]),'generation':int(valid.generation.iloc[i]),'birth':b[i],'alpha':a[i],'time':t[i],'bin_probability':prob[i],'cdf_lower':cdflo[i],'cdf_upper':cdfhi[i],'randomized_pit':pit[i],'effective_sample_size':ess[i]})
        hist,_=np.histogram(pit,bins=np.linspace(0,1,11))
        pitrows.extend({'model':name,'left':j/10,'right':(j+1)/10,'fraction':v/len(t)} for j,v in enumerate(hist))
        print('held out',name,scores[-1],flush=True)
    prob=hill_prob(lo,hi,valid,past['fits']['train']);probabilities['Hill4']=prob
    scores.append({'model':'Hill4','mean_log_bin_probability':float(np.log(np.maximum(prob,1e-300)).mean()),'mean_log_bin_average_density':float(np.log(np.maximum(prob,1e-300)/DELTA).mean()),'n_below_1e_12':int(np.sum(prob<1e-12))})
    # Paired original-lineage bootstrap of held-out score differences.
    lineage=valid.lineage.to_numpy();ids=np.unique(lineage);rng=np.random.default_rng(19907)
    intervals=[]
    for other in ['Hill4','b_only','alpha_only','neither']:
        delta=np.log(np.maximum(probabilities['b_alpha'],1e-300))-np.log(np.maximum(probabilities[other],1e-300))
        sums=np.array([delta[lineage==i].sum() for i in ids]);counts=np.array([np.sum(lineage==i) for i in ids])
        idx=rng.integers(0,len(ids),(1500,len(ids)));boot=sums[idx].sum(1)/counts[idx].sum(1)
        intervals.append({'comparison':'b_alpha minus '+other,'mean_gain_nats_per_cycle':float(delta.mean()),'lo':float(np.quantile(boot,.025)),'hi':float(np.quantile(boot,.975))})
    pd.DataFrame(scores).to_csv(OUT/'heldout_scores.csv',index=False)
    pd.DataFrame(intervals).to_csv(OUT/'heldout_score_differences.csv',index=False)
    pd.DataFrame(pred).to_csv(OUT/'heldout_predictions.csv',index=False)
    pd.DataFrame(pitrows).to_csv(OUT/'heldout_pit_histogram.csv',index=False)
    # Descriptive full-data refit uses the already-selected training bandwidths.
    kde=model(data,choices['b_alpha']);bs=np.quantile(data.length_birth,[.15,.5,.85]);alphas=np.quantile(data.alpha,[.15,.5,.85])
    grid=np.linspace(0,8,1001);rows=[];contexts=[]
    for ai,alpha in enumerate(alphas):
        for bi,birth in enumerate(bs):
            density=kde.density(grid,birth,alpha);ess=float(kde.effective_sample_size(birth,alpha)[0]);mean=float(kde.conditional_mean(birth,alpha)[0])
            # Matched all-data Hill curve, provided only as a reference.
            fake=pd.DataFrame({'length_birth':np.full(len(grid),birth),'alpha':np.full(len(grid),alpha)})
            half=.0001;hpdf=hill_prob(np.maximum(grid-half,0),grid+half,fake,past['fits']['all'])/(half+np.minimum(grid,half))
            for ti,den,hd in zip(grid,density,hpdf):rows.append({'birth_index':bi,'alpha_index':ai,'birth':birth,'alpha':alpha,'time':ti,'density':den,'Hill4_density':hd})
            contexts.append({'birth_index':bi,'alpha_index':ai,'birth':birth,'alpha':alpha,'effective_sample_size':ess,'mean_time':mean,'probability_above_4h':float(1-kde.cdf(4,birth,alpha))})
    pd.DataFrame(rows).to_csv(OUT/'conditional_density_slices.csv',index=False)
    pd.DataFrame(contexts).to_csv(OUT/'conditional_summary.csv',index=False)
    np.savez_compressed(OUT/'empirical_density.npz',birth=data.length_birth.to_numpy(),alpha=data.alpha.to_numpy(),time=data.time.to_numpy(),**{k:v for k,v in choices['b_alpha'].items() if k.startswith('h_')})
    result['heldout_scores']=scores;result['heldout_score_differences']=intervals
    result['density_slice_births']=bs.tolist();result['density_slice_alphas']=alphas.tolist()
    (OUT/'fit_results.json').write_text(json.dumps(result,indent=2))
    print('selected bandwidths',choices,flush=True);print('score differences',intervals,flush=True)


if __name__=='__main__':main()
