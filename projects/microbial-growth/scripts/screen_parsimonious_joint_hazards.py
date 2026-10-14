#!/usr/bin/env python3
"""Screen low-dimensional event-time laws against joint Glu replay statistics.

This is conditional replay, not a generative model for the observed alpha and
effective-factor histories. Missing-generation gaps are split, never bridged.
Entire original lineages, not individual cycles, form the validation split.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from optimize_birth_size_hazard_acf import Replay, PROJECT_DIR, load_data, add_effective_factors

OUT = PROJECT_DIR / 'outputs/model_comparison/Glu/parsimonious_joint_hazard_screen'
METRICS = ['mean_b', 'sd_b', 'mean_t', 'sd_t', 'q95_t', 'q99_t', 'corr_b_t',
           'corr_logb_g', 'corr_loga_logt'] + [f'b_acf{k}' for k in range(1, 7)] + [f't_acf{k}' for k in range(1, 7)]


def segments(groups):
    result = []
    for g in groups:
        breaks = np.r_[0, 1 + np.flatnonzero(np.diff(g.generation) != 1), len(g)]
        result.extend(g.iloc[a:b].reset_index(drop=True) for a, b in zip(breaks[:-1], breaks[1:]))
    return add_effective_factors(result)[0]


def corr(x, y):
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    return np.sum(x*y, axis=-1) / np.sqrt(np.sum(x*x, axis=-1)*np.sum(y*y, axis=-1))


class FastReplay(Replay):
    def __init__(self, groups, bstar, astar, replicates=12, seed=20481):
        super().__init__(groups, bstar, replicates=replicates, seed=seed)
        self.astar = astar
        self.Z = np.random.default_rng(seed).normal(size=self.E.shape)
        self.loga = np.log(self.alpha / astar)
        self.groups = groups
        # Preserve original-lineage centering/weighting. Splitting gaps must not
        # turn a single lineage into several equally weighted short lineages.
        ids=np.concatenate([np.full(len(g),g.lineage.iloc[0]) for g in groups])
        _,self.event_lineage,self.counts=np.unique(ids,return_inverse=True,return_counts=True)
        self.starts=np.r_[0,np.cumsum(self.counts)[:-1]]
        self.pairs={}
        for k in range(1,31):
            left=[];right=[];offset=0
            for g in groups:
                ix=np.arange(max(len(g)-k,0))+offset
                left.extend(ix);right.extend(ix+k);offset+=len(g)
            self.pairs[k]=(np.array(left,dtype=int),np.array(right,dtype=int))

    def acfs(self, values, lag=30):
        flat=values[:,self.mask]
        means=np.add.reduceat(flat,self.starts,axis=1)/self.counts
        centered=flat-means[:,self.event_lineage]
        den=np.add.reduceat(centered*centered,self.starts,axis=1)
        result=np.ones((len(values),lag+1))
        for k in range(1,lag+1):
            left,right=self.pairs[k]
            products=np.zeros_like(flat)
            products[:,left]=centered[:,left]*centered[:,right]
            numer=np.add.reduceat(products,self.starts,axis=1)
            available=np.bincount(self.event_lineage[left],minlength=len(self.counts))>0
            valid=available & (self.counts>=3) & (den[0]>0)
            result[:,k]=np.mean(numer[:,valid]/den[:,valid],axis=1)
        return result

    def simulate_aft(self, params):
        mu, cb, ca, sigma, rho = params
        births = np.zeros(self.E.shape)
        times = np.zeros_like(births)
        logb = np.broadcast_to(np.log(self.birth0), (self.R, len(self.birth0))).copy()
        residual = self.Z[:, :, 0] * sigma / np.sqrt(1-rho*rho)
        for n in range(self.mask.shape[1]):
            active = self.mask[:, n]
            if n:
                residual[:, active] = rho * residual[:, active] + sigma * self.Z[:, active, n]
            y = mu + cb * (logb[:, active]-np.log(self.bstar)) + ca*self.loga[active,n] + residual[:,active]
            if np.any(y>15) or np.any(logb[:,active]>20) or np.any(logb[:,active]<-20):
                raise FloatingPointError('unstable replay')
            # Lognormal threshold in added size, NOT in alpha*t. The latter
            # would give an infinite mean daughter size E[exp(lognormal)].
            g = np.logaddexp(0.,y-logb[:,active])
            births[:,active,n] = np.exp(logb[:,active])
            times[:,active,n] = g / self.alpha[active,n]
            if n+1<self.mask.shape[1]:
                logb[:,active] += g + np.log(self.f[active,n+1])
        return births, times

    def statistics(self, births, times):
        b, t = births[:,self.mask], times[:,self.mask]
        a = self.alpha[self.mask][None,:]
        metrics = np.column_stack([b.mean(1),b.std(1),t.mean(1),t.std(1),
            np.quantile(t,.95,axis=1),np.quantile(t,.99,axis=1),corr(b,t),
            corr(np.log(b),a*t),corr(np.log(a),np.log(t)),
            self.acfs(births,6)[:,1:],self.acfs(times,6)[:,1:]])
        return metrics

    def observed(self):
        b=np.zeros((1,*self.mask.shape)); t=np.zeros_like(b)
        for i,g in enumerate(self.groups):
            b[0,i,:len(g)] = g.length_birth
            t[0,i,:len(g)] = g.time
        old = self.R
        self.R = 1
        stats = self.statistics(b,t)[0]
        self.R = old
        return stats,b,t


def fit_aft(groups, bstar, astar, free_alpha=True, free_rho=False):
    def profile(rho):
        xx=[]; yy=[]; jac=0.
        for gr in groups:
            growth=gr.alpha.to_numpy()*gr.time.to_numpy()
            y=np.log(gr.length_birth.to_numpy()*np.expm1(growth))
            x=np.column_stack([np.ones(len(gr)),np.log(gr.length_birth.to_numpy()/bstar),np.log(gr.alpha.to_numpy()/astar)])
            if not free_alpha:
                x=x[:,:2]
            xx.extend([x[:1]*np.sqrt(1-rho*rho),x[1:]-rho*x[:-1]])
            yy.extend([y[:1]*np.sqrt(1-rho*rho),y[1:]-rho*y[:-1]])
            # dp/dt = alpha*b*exp(alpha*t); lognormal p density supplies -log p.
            jac += np.sum(y-np.log(gr.alpha.to_numpy())-np.log(gr.length_birth.to_numpy())-growth)
        X=np.concatenate(xx); Y=np.concatenate(yy)
        beta=np.linalg.lstsq(X,Y,rcond=None)[0]
        sigma=np.sqrt(np.mean((Y-X@beta)**2))
        ll=-len(Y)*(np.log(sigma)+.5*np.log(2*np.pi)+.5)+.5*len(groups)*np.log(1-rho*rho)-jac
        if not free_alpha:beta=np.r_[beta,0.]
        return np.r_[beta,sigma,rho],float(ll)
    if not free_rho:return profile(0.)
    rs=np.linspace(-.8,.8,81)
    fits=[profile(r) for r in rs]
    best=int(np.argmax([f[1] for f in fits])); rho=rs[best]
    for step in [.005,.001,.0002]:
        rs=np.clip(rho+np.arange(-5,6)*step,-.95,.95)
        fits=[profile(r) for r in rs]
        best=int(np.argmax([f[1] for f in fits]));rho=rs[best]
    return fits[best]


def bootstrap_errors(groups,bstar,astar,n=300):
    ids=np.unique([g.lineage.iloc[0] for g in groups])
    byid={i:[g for g in groups if g.lineage.iloc[0]==i] for i in ids}
    rng=np.random.default_rng(45191)
    values=[]
    for _ in range(n):
        gs=[]
        for j,i in enumerate(rng.choice(ids,len(ids))):
            for g in byid[i]:
                copy=g.copy();copy['lineage']=j;gs.append(copy)
        values.append(FastReplay(gs,bstar,astar,1).observed()[0])
    return np.std(values,axis=0,ddof=1)


def joint_optimize(replay, target, se, starts, free_rho, generations=65):
    dim=5 if free_rho else 4
    # sigma is optimized in log coordinates.
    bounds=np.array([[.1,1.5],[-2.,1.0],[-.7,.4],[np.log(.025),np.log(.65)],[-.8,.95]])[:dim]
    lo,hi=bounds.T
    def physical(x):
        out=np.r_[x,0.] if dim==4 else x.copy()
        out[3]=np.exp(out[3])
        return out
    def obj(x):
        if np.any(x<lo) or np.any(x>hi):return 1e8
        try:
            b,t=replay.simulate_aft(physical(x))
            stats=replay.statistics(b,t).mean(0)
            z=(stats-target)/se
            return np.mean(z*z) if np.all(np.isfinite(z)) else 1e8
        except (FloatingPointError,ValueError):return 1e8
    rng=np.random.default_rng(9161+dim)
    pop=rng.uniform(lo,hi,(32,dim))
    for i,start in enumerate(starts):
        x=start[:dim].copy();x[3]=np.log(x[3]);pop[i]=np.clip(x,lo,hi)
    scores=np.array([obj(p) for p in pop]); history=[]
    for generation in range(generations):
        for i in range(len(pop)):
            a,b,c=rng.choice(np.delete(np.arange(len(pop)),i),3,replace=False)
            trial=np.clip(pop[a]+.7*(pop[b]-pop[c]),lo,hi)
            cross=rng.random(dim)<.8;cross[rng.integers(dim)]=True
            trial=np.where(cross,trial,pop[i]);score=obj(trial)
            if score<scores[i]:pop[i]=trial;scores[i]=score
        j=np.argmin(scores);history.append({'generation':generation,'score':float(scores[j])})
        if generation%10==9:print('joint',dim,generation+1,round(scores[j],4),physical(pop[j]).round(5),flush=True)
    j=np.argmin(scores)
    x=pop[j].copy();score=scores[j];step=np.full(dim,.015)
    for _ in range(70):
        improved=False
        for k in range(dim):
            for d in [-1,1]:
                trial=x.copy();trial[k]+=d*step[k];v=obj(trial)
                if v<score:x,score,improved=trial,v,True
        if not improved:step*=.5
        if step.max()<1e-4:break
    return physical(x),float(score),history


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true');parser.add_argument('--generations',type=int,default=65)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    real,original=load_data();groups=segments(original)
    ids=np.array([g.lineage.iloc[0] for g in original]);rng=np.random.default_rng(40713);rng.shuffle(ids)
    valid_ids=set(ids[:len(ids)//4]);train=[g for g in groups if g.lineage.iloc[0] not in valid_ids];valid=[g for g in groups if g.lineage.iloc[0] in valid_ids]
    trainreal=pd.concat(train);bstar=float(np.exp(np.log(trainreal.length_birth).mean()));astar=float(np.exp(np.log(trainreal.alpha).mean()))
    tr=FastReplay(train,bstar,astar,12,5091);target=tr.observed()[0]
    if args.reuse:
        results=json.loads((OUT/'fit_results.json').read_text());se=np.array(results['bootstrap_se_train']);models=results['models']
    else:
        se=bootstrap_errors(train,bstar,astar)
        models={}
        for name,fa,fr in [('AFT3_MLE',False,False),('AFT4_MLE',True,False),('AFT5_AR_MLE',True,True)]:
            p,ll=fit_aft(train,bstar,astar,fa,fr);models[name]={'params':p.tolist(),'train_log_likelihood':ll,'n_params':3+fa+fr}
            print(name,models[name],flush=True)
        for free_rho,name in [(False,'AFT4_joint'),(True,'AFT5_AR_joint')]:
            starts=[np.array(v['params']) for v in models.values()]
            if free_rho:
                starts.extend([np.array([.6,-1.,-.12,.2,.6]),np.array([.6,-1.5,-.12,.16,.85])])
            p,score,history=joint_optimize(tr,target,se,starts,free_rho,args.generations)
            models[name]={'params':p.tolist(),'train_score':score,'n_params':4+free_rho,'optimization_history':history}
        results={'models':models,'b_star':bstar,'alpha_star':astar,'bootstrap_se_train':se.tolist(),'metrics':METRICS,
                 'train_lineages':sorted(set(ids)-valid_ids),'validation_lineages':sorted(valid_ids),
                 'n_cycles':len(real),'n_lineages':len(original),'n_segments':len(groups),
                 'model':'lognormal licensing threshold: log p = mu + cb*log(b/b_star) + ca*log(alpha/alpha_star) + e; e = rho*previous e + sigma*Z',
                 'design':'75% train lineages; gaps split for dynamics, original-lineage ACF centering and weighting with pairs crossing gaps omitted; fixed alpha and f_eff; initial segment birth conditioned on data; joint objective mean squared error standardized by lineage-bootstrap SE; 12 common-random training replicates; 200 fresh validation replicates'}
        (OUT/'fit_results.json').write_text(json.dumps(results,indent=2))
    rows=[];curves=[];dist=[]
    for split,gs in [('train',train),('validation',valid),('all',groups)]:
        rp=FastReplay(gs,bstar,astar,200,103822)
        obs,ob,ot=rp.observed();obs_se=bootstrap_errors(gs,bstar,astar,200)
        rows.extend({'split':split,'model':'Observed','metric':k,'mean':v,'lo':v-1.96*e,'hi':v+1.96*e} for k,v,e in zip(METRICS,obs,obs_se))
        old=rp.R;rp.R=1
        bac=rp.acfs(ob);tac=rp.acfs(ot);rp.R=old
        for k in range(31):curves.append({'split':split,'model':'Observed','lag':k,'birth_mean':bac[0,k],'time_mean':tac[0,k]})
        for name,entry in models.items():
            b,t=rp.simulate_aft(entry['params']);metrics=rp.statistics(b,t)
            rows.extend({'split':split,'model':name,'metric':k,'mean':v,'lo':lo,'hi':hi} for k,v,lo,hi in zip(METRICS,metrics.mean(0),*np.quantile(metrics,[.025,.975],axis=0)))
            entry[split+'_joint_score']=float(np.mean(((metrics.mean(0)-obs)/obs_se)**2))
            ba=rp.acfs(b);ta=rp.acfs(t)
            for k in range(31):curves.append({'split':split,'model':name,'lag':k,'birth_mean':ba[:,k].mean(),'birth_lo':np.quantile(ba[:,k],.025),'birth_hi':np.quantile(ba[:,k],.975),'time_mean':ta[:,k].mean(),'time_lo':np.quantile(ta[:,k],.025),'time_hi':np.quantile(ta[:,k],.975)})
            if split=='all':
                for var,values,observed,edges in [('birth',b[:,rp.mask],ob[:,rp.mask],np.linspace(0,6,61)),('time',t[:,rp.mask],ot[:,rp.mask],np.arange(0,12+1/6,1/6))]:
                    hist=np.histogram(values,bins=edges)[0]/values.size/np.diff(edges)
                    histobs=np.histogram(observed,bins=edges)[0]/observed.size/np.diff(edges)
                    for a,z,d,o in zip(edges[:-1],edges[1:],hist,histobs):dist.append({'model':name,'variable':var,'left':a,'right':z,'density':d,'observed_density':o})
            print(split,name,'score',entry[split+'_joint_score'],'stats',dict(zip(METRICS[:12],metrics.mean(0)[:12].round(5))),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'joint_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'acf_comparison.csv',index=False)
    pd.DataFrame(dist).to_csv(OUT/'marginal_densities.csv',index=False)
    (OUT/'fit_results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
