#!/usr/bin/env python3
"""Nested five/six-parameter licensing threshold with slow and fast noise."""
import json
import argparse
import numpy as np
import pandas as pd
from screen_parsimonious_joint_hazards import (FastReplay,segments,load_data,OUT,METRICS,bootstrap_errors)


class TwoNoiseReplay(FastReplay):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        seed=kwargs.get('seed',args[4] if len(args)>4 else 20481)
        self.Zslow=np.random.default_rng(782112+seed-5091).normal(size=self.Z.shape)

    def simulate_aft(self,params):
        mu,cb,ca,sigma,rho,slow=params
        births=np.zeros(self.E.shape);times=np.zeros_like(births)
        logb=np.broadcast_to(np.log(self.birth0),(self.R,len(self.birth0))).copy()
        state=self.Zslow[:,:,0]*slow/np.sqrt(1-rho*rho)
        for n in range(self.mask.shape[1]):
            active=self.mask[:,n]
            if n:state[:,active]=rho*state[:,active]+slow*self.Zslow[:,active,n]
            y=mu+cb*(logb[:,active]-np.log(self.bstar))+ca*self.loga[active,n]+state[:,active]+sigma*self.Z[:,active,n]
            if np.any(y>15) or np.any(abs(logb[:,active])>20):raise FloatingPointError
            g=np.logaddexp(0.,y-logb[:,active])
            births[:,active,n]=np.exp(logb[:,active]);times[:,active,n]=g/self.alpha[active,n]
            if n+1<self.mask.shape[1]:logb[:,active]+=g+np.log(self.f[active,n+1])
        return births,times


def optimize(rp,target,se,start,adder=False,generations=55):
    # Six parameters: log u, birth feedback, alpha scaling, fast SD, persistence,
    # slow-innovation SD. The five-parameter adder fixes birth feedback to zero.
    free=np.array([0,2,3,4,5] if adder else list(range(6)))
    bounds=np.array([[.1,1.5],[-2,1],[-.7,.4],[.001,.5],[0,.995],[.005,.35]])[free]
    lo,hi=bounds.T;dim=len(free)
    def physical(x):
        p=np.zeros(6);p[free]=x;return p
    def obj(x):
        if np.any(x<lo) or np.any(x>hi):return 1e8
        try:
            b,t=rp.simulate_aft(physical(x));s=rp.statistics(b,t).mean(0)
            z=(s-target)/se
            return np.mean(z*z) if np.all(np.isfinite(z)) else 1e8
        except (FloatingPointError,ValueError):return 1e8
    rng=np.random.default_rng(773+dim);pop=rng.uniform(lo,hi,(36,dim))
    pop[0]=np.clip(start[free],lo,hi)
    for i,f in enumerate([.03,.07,.12,.18,.24],1):
        p=start.copy();p[3]=f;pop[i]=np.clip(p[free],lo,hi)
    score=np.array([obj(p) for p in pop]);history=[]
    for it in range(generations):
        for i in range(len(pop)):
            a,b,c=rng.choice(np.delete(np.arange(len(pop)),i),3,replace=False)
            trial=np.clip(pop[a]+.7*(pop[b]-pop[c]),lo,hi);cross=rng.random(dim)<.85;cross[rng.integers(dim)]=True
            trial=np.where(cross,trial,pop[i]);s=obj(trial)
            if s<score[i]:pop[i]=trial;score[i]=s
        j=np.argmin(score);history.append({'generation':it+1,'score':float(score[j])})
        if it%10==9:print('two-noise',dim,it+1,round(score[j],4),physical(pop[j]).round(5),flush=True)
    x=pop[np.argmin(score)].copy();best=min(score);step=np.full(dim,.015)
    for _ in range(80):
        improved=False
        for k in range(dim):
            for direction in [-1,1]:
                z=x.copy();z[k]+=direction*step[k];v=obj(z)
                if v<best:x,best,improved=z,v,True
        if not improved:step*=.5
        if step.max()<.0002:break
    return physical(x),best,history


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true');parser.add_argument('--generations',type=int,default=55);args=parser.parse_args()
    old=json.loads((OUT/'fit_results.json').read_text());_,original=load_data();groups=segments(original)
    bstar=old['b_star'];astar=old['alpha_star'];vi=set(old['validation_lineages'])
    train=[g for g in groups if g.lineage.iloc[0] not in vi];valid=[g for g in groups if g.lineage.iloc[0] in vi]
    rp=TwoNoiseReplay(train,bstar,astar,12,5091);target=rp.observed()[0];se=np.array(old['bootstrap_se_train'])
    if args.reuse:models=json.loads((OUT/'two_noise_fits.json').read_text())
    else:
        p=np.array(old['models']['AFT5_AR_joint']['params']);start=np.r_[p[:3],.01,p[4],p[3]];models={}
        for adder,name in [(False,'Threshold6_two_noise'),(True,'Threshold5_adder_two_noise')]:
            params,score,history=optimize(rp,target,se,start,adder,args.generations)
            models[name]={'params':params.tolist(),'n_params':6-adder,'train_score':score,'optimization_history':history}
        (OUT/'two_noise_fits.json').write_text(json.dumps(models,indent=2))
    rows=[];curves=[];densities=[]
    for split,gs in [('train',train),('validation',valid),('all',groups)]:
        rp=TwoNoiseReplay(gs,bstar,astar,200,638111);obs,ob,ot=rp.observed();obsse=bootstrap_errors(gs,bstar,astar,200)
        for name,entry in models.items():
            b,t=rp.simulate_aft(entry['params']);st=rp.statistics(b,t)
            rows.extend({'split':split,'model':name,'metric':k,'mean':v,'lo':a,'hi':z} for k,v,a,z in zip(METRICS,st.mean(0),*np.quantile(st,[.025,.975],axis=0)))
            entry[split+'_joint_score']=float(np.mean(((st.mean(0)-obs)/obsse)**2))
            ba=rp.acfs(b);ta=rp.acfs(t)
            for k in range(31):curves.append({'split':split,'model':name,'lag':k,'birth_mean':ba[:,k].mean(),'birth_lo':np.quantile(ba[:,k],.025),'birth_hi':np.quantile(ba[:,k],.975),'time_mean':ta[:,k].mean(),'time_lo':np.quantile(ta[:,k],.025),'time_hi':np.quantile(ta[:,k],.975)})
            if split=='all':
                for var,v,o,edges in [('birth',b[:,rp.mask],ob[:,rp.mask],np.linspace(0,6,61)),('time',t[:,rp.mask],ot[:,rp.mask],np.arange(0,12+1/6,1/6))]:
                    h=np.histogram(v,bins=edges)[0]/v.size/np.diff(edges);ho=np.histogram(o,bins=edges)[0]/o.size/np.diff(edges)
                    for a,z,d,d0 in zip(edges[:-1],edges[1:],h,ho):densities.append({'model':name,'variable':var,'left':a,'right':z,'density':d,'observed_density':d0})
            print(split,name,'score',entry[split+'_joint_score'],'stats',dict(zip(METRICS,st.mean(0).round(5))),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'two_noise_joint_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'two_noise_acf.csv',index=False)
    pd.DataFrame(densities).to_csv(OUT/'two_noise_marginals.csv',index=False)
    (OUT/'two_noise_fits.json').write_text(json.dumps(models,indent=2))


if __name__=='__main__':main()
