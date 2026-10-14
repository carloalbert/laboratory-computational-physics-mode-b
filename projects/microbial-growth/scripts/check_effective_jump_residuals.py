#!/usr/bin/env python3
"""Tail-shape robustness diagnostic, not a new low-parameter fitted model."""
import json
import numpy as np
import pandas as pd
from fit_conditional_effective_jumps import *


def main():
    result=json.loads((OUT/'fit_results.json').read_text());_,original=load_data()
    gs=segments(original);ids=set(result['validation_ids'])
    train=[g for g in gs if g.lineage.iloc[0] not in ids]
    valid=[g for g in gs if g.lineage.iloc[0] in ids]
    rows=[];moments=[]
    for scope,groups,fitscope,training in [('all',gs,'all',gs),('validation',valid,'train',train)]:
        fitted=result['fits'][fitscope];refs=fitted['references'];tr=transitions(training)
        rp=FastReplay(groups,refs['b'],refs['alpha'],200,58101)
        for name in ['age','birth_growth_alpha']:
            params=dict(fitted['jumps'][name]);res=np.log(tr.f_eff.to_numpy())-matrix(tr,params['features'],refs)@params['coefficients']
            params['empirical_residuals']=res-res.mean()
            b,t,f=simulate(rp,fitted['hazard'],params,refs,seed=581101)
            st=rp.statistics(b,t)
            rows.extend({'scope':scope,'model':name+'_empirical_residuals','metric':key,'mean':v,'lo':lo,'hi':hi} for key,v,lo,hi in zip(METRICS,st.mean(0),*np.quantile(st,[.025,.975],axis=0)))
            ids0=tr.lineage.to_numpy();gen=tr.generation.to_numpy();pairs=(ids0[1:]==ids0[:-1])&(np.diff(gen)==1)
            moments.append({'scope':scope,'model':name,'residual_skewness':float(np.mean((res/res.std())**3)),
                'residual_excess_kurtosis':float(np.mean((res/res.std())**4)-3),
                'residual_lag1':float(np.corrcoef(res[1:][pairs],res[:-1][pairs])[0,1])})
            print(scope,name,dict(zip(METRICS,st.mean(0).round(5))),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'empirical_residual_robustness.csv',index=False)
    pd.DataFrame(moments).to_csv(OUT/'jump_residual_diagnostics.csv',index=False)


if __name__=='__main__':main()
