#!/usr/bin/env python3
"""Paired cluster-bootstrap relative improvements and a compact uncertainty plot."""
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from check_markov_memory import OUT


def main():
    rows=[]
    pairs=[('state_order1','state_order2'),('state_order1','state_order3'),('timing_no_history','timing_one_previous'),('state_order1_age','state_order2_age'),('timing_no_history_age','timing_one_previous_age')]
    for run,path in [('primary',OUT),('384_landmarks',OUT/'sensitivity_384')]:
        tab=pd.read_csv(path/'heldout_predictions.csv')
        for base,extended in pairs:
            if extended not in tab.model.unique():continue
            for family in ['quadratic','nonlinear']:
                a=tab[(tab.model==base)&(tab.family==family)]
                b=tab[(tab.model==extended)&(tab.family==family)]
                merged=a.merge(b,on=['lineage','generation','outcome'],suffixes=('_base','_extended'),validate='one_to_one')
                np.testing.assert_allclose(merged.observed_base,merged.observed_extended)
                for outcome,sub in merged.groupby('outcome'):
                    ea=(sub.observed_base-sub.predicted_base)**2;eb=(sub.observed_base-sub.predicted_extended)**2
                    loss=pd.DataFrame({'lineage':sub.lineage,'base':ea,'extended':eb}).groupby('lineage')[['base','extended']].sum().to_numpy()
                    rng=np.random.default_rng(70155);ix=rng.integers(0,len(loss),(5000,len(loss)))
                    samples=loss[ix].sum(1);delta=100*(1-samples[:,1]/samples[:,0])
                    lo,hi=np.quantile(delta,[.025,.975]);flo,fhi=np.quantile(delta,[.05/6,1-.05/6])
                    rows.append({'run':run,'baseline':base,'extended':extended,'family':family,'outcome':outcome,'MSE_reduction_percent':100*(1-eb.mean()/ea.mean()),'lo95':lo,'hi95':hi,'lo98_333':flo,'hi98_333':fhi})
    result=pd.DataFrame(rows);result.to_csv(OUT/'relative_improvements.csv',index=False)
    print(result[result.family=='nonlinear'].to_string(index=False))
    im=Image.new('RGB',(1550,1050),'white');d=ImageDraw.Draw(im)
    def text(x,y,s,size=23,bold=False,color='#243249'):
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
        d.text((x,y),s,font=font,fill=color)
    text(65,30,'Glu: does history improve held-out prediction?',37,True)
    text(65,86,'1,278 consecutive-cycle windows from 69 held-out lineages; nonlinear predictors',25)
    text(65,132,'Percentage reduction in mean squared error of log outcomes; bars are 95% lineage-bootstrap intervals',22)
    left,right,top,bottom=720,1445,235,840
    def xp(v):return left+(v+8)/24*(right-left)
    for x in [-8,-4,0,4,8,12,16]:
        d.line((xp(x),top,xp(x),bottom),fill='#243249' if x==0 else '#E0E6EB',width=2 if x==0 else 1)
        text(xp(x)-16,bottom+15,str(x),21)
    specs=[('state_order1','state_order2','log_time','Next division time',275),('state_order1','state_order2','log_alpha','Next growth rate',375),('state_order1','state_order2','log_birth','Next birth size',475),('timing_no_history','timing_one_previous','log_time','Timing given current birth and growth rate',690)]
    text(65,209,'Add an older cycle beyond the full current state:',24,True)
    text(65,603,'Separate check of the simpler timing law:',24,True)
    for base,extended,outcome,label,y in specs:
        sub=result[(result.run=='primary')&(result.family=='nonlinear')&(result.baseline==base)&(result.extended==extended)&(result.outcome==outcome)].iloc[0]
        color='#148578' if sub.lo95>0 else '#687B90'
        text(65,y-17,label,25)
        d.line((xp(sub.lo95),y,xp(sub.hi95),y),fill=color,width=5)
        for value in [sub.lo95,sub.hi95]:d.line((xp(value),y-8,xp(value),y+8),fill=color,width=3)
        x=xp(sub.MSE_reduction_percent);d.ellipse((x-7,y-7,x+7,y+7),fill=color)
        text(750,y+25,f'{sub.MSE_reduction_percent:.1f}%  [{sub.lo95:.1f}, {sub.hi95:.1f}]',21,color=color)
    text(790,893,'Improvement (%) — positive is better',23)
    text(65,945,'Predictive evidence, not a distribution-free test of Markovianity. Measurement noise and lineage',23)
    text(65,980,'heterogeneity can also create apparent memory. Wider-basis and generation-drift checks are saved.',23)
    im.save(OUT/'history_prediction_gain.png')


if __name__=='__main__':main()
