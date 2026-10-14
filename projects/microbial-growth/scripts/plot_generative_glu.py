#!/usr/bin/env python3
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fit_generative_glu import OUT
from plot_joint_growth_timing import panel

COLORS={'Observed':'#243249','Recorded_F':'#7291AD','Generative':'#C85633'}
LABELS={'Observed':'Observed','Recorded_F':'AR growth; recorded factors','Generative':'AR growth; generated iid factors'}


def canvas(scope,height,title):
    im=Image.new('RGB',(1920,height),'white');d=ImageDraw.Draw(im)
    def txt(x,y,s,size=23,bold=False,color='#243249'):
        f=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
        d.text((x,y),s,font=f,fill=color)
    txt(60,24,title,36,True)
    scope_label='All-data refit (descriptive)' if scope=='all' else 'Held-out lineages; training-only parameter estimates'
    txt(60,76,scope_label+' | 200 simulations; only initial birth and alpha are observed',23)
    for i,(name,color) in enumerate(COLORS.items()):
        x=65+610*i;d.line((x,140,x+55,140),fill=color,width=5);txt(x+70,123,LABELS[name])
    return im,d,txt


def curves(draw,tab,xcol='lag',bands=True):
    for name,color in COLORS.items():
        sub=tab[tab.model==name].sort_values(xcol)
        lo=sub.lo.to_numpy() if bands and name=='Generative' else None
        hi=sub.hi.to_numpy() if bands and name=='Generative' else None
        draw(sub[xcol].to_numpy(),sub['mean'].to_numpy(),color,lo,hi)


def main():
    ac=pd.read_csv(OUT/'acf_comparison.csv');hist=pd.read_csv(OUT/'marginal_densities.csv');cc=pd.read_csv(OUT/'cross_correlations.csv')
    for scope in ['all','validation']:
        im,d,txt=canvas(scope,1390,'Glu: generated growth rates, division times, and effective factors')
        for i,(var,title,xlim,xlabel,ticks) in enumerate([('birth','Birth size',(1,5),'Birth size',[1,2,3,4,5]),('alpha','Growth rate',(0,1.25),'Growth rate (1/h)',[0,.25,.5,.75,1,1.25]),('time','Division time',(0,5),'Time (h)',[0,1,2,3,4,5])]):
            left=100+630*i;right=left+480
            draw=panel(im,d,txt,(left,235,right,650),title+' ACF',(1,10),(-.15,.55),[1,3,5,7,9],np.arange(-.1,.51,.1),'Lag (generations)')
            curves(draw,ac[(ac.scope==scope)&(ac.variable==var)&ac.lag.between(1,10)])
            sub=hist[(hist.scope==scope)&(hist.variable==var)];ymax=float(np.ceil(sub.density.max()*5.5)/5)
            draw=panel(im,d,txt,(left,850,right,1210),title+' density',xlim,(0,ymax),ticks,np.linspace(0,ymax,5),xlabel)
            for name,color in COLORS.items():
                s=sub[sub.model==name];x=(s.left.to_numpy()+s.right.to_numpy())/2;keep=(x>=xlim[0])&(x<=xlim[1])
                draw(x[keep],s.density.to_numpy()[keep],color,dots=False)
        txt(65,1315,'Generated-factor model draws fresh independent effective factors: birth trajectories are no longer tied to recorded jumps.',22)
        txt(65,1350,'Gaps reset initial conditions. ACFs retain original-lineage weighting; bands show simulation variation, not parameter uncertainty.',21)
        im.save(OUT/f'{scope}_marginals_acfs.png')
        im,d,txt=canvas(scope,900,'Glu: cross-correlations with independently generated factors')
        for i,(v1,v2,title,ylim,yticks) in enumerate([('birth','time','Birth size versus time',(-.5,.25),[-.4,-.2,0,.2]),('alpha','time','Growth rate versus time',(-.85,.1),[-.8,-.6,-.4,-.2,0]),('birth','alpha','Birth size versus growth rate',(-.05,.30),[0,.1,.2,.3])]):
            left=100+630*i;draw=panel(im,d,txt,(left,235,left+480,680),title,(-5,5),ylim,[-5,-3,-1,0,1,3,5],yticks,'Lag k (generations)')
            sub=cc[(cc.scope==scope)&(cc.left_variable==v1)&(cc.right_variable==v2)&(cc.statistic=='pooled_Pearson')&cc.lag.between(-5,5)]
            for name,color in COLORS.items():
                s=sub[sub.model==name].sort_values('lag');draw(s.lag.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy(),s.hi.to_numpy())
        txt(65,790,'Convention: Corr(left variable at n, right variable at n+k). At k=-1, the first panel pairs time with NEXT birth.',22)
        txt(65,832,'Bands: observed whole-lineage bootstrap; fixed-parameter simulation intervals. Same-cycle b-time correlation now matches closely.',22)
        im.save(OUT/f'{scope}_cross_correlations.png')
        im,d,txt=canvas(scope,950,'Glu: checking the iid effective-factor law')
        draw=panel(im,d,txt,(120,235,865,680),'Effective-factor density',(.35,.85),(0,18),[.4,.5,.6,.7,.8],[0,4,8,12,16],'Effective factor F')
        for name,color in COLORS.items():
            s=hist[(hist.scope==scope)&(hist.variable=='factor')&(hist.model==name)];x=(s.left.to_numpy()+s.right.to_numpy())/2;keep=(x>=.35)&(x<=.85)
            draw(x[keep],s.density.to_numpy()[keep],color,dots=False)
        draw=panel(im,d,txt,(1080,235,1825,680),'Effective-factor ACF',(1,10),(-.15,.15),[1,3,5,7,9],[-.1,0,.1],'Lag (generations)')
        curves(draw,ac[(ac.scope==scope)&(ac.variable=='factor')&ac.lag.between(1,10)])
        txt(65,795,'First factors are undefined initial conditions and excluded. The small negative simulated ACF is finite-lineage centering bias.',23)
        txt(65,842,'Factor independence also predicts Corr(t_n, F_(n+1)) = 0; the observed all-data correlation is -0.292.',23)
        txt(65,886,'These effective factors absorb reconstruction/measurement discrepancies; they are not physical partition fractions.',23)
        im.save(OUT/f'{scope}_factor_diagnostics.png')


if __name__=='__main__':main()
