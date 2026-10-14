#!/usr/bin/env python3
"""Compare empirical birth transitions and reconstructed times with Glu data."""
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fit_empirical_birth_transitions import OUT
from plot_joint_growth_timing import panel

COLORS={'Observed':'#243249','Hill_reference':'#7291AD','Empirical_birth':'#C85633'}
LABELS={'Observed':'Observed','Hill_reference':'Previous Hill reference','Empirical_birth':'Empirical birth transition'}


def canvas(scope,height,title):
    im=Image.new('RGB',(1800,height),'white');d=ImageDraw.Draw(im)
    def txt(x,y,s,size=23,bold=False,color='#243249'):
        f=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
        d.text((x,y),s,font=f,fill=color)
    txt(60,24,title,36,True)
    txt(60,76,('All-data refit' if scope=='all' else 'Held-out lineages')+' | Recorded alpha and effective F paths fixed | 200 simulations',23)
    for i,(name,color) in enumerate(COLORS.items()):
        x=65+560*i;d.line((x,140,x+55,140),fill=color,width=5);txt(x+70,123,LABELS[name])
    return im,d,txt


def main(band_model='Empirical_birth',time_note=None):
    ac=pd.read_csv(OUT/'acf_comparison.csv');hist=pd.read_csv(OUT/'marginal_densities.csv');cc=pd.read_csv(OUT/'cross_correlations.csv')
    for scope in ['all','validation']:
        im,d,txt=canvas(scope,1390,'Glu: fit birth transitions, then reconstruct division times')
        for i,(var,title,xlim,xlabel) in enumerate([('birth','Birth size',(.8,5.2),'Birth size'),('time','Reconstructed time',(0,6),'Time (h)')]):
            left=110+860*i;right=left+700
            curve=panel(im,d,txt,(left,235,right,645),title+' ACF',(1,10),(-.15,.57),[1,3,5,7,9],np.arange(-.1,.51,.1),'Lag (generations)')
            for name,color in COLORS.items():
                s=ac[(ac.scope==scope)&(ac.variable==var)&(ac.model==name)&ac.lag.between(1,10)].sort_values('lag');band=name==band_model
                curve(s.lag.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy() if band else None,s.hi.to_numpy() if band else None)
            sh=hist[(hist.scope==scope)&(hist.variable==var)];ymax=float(np.ceil(sh.density.max()*1.1*5)/5)
            curve=panel(im,d,txt,(left,840,right,1200),title+' density',xlim,(0,ymax),[1,2,3,4,5] if var=='birth' else [0,1,2,3,4,5,6],np.linspace(0,ymax,5),xlabel)
            for name,color in COLORS.items():
                s=sh[sh.model==name];x=(s.left.to_numpy()+s.right.to_numpy())/2;keep=(x>=xlim[0])&(x<=xlim[1])
                curve(x[keep],s.density.to_numpy()[keep],color,dots=False)
        txt(65,1300,time_note or 'Time comparisons omit each segment\u2019s last cycle in both data and simulations; no next birth is recorded.',23)
        txt(65,1340,'ACFs: equal original-lineage weighting, no gap-crossing pairs. Shading: 95% fixed-fit simulation intervals.',22)
        im.save(OUT/f'{scope}_marginals_acfs.png')
        im,d,txt=canvas(scope,900,'Glu: birth-size / reconstructed-time cross-correlations')
        for i,(stat,title) in enumerate([('pooled_Pearson','Pooled Pearson'),('within_lineage_Pearson','Within-lineage Pearson')]):
            left=110+860*i;right=left+700
            curve=panel(im,d,txt,(left,235,right,690),title,(-5,5),(-.5,.25),[-5,-3,-1,0,1,3,5],[-.4,-.2,0,.2],'Lag k (generations)')
            for name,color in COLORS.items():
                s=cc[(cc.scope==scope)&(cc.model==name)&(cc.statistic==stat)&cc.lag.between(-5,5)].sort_values('lag')
                curve(s.lag.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy(),s.hi.to_numpy())
        txt(65,790,'Convention: Corr(birth at n, time at n+k). Lag -1 relates a time to the following birth.',23)
        txt(65,835,'Bands: observed whole-lineage bootstrap; 95% fixed-fit simulation intervals.',22)
        im.save(OUT/f'{scope}_cross_correlations.png')


if __name__=='__main__':main()
