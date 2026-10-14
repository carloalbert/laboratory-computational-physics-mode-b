#!/usr/bin/env python3
"""Visual evidence for drift versus lineage-rate heterogeneity in Glu births."""
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from plot_joint_growth_timing import panel
from analyze_acf_heterogeneity_transients import OUT


def canvas(title):
    im=Image.new('RGB',(1800,920),'white');d=ImageDraw.Draw(im)
    def txt(x,y,s,size=23,bold=False,color='#243249'):
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
        d.text((x,y),s,font=font,fill=color)
    txt(60,25,title,36,True)
    return im,d,txt


def main():
    ac=pd.read_csv(OUT/'acf_comparisons.csv');trend=pd.read_csv(OUT/'birth_progress_trend.csv')
    im,d,txt=canvas('Glu birth size: a strong drift within individual lineages')
    txt(60,85,'276 lineages with at least 6 observations | Means and confidence bands resample whole lineages',23)
    curve=panel(im,d,txt,(110,240,800,690),'Birth size relative to each lineage\u2019s mean',(0,1),(-.5,.5),[0,.25,.5,.75,1],[-.4,-.2,0,.2,.4],'Relative progress through lineage')
    curve(trend.progress.to_numpy(),trend['mean'].to_numpy(),'#C85633',trend.lo.to_numpy(),trend.hi.to_numpy())
    curve=panel(im,d,txt,(970,240,1660,690),'Birth-size ACF',(1,8),(-.15,.5),[1,2,3,4,5,6,7,8],[-.1,0,.1,.2,.3,.4,.5],'Lag (generations)')
    styles=[('Within_lineage','Original (lineage means removed)','#243249'),('Common_exponential_removed','Shared trend removed','#C85633'),('Linearly_detrended','Individual linear trends removed','#7291AD')]
    for i,(processing,label,color) in enumerate(styles):
        s=ac[(ac['transform']=='raw')&(ac.model=='Observed')&(ac.processing==processing)&ac.lag.between(1,8)].sort_values('lag')
        curve(s.lag.to_numpy(),s['mean'].to_numpy(),color)
        x=70+570*i;d.line((x,147,x+40,147),fill=color,width=5);txt(x+53,132,label,21)
    txt(65,800,'93% of lineages increase from their first third to their final third; mean increase 0.51 [0.47, 0.55].',24,True)
    txt(65,847,'Detrending induces negative finite-sample ACF bias; disappearance of the slow tail alone is not proof of a mechanism.',22)
    im.save(OUT/'birth_drift_and_acf.png')
    im,d,txt=canvas('Shared drift removes the slow ACF component on held-out lineages')
    txt(60,85,'The shared trend is fitted on training lineages only; no per-lineage slopes are fitted here.',24)
    for i,(scope,title) in enumerate([('train','Training lineages'),('validation','Held-out lineages')]):
        curve=panel(im,d,txt,(110+860*i,240,800+860*i,690),title,(1,8),(-.15,.55),[1,2,3,4,5,6,7,8],[-.1,0,.1,.2,.3,.4,.5],'Lag (generations)')
        for name,color in [('Before','#243249'),('After','#C85633')]:
            s=ac[(ac['transform']=='raw')&(ac.model==scope+'_shared_trend')&(ac.processing==name)&ac.lag.between(1,8)].sort_values('lag')
            curve(s.lag.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy(),s.hi.to_numpy())
    for x,label,color in [(65,'Before removing shared trend','#243249'),(900,'After removing shared trend','#C85633')]:
        d.line((x,147,x+40,147),fill=color,width=5);txt(x+55,132,label,23)
    txt(65,800,'Lineage means are removed in both curves. Pairs use actual generation lags and never cross gaps.',23)
    txt(65,847,'Bands: whole-lineage bootstrap, conditional on the fitted shared trend; the validation set is exploratory.',22)
    im.save(OUT/'heldout_shared_trend_acf.png')


if __name__=='__main__':main()
