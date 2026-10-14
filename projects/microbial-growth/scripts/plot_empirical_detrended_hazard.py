#!/usr/bin/env python3
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fit_empirical_detrended_hazard import OUT
from growth_plotting import panel, font_for_plot


def main():
    ac=pd.read_csv(OUT/'acf_comparison.csv');means=pd.read_csv(OUT/'mean_replay.csv')
    colors={'Observed':'#243249','Original':'#7291AD','Detrended':'#C85633'}
    labels={'Observed':'Observed, empirically centered','Original':'Original hazard replay','Detrended':'Detrended-birth hazard replay'}
    im=Image.new('RGB',(1800,1400),'white');d=ImageDraw.Draw(im)
    def txt(x,y,s,size=23,bold=False,color='#243249'):
        f=font_for_plot(size,bold)
        d.text((x,y),s,font=f,fill=color)
    txt(60,24,'Glu: exact empirical detrending + four-parameter Hill hazard',34,True)
    txt(60,80,'172 complete lineages | first 20 transitions | 300 replays | observed alpha / effective factors / initial births',22)
    for j,(name,color) in enumerate(colors.items()):
        x=60+585*j;d.line((x,145,x+45,145),fill=color,width=5);txt(x+60,130,labels[name],21)
    for j,(var,title,ylim,ticks) in enumerate([('X','Centered log-birth ACF',(-.17,.55),[-.1,0,.1,.2,.3,.4,.5]),
                                            ('G','Centered accumulated-growth ACF',(-.35,.10),[-.3,-.2,-.1,0,.1])]):
        left=100+900*j
        curve=panel(im,d,txt,(left,255,left+700,660),title,(1,10),ylim,[1,3,5,7,10],ticks,'Lag (generations)')
        for name,color in colors.items():
            s=ac[(ac.model==name)&(ac.variable==var)&(ac.lag>=1)]
            curve(s.lag.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy(),s.hi.to_numpy())
    for j,(var,title,ylim,ticks) in enumerate([('X','Mean centered log birth during replay',(-.16,.12),[-.15,-.1,-.05,0,.05,.1]),
                                            ('G','Mean centered accumulated growth during replay',(-.06,.06),[-.06,-.03,0,.03,.06])]):
        left=100+900*j
        curve=panel(im,d,txt,(left,865,left+700,1210),title,(0,20),ylim,[0,5,10,15,20],ticks,'Generation since tracking began')
        for name,color in colors.items():
            s=means[(means.model==name)&(means.variable==var)]
            curve(s.generation.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy(),s.hi.to_numpy())
    txt(60,1310,'The residual correlations still disagree; subtracting empirical mean curves does not resolve the hazard mismatch.',24,True)
    txt(60,1350,'ACF bands: data lineage bootstrap (mean curves fixed); fixed-parameter simulation intervals. No generated-mean recentering.',21)
    im.save(OUT/'detrended_hazard_checks.png')


if __name__=='__main__':main()
