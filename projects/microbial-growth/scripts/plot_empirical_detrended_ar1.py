#!/usr/bin/env python3
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fit_empirical_detrended_ar1 import OUT
from plot_joint_growth_timing import panel


def main():
    ac=pd.read_csv(OUT/'acf_comparison.csv')
    im=Image.new('RGB',(1500,980),'white');d=ImageDraw.Draw(im)
    def txt(x,y,s,size=23,bold=False,color='#243249'):
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
        d.text((x,y),s,font=font,fill=color)
    txt(60,25,'Does one AR(1) describe empirically detrended Glu births?',34,True)
    txt(60,80,'172 lineages x 21 births | exact generation-mean removal | 3,000 simulations per model',23)
    colors={'Observed':'#243249','Zero_offset_AR1':'#7291AD','Offset_AR1':'#C85633'}
    labels={'Observed':'Observed log-birth residuals','Zero_offset_AR1':'Zero-offset AR(1)','Offset_AR1':'AR(1) + static lineage offsets'}
    for j,(name,color) in enumerate(colors.items()):
        x=60+480*j;d.line((x,146,x+45,146),fill=color,width=5);txt(x+60,132,labels[name],21)
    curve=panel(im,d,txt,(130,265,1400,750),'Same finite-record ACF estimator for data and simulations',
                (1,10),(-.17,.56),list(range(1,11)),[-.1,0,.1,.2,.3,.4,.5],'Lag (generations)')
    for name,color in colors.items():
        sub=ac[(ac.model==name)&(ac.lag>=1)]
        curve(sub.lag.to_numpy(),sub['mean'].to_numpy(),color,sub.lo.to_numpy(),sub.hi.to_numpy())
    txt(60,855,'Offset model: phi = 0.4275; correlation length = 1.18 generations. Lag 1 matches; lags 2–4 do not.',24,True)
    txt(60,903,'Generation means are re-estimated in every simulation and data bootstrap; each lineage is then demeaned for its ACF.',21)
    txt(60,939,'Bands are pointwise: observed bootstrap versus fixed-parameter simulation intervals, not simultaneous tests.',21)
    im.save(OUT/'ar1_acf_check.png')


if __name__=='__main__':main()
