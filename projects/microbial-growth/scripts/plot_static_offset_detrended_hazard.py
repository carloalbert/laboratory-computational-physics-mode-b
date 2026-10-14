#!/usr/bin/env python3
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fit_static_offset_detrended_hazard import OUT
from growth_plotting import panel, font_for_plot


def main():
    ac=pd.read_csv(OUT/'acf_comparison.csv');hist=pd.read_csv(OUT/'marginal_densities.csv')
    colors={'Observed':'#243249','Shared':'#7291AD','Population_offsets':'#C85633','Posterior_offsets':'#4B8B68'}
    labels={'Observed':'Observed','Shared':'Shared hazard','Population_offsets':'Offsets: population draws','Posterior_offsets':'Offsets: fitted-lineage replay'}
    for scope in ['all','validation']:
        im=Image.new('RGB',(1800,1400),'white');d=ImageDraw.Draw(im)
        def txt(x,y,s,size=23,bold=False,color='#243249'):
            f=font_for_plot(size,bold)
            d.text((x,y),s,font=f,fill=color)
        txt(60,25,'Glu: static lineage offsets in the detrended Hill hazard',35,True)
        subtitle='172 lineages; all-data refit. Fitted-lineage replay uses full-record posterior offsets (descriptive only).'
        if scope=='validation':subtitle='38 held-out lineages; population and mean curves fitted on 134 training lineages. No fitted validation offsets.'
        txt(60,80,subtitle,22)
        for j,(name,color) in enumerate(colors.items()):
            if scope=='validation' and name=='Posterior_offsets':continue
            x=60+435*j;d.line((x,145,x+40,145),fill=color,width=5);txt(x+50,131,labels[name],20)
        for j,(var,title,ylim,ticks) in enumerate([
                ('X','Centered log-birth ACF',(-.16,.55),[-.1,0,.1,.2,.3,.4,.5]),
                ('G','Centered accumulated-growth ACF',(-.36,.12),[-.3,-.2,-.1,0,.1]),
                ('time','Division-time ACF',(-.14,.25),[-.1,0,.1,.2])]):
            left=100+900*(j%2);top=255+600*(j//2)
            curve=panel(im,d,txt,(left,top,left+700,top+375),title,(1,10),ylim,[1,3,5,7,10],ticks,'Lag (generations)')
            for name,color in colors.items():
                sub=ac[(ac.scope==scope)&(ac.model==name)&(ac.variable==var)&(ac.lag>=1)]
                if sub.empty:continue
                band=name in ['Observed','Population_offsets']
                curve(sub.lag.to_numpy(),sub['mean'].to_numpy(),color,sub.lo.to_numpy() if band else None,sub.hi.to_numpy() if band else None)
        subhist=hist[(hist.scope==scope)&(hist.variable=='time')];ymax=float(np.ceil(subhist.density.max()*1.05*2)/2)
        curve=panel(im,d,txt,(1000,855,1700,1230),'Division-time density',(0,5),(0,ymax),[0,1,2,3,4,5],np.linspace(0,ymax,6),'Time (hours)')
        for name,color in colors.items():
            sub=subhist[subhist.model==name];x=(sub.left.to_numpy()+sub.right.to_numpy())/2;keep=x<=5
            if sub.empty:continue
            curve(x[keep],sub.density.to_numpy()[keep],color,dots=False)
        txt(60,1320,'Static heterogeneity improves lag 1, but the residual ACFs and broad division-time distribution remain mismatched.',23,True)
        txt(60,1360,'500 replays per model; observed initial births, alpha and effective factors fixed. Bands condition on fitted mean curves/parameters.',20)
        im.save(OUT/f'{scope}_offset_hazard_checks.png')


if __name__=='__main__':main()
