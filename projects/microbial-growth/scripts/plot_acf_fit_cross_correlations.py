#!/usr/bin/env python3
"""Plot the saved ACF-fit audit without rerunning any simulations."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from check_acf_fit_cross_correlations import OUT

COLORS={'Observed':'#243249','ACF_only':'#C85633','Likelihood':'#148578'}
LABELS={'Observed':'Observed','ACF_only':'ACF-only fit','Likelihood':'Likelihood fit (reference)'}


def canvas(title, subtitle, height):
    im=Image.new('RGB',(1800,height),'white'); d=ImageDraw.Draw(im)
    def txt(x,y,s,size=23,bold=False,color='#243249'):
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
        d.text((x,y),s,font=font,fill=color)
    txt(70,25,title,37,True);txt(70,79,subtitle,24)
    for i,(name,color) in enumerate(COLORS.items()):
        x=75+i*550;d.line((x,140,x+55,140),fill=color,width=5);txt(x+70,124,LABELS[name])
    return im,d,txt


def panel(im,d,txt,box,title,xlim,ylim,xticks,yticks,xlabel):
    left,top,right,bottom=box
    def xy(x,y):return left+(x-xlim[0])/(xlim[1]-xlim[0])*(right-left),bottom-(y-ylim[0])/(ylim[1]-ylim[0])*(bottom-top)
    txt(left,top-49,title,28,True)
    for y in yticks:
        yy=xy(xlim[0],y)[1];d.line((left,yy,right,yy),fill='#DFE5EB',width=1);txt(left-78,yy-13,f'{y:.2f}',21)
    for x in xticks:txt(xy(x,ylim[0])[0]-12,bottom+12,f'{x:g}',21)
    d.line((left,top,left,bottom,right,bottom),fill='#243249',width=2)
    txt(left+220,bottom+49,xlabel,22)
    def curve(xs,ys,color,lo=None,hi=None,dots=True):
        points=[xy(x,y) for x,y in zip(xs,ys)]
        if lo is not None:
            overlay=Image.new('RGBA',im.size,(0,0,0,0)); od=ImageDraw.Draw(overlay)
            rgb=tuple(int(color[j:j+2],16) for j in (1,3,5))
            band=[xy(x,y) for x,y in zip(xs,lo)]+[xy(x,y) for x,y in zip(xs[::-1],hi[::-1])]
            od.polygon(band,fill=rgb+(30,));im.paste(overlay,(0,0),overlay)
        d.line(points,fill=color,width=4)
        if dots:
            for x,y in points:d.ellipse((x-4,y-4,x+4,y+4),fill=color)
    return curve


def main():
    cc=pd.read_csv(OUT/'cross_correlations.csv')
    cc=cc[(cc.design=='historical_rows')&(cc.birth_transform=='birth')]
    im,d,txt=canvas('Glu: the ACF-focused fit misses the size-time cross-correlations',
                    '200 fresh simulations of the saved fit; recorded growth rates and effective division factors',950)
    for i,(stat,title) in enumerate([('pooled_Pearson','Pooled Pearson correlation'),('within_lineage_Pearson','After subtracting each lineage mean')]):
        curve=panel(im,d,txt,(115+870*i,235,815+870*i,705),title,(-8,8),(-.52,.2),range(-8,9,2),np.arange(-.5,.21,.1),'Lag k (generations)')
        for name,color in COLORS.items():
            sub=cc[(cc.statistic==stat)&(cc.model==name)&(cc.lag.abs()<=8)].sort_values('lag')
            curve(sub.lag.to_numpy(),sub['mean'].to_numpy(),color,sub.lo.to_numpy(),sub.hi.to_numpy())
    txt(75,810,'Definition: Corr(b_n, t_(n+k)). At k = -1, this is the correlation of a division time with the NEXT birth size.',23)
    txt(75,852,'Bands: observed 95% whole-lineage bootstrap; simulations 95% between-replay interval at fixed parameters.',22)
    txt(75,891,'Excluding generation gaps gives the same conclusion; see the saved gap-reset results.',22)
    im.save(OUT/'cross_correlations.png')

    ac=pd.read_csv(OUT/'acf_comparison.csv');ac=ac[ac.design=='historical_rows']
    hist=pd.read_csv(OUT/'marginal_densities.csv')
    im,d,txt=canvas('Glu: a better birth-size ACF is not a joint statistical match',
                    'Same saved ACF-only fit, without refitting; likelihood fit included as a reference',1490)
    for i,(var,title) in enumerate([('birth','Birth-size ACF'),('time','Division-time ACF')]):
        curve=panel(im,d,txt,(115+870*i,235,815+870*i,650),title,(1,10),(-.12,.52),range(1,11),np.arange(-.1,.51,.1),'Lag (generations)')
        for name,color in COLORS.items():
            sub=ac[(ac.variable==var)&(ac.model==name)&ac.lag.between(1,10)].sort_values('lag')
            curve(sub.lag.to_numpy(),sub['mean'].to_numpy(),color,dots=True)
    for i,(var,title,xlim,ylim,xticks,yticks,xlabel) in enumerate([
        ('birth','Birth-size probability density',(1,4.8),(0,1.4),np.arange(1,5),np.arange(0,1.41,.2),'Birth size'),
        ('time','Division-time probability density',(0,5),(0,1.7),np.arange(0,6),np.arange(0,1.61,.4),'Division time (h)')]):
        curve=panel(im,d,txt,(115+870*i,850,815+870*i,1265),title,xlim,ylim,xticks,yticks,xlabel)
        for name,color in COLORS.items():
            sub=hist[(hist.variable==var)&(hist.model==('ACF_only' if name=='Observed' else name))]
            centers=(sub.left.to_numpy()+sub.right.to_numpy())/2
            keep=(centers>=xlim[0])&(centers<=xlim[1])
            values=sub['observed_density' if name=='Observed' else 'density'].to_numpy()
            curve(centers[keep],values[keep],color,dots=False)
    txt(75,1370,'ACFs use the original equal-lineage estimator. Marginals pool all cycles; displayed ranges omit small tails.',22)
    txt(75,1412,'The ACF-only objective targeted birth-size autocorrelation only, not either marginal or the time ACF.',22)
    im.save(OUT/'four_diagnostics.png')


if __name__=='__main__':main()
