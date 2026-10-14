#!/usr/bin/env python3
"""Visual checks for generated growth rates and division times."""
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fit_joint_growth_timing import OUT

COLORS={'Observed':'#243249','Recorded_alpha':'#7291AD','Joint_generated':'#C85633'}
LABELS={'Observed':'Observed','Recorded_alpha':'Recorded-alpha replay','Joint_generated':'Generated alpha and time'}


def canvas(scope,height,title):
    im=Image.new('RGB',(1920,height),'white');d=ImageDraw.Draw(im)
    def txt(x,y,s,size=23,bold=False,color='#243249'):
        f=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
        d.text((x,y),s,font=f,fill=color)
    txt(60,24,title,36,True)
    label='All-data refit (descriptive)' if scope=='all' else 'Held-out lineages; parameters fitted on training lineages only'
    txt(60,76,label+' | 200 simulations; observed initial states and effective division factors',23)
    for i,(name,color) in enumerate(COLORS.items()):
        x=65+610*i;d.line((x,140,x+55,140),fill=color,width=5);txt(x+70,123,LABELS[name])
    return im,d,txt


def panel(im,d,txt,box,title,xlim,ylim,xticks,yticks,xlabel):
    left,top,right,bottom=box
    def xy(x,y):return left+(x-xlim[0])/(xlim[1]-xlim[0])*(right-left),bottom-(y-ylim[0])/(ylim[1]-ylim[0])*(bottom-top)
    txt(left,top-47,title,27,True)
    for y in yticks:
        yy=xy(xlim[0],y)[1];d.line((left,yy,right,yy),fill='#DFE5EB',width=1);txt(left-71,yy-12,f'{y:.2f}',20)
    for x in xticks:txt(xy(x,ylim[0])[0]-10,bottom+13,f'{x:g}',21)
    d.line((left,top,left,bottom,right,bottom),fill='#243249',width=2);txt(left+150,bottom+49,xlabel,22)
    def curve(x,y,color,lo=None,hi=None,dots=True):
        if lo is not None:
            overlay=Image.new('RGBA',im.size,(0,0,0,0));od=ImageDraw.Draw(overlay)
            rgb=tuple(int(color[j:j+2],16) for j in [1,3,5])
            od.polygon([xy(a,b) for a,b in zip(x,lo)]+[xy(a,b) for a,b in zip(x[::-1],hi[::-1])],fill=rgb+(28,));im.paste(overlay,(0,0),overlay)
        pts=[xy(a,b) for a,b in zip(x,y)];d.line(pts,fill=color,width=4)
        if dots:
            for a,b in pts:d.ellipse((a-4,b-4,a+4,b+4),fill=color)
    return curve


def main():
    ac=pd.read_csv(OUT/'acf_comparison.csv');hist=pd.read_csv(OUT/'marginal_densities.csv');cc=pd.read_csv(OUT/'cross_correlations.csv')
    for scope in ['all','validation']:
        im,d,txt=canvas(scope,1390,'Glu: joint growth-rate and division-time model')
        for i,(var,title,xlim,xlabel) in enumerate([('birth','Birth size',(1,5),'Birth size'),('alpha','Growth rate',(0,1.25),'Growth rate (1/h)'),('time','Division time',(0,5),'Time (h)')]):
            left=100+630*i;right=left+480
            curve=panel(im,d,txt,(left,235,right,650),title+' ACF',(1,10),(-.15,.55),[1,3,5,7,9],np.arange(-.1,.51,.1),'Lag (generations)')
            for name,color in COLORS.items():
                sub=ac[(ac.scope==scope)&(ac.variable==var)&(ac.model==name)&ac.lag.between(1,10)].sort_values('lag')
                bands=name=='Joint_generated'
                curve(sub.lag.to_numpy(),sub['mean'].to_numpy(),color,sub.lo.to_numpy() if bands else None,sub.hi.to_numpy() if bands else None)
            subhist=hist[(hist.scope==scope)&(hist.variable==var)]
            ymax=float(np.ceil(subhist.density.max()*1.1*5)/5)
            ticks={'birth':[1,2,3,4,5],'alpha':[0,.25,.5,.75,1,1.25],'time':[0,1,2,3,4,5]}[var]
            curve=panel(im,d,txt,(left,850,right,1210),title+' density',xlim,(0,ymax),ticks,np.linspace(0,ymax,5),xlabel)
            for name,color in COLORS.items():
                sub=subhist[subhist.model==name];x=(sub.left.to_numpy()+sub.right.to_numpy())/2;keep=(x>=xlim[0])&(x<=xlim[1])
                curve(x[keep],sub.density.to_numpy()[keep],color,dots=False)
        txt(65,1315,'Birth curves coincide exactly in paired simulations: alpha changes the clock, not accumulated growth at division.',23)
        txt(65,1350,'Generation gaps excluded. ACFs use equal original-lineage weighting; shaded bands are fixed-parameter simulation intervals.',21)
        im.save(OUT/f'{scope}_marginals_acfs.png')
        im,d,txt=canvas(scope,900,'Glu: cross-correlations under the joint model')
        for i,(leftvar,rightvar,title,ylim) in enumerate([('birth','time','Birth size versus time',(-.55,.20)),('alpha','time','Growth rate versus time',(-.85,.10)),('birth','alpha','Birth size versus growth rate',(-.05,.35))]):
            left=100+630*i;right=left+480
            yticks=[-.5,-.3,-.1,.1] if i==0 else ([-.8,-.6,-.4,-.2,0] if i==1 else [0,.1,.2,.3])
            curve=panel(im,d,txt,(left,235,right,680),title,(-5,5),ylim,[-5,-3,-1,0,1,3,5],yticks,'Lag k (generations)')
            for name,color in COLORS.items():
                sub=cc[(cc.scope==scope)&(cc.model==name)&(cc.left_variable==leftvar)&(cc.right_variable==rightvar)&(cc.statistic=='pooled_Pearson')&cc.lag.between(-5,5)].sort_values('lag')
                curve(sub.lag.to_numpy(),sub['mean'].to_numpy(),color,sub.lo.to_numpy(),sub.hi.to_numpy())
        txt(65,790,'Convention: Corr(left variable at n, right variable at n+k). In the first panel, k=-1 pairs time with NEXT birth.',22)
        txt(65,832,'Bands: observed whole-lineage bootstrap; fixed-parameter simulation intervals. Within-lineage checks are also saved.',22)
        im.save(OUT/f'{scope}_cross_correlations.png')


if __name__=='__main__':main()
