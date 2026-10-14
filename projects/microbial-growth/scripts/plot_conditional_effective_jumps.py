#!/usr/bin/env python3
"""Observed and simulated ACFs/marginals for conditional effective jumps."""
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from fit_conditional_effective_jumps import OUT


def main():
    ac=pd.read_csv(OUT/'acf_comparison.csv');ac=ac[ac.scope=='all']
    hist=pd.read_csv(OUT/'marginal_densities.csv')
    styles={'Observed':('Observed','#253247'),
            'fixed_factors':('Fixed effective factors','#2D6FA6'),
            'age':('Time-dependent jumps (3p)','#D66A32'),
            'birth_growth_alpha':('Size/growth/rate jumps (5p)','#248E79')}
    im=Image.new('RGB',(1800,1320),'white');d=ImageDraw.Draw(im)
    def f(size,b=False):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if b else '')+'.ttf',size)
    def txt(x,y,s,size=23,b=False,color='#253247'):d.text((x,y),s,font=f(size,b),fill=color)
    txt(75,30,'Glu: sampling conditional effective division factors',36,True)
    txt(75,85,'Same four-parameter hazard and recorded growth rates; 200 simulations per model',25)
    for i,(name,(label,color)) in enumerate(styles.items()):
        x=75+(i%2)*860;y=143+(i//2)*45
        d.line((x,y,x+55,y),fill=color,width=5);txt(x+72,y-16,label,24)
    panels=[('birth','acf','Birth-size ACF',1,10,-.12,.56),('time','acf','Division-time ACF',1,10,-.10,.20),
            ('birth','density','Birth-size distribution',.5,5.,0,1.15),('time','density','Division-time distribution',0,4,0,1.55)]
    for index,(variable,kind,title,xmin,xmax,ymin,ymax) in enumerate(panels):
        left=110+(index%2)*875;right=left+710;top=285+(index//2)*510;bottom=top+335
        def xy(x,y):return left+(x-xmin)/(xmax-xmin)*(right-left),bottom-(y-ymin)/(ymax-ymin)*(bottom-top)
        txt(left,top-47,title,29,True)
        yticks=np.linspace(ymin,ymax,5) if kind=='density' else ([-.1,0,.1,.2,.3,.4,.5] if variable=='birth' else [-.1,-.05,0,.05,.1,.15,.2])
        for y in yticks:
            yy=xy(xmin,y)[1];d.line((left,yy,right,yy),fill='#E1E6EB',width=1);txt(left-75,yy-12,f'{y:.2f}',20)
        xticks=range(1,11) if kind=='acf' else np.arange(np.ceil(xmin),xmax+.01,1)
        for x in xticks:txt(xy(x,ymin)[0]-9,bottom+12,f'{x:g}',21)
        d.line((left,top,left,bottom,right,bottom),fill='#253247',width=2)
        txt(left+210,bottom+50,'Lag (generations)' if kind=='acf' else ('Birth size' if variable=='birth' else 'Division time (h)'),23)
        for name,(label,color) in styles.items():
            if kind=='acf':
                sub=ac[(ac.model==name)&(ac.variable==variable)&(ac.lag>=xmin)&(ac.lag<=xmax)]
                pts=[xy(x,y) for x,y in zip(sub.lag,sub['mean'])]
            else:
                sub=hist[(hist.model==('fixed_factors' if name=='Observed' else name))&(hist.variable==variable)]
                column='observed_density' if name=='Observed' else 'density';pts=[]
                for row in sub.to_dict('records'):
                    if row['right']<=xmin or row['left']>=xmax:continue
                    pts.extend([xy(max(xmin,row['left']),row[column]),xy(min(xmax,row['right']),row[column])])
            if pts:d.line(pts,fill=color,width=4)
            if kind=='acf':
                for x,y in pts:d.ellipse((x-4,y-4,x+4,y+4),fill=color)
    txt(80,1220,'Jump-parameter counts exclude the 4 hazard parameters. All-data refits shown; held-out checks also fail.',22)
    txt(80,1255,'Densities use all samples for normalization. Missing-generation gaps reset dynamics and are omitted from lag pairs.',21)
    im.save(OUT/'conditional_jump_comparison.png')


if __name__=='__main__':main()
