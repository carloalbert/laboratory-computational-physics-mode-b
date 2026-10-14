#!/usr/bin/env python3
"""Conditional density slices; fixed birth size by column/growth rate by row."""
import pandas as pd
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from fit_empirical_conditional_times import OUT


def main():
    tab=pd.read_csv(OUT/'conditional_density_slices.csv')
    contexts=pd.read_csv(OUT/'conditional_summary.csv')
    im=Image.new('RGB',(1800,1330),'white');d=ImageDraw.Draw(im)
    def font(size,bold=False):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
    def text(x,y,t,size=22,bold=False,color='#243249'):d.text((x,y),t,font=font(size,bold),fill=color)
    text(75,27,'Glu: empirical conditional division-time densities',36,True)
    text(75,83,'Columns fix birth size; rows fix growth rate. Each curve integrates to one over positive time.',24)
    d.line((80,145,140,145),fill='#236DA4',width=5);text(155,127,'Empirical conditional density',25)
    d.line((785,145,845,145),fill='#CF7239',width=4);text(860,127,'Four-parameter Hill reference',25)
    for ai in range(3):
        row=tab[(tab.alpha_index==ai)&(tab.time<=4)]
        ymax=np.ceil(max(row.density.max(),row.Hill4_density.max())*1.07*2)/2
        for bi in range(3):
            ctx=contexts[(contexts.alpha_index==ai)&(contexts.birth_index==bi)].iloc[0]
            left=105+bi*585;right=left+465;top=270+ai*320;bottom=top+205
            text(left,top-70,f'b = {ctx.birth:.2f};  alpha = {ctx.alpha:.3f} /h',24,True)
            text(left,top-35,f'Mean t = {ctx.mean_time:.2f} h',22)
            def xy(x,y):return left+x/4*(right-left),bottom-y/ymax*(bottom-top)
            for y in np.linspace(0,ymax,4):
                yy=xy(0,y)[1];d.line((left,yy,right,yy),fill='#E1E7ED',width=1);text(left-60,yy-12,f'{y:.1f}',19)
            for x in range(5):text(xy(x,0)[0]-7,bottom+10,str(x),20)
            d.line((left,top,left,bottom,right,bottom),fill='#243249',width=2)
            sub=tab[(tab.alpha_index==ai)&(tab.birth_index==bi)&(tab.time<=4)]
            for field,color,width in [('Hill4_density','#CF7239',4),('density','#236DA4',4)]:
                points=[xy(x,y) for x,y in zip(sub.time,sub[field])]
                if field=='Hill4_density':
                    for j in range(0,len(points)-1,9):d.line(points[j:j+6],fill=color,width=width)
                else:d.line(points,fill=color,width=width)
            if ai==2:text(left+95,bottom+40,'Division time t (hours)',22)
    text(80,1220,'Birth-size and growth-rate values are the 15th, 50th and 85th percentiles. Vertical axes: density per hour.',22)
    text(80,1255,'Bandwidths selected by lineage-level cross-validation; curves use all 6,504 observations. Tail estimates are less reliable.',21)
    im.save(OUT/'conditional_density_slices.png')


if __name__=='__main__':main()
