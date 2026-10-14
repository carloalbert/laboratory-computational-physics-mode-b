#!/usr/bin/env python3
"""Compact visual summary of the joint hazard screen (Pillow only)."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from screen_parsimonious_joint_hazards import OUT


def main():
    ac=pd.read_csv(OUT/'acf_comparison.csv');ac=ac[ac.split=='all']
    hc=pd.read_csv(OUT/'alpha_clock/acf_comparison.csv');hc=hc[hc.model=='hill_alpha_clock_4p']
    metrics=pd.read_csv(OUT/'joint_metrics.csv');metrics=metrics[metrics.split=='all']
    hm=pd.read_csv(OUT/'alpha_clock/joint_metrics.csv');hm=hm[hm.model=='hill_alpha_clock_4p'].set_index('metric')
    im=Image.new('RGB',(1800,1080),'white');d=ImageDraw.Draw(im)
    def font(size,bold=False):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',size)
    def txt(x,y,t,size=23,color='#243249',bold=False):d.text((x,y),t,font=font(size,bold),fill=color)
    txt(75,28,'Glu: improved components, but no joint match yet',38,bold=True)
    txt(75,83,'Recorded growth and effective-factor histories; gaps excluded from lag pairs',25)
    colors=['#243249','#148578','#C85633'];names=['Observed','4-parameter clock correction','5-parameter joint threshold fit']
    for i,(label,color) in enumerate(zip(names,colors)):
        x=75+i*560;d.line((x,144,x+55,144),fill=color,width=5);txt(x+70,127,label,23)
    for panel,(var,title,ylo,yhi) in enumerate([('birth','Birth-size ACF',-.12,.55),('time','Division-time ACF',-.10,.20)]):
        left=110+panel*870;top=235;right=left+710;bottom=655
        def xy(x,y):return (left+(x-1)/9*(right-left),bottom-(y-ylo)/(yhi-ylo)*(bottom-top))
        txt(left,186,title,29,bold=True)
        ticks=np.arange(-.1,.51,.1) if var=='birth' else [-.1,-.05,0,.05,.1,.15,.2]
        for y in ticks:
            yy=xy(1,y)[1];d.line((left,yy,right,yy),fill='#DEE4EA',width=1);txt(left-78,yy-13,f'{y:.2f}',21)
        for x in range(1,11):
            xx=xy(x,ylo)[0];txt(xx-8,bottom+12,str(x),21)
        d.line((left,top,left,bottom,right,bottom),fill='#243249',width=2)
        txt(left+320,bottom+49,'Lag',23)
        for name,color in zip(names,colors):
            if name=='4-parameter clock correction':
                sub=hc[hc.variable==('length_birth' if var=='birth' else 'time')];ys=sub.predictive_median.to_numpy()
            else:
                sub=ac[ac.model==('Observed' if name=='Observed' else 'AFT5_AR_joint')];ys=sub[var+'_mean'].to_numpy()
            pts=[xy(k,ys[k]) for k in range(1,11)]
            d.line(pts,fill=color,width=4)
            for x,y in pts:d.ellipse((x-5,y-5,x+5,y+5),fill=color)
    labels={'sd_b':'Birth-size SD','sd_t':'Time SD (h)','q99_t':'Time 99th percentile (h)','corr_b_t':'Correlation of b and t'}
    xcols=[90,790,1160,1550]
    for x,label in zip(xcols,['Joint check','Observed','Clock, 4p','Threshold, 5p']):txt(x,757,label,24,bold=True)
    obs=metrics[metrics.model=='Observed'].set_index('metric')['mean'];pred=metrics[metrics.model=='AFT5_AR_joint'].set_index('metric')['mean']
    table=[]
    for i,(metric,label) in enumerate(labels.items()):
        y=800+i*44
        if i%2==0:d.rectangle((75,y-3,1730,y+38),fill='#F2F5F8')
        values=[label,f'{obs[metric]:.3f}',f'{hm.loc[metric,"predictive_median"]:.3f}',f'{pred[metric]:.3f}']
        for x,label2 in zip(xcols,values):txt(x,y,label2,24)
        table.append({'metric':metric,'Observed':obs[metric],'Clock4':hm.loc[metric,'predictive_median'],'Threshold5':pred[metric]})
    txt(80,1005,'Clock fit: all-data exploratory MLE, 100 replays. Threshold: fit on 75% of lineages, 200 fresh replays.',21)
    txt(80,1038,'All-data diagnostics shown; neither curve is a successful joint fit. Separate held-out results are saved.',21)
    im.save(OUT/'joint_hazard_screen.png')
    pd.DataFrame(table).to_csv(OUT/'compact_comparison.csv',index=False)


if __name__=='__main__':main()
