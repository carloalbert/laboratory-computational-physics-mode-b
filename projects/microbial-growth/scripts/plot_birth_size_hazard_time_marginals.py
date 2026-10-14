#!/usr/bin/env python3
"""Observed versus simulated division-time marginals for the no-memory hazard.

Both fitted parameter vectors are replayed with observed initial sizes, lineage
lengths, alpha and effective-factor histories. All values enter density
normalization; the first panel zooms to 0--4 h and the survival panel shows tails.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from optimize_birth_size_hazard_acf import (
    PROJECT_DIR, PARAMETER_NAMES, Replay, load_data, add_effective_factors,
)

ROOT = PROJECT_DIR / 'outputs/model_comparison/Glu'
OUT = ROOT / 'birth_size_dependent_time_hazard_acf_only_wide'
COLORS = {'Observed': '#255D9B', 'ACF-only fit': '#C9512C', 'Likelihood fit': '#208879'}


def plot(hist, survival, stats):
    im = Image.new('RGB', (1800, 900), 'white')
    dr = ImageDraw.Draw(im)
    font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    bold_path = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
    font = lambda size, bold=False: ImageFont.truetype(bold_path if bold else font_path, size)
    def text(x, y, label, size=24, fill='#243249', bold=False):
        dr.text((x, y), label, font=font(size, bold), fill=fill)
    text(80, 32, 'Glu: marginal inter-division times', 38, bold=True)
    text(80, 90, 'Same p,b-dependent hazard without memory; 200 simulated datasets per fit', 25)
    for i, (name, color) in enumerate(COLORS.items()):
        x=85+i*460
        dr.line((x,151,x+65,151), fill=color, width=5)
        text(x+80,135,name,26)
    panels=[(95,245,835,735),(1015,245,1755,735)]
    # The histogram bins are one third of an hour to accommodate quantized observations.
    ymax=max(hist.loc[hist.bin_left<4, list(COLORS)].max())*1.1
    ymax=np.ceil(ymax*2)/2
    for panel, (left,top,right,bottom) in enumerate(panels):
        xmax=4 if panel==0 else 10
        def xy(x,y):
            frac=y/ymax if panel==0 else (np.log10(max(y,1e-4))+4)/4
            return left+x/xmax*(right-left),bottom-frac*(bottom-top)
        text(left,192,'Probability density (main range)' if panel==0 else 'Survival probability (including the tail)',27,bold=True)
        yticks=np.linspace(0,ymax,5) if panel==0 else [1e-4,1e-3,1e-2,.1,1]
        for y in yticks:
            yy=xy(0,y)[1]
            dr.line((left,yy,right,yy),fill='#E1E6EE',width=1)
            label=f'{y:.2f}' if panel==0 else {1e-4:'0.0001',1e-3:'0.001',1e-2:'0.01',.1:'0.1',1:'1'}[y]
            text(left-85,yy-13,label,21)
        for x in np.arange(0,xmax+.01,.5 if panel==0 else 2):
            xx=xy(x,0 if panel==0 else 1e-4)[0]
            dr.line((xx,bottom,xx,bottom+8),fill='#243249',width=2)
            text(xx-12,bottom+15,f'{x:g}',22)
        dr.line((left,top,left,bottom,right,bottom),fill='#243249',width=2)
        text(left+180,bottom+57,'Inter-division time t (hours)',26)
        for name,color in COLORS.items():
            points=[]
            if panel==0:
                for row in hist.itertuples(index=False, name=None):
                    a,b=row[:2]
                    if a>=4:break
                    val=hist.loc[hist.bin_left==a,name].iloc[0]
                    points.extend([xy(a,val),xy(min(b,4),val)])
            else:
                for x,y in zip(survival.time_hours,survival[name]):
                    if y>=1e-4:points.append(xy(x,y))
            dr.line(points,fill=color,width=4)
    text(90,840,'Densities use all observations for normalization. Right: fraction of cycles longer than t (log scale).',23)
    im.save(OUT/'division_time_marginals.png')
    im.save(OUT/'division_time_marginals.pdf', 'PDF', resolution=180)


def main():
    real,groups=load_data()
    groups,_,_=add_effective_factors(groups)
    d=json.loads((OUT/'diagnostics.json').read_text())
    old=json.loads((ROOT/'birth_size_dependent_time_hazard_effective_f/diagnostics.json').read_text())
    parameters={'ACF-only fit':d['parameters'],'Likelihood fit':old['mle']}
    samples={'Observed':real.time.to_numpy(float)}
    replicates=200
    for mi,(name,params) in enumerate(parameters.items()):
        chunks=[]
        for i in range(0,replicates,10):
            replay=Replay(groups,d['b_star'],replicates=10)
            chunks.append(replay.exact_thinning([params[n] for n in PARAMETER_NAMES],672340+mi*10000+i,return_times=True))
        samples[name]=np.concatenate(chunks).ravel()
        print(name,'simulated times',len(samples[name]),flush=True)
    edges=np.arange(0, max(max(v) for v in samples.values())+1/3, 1/3)
    hist=pd.DataFrame({'bin_left':edges[:-1],'bin_right':edges[1:]})
    survival=pd.DataFrame({'time_hours':np.linspace(0,10,1001)})
    stats=[]
    for name,values in samples.items():
        counts,_=np.histogram(values,bins=edges)
        hist[name]=counts/(len(values)*np.diff(edges))
        ordered=np.sort(values)
        survival[name]=(len(values)-np.searchsorted(ordered,survival.time_hours,side='right'))/len(values)
        stats.append({'dataset':name,'n_values':len(values),'mean_hours':float(values.mean()),
                      'sd_hours':float(values.std()),'median_hours':float(np.median(values)),
                      'q95_hours':float(np.quantile(values,.95)), 'q99_hours':float(np.quantile(values,.99)),
                      'fraction_above_4h':float(np.mean(values>4))})
    stats=pd.DataFrame(stats)
    hist.to_csv(OUT/'division_time_histogram.csv',index=False)
    survival.to_csv(OUT/'division_time_survival.csv',index=False)
    stats.to_csv(OUT/'division_time_marginal_summary.csv',index=False)
    (OUT/'division_time_marginal_config.json').write_text(json.dumps({'replicates':replicates,'seed_base':672340,
        'seed_rule':'seed_base + model_index*10000 + batch_start_index','parameters':parameters,
        'sampling':'exact thinning, observed alpha and effective factors, same retained-row pairing as ACF replay'},indent=2))
    plot(hist,survival,stats)
    print(stats.to_string(index=False))


if __name__=='__main__':main()
