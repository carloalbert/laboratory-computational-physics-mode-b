#!/usr/bin/env python3
"""Matched first/second-order conditional birth-size replay figures."""
import plot_empirical_birth_transitions as plot
from fit_second_order_birth_transitions import OUT


if __name__=='__main__':
    plot.OUT=OUT
    plot.COLORS={'Observed':'#243249','First_order':'#7291AD','Second_order':'#C85633'}
    plot.LABELS={'Observed':'Observed','First_order':'First order (matched control)','Second_order':'Second order (+ older birth)'}
    plot.main(band_model='Second_order',time_note='Both models fix the first TWO births. Time comparisons exclude the first and last cycle of each segment.')
    ac=plot.pd.read_csv(OUT/'acf_comparison.csv')
    for scope in ['all','validation']:
        im,d,txt=plot.canvas(scope,900,'Glu: birth ACF after excluding both initial births')
        for i,(var,title) in enumerate([('birth','Full trajectories (two observed seeds)'),('birth_generated_only','Generated births only (seeds excluded)')]):
            left=110+860*i;right=left+700
            curve=plot.panel(im,d,txt,(left,235,right,690),title,(1,10),(-.15,.57),[1,3,5,7,9],plot.np.arange(-.1,.51,.1),'Lag (generations)')
            for name,color in plot.COLORS.items():
                s=ac[(ac.scope==scope)&(ac.variable==var)&(ac.model==name)&ac.lag.between(1,10)].sort_values('lag');band=name=='Second_order'
                curve(s.lag.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy() if band else None,s.hi.to_numpy() if band else None)
        txt(65,790,'Observed ACF is recomputed on the same retained positions in each panel.',23)
        txt(65,835,'Equal original-lineage weighting; no gap-crossing pairs. Shading: 95% fixed-fit simulation intervals.',22)
        im.save(OUT/f'{scope}_seed_exclusion_acfs.png')
