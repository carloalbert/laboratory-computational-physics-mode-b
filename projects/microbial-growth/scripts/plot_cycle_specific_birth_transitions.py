#!/usr/bin/env python3
"""Cycle-specific birth fit versus the shared elapsed-time reference."""
import plot_empirical_birth_transitions as plot
from fit_cycle_specific_birth_transitions import OUT


if __name__=='__main__':
    plot.OUT=OUT;plot.COLORS={'Observed':'#243249','Shared_elapsed':'#7291AD','Cycle_specific':'#C85633'}
    plot.LABELS={'Observed':'Observed','Shared_elapsed':'Shared elapsed-time model','Cycle_specific':'Separate model for each cycle'}
    plot.main(band_model='Cycle_specific',time_note='Same gap-free lineages, fixed alpha/F paths and two birth seeds. Elapsed time is generated, not replayed.')
    ac=plot.pd.read_csv(OUT/'acf_comparison.csv');tr=plot.pd.read_csv(OUT/'birth_progress_trend.csv')
    for scope in ['all','validation']:
        im,d,txt=plot.canvas(scope,900,'Glu: independent cell-cycle transition densities')
        curve=plot.panel(im,d,txt,(110,235,810,690),'Birth-size drift (lineage means removed)',(0,1),(-.45,.6),[0,.25,.5,.75,1],[-.4,-.2,0,.2,.4,.6],'Relative progress through lineage')
        for name,color in plot.COLORS.items():
            s=tr[(tr.scope==scope)&(tr.model==name)].sort_values('progress');band=name=='Cycle_specific'
            curve(s.progress.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy() if band else None,s.hi.to_numpy() if band else None)
        curve=plot.panel(im,d,txt,(970,235,1670,690),'Birth ACF with two initial births excluded',(1,10),(-.15,.65),[1,3,5,7,9],plot.np.arange(-.1,.61,.1),'Lag (generations)')
        for name,color in plot.COLORS.items():
            s=ac[(ac.scope==scope)&(ac.variable=='birth_generated_only')&(ac.model==name)&ac.lag.between(1,10)].sort_values('lag');band=name=='Cycle_specific'
            curve(s.lag.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy() if band else None,s.hi.to_numpy() if band else None)
        txt(65,790,'Independent cycle fits; no smoothing across cycles. Singleton cycle 46 uses fixed reference bandwidths.',23)
        txt(65,835,'Late cycles are sparse: 9 training examples at cycle 32, and only 1 at cycle 46.',23)
        im.save(OUT/f'{scope}_drift_and_seed_exclusion.png')
