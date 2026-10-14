#!/usr/bin/env python3
"""Elapsed-time birth transition diagnostics."""
import plot_empirical_birth_transitions as plot
from fit_elapsed_birth_transitions import OUT


if __name__=='__main__':
    plot.OUT=OUT
    plot.COLORS={'Observed':'#243249','No_elapsed':'#7291AD','Elapsed':'#C85633'}
    plot.LABELS={'Observed':'Observed (gap-free lineages)','No_elapsed':'Second order, no elapsed time','Elapsed':'Second order + elapsed time'}
    plot.main(band_model='Elapsed',time_note='Both models fix two initial births. The elapsed-time clock is advanced with generated times, not observed times.')
    ac=plot.pd.read_csv(OUT/'acf_comparison.csv');tr=plot.pd.read_csv(OUT/'birth_progress_trend.csv')
    for scope in ['all','validation']:
        im,d,txt=plot.canvas(scope,900,'Glu: endogenous elapsed-time clock and birth-size drift')
        curve=plot.panel(im,d,txt,(110,235,810,690),'Birth size relative to each lineage\u2019s mean',(0,1),(-.45,.5),[0,.25,.5,.75,1],[-.4,-.2,0,.2,.4],'Relative progress through lineage')
        for name,color in plot.COLORS.items():
            s=tr[(tr.scope==scope)&(tr.model==name)].sort_values('progress')
            curve(s.progress.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy() if name=='Elapsed' else None,s.hi.to_numpy() if name=='Elapsed' else None)
        curve=plot.panel(im,d,txt,(970,235,1670,690),'Birth ACF with both initial births excluded',(1,10),(-.15,.57),[1,3,5,7,9],plot.np.arange(-.1,.51,.1),'Lag (generations)')
        for name,color in plot.COLORS.items():
            s=ac[(ac.scope==scope)&(ac.variable=='birth_generated_only')&(ac.model==name)&ac.lag.between(1,10)].sort_values('lag')
            curve(s.lag.to_numpy(),s['mean'].to_numpy(),color,s.lo.to_numpy() if name=='Elapsed' else None,s.hi.to_numpy() if name=='Elapsed' else None)
        txt(65,790,'The clock starts at the first recorded birth; all subsequent non-seed durations are reconstructed from generated births.',22)
        txt(65,835,'Gap-free lineages only. Shading: 95% fixed-fit simulation intervals; observed curves use matching positions.',22)
        im.save(OUT/f'{scope}_drift_and_seed_exclusion.png')
