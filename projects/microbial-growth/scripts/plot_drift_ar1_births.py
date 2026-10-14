#!/usr/bin/env python3
"""Static diagnostic plots for a shared linear birth drift plus AR(1)."""
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from fit_drift_ar1_births import OUT
from plot_joint_growth_timing import panel

COLORS = {'Observed': '#243249', 'No_drift': '#7291AD', 'Linear': '#C85633'}
LABELS = {'Observed': 'Observed Glu', 'No_drift': 'Stationary AR(1)', 'Linear': 'Linear drift + AR(1)'}


def canvas(title, subtitle, height=950):
    im = Image.new('RGB', (1800, height), 'white'); d = ImageDraw.Draw(im)
    def txt(x, y, s, size=23, bold=False, color='#243249'):
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf', size)
        d.text((x, y), s, font=font, fill=color)
    txt(60, 25, title, 36, True); txt(60, 80, subtitle, 22)
    for i, (name, color) in enumerate(COLORS.items()):
        x = 65+560*i; d.line((x, 145, x+60, 145), fill=color, width=5); txt(x+75, 130, LABELS[name], 23)
    return im, d, txt


def main():
    ac = pd.read_csv(OUT/'acf_comparison.csv'); tr = pd.read_csv(OUT/'centered_drift.csv')
    hist = pd.read_csv(OUT/'marginal_densities.csv'); residual = pd.read_csv(OUT/'residual_acf.csv')
    im, d, txt = canvas('Glu: a shared linear drift + AR(1) fluctuations',
                       '276 lineages | 6,441 births | likelihood fit, not ACF matching | 1,000 simulated datasets')
    curve = panel(im, d, txt, (110, 260, 820, 720), 'Mean-centered birth-size trajectory', (0, 32), (-.6, .7),
                  [0, 5, 10, 15, 20, 25, 30], [-.6, -.3, 0, .3, .6], 'Generation since tracking began (at least 20 lineages)')
    for name, color in COLORS.items():
        sub = tr[(tr.model==name)&(tr.generation<=32)]
        curve(sub.generation.to_numpy(), sub['mean'].to_numpy(), color, sub.lo.to_numpy(), sub.hi.to_numpy())
    curve = panel(im, d, txt, (1020, 260, 1730, 720), 'Birth-size ACF (same finite-record estimator)', (1, 10), (-.16, .56),
                  [1, 3, 5, 7, 10], [-.1, 0, .1, .2, .3, .4, .5], 'Lag (generations)')
    for name, color in COLORS.items():
        sub = ac[(ac.model==name)&(ac.variable=='birth')&(ac.lag>=1)]
        curve(sub.lag.to_numpy(), sub['mean'].to_numpy(), color, sub.lo.to_numpy(), sub.hi.to_numpy())
    txt(65, 835, 'Drift = 0.0309 per generation; residual AR coefficient = 0.475; innovation SD = 0.300.', 25, True)
    txt(65, 880, 'Bands: whole-lineage bootstrap for data; fixed-parameter simulations for models. Lineage offsets fitted descriptively.', 22)
    im.save(OUT/'birth_drift_and_acf.png')

    im, d, txt = canvas('Glu: remaining discrepancies after adding linear drift',
                       'Recorded growth rates and effective factors fixed | negative reconstructed times retained', 1420)
    settings = [
        ('birth', 'Birth-size density', (100, 255, 800, 650), (0, 5), (0, 1), [0, 1, 2, 3, 4, 5], np.arange(0, 1.01, .25), 'Birth size'),
        ('time', 'Reconstructed division-time density', (1010, 255, 1710, 650), (-1, 5), (0, 2.6), [-1, 0, 1, 2, 3, 4, 5], [0, .5, 1, 1.5, 2, 2.5], 'Time (h)')]
    for var, title, box, xlim, ylim, xticks, yticks, xlabel in settings:
        curve = panel(im, d, txt, box, title, xlim, ylim, xticks, yticks, xlabel)
        for name, color in COLORS.items():
            sub = hist[(hist.model==name)&(hist.variable==var)]
            x = (sub.left.to_numpy()+sub.right.to_numpy())/2; keep = (x>=xlim[0])&(x<=xlim[1])
            curve(x[keep], sub.density.to_numpy()[keep], color, dots=False)
    curve = panel(im, d, txt, (100, 865, 800, 1215), 'ACF after removing the shared linear drift', (1, 10), (-.15, .4),
                  [1, 3, 5, 7, 10], [-.1, 0, .1, .2, .3, .4], 'Lag (generations)')
    for name in ['Observed', 'Linear']:
        sub = residual[(residual.model==name)&(residual.lag>=1)]
        curve(sub.lag.to_numpy(), sub['mean'].to_numpy(), COLORS[name], sub.lo.to_numpy(), sub.hi.to_numpy())
    curve = panel(im, d, txt, (1010, 865, 1710, 1215), 'Reconstructed division-time ACF', (1, 10), (-.1, .14),
                  [1, 3, 5, 7, 10], [-.1, -.05, 0, .05, .1], 'Lag (generations)')
    for name, color in COLORS.items():
        sub = ac[(ac.model==name)&(ac.variable=='time')&(ac.lag>=1)]
        curve(sub.lag.to_numpy(), sub['mean'].to_numpy(), color)
    txt(65, 1320, 'The residual ACF is not exactly AR(1). Time SD: observed 0.690 h; model 0.635 h.', 25, True)
    txt(65, 1360, '0.080% of reconstructed durations are nonpositive: useful statistical approximation, not yet a physical timing model.', 22)
    im.save(OUT/'remaining_diagnostics.png')


if __name__=='__main__':
    main()
