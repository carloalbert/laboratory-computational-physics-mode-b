# Shared cycle environment in the effective division jump

This Glu experiment places the environmental effect directly in the observed
effective log jump,

\[
q_{\ell i}=\log b_{\ell,i+1}-\log b_{\ell i}-\alpha_{\ell i}\tau_{\ell i}
             =\mu_q+E_i+\xi_{\ell i},
\]

with

\[
E_i\stackrel{\mathrm{iid}}{\sim}N(0,\sigma_E^2),\qquad
\xi_{\ell i}\stackrel{\mathrm{iid}}{\sim}N(0,\sigma_q^2).
\]

There is one common latent `E_i` for each of the 20 observed transitions in the
balanced Glu cohort. The prior is independent across cycles; the fitted sequence
is not constrained to be a trend or a random walk.

## Inference

The normal-normal hierarchy is fitted by empirical Bayes. The hyperparameters
`mu_q`, `sigma_E`, and `sigma_q` maximize the likelihood after analytically
integrating every `E_i`. Conditional on these point estimates and the observations
in cycle `i`,

\[
V_i=\left(\sigma_E^{-2}+n_i\sigma_q^{-2}\right)^{-1},\qquad
E_i\mid q\sim N\left[
V_i\frac{n_i(\bar q_i-\mu_q)}{\sigma_q^2},V_i
\right].
\]

The fit gives `mu_q = -0.52074`, `sigma_E = 0.01445`, and
`sigma_q = 0.07342`. The shared effect accounts for 3.73% of the marginal
single-cell q variance. Nevertheless it is accurately estimable because 172
lineages contribute to each cycle. Thirteen of the 20 marginal 95% posterior
intervals exclude zero. The posterior means rise from `-0.02055` at cycle 0 to
`+0.02506` at cycle 19, with a linear slope of `0.00223` per generation.

Alpha, the protein Hill hazard, and the initial-size law are unchanged from the
`fast_slow_ar_q_protein_hazard` experiment. Posterior diagnostics draw one `E_i`
from its posterior in each replicate and share that draw across all simulated
lineages; cell residuals are drawn independently. No observed q, alpha, time, or
birth-size paths are replayed.

## Result

The environmental path improves the balanced-cohort generation-20 birth-size
change from `0.3729` in the original factorized simulation to `0.4786`; the
observed change is `0.5823`. Mean-curve RMSE is `0.0863`.

It does not explain the long birth-size correlation tail. Birth-ACF RMSE over
lags 1--15 is `0.0818`, only slightly below the original model's `0.0843`, and
q-ACF RMSE is `0.0861`. Thus an inferred shared mean trajectory explains much of
the size drift, while persistent within-lineage q structure is still required.

This is an in-sample posterior-conditioned diagnostic. If future `E_i` values are
drawn iid from their fitted prior, their expected path has no drift. A predictive
test should infer `E_i` using training lineages and assess held-out lineages from
the same cycles.

## Files

- `environment_posterior.csv`: inferred environmental effects and intervals.
- `balanced_generation_means.csv`: observed and posterior-predictive mean sizes.
- `birth_size_acf.csv` and `q_acf.csv`: ACF diagnostics.
- `fit_results.json`: fitted values and summary metrics.
- `shared_cycle_environment_q_diagnostics.png`: combined diagnostic figure.

Reproduce with:

```bash
python scripts/fit_shared_cycle_environment_q_glu.py --replicates 500
```
