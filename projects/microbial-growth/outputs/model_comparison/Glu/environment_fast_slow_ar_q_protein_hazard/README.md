# Shared cycle environment plus lineage q memory

This Glu experiment combines the two effects supported by the preceding tests:

\[
q_{\ell i}=\mu_q+E_i+r^{(f)}_{\ell i}+r^{(s)}_{\ell i},
\]

\[
E_i\stackrel{\mathrm{iid}}{\sim}N(0,\sigma_E^2),\qquad
r^{(k)}_{\ell,i+1}=\rho_k r^{(k)}_{\ell i}+\zeta^{(k)}_{\ell i},qquad
\zeta^{(k)}_{\ell i}\sim N(0,\sigma_k^2).
\]

The environmental effect is common to all lineages in generation `i`. The fast
and slow AR modes are independent between lineages and retain the within-lineage
q memory. The birth-size recursion is

\[
\log b_{\ell,i+1}=\log b_{\ell i}+\alpha_{\ell i}\tau_{\ell i}+q_{\ell i}.
\]

Alpha, the protein Hill timing hazard, and the initial-size distribution are
unchanged from `fast_slow_ar_q_protein_hazard`.

## Joint likelihood and posterior effects

For the balanced panel with `n=172` lineages and `T=20` transitions, let `K` be
the stationary `T x T` covariance of the sum of the two AR modes. Integrating
the complete environmental vector gives

\[
\operatorname{Cov}[\operatorname{vec}(q)]
=I_n\otimes K+\mathbf 1_n\mathbf 1_n^\top\otimes\sigma_E^2I_T.
\]

The implementation evaluates this Gaussian likelihood through the orthogonal
lineage-mean and within-lineage subspaces, avoiding construction of its full
`3440 x 3440` covariance matrix. All q population parameters, including
`sigma_E`, are fitted jointly. The nested no-environment comparison is refitted
on the same balanced panel.

Conditional on the fitted hyperparameters, with
`C = sigma_E^2 I + K/n`,

\[
E\mid q\sim N\!\left[
\sigma_E^2C^{-1}(\bar q-\mu_q\mathbf1),
\sigma_E^2I-\sigma_E^4C^{-1}
\right].
\]

The full posterior covariance is retained when drawing the 20 environmental
effects for autonomous simulations.

## Fit

| Parameter | Estimate |
|---|---:|
| `mu_q` | -0.52074 |
| `rho_fast` | -0.11221 |
| fast innovation SD | 0.06811 |
| stationary fast SD | 0.06854 |
| `rho_slow` | 0.95452 |
| slow innovation SD | 0.00790 |
| stationary slow SD | 0.02649 |
| environmental prior SD | 0.01431 |

Adding the shared environmental variance improves AIC by `61.95` relative to
the jointly refitted no-environment model. The inferred `E_i` path is almost
unchanged from the environment-only analysis: it rises from `-0.02023` at cycle
0 to `+0.02482` at cycle 19.

## Autonomous diagnostic

Five hundred simulations generate q, alpha, division time, and birth size;
nothing except the observed lineage-length pattern is replayed.

| Statistic | AR modes only | Environment only | Combined | Observed |
|---|---:|---:|---:|---:|
| Birth-size change, generations 0--20 | 0.3729 | 0.4786 | 0.4862 | 0.5823 |
| Mean birth-curve RMSE | 0.1585 | 0.0863 | 0.0901 | -- |
| Birth-ACF RMSE, lags 1--15 | 0.0843 | 0.0818 | 0.0766 | -- |
| q-ACF RMSE, lags 1--15 | 0.0249 | 0.0861 | 0.0245 | -- |

The combined model succeeds at preserving both the improved mean drift and the
q ACF. Its birth-size ACF is better than either component model, but it still
decays too rapidly at lags 3--8. Therefore the missing birth-size correlation is
not explained entirely by the marginal q memory plus a generation-shared q
shift; dependence involving growth/timing or the hazard remains a candidate.

This is an in-sample posterior-conditioned environmental history. Fresh future
`E_i` values drawn iid from the prior have zero expected drift. A stringent
predictive test would infer the same `E_i` from training lineages and evaluate
held-out lineages observed in those cycles.

Reproduce with:

```bash
python scripts/fit_environment_fast_slow_ar_q_glu.py --replicates 500
```
