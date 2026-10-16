# All-parameter birth-ACF calibration

This is an intentionally unrestricted structural stress test. It minimizes only
the mean squared error of the mean within-lineage birth-size ACF at lags 1--15.
There is no likelihood, prior, marginal-distribution, mean-drift, or parameter
regularization term in the objective.

The calibrated quantities are the 13 population parameters

- `u`, `omega2`, and `eta` of the protein Hill hazard;
- the alpha marginal mean and SD and its AR(1) coefficient;
- `mu_q`, both q AR coefficients, and both stationary q-mode SDs;
- the initial log-birth mean and SD;

plus all 47 generation-specific effects `E_0,...,E_46`. Because `mu_q` and a
constant shift of every `E_i` are redundant, the fit imposes

\[
\sum_{i=0}^{46}E_i=0.
\]

The likelihood-fit point is used only as the optimizer initialization. Six
common-seed simulations per objective evaluation make the stochastic objective
reproducible. Candidate solutions are selected using 32 simulations and assessed
with 500 fresh simulations.

## Result

| Diagnostic | Likelihood calibration | Birth-ACF calibration |
|---|---:|---:|
| Birth-ACF RMSE, lags 1--15 | 0.07645 | **0.01114** |
| q-ACF RMSE, lags 1--15 | **0.02428** | 0.07883 |
| Mean birth-curve RMSE, generations 0--20 | **0.08861** | 0.75812 |
| Generation-20 birth-size change | 0.49152 | 1.66906 |
| Comparable log-likelihood change | -- | -35,105.99 |

The observed generation-20 change is 0.58225. The ACF-only calibration therefore
matches the targeted curve very well but produces almost three times too much
mean drift.

Marginal checks show the same trade-off:

| Variable | Observed mean | Likelihood-simulated mean | ACF-simulated mean | ACF-fit KS distance |
|---|---:|---:|---:|---:|
| Birth size | 2.5583 | 2.5632 | 3.2602 | 0.5395 |
| q | -0.5193 | -0.5222 | -0.6041 | 0.4581 |
| alpha | 0.5436 | 0.5401 | 0.4373 | 0.4435 |
| Division time | 1.1047 | 1.0652 | 1.5061 | 0.5889 |

The comparable log-likelihood loss decomposes into -1,809.90 for q, -2,367.60
for alpha, -30,907.42 for the timing hazard, and -21.07 for initial births.

The fit uses a nearly non-decaying slow q mode (`rho_slow=0.99268`, close to its
0.995 bound), a stronger negative fast mode (`rho_fast=-0.43192`), and much
larger environmental variation. The RMS of the fitted `E_i` values is 0.04380,
compared with the likelihood-fit environmental prior SD 0.01431. It also moves
the hazard from `(u, omega2, eta)=(1.449, 11.156, 8.516)` to
`(2.546, 18.456, 16.507)`.

## Interpretation

The observed birth ACF is reachable within this model family, but only because
an ACF-only objective can exploit weakly identified parameters and an arbitrary
47-dimensional environmental path. The severe failures in drift, marginals,
q-ACF, and likelihood show that this is not a viable generative fit. A useful
next calibration would be a likelihood or moment-constrained ACF objective,
rather than treating the birth ACF as the sole statistic.

Reproduce with:

```bash
python scripts/fit_all_parameters_birth_acf_glu.py \
  --training-replicates 6 --validation-replicates 500 --iterations 180
```
