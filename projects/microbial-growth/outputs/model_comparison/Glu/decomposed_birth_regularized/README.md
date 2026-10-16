# Decomposed mean-drift and residual-ACF fit for Glu

The fit separates the common generation-wise birth-size trajectory from
within-generation lineage fluctuations. The objective retains the complete
likelihood, targets the balanced-cohort mean birth-size curve, and applies
the ACF penalty only after subtracting the common mean at every generation.

All posterior-predictive models use the same conditional posterior
covariance for the inferred environmental path `E`; no model uses the old
near-zero covariance placeholder.

## Validation metrics

| Model | Raw b-ACF RMSE | Residual b-ACF RMSE | Mean-b RMSE | q-ACF RMSE | Gen.-20 change | Log-density loss / datum |
|---|---:|---:|---:|---:|---:|---:|
| likelihood | 0.07604 | 0.06273 | 0.10007 | 0.02271 | 0.4667 | -0.00000 |
| E only | 0.06277 | 0.06061 | 0.02250 | 0.03385 | 0.5389 | 0.01096 |
| full | 0.06389 | 0.05771 | 0.03657 | 0.04368 | 0.4958 | 0.01379 |

The observed generation-20 change is 0.5823.

`E only` changes only the inferred shared cycle path. `full` starts from
that solution and then relaxes all population parameters.

## Interpretation

Changing only `E` reduces mean-trajectory RMSE by 77.5% and raw birth-ACF RMSE
by 17.5%. The residual-ACF RMSE changes much less because a generation-shared
effect largely cancels when the generation mean is removed.

The fully relaxed solution does not improve the raw birth ACF or the mean
trajectory relative to the `E only` solution, and loses more likelihood. It is
therefore not preferred. The remaining ACF mismatch is in the
generation-demeaned lineage fluctuations rather than in the common mean drift
alone.
