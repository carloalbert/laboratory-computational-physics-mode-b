# Glu shared-environment plus lineage-offset q model

The proposed process is $q_{\ell i}=\mu_q+E_i+A_\ell+r_{\ell i}$,
where $E_i$ is shared across lineages, $A_\ell$ is a Gaussian lineage
offset integrated out of the likelihood, and `r` is one stationary AR(1).
The comparator is the current shared-environment fast-plus-slow AR model,
refitted to the same variable-length contiguous q sequences.

## Fitted q parameters

- Mean q: -0.513679
- Environmental prior SD: 0.012219
- Lineage-offset SD: 0.021078
- Fast AR coefficient: -0.031899
- Fast stationary SD: 0.078068
- Offset fraction of total q variance: 0.066

## Autonomous validation

| Model | Raw b-ACF RMSE | Demeaned b-ACF RMSE | q-ACF RMSE | Mean-b RMSE | Birth KS | q KS |
|---|---:|---:|---:|---:|---:|---:|
| current fast slow | 0.07579 | 0.06245 | 0.02244 | 0.09940 | 0.0257 | 0.0799 |
| proposed offset | 0.07732 | 0.06176 | 0.03152 | 0.08771 | 0.0232 | 0.0811 |

Proposed minus current q joint log density: -19.800.
AIC(proposed) - AIC(current): 37.599 (negative favors proposed).

The lineage-slope comparison uses birth sizes after removing the
generation-wise common mean in every observed or simulated replicate.

## Interpretation

The lineage offset is clearly nonzero and the remaining AR process is
essentially iid. Nevertheless, replacing the slow AR mode by an independent
lineage offset does not improve the raw birth-size ACF. It slightly improves
the generation-demeaned ACF and mean-size trajectory, but worsens the q-ACF and
is disfavored by both joint q density and AIC.

A static q offset mostly shifts the lineage's size setpoint under the hazard
feedback; the per-lineage centering in the ACF then removes much of its effect.
The observed slope heterogeneity therefore cannot be represented adequately by
an independent q intercept alone.
