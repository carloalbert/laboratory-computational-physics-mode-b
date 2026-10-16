# Likelihood plus birth-ACF regularization

This analysis performs penalized-MAP inference for the complete autonomous Glu
model. All population parameters and all 47 cycle-specific environmental effects
are calibrated. Unlike the ACF-only experiment, every environmental effect has
its Gaussian prior penalty.

The objective is

\[
J_\lambda(\theta,E)=
\frac{\ell(\hat\theta_{\rm ML},\hat E_{\rm ML})-\ell(\theta,E)}{N}
+\lambda\frac{\operatorname{MSE}_{b\text{-ACF}}(\theta,E)}
{\operatorname{MSE}_{b\text{-ACF}}(\hat\theta_{\rm ML},\hat E_{\rm ML})}.
\]

The likelihood contains the q sequence density and environmental prior, the
Gamma-copula alpha process, the protein Hill timing density, and the initial-size
density. The pure-likelihood q model was refitted to all contiguous Glu sequences
under the same 47-effect representation before applying the penalty.

Four weights were fitted. The weakest value, `lambda=0.03`, is the most defensible
regularizer; larger values expose the trade-off but no longer behave as a mild
perturbation.

| Weight | Birth-ACF RMSE | q-ACF RMSE | Mean-size RMSE | Gen.-20 change | Lost log density / datum |
|---:|---:|---:|---:|---:|---:|
| 0 (likelihood) | 0.07628 | 0.02259 | 0.09914 | 0.4670 | 0 |
| 0.03 | 0.06017 | 0.04737 | 0.14305 | 0.6833 | 0.00870 |
| 0.10 | 0.04497 | 0.04588 | 0.14474 | 0.9655 | 0.02720 |
| 0.30 | 0.02749 | 0.10092 | 0.15180 | 1.0756 | 0.06020 |
| 1.00 | 0.01995 | 0.11134 | 0.13573 | 1.1333 | 0.09681 |

The observed generation-20 change is 0.5823. Even weak ACF regularization moves
the endpoint drift through the observed value and then overshoots rapidly.

## Weak regularization (`lambda=0.03`)

The largest changes relative to the pure likelihood fit are:

| Quantity | Likelihood | Regularized | Change |
|---|---:|---:|---:|
| fast q AR coefficient | -0.0805 | -0.1990 | -0.1185 |
| RMS of inferred E path | 0.01009 | 0.01885 | +86.8% |
| environmental prior SD | 0.01009 | 0.01406 | +39.3% |
| stationary slow-q SD | 0.02808 | 0.03637 | +29.5% |
| hazard eta | 8.516 | 7.989 | -6.2% |
| alpha AR coefficient | 0.5175 | 0.5492 | +6.1% |
| hazard omega2 | 11.156 | 11.687 | +4.8% |

The inferred E path correlation with the likelihood path is 0.770. The alpha
marginal and timing parameters remain comparatively stable. The q component
accounts for 152.8 of the total 161.9 log-density loss, so the birth ACF chiefly
pulls on q memory and environmental heterogeneity rather than on alpha or the
division hazard.

At this weak weight the birth-ACF error improves by 21%, but q-ACF RMSE more than
doubles. Birth marginal KS distance rises from 0.024 to 0.079; q KS distance
rises from 0.080 to 0.099. Alpha remains almost unchanged, and the division-time
KS distance improves slightly from 0.167 to 0.150.

## Interpretation

The stable response across the path is stronger generation-specific variation
and a substantially more negative fast q correlation. These changes improve the
birth ACF, but they conflict with the directly observed q ACF and induce excessive
mean-size drift. Therefore the birth-ACF discrepancy cannot be repaired cleanly
by moving within the current parameter family: even a genuine likelihood
regularizer reveals tension between the birth ACF and the observed q process.

The folders `weight_0.03`, `weight_0.1`, `weight_0.3`, and `weight_1` contain all
parameters, component likelihoods, marginals, drift curves, and fresh-simulation
diagnostics.
