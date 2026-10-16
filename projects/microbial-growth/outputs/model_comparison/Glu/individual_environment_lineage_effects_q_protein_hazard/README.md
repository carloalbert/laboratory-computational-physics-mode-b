# Glu model with individually inferred lineage and environment effects

The effective jump is parameterized as $q_{\ell i}=A_\ell+E_i+r_{\ell i}$.
Here $A_\ell\sim\mathcal N(\mu_A,\sigma_A^2)$ is the absolute lineage mean
(so the former $\mu_q$ is absorbed into $A_\ell$),
$E_i\sim\mathcal N(0,\sigma_E^2)$ is shared by all lineages at cycle $i$,
and $r_{\ell i}$ is a stationary Gaussian AR(1). At fixed fitted
hyperparameters the complete vector of all 303 lineage effects and 47
cycle effects has an exact joint Gaussian posterior. The tables in this
directory report every effect and its posterior uncertainty.

For display, the gauge is chosen so $\sum_i n_iE_i=0$, with $n_i$ the
number of observed transitions at cycle $i$. Shifting all $A_\ell$ by a
constant and all $E_i$ by its negative leaves every $A_\ell+E_i$ unchanged.

## Posterior predictive checks

`conditional A E` draws the fitted effects jointly from their posterior,
uses each measured initial birth size and lineage length, but generates new
AR residuals, growth rates, and division times. `new lineages` draws new $A_\ell$
and initial sizes, while retaining posterior uncertainty in the shared environment.

| Check | New lineages | Conditional A,E |
|---|---:|---:|
| Raw b-ACF RMSE | 0.07749 | 0.07653 |
| Demeaned b-ACF RMSE | 0.06184 | 0.06142 |
| q-ACF RMSE | 0.03127 | 0.03004 |
| Mean-b trajectory RMSE | 0.08810 | 0.09598 |
| Birth KS | 0.02357 | 0.02678 |
| q KS | 0.08134 | 0.07917 |
| alpha KS | 0.10999 | 0.10958 |
| tau KS | 0.16798 | 0.16677 |

The conditional check is deliberately in-sample for the persistent
effects and initial sizes. Its improvement measures how much information
the fitted identities contain; only the new-lineage check measures
unconditional generalization.

## Result

The inferred $A_\ell$ values do track the observed lineage mean q (correlation 0.883), but their fitted between-lineage SD is only 0.0157, versus a residual stationary SD of 0.0781. Conditioning on every fitted effect therefore changes the raw b-ACF RMSE only from 0.07749 to 0.07653; the demeaned b-ACF RMSE changes from 0.06184 to 0.06142.

Only 2 of 15 nonzero raw-ACF lags and 4 of 15 demeaned-ACF lags lie in the conditional 95% posterior predictive bands. Thus the failure is not caused by having drawn new
lineage offsets in the earlier check: an individually inferred static
offset shifts a lineage's size setpoint, but the hazard feedback still
removes birth-size memory too quickly.
