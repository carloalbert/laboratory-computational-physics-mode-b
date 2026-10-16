# Glu q-environment model with explicit birth-size Hill feedback

The q process remains $q_{\ell i}=A_\ell+E_i+r_{\ell i}$, with every
$A_\ell$ and $E_i$ inferred. The alpha-environment $G_i$ is removed.
The timing hazard is

$$h(t\mid b,\alpha)=\alpha\omega_2\,\operatorname{sigmoid}\!\left[
\eta\log\frac{b(e^{\alpha t}-1)}{u}+\kappa\log\frac{b}{b_\star}\right].$$

The nested protein-only model has $\kappa=0$. The fixed $b_\star$ is the
geometric mean birth size in the fitted transition panel.

## Hazard fit

- u: 1.446644
- omega2: 11.219700
- eta: 8.589284
- kappa: -0.006149
- b_star: 2.496167
- Twice log-likelihood gain over kappa=0: 0.001
- Delta BIC (extended minus nested): 8.686

## Posterior checks

| Metric | Protein-only Hill | Explicit-b Hill |
|---|---:|---:|
| Mean-b trajectory RMSE | 0.09218 | 0.09201 |
| Alpha-tau trajectory RMSE | 0.00981 | 0.00990 |
| Tau trajectory RMSE | 0.05888 | 0.05940 |
| Raw b-ACF RMSE | 0.07697 | 0.07703 |
| Demeaned b-ACF RMSE | 0.06152 | 0.06180 |
| Tau-ACF RMSE | 0.02896 | 0.02880 |
| Birth KS | 0.02246 | 0.02185 |
| Tau KS | 0.16771 | 0.16741 |

## Interpretation

The fitted explicit-birth coefficient is kappa=-0.00615, essentially zero.
The likelihood-ratio statistic is only 0.0013, and adding kappa raises BIC by
8.686. The posterior checks are correspondingly indistinguishable from the
protein-only hazard. The protein variable $p=b(\exp(\alpha t)-1)$ already
contains birth size; these data do not support an additional static birth-size
exponent. More useful timing flexibility must therefore be time/cycle dependent
or affect another part of the hazard.
