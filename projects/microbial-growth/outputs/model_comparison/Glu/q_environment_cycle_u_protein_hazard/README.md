# Glu q environment plus cycle-specific hazard threshold

The q model remains $q_{\ell i}=A_\ell+E_i+r_{\ell i}$, with every
$A_\ell$ and $E_i$ inferred. There is no alpha $G_i$ process and no
static explicit-birth coefficient $\kappa$. Instead,

$$h_i(t\mid b,\alpha)=\alpha\omega_2\,\operatorname{sigmoid}\!\left[
\eta\log\frac{b(e^{\alpha t}-1)}{u_i}\right],$$

$$\log u_i\sim\mathcal N(\mu_u,\sigma_u^2).$$

All $u_i$ are integrated out by Gauss-Hermite quadrature when fitting
$(\mu_u,\sigma_u,\omega_2,\eta)$, then each posterior $p(u_i\mid\mathrm{data})$
is recovered and propagated through simulation.

## Fit

- Population geometric-mean threshold: 1.460256
- Between-cycle SD of log u: 0.026966
- omega2: 11.291453
- eta: 8.623030
- Twice marginal log-likelihood gain: 18.357
- Delta BIC (cycle-u minus shared-u): -9.670

## Posterior checks

| Metric | Shared u | Inferred u_i |
|---|---:|---:|
| Mean-b trajectory RMSE | 0.09083 | 0.06385 |
| Alpha-tau trajectory RMSE | 0.00978 | 0.00629 |
| Tau trajectory RMSE | 0.05875 | 0.05720 |
| Birth-ACF RMSE | 0.07693 | 0.06874 |
| Tau-ACF RMSE | 0.02880 | 0.02888 |
| Birth KS | 0.02161 | 0.02145 |
| Tau KS | 0.16719 | 0.16953 |

## Interpretation

The fitted log-threshold SD corresponds to about 2.7% multiplicative
cycle-to-cycle variation. Relative to the refitted shared-u comparator,
inferred $u_i$ reduce the mean-birth trajectory RMSE by 29.7% and the birth-ACF
RMSE by 10.6%. The birth marginal is essentially unchanged, and the
division-time marginal does not improve.

This is an in-sample empirical-Bayes diagnostic: the same division-time data
identify each $u_i$ and assess its posterior replay. A held-out-cycle or
future-cycle check would be required to establish genuine predictive
improvement.
