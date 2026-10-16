# Glu q environment plus cycle-specific omega2

The q model remains $q_{\ell i}=A_\ell+E_i+r_{\ell i}$. There is no
alpha $G_i$ process and no explicit-birth coefficient $\kappa$. The hazard is

$$h_i(t\mid b,\alpha)=\alpha\omega_{2,i}\,\operatorname{sigmoid}\!\left[
\eta\log\frac{b(e^{\alpha t}-1)}{u}\right],$$

$$\log\omega_{2,i}\sim\mathcal N(\mu_{\log\omega_2},\sigma_{\log\omega_2}^2).$$

Each cycle rate is integrated by Gauss-Hermite quadrature during hyperparameter fitting,
then its posterior is recovered and propagated through simulation.

## Fit

- Shared threshold u: 1.462551
- Population geometric-mean omega2: 11.088140
- Between-cycle SD of log omega2: 0.147305
- eta: 8.460331
- Twice marginal log-likelihood gain: 59.873
- Delta BIC: -51.186

## Posterior checks

| Metric | Shared omega2 | Inferred omega2_i |
|---|---:|---:|
| Mean-b trajectory RMSE | 0.09115 | 0.03836 |
| Alpha-tau trajectory RMSE | 0.00999 | 0.00411 |
| Tau trajectory RMSE | 0.05989 | 0.05652 |
| Birth-ACF RMSE | 0.07695 | 0.06255 |
| Tau-ACF RMSE | 0.02878 | 0.02879 |
| Birth KS | 0.02088 | 0.01527 |
| Tau KS | 0.16707 | 0.17169 |

## Interpretation

Mean-birth trajectory RMSE changes by 57.9% and birth-ACF RMSE by 18.7%
relative to the refitted shared-rate model (positive means improvement). This is an
in-sample empirical-Bayes diagnostic; held-out cycles are needed for a predictive test.
