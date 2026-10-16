# Glu q and alpha environmental effects

This model retains $q_{\ell i}=A_\ell+E_i+r_{\ell i}$ and adds
$z^{(\alpha)}_{\ell i}=G_i+x_{\ell i}$ with
$\alpha_{\ell i}=F_\Gamma^{-1}[\Phi(z^{(\alpha)}_{\ell i})]$.
Every $A_\ell$, $E_i$, and $G_i$ is inferred. The constraint
$\sigma_G^2+\sigma_x^2=1$ preserves the fitted Gamma marginal on the
unconditional latent-normal scale.

## Fitted alpha process

- Shared alpha-environment SD on the latent scale: 0.09880
- Residual alpha AR coefficient: 0.51381
- Residual stationary SD: 0.99511

## Posterior checks

| Metric | Stationary alpha | Inferred G_i |
|---|---:|---:|
| Alpha trajectory RMSE | 0.01904 | 0.01165 |
| Tau trajectory RMSE | 0.05875 | 0.03768 |
| Alpha-tau trajectory RMSE | 0.00990 | 0.00995 |
| Mean-b trajectory RMSE | 0.09711 | 0.09719 |
| Alpha ACF RMSE | 0.02909 | 0.02882 |
| Birth ACF RMSE | 0.07673 | 0.07735 |
| Alpha KS | 0.11025 | 0.11252 |
| Tau KS | 0.16674 | 0.16768 |

## Interpretation

The inferred $G_i$ path can reproduce the growth-rate drift and thereby
change the division-time trajectory. It cannot repair the growth-increment
or birth-size trajectories in this hazard model: simulation samples
$s=\alpha\tau$ directly from a distribution depending on birth size and
the hazard parameters, after which $\tau=s/\alpha$. Consequently changing
$\alpha$ leaves both $s$ and $b_{\ell,i+1}=b_{\ell i}\exp(s+q)$ invariant
apart from Monte Carlo differences between runs.
