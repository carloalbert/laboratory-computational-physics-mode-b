# Joint alpha-q marginal model for Glu

This directory contains the fit and autonomous posterior-predictive-style
diagnostics for a directional joint model of the growth rate `alpha` and the
effective logarithmic division jump `q`.

The effective jump is decomposed into two independent stationary Gaussian
AR(1) modes,

```text
q[l,i] = mu_q + r_fast[l,i] + r_slow[l,i],
r_k[l,i] = rho_k r_k[l,i-1] + sigma_k z_k[l,i].
```

The empirical marginal distribution of `alpha` is represented by a
rank-Gaussian score `z_alpha`.  Conditional on the q path, that score follows
the directional ARX(2) model

```text
z_alpha[l,i] = c
             + phi_1 z_alpha[l,i-1] + phi_2 z_alpha[l,i-2]
             + beta_0 (q[l,i]   - mu_q)
             + beta_1 (q[l,i-1] - mu_q)
             + beta_2 (q[l,i-2] - mu_q)
             + sigma_alpha e[l,i].
```

Thus the joint path density factorizes as

```text
p(q[0:n], alpha[0:n]) = p(q[0:n]) p(alpha[0:n] | q[0:n]).
```

The simulated Gaussian score is mapped back through the empirical alpha
quantile function.  This preserves the observed non-Gaussian alpha marginal
without adding a fragile parametric tail model.  The birth-size process is
then simulated autonomously with the fitted protein Hill hazard and

```text
log b[l,i+1] = log b[l,i] + alpha[l,i] tau[l,i] + q[l,i].
```

No measured alpha, q, or division-time values are replayed.  Only the observed
contiguous-prefix lineage lengths are reused.

Files:

- `fit_results.json`: fitted parameters and numerical goodness-of-fit summaries.
- `alpha_acf.csv`, `q_acf.csv`: observed and simulated marginal ACFs.
- `alpha_q_cross_correlation.csv`: observed and simulated lagged cross-correlation.
- `birth_size_acf.csv`: resulting autonomous birth-size ACF.
- `joint_alpha_q_diagnostics.{png,pdf}`: combined diagnostic figure.
