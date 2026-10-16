# Hazard-free joint q-alpha-tau model for Glu

This directory contains an exploratory joint statistical model of the three
transition variables. It does not use a division hazard.

Let

```text
x[l,i] = log b[l,i],
q[l,i] = x[l,i+1] - x[l,i] - alpha[l,i] tau[l,i].
```

The effective jump has fast and slow Gaussian AR(1) components,

```text
q[l,i] = mu_q + r_fast[l,i] + r_slow[l,i],
r_k[l,i] = rho_k r_k[l,i-1] + sigma_k xi_k[l,i].
```

Alpha and tau retain their empirical marginal distributions. Their normal
scores are

```text
w[l,i] = (Phi^-1(F_alpha(alpha[l,i])),
          Phi^-1(F_tau(tau[l,i]))).
```

Conditional dependence is represented by

```text
w[l,i] = c
       + B1 w[l,i-1] + B2 w[l,i-2]
       + D0 (q[l,i]   - mu_q)
       + D1 (q[l,i-1] - mu_q)
       + D2 (q[l,i-2] - mu_q)
       + K (x[l,i] - m[l])
       + e[l,i].
```

Here `m[l]` is a random lineage size set point. Paired residual vectors `e`
are sampled from the empirical fitted residual distribution, retaining their
non-Gaussian shape and alpha-tau dependence. The exact size identity closes
the autonomous simulation:

```text
x[l,i+1] = x[l,i] + q[l,i] + alpha[l,i] tau[l,i].
```

The resulting factorization is

```text
p(m[l]) p(q[l,0:n]) p(alpha[l,0:n], tau[l,0:n] | q[l,0:n], x[l,0:n], m[l]).
```

Only observed lineage lengths are reused in the diagnostics. No measured
transition trajectory is replayed.

Files:

- `fit_results.json`: parameters and numerical predictive diagnostics.
- `*_acf.csv`: observed and simulated marginal ACFs.
- `*_cross_correlation.csv`: all three pairwise lagged cross-correlations.
- `birth_size_acf.csv`: birth-size ACF implied by the exact recurrence.
- `joint_q_alpha_tau_diagnostics.{png,pdf}`: combined diagnostic figure.
