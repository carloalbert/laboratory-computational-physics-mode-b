# Iid recorded fractions with shared cycle epsilon

This experiment tests the literal model

```text
q[l,i] = log f[l,i] + epsilon[i],
f[l,i] ~ Beta(a,b),
epsilon[i] ~ Normal(mu_epsilon, sigma_environment),
```

where `epsilon[i]` is inferred once per generation and shared by all lineages.
The recorded fractions are aligned to the daughter row of each transition.

The equality is not exact in the observations: `q-log(f)` has substantial
within-generation scatter. A normal observation model is therefore used only
to infer each shared epsilon. In accordance with the requested strict model,
that observation scatter is not propagated as a third q component in the
simulations.

The model does not improve the Glu birth-size ACF. The inferred shared epsilon
decreases over generations, while the observed increase in q is primarily in
the generation mean of the recorded log fractions. Drawing fractions iid
removes that trend.

Files:

- `fit_results.json`: beta fit, epsilon posterior, and predictive metrics.
- `epsilon_posterior.csv`: raw generation means and posterior shared effects.
- `balanced_generation_means.csv`: observed and simulated size drift.
- `birth_size_acf.csv`, `q_acf.csv`: numerical ACF comparisons.
- `shared_cycle_epsilon_diagnostics.{png,pdf}`: combined diagnostic figure.
