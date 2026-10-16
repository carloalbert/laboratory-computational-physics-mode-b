# Generation-shared random-walk q model for Glu

This experiment modifies the original factorized model only in its q process:

```text
q[l,i] = mu_q + m[i] + r[l,i],
m[i+1] = m[i] + delta + sigma_m z[i],
r[l,i+1] = rho_f r[l,i] + sigma_f e[l,i].
```

The latent `m[i]` is drawn once per replicate and shared by every lineage.
Growth rates retain the original Gamma-marginal Gaussian-copula AR(1), and
division times retain the original protein Hill hazard. Birth size obeys the
same exact recursion.

The shared process is fitted to the generation means of q in the balanced
172-lineage, 20-transition Glu cohort. Its innovation standard deviation fits
at essentially zero, while its drift is positive. Thus the selected model is
effectively a common linear generation trend rather than a noisy random walk.

The shared trend improves the mean birth-size trajectory and modestly improves
the birth-size ACF. Removing the original slow lineage component worsens the
q-ACF, however, so a complete descriptive model would need both the shared
trend and a smaller lineage-specific slow component.

Files:

- `fit_results.json`: fitted parameters and numerical comparisons.
- `balanced_generation_means.csv`: observed, original-model, and shared-walk
  mean birth sizes by generation.
- `generation_mean_q.csv`: observed and fitted common q path.
- `birth_size_acf.csv`, `q_acf.csv`: numerical ACF checks.
- `shared_random_walk_q_diagnostics.{png,pdf}`: combined diagnostic figure.
