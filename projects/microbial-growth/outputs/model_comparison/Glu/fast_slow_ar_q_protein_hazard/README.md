# Glu fast + slow AR(1) q model

This model uses

\[
\log b_{\ell,i+1}=\log b_{\ell i}+\alpha_{\ell i}\tau_{\ell i}+q_{\ell i},
\qquad
q_{\ell i}=\mu_q+r^{(f)}_{\ell i}+r^{(s)}_{\ell i},
\]

with independent stationary Gaussian fast and slow AR(1) components. The exact
Gaussian lineage likelihood profiles the shared mean and resets both components
only across observed generation gaps.

The 300 autonomous simulations draw all initial sizes, growth-rate paths,
division times, and both q paths anew. Only contiguous-prefix lineage lengths
are reused. Bands are fixed-MLE process intervals and exclude parameter
uncertainty.

- `fit_results.json`: fitted parameters, likelihood comparison with the static
  offset model, and diagnostic summaries.
- `birth_size_acf.csv` and `q_acf.csv`: numerical ACF comparisons.
- `birth_and_q_acf_autonomous.{png,pdf}`: comparison with observations and the
  static-offset q model.
- `autonomous_example_trajectories.csv`: three replicates for 20 lineages.
- `mean_size_drift/`: balanced-cohort and paired-lineage checks of the
  generation-dependent mean size drift. These show rapid simulated relaxation
  followed by a plateau, rather than the slower continuing drift in the data.
