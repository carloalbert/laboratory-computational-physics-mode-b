# Glu autonomous AR-residual fit

This directory contains the maximum-likelihood fit and 300 autonomous
fixed-parameter simulations from `fit_autoregressive_residual_glu.py`. No
measured `f`, `alpha`, `epsilon`, `tau`, or initial birth-size paths are replayed.
Only the observed contiguous-prefix lineage lengths are reused so that the
short-series ACF estimator has the same sampling design.

- `fit_results.json`: fitted hazard, initial-size, AR-marginal, and beta
  regression parameters plus simulation checks.
- `birth_size_acf.csv`: observed and simulated mean within-lineage ACF values,
  simulation intervals, and the number of contributing lineages at every lag.
- `birth_size_acf_autonomous.{png,pdf}`: the principal ACF comparison.
- `reconstructed_transitions.csv`: consecutive observed transitions with the
  actual daughter-row division fraction and reconstructed endpoint residual.
- `autonomous_example_trajectories.csv`: three simulation replicates for 20
  lineages, retained as a recursion audit.

The simulation intervals describe finite-data process variation at the fitted
point estimate; they do not include parameter uncertainty. The untruncated
Student-t residual generated nonpositive endpoint log-growth in only 0.0042% of
simulated transitions.
