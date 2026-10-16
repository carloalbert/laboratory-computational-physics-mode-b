# Glu lineage-offset AR(1) q model

This fit replaces the separate physical-fraction and endpoint-residual laws by

\[
\log b_{\ell,i+1}=\log b_{\ell i}+\alpha_{\ell i}\tau_{\ell i}+q_{\ell i},
\qquad
q_{\ell i}=\mu_q+a_\ell+r_{\ell i},
\]

where the lineage offset is Gaussian and the residual is a stationary Gaussian
AR(1). The likelihood integrates the offset exactly over each full lineage and
resets only the AR residual across observed generation gaps.

The 300 autonomous simulations draw new initial sizes, alpha paths, division
times, q offsets, and q residual paths. Only contiguous-prefix lineage lengths
are retained as an observation design. Simulation bands are fixed-MLE process
intervals and do not include parameter uncertainty.

- `fit_results.json`: model definition, fitted parameters, likelihood evidence,
  and marginal/ACF summaries.
- `birth_size_acf.csv` and `q_acf.csv`: numerical ACF comparisons.
- `birth_and_q_acf_autonomous.{png,pdf}`: principal diagnostic figure.
- `autonomous_example_trajectories.csv`: three replicates for 20 lineages.
