# Summary outputs

This folder contains the aggregate results from the current re-parameterized
log-logistic endpoint-residual model for all nine sugars.  The detailed files
for each fit remain under
`outputs/model_comparison/<sugar>/loglogistic_endpoint_residual/`.

- `all_sugars_diagnostics.csv`: convergence, re-parameterization correlations,
  and observation-matched lag-1 ACF coverage.
- `all_sugars_parameter_modes.csv`: one row per sugar and one column per sampled
  or derived parameter.
- `all_sugars_posterior_summary.csv`: long-form posterior modes, means, medians,
  standard deviations, and 95% credible intervals.  The physical scale `u` is
  included as a derived parameter.
- `all_sugars_predictive_summary.csv`: long-form summaries of the 50 replicated
  posterior-predictive diagnostics per sugar.
- `sugar_parameter_scatter.png`: posterior medians and 95% credible intervals.
- `all_sugars_acf_lag1.png`: observed lag-1 ACFs against the 500-replicate,
  observation-matched posterior-predictive intervals.
- `all_sugars_acf_sheet.png` and `.pdf`: the complete division-time and
  birth-length ACF curves for every sugar on one sheet.
- `all_sugars_acf_pairing_comparison.csv`: lag-1 predictive intervals before
  and after preserving the exact observed `(b0, lineage length)` pairing.
- `all_sugars_acf_four_variables.png` and `.pdf`: division-time, birth-length,
  growth-rate, and division-factor ACFs for every sugar.
- `all_sugars_acf_four_variables_lag1.csv`: observed and predictive lag-1 ACF
  summaries for those four variables.
- `run_config.json`: common sampler settings and the expected per-sugar files.

Regenerate this folder after completing all nine fits with:

```bash
python scripts/summarize_loglogistic_endpoint_residual.py
```
