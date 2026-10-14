# Reference summaries and empirical diagnostics

These outputs describe the historical nine-sugar endpoint-residual analysis and
selected empirical checks. They are **not** results from the current Glu
static-offset model; that model's outputs live
[here](../model_comparison/Glu/static_offset_detrended_hazard/).

Retained summary families:

- Cross-sugar tables/figures named all_sugars_*, the parameter scatter, and
  run configuration.
- Selected parameter relationships.
- Lineage half-size persistence.
- Observed Glu birth-size ACF and outlier diagnostics.
- [Glu ACF heterogeneity/transients](Glu_acf_heterogeneity_transients/).
- [Glu interdivision-time drift](Glu_interdivision_time_drift/).

The reference endpoint-residual summary can be regenerated from the repository
root with:

~~~bash
python projects/microbial-growth/scripts/summarize_loglogistic_endpoint_residual.py
~~~

Superseded hierarchical/latent-setpoint replay and ACF-versus-threshold artifacts
were removed; see the [cleanup manifest](../../docs/cleanup_manifest.json).
The [project README](../../README.md) describes the retained current workflow.
