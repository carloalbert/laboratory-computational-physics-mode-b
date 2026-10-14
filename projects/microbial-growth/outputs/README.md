# Retained outputs

## Current Glu analysis

- [Static-offset detrended hazard](model_comparison/Glu/static_offset_detrended_hazard/):
  the current five-population-parameter model, held-out scores, posterior offsets,
  conditional replay metrics, example trajectories, and figures.
- [Shared detrended hazard](model_comparison/Glu/empirical_detrended_hazard/):
  the four-parameter comparator, empirical means, detrended transitions, and
  replay checks. Run this first when rebuilding the current analysis.

See the [project README](../README.md) for equations, quantitative conclusions,
limitations, and commands. Population-offset replay and full-record
posterior-offset replay have different predictive interpretations.

## Historical reference

The loglogistic_endpoint_residual directory for each of the nine sugars is
retained, together with cross-sugar [summaries](summary/) and selected empirical
diagnostics. These are reference results, not the current Glu model.

Superseded Glu experimental output directories and associated summary artifacts
were removed. The [cleanup manifest](../docs/cleanup_manifest.json) records exact
paths and the temporary recovery copy outside the repository. Original data and
historical implementation code were not removed.
