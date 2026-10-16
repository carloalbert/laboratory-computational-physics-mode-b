# Source modules

## Current Glu workflow

- [static_offset_growth_hazard.py](static_offset_growth_hazard.py): four-parameter
  Hill hazard plus one static Gaussian lineage offset; marginal likelihood,
  analytic gradients, quadrature, and bounded optimization.
- [glu_workflow.py](glu_workflow.py): shared Glu data filtering, fixed validation
  IDs, configuration access, and correlation helper.
- [growth_plotting.py](growth_plotting.py): shared plotting panel and portable
  font selection.
- [autoregressive_residual_model.py](autoregressive_residual_model.py):
  exploratory autonomous model with Gamma-copula AR growth rates, Student-t
  copula AR endpoint residuals, actual beta-distributed fractions, and exact
  thinning for the protein Hill hazard.

The balanced-cohort and detrending helpers live in
[fit_empirical_detrended_hazard.py](../scripts/fit_empirical_detrended_hazard.py).
The [project README](../README.md) is the authoritative model description and
reproduction guide. Current entry points no longer import historical runners or
read discarded experiment outputs.

## Historical modules

Other modules preserve the earlier endpoint-residual, hierarchical,
decaying-memory, stationary-memory, empirical birth-transition, and conditional
timing-density implementations. They are research history, not a specification
of the current Glu model. Nine-sugar endpoint-residual fits are retained as
reference results.

Historical modules and scripts remain in their original locations to avoid
breaking their import relationships. Regenerating an old analysis may require
running its prerequisite fits first.
