# Analysis entry points

## Current workflow

Run from the repository root. Install the project's minimal requirements first.

| Script | Role |
|---|---|
| fit_empirical_detrended_hazard.py | Prepare balanced cohort and exact means; fit shared comparator |
| fit_static_offset_detrended_hazard.py | Fit static lineage offsets and perform validation/replays |
| plot_empirical_detrended_hazard.py | Plot shared-comparator checks |
| plot_static_offset_detrended_hazard.py | Plot current-model checks |
| test_glu_workflow.py | Check data/split invariants and independence from historical runners |
| test_*detrended*.py | Test detrending, likelihoods, gradients, and offset behavior |

Run the comparator before the static-offset fit. Both use the fixed
[workflow configuration](../config/glu_current_workflow.json); no retired output
bundle is required.

The [project README](../README.md) provides exact commands, equations, results,
and limitations.

## Historical experiments

Other scripts are retained as research provenance, including fitting, replay,
diagnostic, plotting, and test entry points. Some expect outputs from other
historical scripts and require those prerequisite fits to be regenerated.
Their generated Glu output directories were pruned; they are not dependencies
of the current workflow. See the project's compact earlier-attempts list.
