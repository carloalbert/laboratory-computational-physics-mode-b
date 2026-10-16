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
| fit_autoregressive_residual_glu.py | Exploratory autonomous fit with AR growth rates and endpoint residuals, actual beta-distributed fractions, and the protein Hill hazard |
| test_autoregressive_residual_model.py | Unit checks for the autonomous model's hazard, fraction law, and short-lineage ACF |
| analyze_glu_transition_acf_scales.py | Compare one- and two-scale ACF descriptions of observed Glu q, alpha, tau, and epsilon |
| fit_lineage_offset_ar_q_glu.py | Autonomous Glu fit replacing f and epsilon by an AR(1) effective jump q with a lineage random offset |
| test_lineage_offset_ar_q.py | Check the integrated random-offset AR(1) likelihood and gap handling |
| fit_fast_slow_ar_q_glu.py | Autonomous Glu fit with independent fast and slow Gaussian AR(1) modes for q |
| test_fast_slow_ar_q.py | Check the two-mode Gaussian likelihood and gap handling |
| fit_joint_alpha_q_glu.py | Directional empirical-copula alpha and two-mode q joint marginal, autonomous simulation, and joint diagnostics |
| test_joint_alpha_q.py | Check the alpha-q cross-correlation lag convention |
| fit_joint_q_alpha_tau_glu.py | Hazard-free, error-correcting semiparametric joint model for q, alpha, and tau, with autonomous birth-size reconstruction |
| test_joint_q_alpha_tau.py | Check the empirical marginal transform and cross-correlation convention |
| check_factorized_model_mean_size_drift.py | Compare observed and autonomous factorized-model mean birth-size drift in balanced and paired cohorts |
| fit_shared_random_walk_q_glu.py | Replace the original model's slow lineage q mode by a generation-shared random walk and check drift and ACFs |
| fit_shared_cycle_epsilon_glu.py | Test iid recorded fractions plus Bayesian shared cycle-specific epsilon effects in the original hazard model |
| fit_shared_cycle_environment_q_glu.py | Infer one iid-prior environmental q effect per cycle and use it jointly across all lineages in posterior simulation |
| fit_environment_fast_slow_ar_q_glu.py | Jointly fit shared cycle-specific q effects and fast/slow lineage AR(1) q modes, then run autonomous drift and ACF checks |
| test_environment_fast_slow_ar_q.py | Verify the structured joint q likelihood against the equivalent full multivariate Gaussian |
| fit_all_parameters_birth_acf_glu.py | Calibrate every autonomous-model parameter and each generation-specific environmental effect to the birth-size ACF only, then quantify likelihood and marginal trade-offs |
| fit_likelihood_acf_regularized_glu.py | Full penalized-MAP inference for all population parameters and environmental effects using normalized likelihood loss plus birth-ACF error |
| summarize_likelihood_acf_regularization.py | Assemble the fitted ACF-weight path and plot ACF, drift, and likelihood trade-offs |
| fit_decomposed_birth_regularized_glu.py | Separate the common mean-birth trajectory from generation-demeaned birth-size memory; fit the shared environmental path first and then test a full parameter relaxation with consistent posterior uncertainty in the environmental effects |
| fit_environment_lineage_offset_ar_q_glu.py | Test a shared environmental effect plus Gaussian lineage offset and one fast AR(1) q residual against the shared-environment fast/slow q model, including autonomous ACF, drift, marginal, and lineage-slope diagnostics |
| fit_individual_environment_lineage_effects_glu.py | Infer every absolute lineage effect A_l and shared cycle effect E_i jointly for the Glu q model, then contrast conditional observed-lineage and autonomous new-lineage posterior checks |
| test_individual_environment_lineage_effects.py | Verify the explicit joint Gaussian A_l/E_i posterior and the zero-mean reporting gauge |
| fit_individual_q_alpha_environment_effects_glu.py | Add one individually inferred shared cycle effect G_i to the latent Gaussian alpha process, alongside the individually inferred q effects, and test alpha, tau, growth-increment, birth-drift, and ACF predictions |
| test_individual_q_alpha_environment_effects.py | Verify the exact shared-alpha-effect posterior and latent variance constraint |
| fit_q_environment_birth_hill_glu.py | Remove the alpha G_i process, retain individually inferred q lineage/environment effects, and fit the four-parameter Hill timing hazard with an explicit birth-size term kappa log(b/b_star) |
| test_q_environment_birth_hill.py | Verify nesting at kappa=0 and the direction of explicit birth-size feedback |
| fit_q_environment_cycle_u_glu.py | Remove G_i and kappa, retain individually inferred q effects, and infer one hierarchical protein-Hill threshold u_i for every experimental cycle with exact random-effect quadrature |
| test_q_environment_cycle_u.py | Verify the shared-u limit and posterior normalization/sampling for the cycle-specific thresholds |
| fit_q_environment_cycle_omega2_glu.py | Restore shared u and infer one hierarchical maximal hazard rate omega2_i for every experimental cycle, retaining the inferred q effects |
| test_q_environment_cycle_omega2.py | Verify the shared-omega2 limit and posterior normalization/sampling for cycle-specific rates |

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
