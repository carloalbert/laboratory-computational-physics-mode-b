# Microbial Growth Analysis

This project analyzes microbial growth and division dynamics using lineage data
and a modified hazard-function model.  The current implementation reconstructs
measured division endpoints from consecutive rows and explicitly models the
residual between fitted within-cycle growth and endpoint growth.

The analysis compares real lineage data with simulated lineages through:

- survival-function fits,
- posterior summaries of model parameters,
- posterior-predictive simulations,
- real-versus-simulated histograms,
- autocorrelation function comparisons,
- MCMC diagnostics including trace and corner plots,
- likelihood scans.

## Current endpoint-residual model

The primary input remains the common all-sugar file:

```text
data/processed/processed_input_data.csv
```

No sugar-specific endpoint file is required.  For two consecutive observations
in the same lineage, the measured final length, measured added length and
endpoint residual are reconstructed as

```text
L_final,n = b_(n+1) / f_(n+1)
p_n       = L_final,n - b_n
epsilon_n = log(L_final,n / b_n) - alpha_n * tau_n
```

The shift to `f_(n+1)` is intentional: the recorded division fraction is
attached to the daughter row.  On Ace, this reconstruction agrees with the
independently supplied `UMIK_Ace.csv` endpoints to floating-point precision.

Division is fitted to the measured `p_n`, rather than to the surrogate
`b_n * (exp(alpha_n*tau_n) - 1)`.  The normalized log-logistic mixed
adder-sizer survival law is

```text
S(p | b) = (1 + (eta*b/u)^omega2)
           / (1 + ((p + eta*b)/u)^omega2)
```

and the measured-added-size density is `h_p(p | b) * S(p | b)`.  The old
Jacobian `dp/dtau = b*alpha*exp(alpha*tau)` is not used.  Growth rates retain a
Gamma marginal parameterized by mean and standard deviation, and division
fractions retain a Beta marginal parameterized in the same way.  Endpoint
residuals use a robust Student-t location-scale distribution with four fixed
degrees of freedom, truncated at `epsilon < log(1 + p/b)` so that
reconstructed division times remain positive.

The physical scale \(u\) and feedback \(\eta\) are nearly degenerate because
the data principally identify \(u-\eta b\).  The sampler therefore uses the
data-defined pivot

\[
b_\star=\frac{1}{|\mathcal T|}\sum_{n\in\mathcal T}b_n,
\qquad
p_\star=u-\eta b_\star,
\qquad
u=p_\star+\eta b_\star.
\]

For Ace, \(b_\star=1.791927\).  The transformation has unit Jacobian and does
not change the physical model or posterior.  The nine sampled parameters are:

```text
p_star, omega2, eta,
mu_alpha, sigma_alpha,
mu_f, sigma_f,
mu_epsilon, sigma_epsilon
```

The derived physical \(u\) posterior is saved separately in
`derived_physical_posterior_summary.csv`.

### Full likelihood

Let \(\mathcal I\) denote all retained complete observations and let
\(\mathcal T\) denote transitions for which rows \(n\) and \(n+1\) are
consecutive members of the same lineage.  For each \(n\in\mathcal T\), define

\[
L_n=\frac{b_{n+1}}{f_{n+1}},\qquad
p_n=L_n-b_n,\qquad
g_n=\log\!\left(1+\frac{p_n}{b_n}\right),\qquad
\epsilon_n=g_n-\alpha_n\tau_n.
\]

For the division component, write

\[
x_n=\frac{p_n+\eta b_n}{u},\qquad
x_{0n}=\frac{\eta b_n}{u}.
\]

The conditional survival, hazard and measured-added-size density are

\[
S_p(p_n\mid b_n)
=\frac{1+x_{0n}^{\omega_2}}{1+x_n^{\omega_2}},
\]

\[
h_p(p_n\mid b_n)
=\frac{\omega_2}{u}
  \frac{x_n^{\omega_2-1}}{1+x_n^{\omega_2}},
\]

\[
q_p(p_n\mid b_n)
=h_p(p_n\mid b_n)S_p(p_n\mid b_n)
=\frac{\omega_2}{u}
  \frac{x_n^{\omega_2-1}(1+x_{0n}^{\omega_2})}
       {(1+x_n^{\omega_2})^2},
\qquad p_n>0.
\]

The growth-rate density is Gamma with mean \(\mu_\alpha\) and standard
deviation \(\sigma_\alpha\):

\[
q_\alpha(\alpha_i\mid\mu_\alpha,\sigma_\alpha)
=\frac{\alpha_i^{k_\alpha-1}e^{-\alpha_i/\theta_\alpha}}
       {\Gamma(k_\alpha)\theta_\alpha^{k_\alpha}},
\qquad
k_\alpha=\left(\frac{\mu_\alpha}{\sigma_\alpha}\right)^2,
\qquad
\theta_\alpha=\frac{\sigma_\alpha^2}{\mu_\alpha}.
\]

The division-fraction density is Beta with mean \(\mu_f\) and standard
deviation \(\sigma_f\).  Defining

\[
\kappa_f=\frac{\mu_f(1-\mu_f)}{\sigma_f^2}-1,
\qquad
a_f=\mu_f\kappa_f,
\qquad
c_f=(1-\mu_f)\kappa_f,
\]

gives

\[
q_f(f_i\mid\mu_f,\sigma_f)
=\frac{f_i^{a_f-1}(1-f_i)^{c_f-1}}{B(a_f,c_f)},
\qquad 0<f_i<1.
\]

Let \(t_\nu(z;\mu,\sigma)\) be the Student-t location-scale density

\[
t_\nu(z;\mu,\sigma)
=\frac{\Gamma((\nu+1)/2)}
       {\Gamma(\nu/2)\sqrt{\nu\pi}\,\sigma}
 \left[1+\frac{(z-\mu)^2}{\nu\sigma^2}\right]^{-(\nu+1)/2}.
\]

The endpoint-residual factor uses \(\nu=4\) and is upper-truncated at \(g_n\):

\[
q_\epsilon(\epsilon_n\mid g_n,\mu_\epsilon,\sigma_\epsilon)
=\frac{t_4(\epsilon_n;\mu_\epsilon,\sigma_\epsilon)}
       {T_4\!\left((g_n-\mu_\epsilon)/\sigma_\epsilon\right)}
  \mathbf 1(\epsilon_n<g_n),
\]

where \(T_4\) is the CDF of a standard Student-t variable with four degrees of
freedom.  The transformation from the recorded pair \((b_{n+1},\tau_n)\) to
\((p_n,\epsilon_n)\), conditional on \((b_n,\alpha_n,f_{n+1})\), has Jacobian

\[
\left|
\frac{\partial(p_n,\epsilon_n)}
     {\partial(b_{n+1},\tau_n)}
\right|
=\frac{\alpha_n}{f_{n+1}}.
\]

Consequently, for the sampled parameter vector

\[
\Phi=(p_\star,\omega_2,\eta,\mu_\alpha,\sigma_\alpha,
      \mu_f,\sigma_f,\mu_\epsilon,\sigma_\epsilon),
\qquad
u(\Phi)=p_\star+\eta b_\star,
\]

the full likelihood implemented by the runner is

\[
\boxed{
\mathcal L(\Phi)
=
\prod_{i\in\mathcal I}
 q_\alpha(\alpha_i\mid\mu_\alpha,\sigma_\alpha)
 q_f(f_i\mid\mu_f,\sigma_f)
\prod_{n\in\mathcal T}
 q_p(p_n\mid b_n)
 q_\epsilon(\epsilon_n\mid g_n,\mu_\epsilon,\sigma_\epsilon)
 \frac{\alpha_n}{f_{n+1}}.
}
\]

Equivalently,

\[
\log\mathcal L
=\sum_{i\in\mathcal I}
  \left[\log q_\alpha(\alpha_i)+\log q_f(f_i)\right]
 +\sum_{n\in\mathcal T}
  \left[\log q_p(p_n\mid b_n)
       +\log q_\epsilon(\epsilon_n\mid g_n)
       +\log\alpha_n-\log f_{n+1}\right].
\]

The observation-transformation Jacobian term is data-dependent and therefore
does not change parameter estimates within this model, but it is retained in
the reported likelihood.  The original prior is bounded uniform in the
physical \((u,\eta)\) coordinates.  Since
\(|\partial u/\partial p_\star|=1\), its exact transformed scale-feedback factor
is

\[
\pi(p_\star,\eta)
\propto
\mathbf 1(\eta_{\min}<\eta<\eta_{\max})
\mathbf 1\!\left(u_{\min}<p_\star+\eta b_\star<u_{\max}\right).
\]

Thus the prior support is slanted in \((p_\star,\eta)\); treating their bounds
as an independent rectangle would change the prior.  With the remaining
bounded-uniform factors denoted by \(\pi_{\rm rest}\), the posterior is

\[
p(\Phi\mid\text{data})\propto
\mathcal L(\Phi)\,\pi(p_\star,\eta)\,\pi_{\rm rest}(\Phi).
\]

All numerical bounds are listed in each run's `diagnostics.json`.

The posterior is sampled with `emcee`.  In posterior-predictive lineages the
model samples measured added size first and uses the exact size recursion

```text
L_final,n = b_n + p_n
b_(n+1)   = f_(n+1) * L_final,n
tau_n     = (log(1 + p_n/b_n) - epsilon_n) / alpha_n.
```

This prevents the systematic size drift caused by treating the fitted growth
rate as an exact endpoint growth rate.  The current residual law is marginal:
Ace still shows residual dependence on birth and added size, and the model
may retain long-lineage birth-size correlation.  The primary ACF posterior
diagnostic therefore simulates the exact empirical lineage-length design and
applies the same sample-mean-subtracted estimator to data and simulations.
`acf_observation_matched.png` shows the posterior-predictive band; each
simulated lineage preserves the exact observed `(b0, lineage length)` pair. The
80-generation ACF is retained separately as a long-lineage model property.

## Experimental alpha-scaled time-hazard model

For Glu, the repository also contains an experimental three-parameter hazard
whose clock is scaled by the observed single-cell growth rate.  With growth
time \(s=\alpha t\) and modeled added size

\[
p_h(s)=b(e^s-1),
\]

the physical-time hazard and cumulative hazard are

\[
h(t\mid b,\alpha)
=\alpha\omega_2
\frac{[p_h(\alpha t)/u]^\eta}
     {1+[p_h(\alpha t)/u]^\eta},
\]

\[
H(t\mid b,\alpha)
=\omega_2\int_0^{\alpha t}
\frac{[b(e^r-1)/u]^\eta}
     {1+[b(e^r-1)/u]^\eta}\,dr.
\]

Consequently, the conditional event-time density is

\[
q_\tau(t\mid b,\alpha,u,\omega_2,\eta)
=h(t\mid b,\alpha)e^{-H(t\mid b,\alpha)}.
\]

The endpoint construction is unchanged.  For every consecutive pair within a
lineage,

\[
L_i=\frac{b_{i+1}}{f_{i+1}},\qquad
\epsilon_i=\log\frac{L_i}{b_i}-\alpha_i\tau_i,
\]

and the residual has a Student-t location-scale density with fixed
\(\nu=4\), lower-truncated at \(-\alpha_i\tau_i\):

\[
q_\epsilon(\epsilon_i\mid s_i,\mu_\epsilon,\sigma_\epsilon)
=\frac{
  \sigma_\epsilon^{-1}t_4\!\left(
    \frac{\epsilon_i-\mu_\epsilon}{\sigma_\epsilon}
  \right)
}{
  1-T_4\!\left(
    \frac{-s_i-\mu_\epsilon}{\sigma_\epsilon}
  \right)
}\,\mathbf 1(\epsilon_i>-s_i),
\qquad s_i=\alpha_i\tau_i.
\]

Writing \(g_\Gamma\) for the Gamma growth-rate density and \(g_B\) for
the Beta division-fraction density used by the current model, the full
likelihood is

\[
\mathcal L(\Theta)=
\prod_{i\in\mathcal O}
q_\tau(\tau_i\mid b_i,\alpha_i,u,\omega_2,\eta)
g_\Gamma(\alpha_i\mid\mu_\alpha,\sigma_\alpha)
g_B(f_i\mid\mu_f,\sigma_f)
\prod_{i\in\mathcal T}
q_\epsilon(\epsilon_i\mid\alpha_i\tau_i,
\mu_\epsilon,\sigma_\epsilon),
\]

where \(\mathcal O\) contains all observations and \(\mathcal T\) only valid
within-lineage transitions.  Prediction uses

\[
L_i=b_i e^{s_i+\epsilon_i},\qquad b_{i+1}=f_{i+1}L_i.
\]

The likelihood factorizes into hazard, endpoint-residual, and unchanged
Gamma/Beta nuisance blocks.  The Glu runner therefore samples the first two
blocks and combines them with independent draws from the existing Glu
nuisance posterior; this is equivalent to resampling all blocks jointly under
the same independent priors.  Run it with

```bash
python projects/microbial-growth/scripts/run_alpha_scaled_hill_time_hazard.py --sugar Glu
```

Results are written to
`outputs/model_comparison/Glu/alpha_scaled_hill_time_hazard/` without
overwriting the current model.

For the ACF-targeted sanity check, replaying each Glu lineage's measured
growth-rate and division-factor sequences while optimizing only
`u`, `omega2`, and `eta`, run

```bash
python projects/microbial-growth/scripts/optimize_alpha_scaled_acf_measured_inputs.py
```

By default this minimizes the mean squared birth-length ACF error over lags
1--15.  The endpoint-residual parameters remain fixed at their Glu posterior
medians, and common random numbers are used during optimization.  This is a
diagnostic calibration, not a likelihood or posterior fit.  Passing
`--objective both` instead gives equal weight to the birth-length and
division-time ACF errors.

The fixed-parameter critical replay requested for Glu is reproduced with

```bash
python projects/microbial-growth/scripts/simulate_alpha_scaled_critical_replay.py
```

It uses the positive scale `u=-1/log(eta)` (the literal `1/log(eta)` is
negative for the default `eta=0.1`) and computes the critical `omega2` from
the measured replayed division factors and the fitted endpoint-residual law.

### Slow lineage-state diagnostic

To test whether the long birth-size shoulder reflects a persistent lineage
state, run

```bash
python projects/microbial-growth/scripts/analyze_lineage_size_persistence.py --replicates 100
```

For every lineage of at least eight observations, this correlates its first-
and second-half mean log birth sizes.  The observed statistic is compared with
observation-matched posterior simulations using the same initial sizes and
lineage lengths.  The all-sugar table and figure are written to
`outputs/summary/lineage_half_size_persistence.*`.

`latent_setpoint_endpoint_residual.py` provides the corresponding minimal
extension

\[
u_n=u\exp(z_n-\sigma_z^2/2),\qquad
z_{n+1}=\phi z_n+\sigma_z\sqrt{1-\phi^2}\,\xi_n.
\]

It preserves the mean division scale while adding a stationary, slowly
evolving lineage state.  The measured-input Glu/Lac calibration is reproduced
with

```bash
python projects/microbial-growth/scripts/calibrate_latent_setpoint_replay.py
```

This is a summary-statistic diagnostic, not Bayesian inference.  It shows that
the slow state explains much of the first-half/second-half persistence but not
the per-lineage-mean-subtracted ACF shoulder: for Glu and Lac the birth-ACF
errors remain essentially unchanged while the simulated birth-size marginal
becomes too broad.

### Hierarchical lineage-specific division scale

The Glu analysis also fits a persistent scale for each observed lineage,

\[
\log u_\ell\mid \bar u,\sigma_{\log u}
\sim \mathcal N\!\left(\log\bar u,\sigma_{\log u}^2\right),
\qquad
\bar u=p_\star+\eta b_\star.
\]

The scale \(u_\ell\) is shared by every transition in lineage \(\ell\); all
other model factors are unchanged.  If \(\mathcal T_\ell\) is the set of
reconstructed transitions in that lineage, its division contribution is
marginalized as

\[
\mathcal L_{p,\ell}
=\int_0^\infty
  \left[\prod_{n\in\mathcal T_\ell}
    q_p(p_n\mid b_n,u_\ell,\omega_2,\eta)
  \right]
  \operatorname{LogNormal}
  \!\left(u_\ell;\log\bar u,\sigma_{\log u}\right)
  \,du_\ell.
\]

Consequently the full hierarchical likelihood is

\[
\boxed{
\mathcal L_{\rm hier}
=
\left[\prod_{i\in\mathcal I}
q_\alpha(\alpha_i)q_f(f_i)\right]
\left[\prod_{n\in\mathcal T}
q_\epsilon(\epsilon_n\mid g_n)
\frac{\alpha_n}{f_{n+1}}\right]
\prod_\ell \mathcal L_{p,\ell}.
}
\]

The one-dimensional lineage integrals are evaluated with Gauss--Hermite
quadrature.  This leaves only four division hyperparameters
\((p_\star,\omega_2,\eta,\sigma_{\log u})\) in the MCMC.  Because the nuisance
blocks factorize exactly, their existing Glu posterior draws are reused rather
than resampled.  Conditional quadrature weights then give
\(p(u_\ell\mid\mathcal T_\ell,\text{hyperparameters})\) for posterior
prediction.

Run the fit and its posterior ACF and marginal checks with

```bash
MPLCONFIGDIR=/tmp/mpl /private/tmp/microbial-growth-venv/bin/python \
  projects/microbial-growth/scripts/run_hierarchical_lineage_u.py
```

The ACF checks use the exact observed \((b_{\ell0},N_\ell)\) design.  The same
posterior lineage-scale draw is used in an IID-input simulation and in a second
simulation that replays the lineage's measured \(f\) and \(\alpha\) sequences.
Results are written to
`outputs/model_comparison/Glu/hierarchical_lineage_u_endpoint_residual/`.

### Hierarchical lineage-specific hazard shape

The alternative Glu hierarchy restores a single global size scale
\(u=p_\star+\eta b_\star\) and instead assigns a persistent log-logistic shape
to each lineage,

\[
\log\omega_{2,\ell}\mid\bar\omega_2,\sigma_{\log\omega_2}
\sim\mathcal N\!\left(
  \log\bar\omega_2,\sigma_{\log\omega_2}^2
\right).
\]

The lineage division contribution becomes

\[
\mathcal L_{p,\ell}^{(\omega)}
=\int_0^\infty
  \left[\prod_{n\in\mathcal T_\ell}
    q_p(p_n\mid b_n,u,\omega_{2,\ell},\eta)
  \right]
  \operatorname{LogNormal}\!\left(
    \omega_{2,\ell};
    \log\bar\omega_2,\sigma_{\log\omega_2}
  \right)
  \,d\omega_{2,\ell}.
\]

The full likelihood is the same boxed hierarchical likelihood above with
\(\mathcal L_{p,\ell}\) replaced by
\(\mathcal L_{p,\ell}^{(\omega)}\).  The lineage integrals are again evaluated
by Gauss--Hermite quadrature and all nuisance posterior blocks are reused by
exact factorization.  Run the fit with

```bash
MPLCONFIGDIR=/tmp/mpl /private/tmp/microbial-growth-venv/bin/python \
  projects/microbial-growth/scripts/run_hierarchical_lineage_omega2.py
```

Results are written to
`outputs/model_comparison/Glu/hierarchical_lineage_omega2_endpoint_residual/`.
The common, lineage-\(u\), and lineage-\(\omega_2\) comparison is regenerated
with

```bash
MPLCONFIGDIR=/tmp/mpl /private/tmp/microbial-growth-venv/bin/python \
  projects/microbial-growth/scripts/compare_glu_lineage_hierarchies.py
```

### Joint lineage-specific scale and shape

The Glu-only joint hierarchy lets both division parameters persist by lineage,
with independent population laws

\[
\log u_\ell\sim\mathcal N(\log\bar u,\sigma_{\log u}^2),\qquad
\log\omega_{2,\ell}\sim
\mathcal N(\log\bar\omega_2,\sigma_{\log\omega_2}^2).
\]

Their two-dimensional lineage integral is evaluated with product
Gauss--Hermite quadrature. Posterior ACF prediction preserves every observed
lineage's initial birth size, length, growth-rate sequence, and division-factor
sequence. Run it with

```bash
MPLCONFIGDIR=/tmp/mpl /private/tmp/microbial-growth-venv/bin/python \
  projects/microbial-growth/scripts/run_hierarchical_lineage_u_omega2.py
```

Results are written to
`outputs/model_comparison/Glu/hierarchical_lineage_u_omega2_endpoint_residual/`.

### Hierarchical alpha-scaled time hazard

For the physical-time hazard

\[
h_\ell(t\mid b,\alpha)=\alpha\omega_{2,\ell}
\frac{[p(t)/u_\ell]^{\eta_\ell}}
{1+[p(t)/u_\ell]^{\eta_\ell}},
\qquad p(t)=b(e^{\alpha t}-1),
\]

the Glu-only hierarchical fit lets all three hazard parameters vary by lineage.
Their logarithms follow a joint Gaussian population distribution with a full
covariance matrix. The scalable fit uses a Laplace empirical-Bayes
approximation: individual lineage likelihoods and Hessians are combined by
marginal maximum likelihood, retaining uncertainty for short lineages.
Posterior prediction replays each lineage's measured alpha and division-factor
sequences. Run it with

```bash
MPLCONFIGDIR=/tmp/mpl /private/tmp/microbial-growth-venv/bin/python \
  projects/microbial-growth/scripts/run_hierarchical_alpha_scaled_time_hazard.py
```

Results are written to
`outputs/model_comparison/Glu/hierarchical_alpha_scaled_time_hazard_all_parameters/`.

## Sugars analyzed

```text
Ace, DMa, Glu, Gly, Lac, Pyr, Rha, Suc, Xyl
```

## Dataset summary

| Sugar | Observations | Lineages |
|---|---:|---:|
| Ace | 597 | 101 |
| DMa | 2471 | 131 |
| Glu | 6504 | 303 |
| Gly | 1647 | 91 |
| Lac | 6063 | 285 |
| Pyr | 1504 | 108 |
| Rha | 4802 | 424 |
| Suc | 2107 | 196 |
| Xyl | 6992 | 390 |

## Folder structure

```text
microbial-growth/
├── README.md
├── notebooks/
│   ├── microbial_growth_modified_hazard_one_sugar.ipynb
│   └── microbial_growth_modified_hazard_one_sugar_clean.ipynb
├── data/
│   ├── raw/
│   └── processed/
│       └── processed_input_data.csv
├── outputs/
│   ├── summary/
│   └── model_comparison/
│       ├── <sugar>/loglogistic_endpoint_residual/
│       └── Glu/<Glu-specific model>/
├── src/
└── report/
```

## Notebooks

The main notebook is:

```text
notebooks/microbial_growth_modified_hazard_one_sugar.ipynb
```

A clean version without executed outputs is also included:

```text
notebooks/microbial_growth_modified_hazard_one_sugar_clean.ipynb
```

To analyze a different sugar, change the `SUGAR` variable in the settings cell, for example:

```python
SUGAR = "Glu"
```

## Results

Global summary outputs are stored in:

```text
outputs/summary/
```

Important summary files:

```text
all_sugars_diagnostics.csv
all_sugars_parameter_modes.csv
all_sugars_posterior_summary.csv
all_sugars_predictive_summary.csv
all_sugars_acf_lag1.png
all_sugars_acf_pairing_comparison.csv
all_sugars_acf_four_variables.png
all_sugars_acf_four_variables.pdf
all_sugars_acf_four_variables_lag1.csv
all_sugars_acf_sheet.png
all_sugars_acf_sheet.pdf
sugar_parameter_scatter.png
run_config.json
```

The current per-sugar results are stored in:

```text
outputs/model_comparison/<sugar>/loglogistic_endpoint_residual/
```

Each sugar folder contains files such as:

```text
acf_comparison.png
acf_observation_matched.csv
acf_observation_matched.png
added_size_survival.png
comparison_metrics.csv
corner_plot.png
derived_physical_posterior_summary.csv
diagnostics.json
endpoint_residual_diagnostics.png
flat_chain.csv
flat_log_probability.csv
posterior_predictive_comparison.png
posterior_predictive_simulation.csv
posterior_summary.csv
reconstructed_transitions.csv
replicated_predictive_metrics.csv
replicated_predictive_summary.csv
trace_plots.png
```

## How to run

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook projects/microbial-growth/notebooks/microbial_growth_modified_hazard_one_sugar.ipynb
```

The notebook is written to work whether it is launched from the repository root, from the project folder, or from the notebook folder.

### Historical Weibull adder-sizer comparison

The historical comparison model can still be fitted without overwriting the
retained results:

```bash
python projects/microbial-growth/scripts/run_weibull_adder_sizer.py --sugar Ace
```

The normalized log-logistic alternative, which retains a power-law tail, is:

```bash
python projects/microbial-growth/scripts/run_loglogistic_adder_sizer.py --sugar Ace
```

The current endpoint-residual analysis is run with:

```bash
python projects/microbial-growth/scripts/run_loglogistic_endpoint_residual.py --sugar Ace
```

Changing `--sugar` is sufficient to run the same reconstruction and inference
for any sugar in `processed_input_data.csv`.

To reproduce the complete nine-sugar run from the repository root:

```bash
for sugar in Ace DMa Glu Gly Lac Pyr Rha Suc Xyl; do
    python projects/microbial-growth/scripts/run_loglogistic_endpoint_residual.py --sugar "$sugar"
done
```

The completed fits for all nine sugars can be consolidated into the current
`outputs/summary/` bundle with:

```bash
python projects/microbial-growth/scripts/summarize_loglogistic_endpoint_residual.py
```

The main per-sugar figures can be regenerated from the saved chains and
simulations, without repeating inference, with:

```bash
python projects/microbial-growth/scripts/regenerate_endpoint_residual_plots.py
```

To recompute an observation-matched ACF table from the saved posterior while
preserving exact initial-size/lineage-length pairs:

```bash
python projects/microbial-growth/scripts/recompute_paired_acf_diagnostics.py --sugar Lac
```

To compare a process-only ACF band at fixed MAP parameters with the full
posterior-predictive band:

```bash
python projects/microbial-growth/scripts/compare_acf_parameter_uncertainty.py --sugar Glu
```

To test whether the observed temporal structure in the measured growth rates
and division factors accounts for the state-variable ACFs, replay the measured
`alpha_n` and `f_n` sequence of every lineage while sampling only added size
and the endpoint residual from posterior draws:

```bash
python projects/microbial-growth/scripts/validate_acf_with_measured_inputs.py
```

This conditional check writes `acf_measured_inputs_validation.{csv,png}` in
each sugar's output directory and the combined
`outputs/summary/all_sugars_acf_measured_inputs.{png,pdf}` sheet.  It is a
diagnostic intervention rather than a draw from the fitted IID-input model:
the observed input histories are held fixed to isolate how much of the
division-time and birth-size ACF discrepancy they can transmit.

This updates `acf_observation_matched.png` (division-time and birth-length ACF),
`posterior_predictive_comparison.png` (the seven predictive marginals), and
`corner_plot.png` in every current per-sugar output folder.

The three comparison runners write results under:

```text
outputs/model_comparison/<sugar>/weibull_adder_sizer/
outputs/model_comparison/<sugar>/loglogistic_adder_sizer/
outputs/model_comparison/<sugar>/loglogistic_endpoint_residual/
```

The first two comparison scripts retain the earlier exact-exponential endpoint
assumption for reproducibility, but their generated outputs are not retained in
the cleaned repository.  The endpoint-residual script is the current
recommended nine-sugar analysis.  All Glu-specific model results are retained.

## Reproducibility

The batch-run settings are stored in:

```text
outputs/summary/run_config.json
```

The included configuration uses:

```text
40 walkers
1000 burn-in steps
3000 production steps
thin = 5
random seed = 42
```

## Data note

The processed data file is included as:

```text
data/processed/processed_input_data.csv
```

Before making the repository public, make sure you are allowed to share this dataset and all generated outputs.
