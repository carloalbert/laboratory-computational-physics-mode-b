# Microbial Growth Analysis

This project analyzes microbial growth and division dynamics using lineage data and a modified hazard-function model.

The analysis compares real lineage data with simulated lineages through:

- survival-function fits,
- posterior summaries of model parameters,
- posterior-predictive simulations,
- real-versus-simulated histograms,
- autocorrelation function comparisons,
- MCMC diagnostics including trace and corner plots,
- likelihood scans.

## Model idea

The division process is represented through a modified hazard function. The fitted parameters are:

```text
u, v, omega2, mu_alpha, sigma_alpha, mu_f, sigma_f
```

The notebook performs a one-sugar Bayesian analysis with `emcee`. The uploaded results contain a batch run for all sugars.

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
│   ├── by_sugar/
│   └── results_manifest.csv
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
sugar_parameter_scatter.png
run_config.json
```

Per-sugar results are stored in:

```text
outputs/by_sugar/
```

Each sugar folder contains files such as:

```text
acf.png
corner_plot.png
diagnostics.json
flat_chain.csv
flat_log_probability.csv
likelihood_scan_u.png
log_probability.npy
posterior_predictive_histograms.png
posterior_predictive_simulation.csv
posterior_summary.csv
survival_fit.png
trace_plots.png
```

Note: the Xyl folder in the uploaded results did not include a likelihood-scan figure, so that file is absent for Xyl.

## How to run

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook projects/microbial-growth/notebooks/microbial_growth_modified_hazard_one_sugar.ipynb
```

The notebook is written to work whether it is launched from the repository root, from the project folder, or from the notebook folder.

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
