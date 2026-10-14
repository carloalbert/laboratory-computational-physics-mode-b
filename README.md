# Laboratory of Computational Physics — Mode B

The active project studies microbial growth and stochastic cell division.

Start with the [microbial-growth README](projects/microbial-growth/README.md).
It describes the current Glu model: an empirically detrended Hill timing hazard
with a static lineage offset, its fitted parameters, checks, and limitations.
Earlier experiments are listed briefly; their source remains available.

## Layout

~~~text
projects/microbial-growth/
├── config/        Fixed cohort and validation configuration
├── data/          Original and processed observations
├── docs/          Repository map and cleanup manifest
├── outputs/       Current model, comparator, and reference results
├── scripts/       Reproducible analysis and historical experiments
└── src/           Model implementations and shared helpers
~~~

## Current workflow

From this directory:

~~~bash
python -m pip install -r projects/microbial-growth/requirements.txt
python projects/microbial-growth/scripts/fit_empirical_detrended_hazard.py
python projects/microbial-growth/scripts/fit_static_offset_detrended_hazard.py
python projects/microbial-growth/scripts/plot_static_offset_detrended_hazard.py
~~~

See the project README for tests and the full reproduction sequence.
The root requirements additionally support historical analyses and MCMC.
