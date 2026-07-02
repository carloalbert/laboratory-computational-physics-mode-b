# Laboratory of Computational Physics — Mode B

This repository contains coursework and project material for **Laboratory of Computational Physics — Mode B**.

## Projects

### Microbial Growth

A Bayesian analysis of microbial growth and division dynamics using lineage data and a modified hazard-function model.

Project folder:

```text
projects/microbial-growth/
```

## Repository structure

```text
laboratory-computational-physics-mode-b/
├── README.md
├── requirements.txt
├── .gitignore
└── projects/
    └── microbial-growth/
        ├── README.md
        ├── notebooks/
        ├── data/
        ├── outputs/
        ├── src/
        └── report/
```

## Environment

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Then open the notebook inside:

```text
projects/microbial-growth/notebooks/
```

## Notes

Large posterior-chain files and generated outputs are stored under the project `outputs/` folder.
If GitHub rejects a file because it is too large, use Git LFS for that file type.
