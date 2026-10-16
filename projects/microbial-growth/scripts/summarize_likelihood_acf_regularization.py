#!/usr/bin/env python3
"""Summarize the fitted likelihood/ACF regularization path."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT = PROJECT_DIR / "outputs/model_comparison/Glu/likelihood_birth_acf_regularized"
WEIGHTS = [0.03, 0.1, 0.3, 1.0]


def main():
    rows, parameter_rows = [], []
    for weight in WEIGHTS:
        directory = OUT / f"weight_{weight:g}"
        diagnostics = json.loads((directory / "diagnostics.json").read_text())
        validation = diagnostics["validation"]
        rows.append({"acf_weight": weight, **validation})
        parameters = pd.read_csv(directory / "parameter_comparison.csv")
        parameters["acf_weight"] = weight
        parameter_rows.append(parameters)
    path = pd.DataFrame(rows)
    baseline = {
        "acf_weight": 0.0,
        "birth_acf_rmse_likelihood": path.birth_acf_rmse_likelihood.iloc[0],
        "birth_acf_rmse_regularized": path.birth_acf_rmse_likelihood.iloc[0],
        "q_acf_rmse_likelihood": path.q_acf_rmse_likelihood.iloc[0],
        "q_acf_rmse_regularized": path.q_acf_rmse_likelihood.iloc[0],
        "mean_birth_rmse_likelihood": path.mean_birth_rmse_likelihood.iloc[0],
        "mean_birth_rmse_regularized": path.mean_birth_rmse_likelihood.iloc[0],
        "generation_20_change_observed": path.generation_20_change_observed.iloc[0],
        "generation_20_change_likelihood": path.generation_20_change_likelihood.iloc[0],
        "generation_20_change_regularized": path.generation_20_change_likelihood.iloc[0],
        "log_density_change": 0.0,
        "log_density_change_per_observation": 0.0,
    }
    path = pd.concat([pd.DataFrame([baseline]), path], ignore_index=True)
    path.to_csv(OUT / "regularization_path.csv", index=False)
    pd.concat(parameter_rows, ignore_index=True).to_csv(
        OUT / "parameter_regularization_path.csv", index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    x = np.arange(len(path))
    labels = [f"{value:g}" for value in path.acf_weight]
    axes[0].plot(x, path.birth_acf_rmse_regularized, "o-", label="Birth ACF")
    axes[0].plot(x, path.q_acf_rmse_regularized, "s-", label="q ACF")
    axes[0].set(title="ACF error", ylabel="RMSE")
    axes[0].legend(frameon=False)
    axes[1].plot(x, path.mean_birth_rmse_regularized, "o-", color="#c4512c", label="Mean-curve RMSE")
    axes[1].plot(x, path.generation_20_change_regularized, "s-", color="#718096", label="Generation-20 change")
    axes[1].axhline(path.generation_20_change_observed.iloc[0], color="#23364d", ls=":", label="Observed change")
    axes[1].set(title="Birth-size drift")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(x, -path.log_density_change_per_observation, "o-", color="#c4512c")
    axes[2].set(title="Likelihood cost", ylabel="Lost log density / observation")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel(r"Normalized ACF weight $\lambda$")
        axis.grid(alpha=0.16)
    fig.suptitle("Glu likelihood + birth-ACF regularization path")
    fig.savefig(OUT / "regularization_path.png", dpi=190)
    fig.savefig(OUT / "regularization_path.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
