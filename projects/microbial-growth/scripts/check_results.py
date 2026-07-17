from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
summary = PROJECT_DIR / "outputs" / "summary" / "all_sugars_diagnostics.csv"
model_root = PROJECT_DIR / "outputs" / "model_comparison"
model_name = "loglogistic_endpoint_residual"

print("Project:", PROJECT_DIR)
print("Summary file exists:", summary.exists())
print(
    "Completed sugar folders:",
    [
        path.parents[1].name
        for path in sorted(model_root.glob(f"*/{model_name}/diagnostics.json"))
    ],
)

if summary.exists():
    df = pd.read_csv(summary)
    print(
        df[
            [
                "sugar",
                "n_observations",
                "n_transitions",
                "n_transition_lineages",
                "mean_acceptance_fraction",
                "tau_max",
                "time_acf_covered",
                "birth_acf_covered",
            ]
        ]
    )
