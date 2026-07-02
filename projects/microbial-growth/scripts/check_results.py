from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
summary = PROJECT_DIR / "outputs" / "summary" / "all_sugars_diagnostics.csv"
by_sugar = PROJECT_DIR / "outputs" / "by_sugar"

print("Project:", PROJECT_DIR)
print("Summary file exists:", summary.exists())
print("Sugar folders:", [p.name for p in sorted(by_sugar.iterdir()) if p.is_dir()])

if summary.exists():
    df = pd.read_csv(summary)
    print(df[["sugar", "n_observations", "n_lineages"]])
