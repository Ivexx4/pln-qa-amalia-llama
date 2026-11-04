"""Global averages by model."""

import pandas as pd

from src.utils.io import write_csv

METRICS = ["exact_match", "rougeL", "bertscore"]


def analyze_overall(metrics: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Calculate and save global metric averages and sample counts."""
    summary = (
        metrics.groupby("modelo", as_index=False)
        .agg(n=("modelo", "size"), **{name: (name, "mean") for name in METRICS})
        .sort_values("modelo")
    )
    write_csv(summary, output_path)
    return summary
