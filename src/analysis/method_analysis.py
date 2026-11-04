"""Metric averages by question origin (HUM or LLM) and model."""

import pandas as pd

from src.analysis.overall_analysis import METRICS
from src.utils.io import write_csv


def analyze_by_method(metrics: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Calculate and save metric averages grouped by source and model."""
    summary = (
        metrics.groupby(["metodo", "modelo"], as_index=False)
        .agg(n=("modelo", "size"), **{name: (name, "mean") for name in METRICS})
        .sort_values(["metodo", "modelo"])
    )
    write_csv(summary, output_path)
    return summary
