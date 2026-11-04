"""CSV input and output helpers."""

from pathlib import Path

import pandas as pd


def write_csv(data: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame as a UTF-8-with-BOM CSV for spreadsheet compatibility."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False, encoding="utf-8-sig")


def read_responses(path: str | Path) -> pd.DataFrame:
    """Read a generated responses CSV."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Ficheiro de respostas não encontrado: {source}")
    return pd.read_csv(source, encoding="utf-8-sig")
