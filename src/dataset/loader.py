"""Load and validate CSV or TSV question-answer datasets."""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .models import DATASET_COLUMNS

_COLUMN_ALIASES = {
    "tema": "tema",
    "category": "tema",
    "pergunta": "pergunta",
    "question": "pergunta",
    "metodo": "metodo",
    "method": "metodo",
    "origem": "metodo",
    "source": "metodo",
    "resposta": "resposta",
    "answer": "resposta",
    "groundtruth": "resposta",
    "respostacorreta": "resposta",
    "reference": "resposta",
}


def _normalize_header(value: str) -> str:
    without_accents = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]", "", without_accents.lower())


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read a dataset, map known column names and validate each row."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as source:
        sample = source.read(4096)
        source.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = (
                csv.excel_tab
                if dataset_path.suffix.lower() == ".tsv"
                else csv.excel
            )
        rows = list(csv.DictReader(source, dialect=dialect))

    if not rows:
        raise ValueError(f"O dataset está vazio: {dataset_path}")

    header_mapping: dict[str, str] = {}
    for header in rows[0]:
        if header is not None:
            canonical = _COLUMN_ALIASES.get(_normalize_header(header))
            if canonical:
                header_mapping[header] = canonical

    missing = set(DATASET_COLUMNS) - set(header_mapping.values())
    if missing:
        raise ValueError(
            f"Colunas obrigatórias em falta: {', '.join(sorted(missing))}."
        )

    canonical_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        canonical = {
            column: row.get(original, "") or ""
            for original, column in header_mapping.items()
        }
        for column in DATASET_COLUMNS:
            if not canonical[column].strip():
                raise ValueError(
                    f"Valor vazio em '{column}', linha {row_number}."
                )
        canonical["metodo"] = canonical["metodo"].strip().upper()
        if canonical["metodo"] not in {"HUM", "LLM"}:
            raise ValueError(
                f"Método inválido na linha {row_number}: {canonical['metodo']!r}."
            )
        canonical_rows.append(canonical)

    return pd.DataFrame(canonical_rows, columns=DATASET_COLUMNS)
