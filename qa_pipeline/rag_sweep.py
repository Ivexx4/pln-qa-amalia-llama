"""Varrimento de parâmetros: expande uma grelha em variantes e resume os resultados lado a lado."""
from __future__ import annotations

import re
from itertools import product
from pathlib import Path

import pandas as pd

from . import rag_settings
from .rag_settings import RagSettings

SWEEP_METRICS = ["Judge", "Contains", "ROUGE_L", "BERTScore_F1"]


def variant_name(changes: dict) -> str:
    text = "_".join(f"{key}-{value}" for key, value in changes.items())
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def expand_sweep(spec: dict, lower: list[dict] | None = None,
                 upper: list[dict] | None = None) -> list[tuple[str, RagSettings]]:
    """spec = {"name", "base": {...}, "grid": {parametro: [valores]}}; devolve (nome_da_variante, parâmetros).

    Precedência: lower (p. ex. --config) < spec["base"] < upper (opções da CLI e --set) < valores da grelha.
    """
    grid = spec.get("grid") or {}
    if not grid or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("'grid' deve mapear cada parâmetro para uma lista não vazia de valores.")
    variants = []
    keys = list(grid)
    for values in product(*(grid[key] for key in keys)):
        changes = dict(zip(keys, values))
        settings = rag_settings.build_settings(*(lower or []), spec.get("base", {}), *(upper or []), changes)
        variants.append((variant_name(changes), settings))
    names = [name for name, _ in variants]
    if len(names) != len(set(names)):
        raise ValueError("A grelha gera variantes com o mesmo nome.")
    return variants


def summarize_variant(name: str, settings: RagSettings, analysis_dir: Path) -> list[dict]:
    """Uma linha por modelo (ou uma só, se ainda não houver geração) com recuperação e qualidade."""
    base = {"variante": name, "fingerprint": settings.fingerprint(), **{
        key: value for key, value in settings.to_dict().items() if key not in ("models", "judge_model")}}
    retrieval_path = analysis_dir / "recuperacao_resumo.csv"
    if retrieval_path.is_file():
        summary = pd.read_csv(retrieval_path)
        overall = summary[summary["grupo"] == "todas"].iloc[0].drop(["grupo"]).to_dict()
        base.update({("n_perguntas" if key == "n" else key): value for key, value in overall.items()})
    results_path = analysis_dir / "all_results.csv"
    if not results_path.is_file():
        return [base]
    results = pd.read_csv(results_path)
    rows = []
    for model, group in results.groupby("Modelo"):
        row = {**base, "Modelo": model}
        for metric in SWEEP_METRICS:
            if metric in group:
                row[metric] = pd.to_numeric(group[metric], errors="coerce").mean()
        rows.append(row)
    return rows


def write_summary(rows: list[dict], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(path, index=False, encoding="utf-8")
    return table
