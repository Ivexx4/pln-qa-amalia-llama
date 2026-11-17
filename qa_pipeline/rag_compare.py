"""Comparação emparelhada entre a Meta 2 (sem RAG) e a Meta 3 (com RAG) nas mesmas perguntas."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, stats

COLUMNS = ["Modelo", "metrica", "n", "sem_rag", "com_rag", "diferenca", "ic_95_inferior", "ic_95_superior",
           "teste", "p_valor", "so_rag_certo", "so_baseline_certo"]


def compare_runs(baseline: pd.DataFrame, rag: pd.DataFrame) -> pd.DataFrame:
    """Para cada modelo e métrica, compara as perguntas presentes (e válidas) em ambas as execuções."""
    rng = np.random.default_rng(stats.BOOTSTRAP_SEED)
    rows = []
    for model in sorted(set(baseline["Modelo"]) & set(rag["Modelo"])):
        left = baseline[baseline["Modelo"] == model].set_index("id")
        right = rag[rag["Modelo"] == model].set_index("id")
        for metric in config.METRICS:
            pair = pd.concat([pd.to_numeric(left[metric], errors="coerce").rename("a"),
                              pd.to_numeric(right[metric], errors="coerce").rename("b")], axis=1,
                             join="inner").dropna().sort_index()
            if pair.empty:
                continue
            a, b = pair["a"].to_numpy(), pair["b"].to_numpy()
            mean, lower, upper = stats.bootstrap_mean(b - a, rng)
            row = {"Modelo": model, "metrica": metric, "n": len(pair), "sem_rag": a.mean(), "com_rag": b.mean(),
                   "diferenca": mean, "ic_95_inferior": lower, "ic_95_superior": upper}
            # paired_test(reference, other) testa reference − other: aqui, RAG − baseline.
            test = stats.paired_test(metric, b, a)
            if test:
                row.update({"teste": test["teste"], "p_valor": test["p_valor"]})
                if metric in ("Judge", "Contains"):
                    row.update({"so_rag_certo": test["discordantes_amalia"],
                                "so_baseline_certo": test["discordantes_outro"]})
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def export_comparison(baseline_csv: Path, rag_csv: Path, retrieval_csv: Path | None, out_dir: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_csv, dtype={"Autor": str})
    rag = pd.read_csv(rag_csv, dtype={"Autor": str})
    baseline = baseline[baseline["id"].isin(rag["id"].unique())]
    if baseline.empty:
        raise ValueError("A baseline não tem nenhuma das perguntas avaliadas com RAG.")
    table = compare_runs(baseline, rag)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out_dir / "comparacao_rag_vs_baseline.csv", index=False, encoding="utf-8")

    # Perguntas em que o RAG ajudou ou prejudicou, cruzadas com o sucesso da recuperação.
    detail = []
    for model in sorted(set(baseline["Modelo"]) & set(rag["Modelo"])):
        a = baseline[baseline["Modelo"] == model].set_index("id")
        b = rag[rag["Modelo"] == model].set_index("id")
        both = a.join(b[["Pred", "Judge", "Contains"]], how="inner", lsuffix="_base", rsuffix="_rag")
        both["Modelo"] = model
        detail.append(both.reset_index())
    detail = pd.concat(detail, ignore_index=True)
    detail["efeito"] = np.select(
        [(detail["Judge_base"] == 0) & (detail["Judge_rag"] == 1),
         (detail["Judge_base"] == 1) & (detail["Judge_rag"] == 0)], ["RAG ajudou", "RAG prejudicou"], "sem mudança")
    if retrieval_csv is not None and retrieval_csv.is_file():
        retrieval = pd.read_csv(retrieval_csv)[["id", "rank_gold", "token_recall", "fontes"]]
        detail = detail.merge(retrieval, on="id", how="left")
    keep = [c for c in ("id", "Modelo", "Tema", "Pergunta", "Gold", "Pred_base", "Pred_rag", "Judge_base",
                        "Judge_rag", "efeito", "rank_gold", "token_recall", "fontes") if c in detail.columns]
    detail[keep].to_csv(out_dir / "efeito_rag_por_pergunta.csv", index=False, encoding="utf-8")
    return table
