"""Consolida as execuções RAG (results/meta3) numa tabela única para o relatório.

Lê, para cada execução, o manifesto (parâmetros, corpus), a recuperação e o all_results.csv; compara com
a baseline sem RAG (Meta 2) nas mesmas perguntas, emparelhando por id, e separa treino/teste.

Saídas (em --out, por omissão results/meta3):
  consolidado.csv       uma linha por execução × modelo × subconjunto (todas/train/test)
  tabela_relatorio.md   tabela larga do subconjunto "todas", pronta a colar no relatório
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["Judge", "Contains", "ROUGE_L", "BERTScore_F1"]
PARAMS = ["embed_model", "chunk_chars", "chunk_overlap", "top_k", "min_score", "store", "prompt"]
EXPERIMENTS = ("chunks-topk", "prompt-minscore", "overlap", "corpus")
DEFAULT_BASELINE = Path("results/meta2/principal-ptpt/analises/both_todas_completa/all_results.csv")


def experiment_of(run_id: str) -> str:
    return next((name for name in EXPERIMENTS if run_id.startswith(name + "-")), "outro")


def short_model(name: str) -> str:
    lowered = name.lower()
    return "amalia" if "amalia" in lowered else "llama" if "llama" in lowered else name.split(":")[0]


def paired_means(base: pd.DataFrame, rag: pd.DataFrame) -> dict:
    """Médias de cada métrica nas perguntas válidas em ambas as execuções (emparelhadas por id)."""
    row = {}
    for metric in METRICS:
        if metric not in rag or metric not in base:
            continue
        a = pd.to_numeric(base.set_index("id")[metric], errors="coerce").rename("sem")
        b = pd.to_numeric(rag.set_index("id")[metric], errors="coerce").rename("com")
        pair = pd.concat([a, b], axis=1, join="inner").dropna()
        if pair.empty:
            continue
        row[metric] = pair["com"].mean()
        row[f"{metric}_sem_rag"] = pair["sem"].mean()
        row[f"{metric}_dif"] = (pair["com"] - pair["sem"]).mean()
        row[f"n_{metric}"] = len(pair)
    return row


def subsets(df: pd.DataFrame):
    yield "todas", df
    if "split" in df:
        for name in ("train", "test"):
            part = df[df["split"] == name]
            if len(part):
                yield name, part


def collect_run(run_dir: Path, baseline: pd.DataFrame) -> list[dict]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = manifest.get("settings", {})
    corpus = manifest.get("corpus", {})
    rows = []
    for analysis in sorted((run_dir / "analises").glob("*_completa")):
        if "limite" in analysis.name or not (analysis / "all_results.csv").is_file():
            continue
        results = pd.read_csv(analysis / "all_results.csv", dtype={"Autor": str})
        retrieval = {}
        summary_path = analysis / "recuperacao_resumo.csv"
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            overall = summary[summary["grupo"] == "todas"]
            if len(overall):
                retrieval = {k: v for k, v in overall.iloc[0].to_dict().items() if k not in ("grupo", "n")}
        for model, group in results.groupby("Modelo"):
            for subset_name, part in subsets(group):
                base = baseline[(baseline["Modelo"] == model) & baseline["id"].isin(part["id"])]
                row = {"run_id": run_dir.name, "experiencia": experiment_of(run_dir.name), "escopo": analysis.name,
                       **{p: settings.get(p) for p in PARAMS},
                       "corpus_sha": str(corpus.get("sha256", ""))[:8], "n_excertos": corpus.get("chunks"),
                       "Modelo": model, "subconjunto": subset_name, "n_perguntas": part["id"].nunique(),
                       **retrieval, **paired_means(base, part)}
                rows.append(row)
    return rows


def baseline_rows(baseline: pd.DataFrame, ids: set) -> list[dict]:
    rows = []
    for model, group in baseline[baseline["id"].isin(ids)].groupby("Modelo"):
        for subset_name, part in subsets(group):
            row = {"run_id": "baseline-sem-rag", "experiencia": "baseline", "Modelo": model,
                   "subconjunto": subset_name, "n_perguntas": part["id"].nunique()}
            for metric in METRICS:
                if metric in part:
                    row[metric] = pd.to_numeric(part[metric], errors="coerce").mean()
            rows.append(row)
    return rows


def to_markdown(df: pd.DataFrame) -> str:
    def fmt(value):
        if pd.isna(value):
            return ""
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def report_table(long: pd.DataFrame) -> pd.DataFrame:
    runs = long[(long["subconjunto"] == "todas") & (long["experiencia"] != "baseline")].copy()
    runs["m"] = runs["Modelo"].map(short_model)
    index = ["experiencia", "run_id", "chunk_chars", "chunk_overlap", "top_k", "min_score", "prompt",
             "corpus_sha", "n_excertos"]
    retrieval_cols = [c for c in runs.columns if c.startswith("hit@")] + ["mrr", "token_recall"]
    keep = index + [c for c in retrieval_cols if c in runs]
    base = runs.drop_duplicates("run_id")[keep].set_index("run_id", drop=False).rename_axis("indice")
    for metric in ("Judge", "Contains", "ROUGE_L", "BERTScore_F1"):
        for col, label in ((metric, metric), (f"{metric}_dif", f"Δ{metric}")):
            if col not in runs:
                continue
            wide = runs.pivot_table(index="run_id", columns="m", values=col, aggfunc="first")
            for model in wide.columns:
                base[f"{label} ({model})"] = base["run_id"].map(wide[model])
    for col in ("chunk_chars", "chunk_overlap", "top_k", "n_excertos"):
        base[col] = pd.to_numeric(base[col], errors="coerce").astype("Int64")
    return base.sort_values(["experiencia", "run_id"]).reset_index(drop=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta3", type=Path, default=Path("results/meta3"))
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--out", type=Path, default=None, help="pasta de saída (por omissão, o --meta3)")
    args = ap.parse_args(argv)
    if not args.baseline.is_file():
        raise SystemExit(f"Baseline inexistente: {args.baseline}")
    baseline = pd.read_csv(args.baseline, dtype={"Autor": str})
    rows = []
    for run_dir in sorted(p for p in args.meta3.iterdir() if p.is_dir()):
        rows.extend(collect_run(run_dir, baseline))
    if not rows:
        raise SystemExit(f"Nenhuma execução completa encontrada em {args.meta3}.")
    ids = set()
    for run_dir in args.meta3.iterdir():
        for path in (run_dir / "analises").glob("*_completa/all_results.csv") if run_dir.is_dir() else []:
            ids |= set(pd.read_csv(path, usecols=["id"])["id"])
    long = pd.DataFrame(baseline_rows(baseline, ids) + rows)
    out = args.out or args.meta3
    out.mkdir(parents=True, exist_ok=True)
    long.round(4).to_csv(out / "consolidado.csv", index=False, encoding="utf-8")
    table = report_table(long)
    (out / "tabela_relatorio.md").write_text(
        "# Execuções RAG (subconjunto: todas as perguntas)\n\n"
        "Δ = RAG − baseline sem RAG, emparelhado por pergunta. Referência sem RAG:\n\n"
        + to_markdown(long[(long["experiencia"] == "baseline") & (long["subconjunto"] == "todas")]
                      [["Modelo", "n_perguntas"] + METRICS]) + "\n\n" + to_markdown(table) + "\n",
        encoding="utf-8")
    print(f"{long['run_id'].nunique() - 1} execuções consolidadas -> {out / 'consolidado.csv'}")
    print(f"Tabela para o relatório -> {out / 'tabela_relatorio.md'}")


if __name__ == "__main__":
    main()
