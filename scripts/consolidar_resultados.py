from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

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


def compute_behavioral_counts(part: pd.DataFrame) -> dict:
    """Calcula estimativas comportamentais (alucinações vs recusas) com base no texto das respostas."""
    if "Resposta" not in part.columns:
        return {"alucinacoes": 0, "recusas": 0}

    # Exemplo heurístico robusto com base em palavras-chave comuns em português/inglês
    respostas = part["Resposta"].astype(str).str.lower()

    # Recusas: o modelo indica que não sabe / não tem informação
    recusas_keywords = ["não sei", "não tenho", "desconheço", "não é possível", "no context", "not found", "não consta"]
    num_recusas = respostas.apply(lambda x: any(kw in x for kw in recusas_keywords)).sum()

    # Alucinações estimadas: baixa pontuação de Judge (< 0.2) mas o modelo deu uma resposta longa/afirmativa
    if "Judge" in part.columns:
        judge_vals = pd.to_numeric(part["Judge"], errors="coerce")
        num_alucinacoes = ((judge_vals < 0.2) & (respostas.str.len() > 50)).sum()
    else:
        num_alucinacoes = 0

    return {"alucinacoes": int(num_alucinacoes), "recusas": int(num_recusas)}


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
                behav = compute_behavioral_counts(part)
                row = {"run_id": run_dir.name, "experiencia": experiment_of(run_dir.name), "escopo": analysis.name,
                       **{p: settings.get(p) for p in PARAMS},
                       "corpus_sha": str(corpus.get("sha256", ""))[:8], "n_excertos": corpus.get("chunks"),
                       "Modelo": model, "subconjunto": subset_name, "n_perguntas": part["id"].nunique(),
                       **retrieval, **paired_means(base, part), **behav}
                rows.append(row)
    return rows


def baseline_rows(baseline: pd.DataFrame, ids: set) -> list[dict]:
    rows = []
    for model, group in baseline[baseline["id"].isin(ids)].groupby("Modelo"):
        for subset_name, part in subsets(group):
            behav = compute_behavioral_counts(part)
            row = {"run_id": "baseline-sem-rag", "experiencia": "baseline", "Modelo": model,
                   "subconjunto": subset_name, "n_perguntas": part["id"].nunique(), **behav}
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
    for metric in METRICS:
        for col, label in ((metric, metric), (f"{metric}_dif", f"Δ{metric}")):
            if col not in runs:
                continue
            wide = runs.pivot_table(index="run_id", columns="m", values=col, aggfunc="first")
            for model in wide.columns:
                base[f"{label} ({model})"] = base["run_id"].map(wide[model])
    for col in ("chunk_chars", "chunk_overlap", "top_k", "n_excertos"):
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce").astype("Int64")
    return base.sort_values(["experiencia", "run_id"]).reset_index(drop=True)


def best_config_by_model_table(long: pd.DataFrame) -> str:
    df_rag = long[(long["subconjunto"] == "todas") & (long["experiencia"] != "baseline")]
    if df_rag.empty or "Judge_dif" not in df_rag.columns:
        return "Sem dados suficientes."
    rows = []
    for modelo, grupo in df_rag.groupby("Modelo"):
        if grupo.empty:
            continue
        best_row = grupo.loc[grupo["Judge_dif"].idxmax()]
        rows.append({
            "Modelo": modelo,
            "Melhor Run ID": best_row["run_id"],
            "Experiência": best_row["experiencia"],
            "Δ Judge": best_row.get("Judge_dif", 0),
            "top_k": best_row.get("top_k"),
            "chunk_chars": best_row.get("chunk_chars")
        })
    return to_markdown(pd.DataFrame(rows))


def generate_insights_text(long: pd.DataFrame) -> str:
    df_rag = long[(long["subconjunto"] == "todas") & (long["experiencia"] != "baseline")]
    if df_rag.empty:
        return "Sem dados suficientes."

    insights = []
    insights.append("## 📌 Sumário Executivo e Comportamento dos Modelos\n")
    insights.append(
        "Abaixo encontram-se as configurações ótimas e o comportamento geral mapeado nas execuções:\n"
    )

    if "Judge_dif" in df_rag.columns:
        for modelo, grupo in df_rag.groupby("Modelo"):
            if grupo.empty:
                continue
            best_row = grupo.loc[grupo["Judge_dif"].idxmax()]
            insights.append(
                f"- **Modelo `{modelo}`:** Pico de eficácia no `run_id = **{best_row['run_id']}**` "
                f"(Δ Judge = `{best_row.get('Judge_dif', 0):.3f}`)."
            )
    return "\n".join(insights)


def generate_plots(long: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "graficos"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    df_rag = long[(long["subconjunto"] == "todas") & (long["experiencia"] != "baseline")].copy()
    if df_rag.empty:
        return

    # a) Evolução dos Modelos por Múltiplas Métricas (Radar / Barras agrupadas das 3 principais)
    plt.figure(figsize=(11, 6))
    metrics_subset = [m for m in ["Judge", "ROUGE_L", "BERTScore_F1"] if m in df_rag.columns]
    if metrics_subset:
        df_melt = df_rag.melt(id_vars=["Modelo", "run_id"], value_vars=metrics_subset, var_name="Métrica", value_name="Valor")
        if HAS_SEABORN:
            sns.barplot(data=df_melt, x="run_id", y="Valor", hue="Métrica", errorbar=None)
            plt.xticks(rotation=45, ha="right")
        else:
            df_melt.groupby(["run_id", "Métrica"])["Valor"].mean().unstack().plot(kind="bar", figsize=(11, 6))
        plt.title("a) Evolução de Desempenho por Indicador (Judge, ROUGE-L, BERTScore)", fontsize=11, fontweight="bold")
        plt.xlabel("Execução (run_id)")
        plt.ylabel("Média da Métrica")
        plt.tight_layout()
        plt.savefig(plots_dir / "evolucao_modelos_3_indicadores.png", dpi=300)
        plt.close()

    # b) Quantidade de vezes que os modelos dizem saber coisas sem o saber (Alucinações estimadas)
    if "alucinacoes" in long.columns:
        plt.figure(figsize=(9, 5))
        if HAS_SEABORN:
            sns.barplot(data=long, x="experiencia", y="alucinacoes", hue="Modelo", estimator=sum, errorbar=None)
        else:
            long.groupby(["experiencia", "Modelo"])["alucinacoes"].sum().unstack().plot(kind="bar", figsize=(9, 5))
        plt.title("b) Alucinações Estimadas (Dizer saber sem fundamento)", fontsize=11, fontweight="bold")
        plt.xlabel("Experiência")
        plt.ylabel("Contagem de Ocorrências")
        plt.tight_layout()
        plt.savefig(plots_dir / "alucinacoes_por_experiencia.png", dpi=300)
        plt.close()

    # c) Quantidade de vezes que os modelos indicam que não sabem (Recusas / Fallbacks)
    if "recusas" in long.columns:
        plt.figure(figsize=(9, 5))
        if HAS_SEABORN:
            sns.barplot(data=long, x="experiencia", y="recusas", hue="Modelo", estimator=sum, errorbar=None)
        else:
            long.groupby(["experiencia", "Modelo"])["recusas"].sum().unstack().plot(kind="bar", figsize=(9, 5))
        plt.title("c) Recusas Explícitas de Resposta (Indicam que não sabem)", fontsize=11, fontweight="bold")
        plt.xlabel("Experiência")
        plt.ylabel("Contagem de Recusas")
        plt.tight_layout()
        plt.savefig(plots_dir / "recusas_por_experiencia.png", dpi=300)
        plt.close()

    # d) Métricas originais avaliadas (Comparativo global RAG vs Baseline)
    base_df = long[(long["subconjunto"] == "todas") & (long["experiencia"] == "baseline")]
    if not base_df.empty:
        plt.figure(figsize=(10, 6))
        metrics_to_plot = [m for m in METRICS if m in df_rag.columns and m in base_df.columns]
        if metrics_to_plot:
            rag_means = df_rag[metrics_to_plot].mean()
            base_means = base_df[metrics_to_plot].mean()
            comp_df = pd.DataFrame({"Com RAG": rag_means, "Sem RAG (Baseline)": base_means})
            comp_df.plot(kind="bar", figsize=(10, 6), rot=0)
            plt.title("d) Métricas Avaliadas no Original (RAG vs Baseline)", fontsize=11, fontweight="bold")
            plt.ylabel("Média da Métrica")
            plt.ylim(0, 1.0)
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(plots_dir / "metricas_originais_comparativo.png", dpi=300)
            plt.close()

    print(f"📈 Gráficos analíticos gerados e guardados em -> {plots_dir}")

def generate_table_comparison_plots(long: pd.DataFrame, out_dir: Path) -> None:
    """Gera gráficos comparativos detalhados por modelo com base nos dados da tabela de relatório."""
    plots_dir = out_dir / "graficos"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if HAS_SEABORN:
        sns.set_theme(style="whitegrid")

    df_rag = long[(long["subconjunto"] == "todas") & (long["experiencia"] != "baseline")].copy()
    if df_rag.empty:
        return

    # 1. Comparativo de Deltas (Δ Judge, Δ BERTScore, etc.) por Modelo e Experiência
    delta_cols = [c for c in df_rag.columns if c.endswith("_dif")]
    if delta_cols:
        plt.figure(figsize=(12, 6))
        # Derreter o dataframe para o seaborn plotar limpo por modelo e métrica de delta
        df_melt_delta = df_rag.melt(
            id_vars=["run_id", "Modelo", "experiencia"],
            value_vars=delta_cols,
            var_name="Metrica_Dif",
            value_name="Valor_Dif"
        )
        if HAS_SEABORN:
            sns.barplot(data=df_melt_delta, x="run_id", y="Valor_Dif", hue="Modelo", errorbar=None)
            plt.xticks(rotation=45, ha="right")
        else:
            df_melt_delta.pivot_table(index="run_id", columns="Modelo", values="Valor_Dif").plot(kind="bar", figsize=(12, 6))

        plt.title("Comparativo de Deltas (RAG vs Baseline) por Execução e Modelo", fontsize=12, fontweight="bold")
        plt.xlabel("Execução (run_id)")
        plt.ylabel("Diferença Média (Δ)")
        plt.axhline(0, color="grey", linestyle="--", linewidth=1)
        plt.legend(title="Modelo")
        plt.tight_layout()
        plt.savefig(plots_dir / "comparativo_deltas_por_modelo.png", dpi=300)
        plt.close()

    # 2. Desempenho absoluto das métricas principais por Modelo em cada Experiência
    for metric in ["Judge", "BERTScore_F1", "ROUGE_L"]:
        if metric in df_rag.columns:
            plt.figure(figsize=(10, 5))
            if HAS_SEABORN:
                sns.barplot(data=df_rag, x="experiencia", y=metric, hue="Modelo", errorbar=None)
            else:
                df_rag.groupby(["experiencia", "Modelo"])[metric].mean().unstack().plot(kind="bar", figsize=(10, 5))

            plt.title(f"Desempenho Médio de {metric} por Experiência e Modelo", fontsize=12, fontweight="bold")
            plt.xlabel("Tipo de Experiência")
            plt.ylabel(f"Média de {metric}")
            plt.ylim(0, 1.0)
            plt.legend(title="Modelo", loc="lower right")
            plt.tight_layout()
            plt.savefig(plots_dir / f"comparativo_modelo_{metric.lower()}.png", dpi=300)
            plt.close()

    print(f"📊 Gráficos comparativos por modelo gerados com sucesso na pasta -> {plots_dir}")

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta3", type=Path, default=Path("results/meta3"))
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--out", type=Path, default=None, help="pasta de saída")
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
    generate_table_comparison_plots(long, out)
    long.round(4).to_csv(out / "consolidado.csv", index=False, encoding="utf-8")
    table = report_table(long)
    insights_md = generate_insights_text(long)
    model_table = best_config_by_model_table(long)

    relatorio_conteudo = (
            "# 📊 Relatório Avançado de Avaliação RAG\n\n"
            f"{insights_md}\n\n"
            "## 🏆 Configurações Ótimas Isoladas por Modelo\n\n"
            f"{model_table}\n\n"
            "---\n\n"
            "## 📋 Tabela Detalhada de Execuções\n\n"
            + to_markdown(table) + "\n\n"
                                   "---\n*Gerado automaticamente pelo script analítico.*\n"
    )

    (out / "tabela_relatorio.md").write_text(relatorio_conteudo, encoding="utf-8")
    generate_plots(long, out)

    print(f"✨ Concluído! Relatório e gráficos gerados com sucesso na diretoria: {out}")


if __name__ == "__main__":
    main()