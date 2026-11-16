"""Fase 4: comparação emparelhada, gráficos e análise de erros."""
import pandas as pd

from . import config, stats


def paired_results(res: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Compara apenas perguntas e métricas disponíveis para todos os modelos."""
    if res.duplicated(["id", "Modelo"]).any():
        raise ValueError("Existem respostas duplicadas para a mesma pergunta e modelo.")
    res = res[res["Modelo"].isin(models)].copy()
    if not models or len(set(models)) != len(models):
        raise ValueError("É necessária uma lista de modelos sem duplicados.")
    for col in ("Pergunta", "Gold", "Tema", "Metodo", "Autor", "split"):
        if res.groupby("id", dropna=False)[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Dados divergentes entre modelos na coluna {col}.")
    coverage = res.groupby("id")["Modelo"].nunique()
    res = res[res["id"].isin(coverage[coverage == len(models)].index)].copy()
    if res.empty:
        raise ValueError("Não há perguntas comuns aos modelos selecionados.")
    for metric in config.METRICS:
        res[metric] = pd.to_numeric(res[metric], errors="coerce")
        available = res.groupby("id")[metric].transform("count") == len(models)
        res.loc[~available, metric] = float("nan")
    return res


def summary_tables(res: pd.DataFrame) -> dict:
    dimensions = {
        "por_modelo": ["Modelo"],
        "por_tema": ["Tema", "Modelo"],
        "por_metodo": ["Metodo", "Modelo"],
        "por_autor": ["Autor", "Modelo"],
    }
    tables = {}
    for name, keys in dimensions.items():
        grouped = res.groupby(keys, dropna=False)
        means = grouped[config.METRICS].mean()
        counts = grouped[config.METRICS].count().add_prefix("n_")
        means.insert(0, "n_respostas", grouped.size())
        tables[name] = means.join(counts)
    return tables


def plot_bars(res: pd.DataFrame, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ("Judge", "BERTScore_F1"):
        for dim in ("Tema", "Metodo"):
            piv = res.pivot_table(index=dim, columns="Modelo", values=metric, aggfunc="mean")
            if piv.empty:
                continue
            ax = piv.sort_values(piv.columns[0]).plot.barh(figsize=(8, max(3, 0.35 * len(piv))))
            ax.set_xlabel(metric)
            plt.tight_layout()
            plt.savefig(out / f"fig_{metric}_{dim}.png", dpi=150)
            plt.close()


def export_error_files(res: pd.DataFrame, out) -> None:
    errors = res[res["Judge"] == 0].sort_values(["Modelo", "Tema"])
    errors[["id", "Modelo", "Tema", "Metodo", "Pergunta", "Gold", "Pred", "ROUGE_L", "BERTScore_F1"]] \
        .to_csv(out / "erros_juiz.csv", index=False, encoding="utf-8")
    disc = res[(res["Judge"] == 1) & (res["EM"] == 0) & (res["ROUGE_L"] < 0.3)]
    disc[["id", "Modelo", "Pergunta", "Gold", "Pred", "ROUGE_L"]] \
        .to_csv(out / "juiz_certo_metrica_lexical_baixa.csv", index=False, encoding="utf-8")
    print(f"Erros (juiz): {len(errors)} | Corretos mas com ROUGE-L<0,3: {len(disc)}")


def export_statistics(out) -> None:
    """Recalcula apenas as tabelas estatísticas de uma análise já exportada."""
    path = out / "all_results.csv"
    if not path.exists():
        raise ValueError(f"Análise inexistente em {out}; execute a pontuação primeiro.")
    res = pd.read_csv(path, dtype={"Autor": str})
    paired = paired_results(res, sorted(res["Modelo"].unique()))
    stats.export_statistical_tables(paired, out)


def analyze(run_dir=None, models=None) -> None:
    out = run_dir or config.RESULTS_DIR
    res = pd.read_csv(out / "all_results.csv", dtype={"Autor": str})
    if res["Estado_experiencia"].eq("parcial").any():
        print("Aviso: análise de uma execução parcial, não de resultados completos.")
    print("Estado dos vereditos:", res["Judge_status"].value_counts().to_dict())
    paired = paired_results(res, models or sorted(res["Modelo"].unique()))
    print(f"Perguntas comuns aos modelos: {paired['id'].nunique()}")
    for name, table in summary_tables(paired).items():
        table = table.round(3)
        table.to_csv(out / f"tabela_{name}.csv", encoding="utf-8")
        print(f"\n== {name} ==\n{table.to_string()}")
    print("\n== Correlação entre métricas ==")
    print(paired[config.METRICS].corr().round(2).to_string())
    plot_bars(paired, out)
    export_error_files(res, out)
    stats.export_statistical_tables(paired, out)