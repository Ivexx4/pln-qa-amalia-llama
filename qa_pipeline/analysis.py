"""Fase 4: tabelas, gráficos e ficheiros de apoio à análise de erros."""
import pandas as pd

from . import config

OUT = config.RESULTS_DIR


def summary_tables(res: pd.DataFrame) -> dict:
    m = config.METRICS
    return {
        "por_modelo": res.groupby("Modelo")[m].mean(),
        "por_tema": res.groupby(["Tema", "Modelo"])[m].mean(),
        # "Autor da pergunta" = HUM vs LLM (coluna Método)
        "por_metodo": res.groupby(["Metodo", "Modelo"])[m].mean(),
    }


def plot_bars(res: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ("Judge", "BERTScore_F1"):
        for dim in ("Tema", "Metodo"):
            piv = res.pivot_table(index=dim, columns="Modelo", values=metric, aggfunc="mean")
            ax = piv.sort_values(piv.columns[0]).plot.barh(
                figsize=(8, max(3, 0.35 * len(piv))))
            ax.set_xlabel(metric)
            plt.tight_layout()
            plt.savefig(OUT / f"fig_{metric}_{dim}.png", dpi=150)
            plt.close()


def export_error_files(res: pd.DataFrame) -> None:
    errors = res[res["Judge"] == 0].sort_values(["Modelo", "Tema"])
    errors[["Modelo", "Tema", "Metodo", "Pergunta", "Gold", "Pred", "ROUGE_L", "BERTScore_F1"]] \
        .to_csv(OUT / "erros_juiz.csv", index=False, encoding="utf-8")
    disc = res[(res["Judge"] == 1) & (res["EM"] == 0) & (res["ROUGE_L"] < 0.3)]
    disc[["Modelo", "Pergunta", "Gold", "Pred", "ROUGE_L"]] \
        .to_csv(OUT / "juiz_certo_metrica_lexical_baixa.csv", index=False, encoding="utf-8")
    print(f"\nErros (juiz): {len(errors)} | Corretos mas com ROUGE-L<0.3: {len(disc)}")


def analyze() -> None:
    res = pd.read_csv(OUT / "all_results.csv")
    res["Judge"] = pd.to_numeric(res["Judge"], errors="coerce")
    print(f"Respostas sem veredito do juiz: {res['Judge'].isna().sum()}")

    for name, table in summary_tables(res).items():
        table = table.round(3)
        table.to_csv(OUT / f"tabela_{name}.csv", encoding="utf-8")
        print(f"\n== {name} ==\n{table.to_string()}")

    print("\n== Correlação entre métricas ==")
    print(res[config.METRICS].corr().round(2).to_string())

    plot_bars(res)
    export_error_files(res)
