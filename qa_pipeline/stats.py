"""Estatísticas descritivas e testes emparelhados sobre perguntas comuns."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from . import config

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42

STAT_COLUMNS = [
    "tipo", "metrica", "modelo", "modelo_comparado", "n", "estimativa",
    "ic_95_inferior", "ic_95_superior", "teste", "estatistica", "p_valor",
    "discordantes_amalia", "discordantes_outro",
]
THEME_COLUMNS = [
    "Tema", "Modelo", "metrica", "n_tema", "n", "aviso", "media",
    "ic_95_inferior", "ic_95_superior",
]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Média e IC percentil de 95%; menos de duas observações não permite estimar IC."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) < 2:
        return mean, np.nan, np.nan

    means = np.empty(BOOTSTRAP_SAMPLES)
    # Limita a memória temporária, mesmo para conjuntos de perguntas maiores.
    batch_size = max(1, min(BOOTSTRAP_SAMPLES, 1_000_000 // len(values)))
    for start in range(0, BOOTSTRAP_SAMPLES, batch_size):
        size = min(batch_size, BOOTSTRAP_SAMPLES - start)
        indexes = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[indexes].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return mean, float(lower), float(upper)


def paired_test(metric: str, reference: np.ndarray, other: np.ndarray) -> dict:
    """Testes bicaudais; as duas sequências têm de estar alinhadas por pergunta."""
    if metric not in ("Judge", "Contains", "ROUGE_L", "BERTScore_F1"):
        return {}
    test = "McNemar exato" if metric in ("Judge", "Contains") else "Wilcoxon"
    result = {"teste": test, "estatistica": np.nan, "p_valor": np.nan}
    if not len(reference):
        return result
    if metric in ("Judge", "Contains"):
        if not (np.isin(reference, [0, 1]).all() and np.isin(other, [0, 1]).all()):
            raise ValueError(f"A métrica {metric} tem de ser binária para o teste de McNemar.")
        b = int(((reference == 1) & (other == 0)).sum())
        c = int(((reference == 0) & (other == 1)).sum())
        result.update({"discordantes_amalia": b, "discordantes_outro": c,
                       "estatistica": min(b, c),
                       "p_valor": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0})
    else:
        differences = reference - other
        if np.all(differences == 0):
            result.update({"estatistica": 0.0, "p_valor": 1.0})
        else:
            test_result = wilcoxon(differences, alternative="two-sided", zero_method="wilcox", method="auto")
            result.update({"estatistica": float(test_result.statistic), "p_valor": float(test_result.pvalue)})
    return result


def statistical_tables(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula médias, diferenças e IC a partir da saída de analysis.paired_results."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    models = sorted(paired["Modelo"].unique())
    reference = config.DEFAULT_AMALIA_MODEL
    # Ordenar por id torna a reamostragem independente da ordem das linhas de entrada.
    questions = paired.drop_duplicates("id").set_index("id").sort_index()
    themes = list(questions.groupby("Tema", dropna=False, sort=True))
    global_rows, theme_rows = [], []

    for metric in config.METRICS:
        wide = paired.pivot(index="id", columns="Modelo", values=metric).sort_index()
        wide = wide.reindex(columns=models).apply(pd.to_numeric, errors="coerce")
        # Todos os modelos usam exatamente as mesmas perguntas para cada métrica.
        wide = wide.where(np.isfinite(wide)).dropna()
        for model in models:
            mean, lower, upper = bootstrap_mean(wide[model].to_numpy(), rng)
            global_rows.append({"tipo": "media", "metrica": metric, "modelo": model,
                                "n": len(wide), "estimativa": mean,
                                "ic_95_inferior": lower, "ic_95_superior": upper})

        if reference in models:
            for model in models:
                if model == reference:
                    continue
                ref_values = wide[reference].to_numpy()
                other_values = wide[model].to_numpy()
                # Reamostrar diferenças preserva o par da mesma pergunta.
                mean, lower, upper = bootstrap_mean(ref_values - other_values, rng)
                row = {"tipo": "diferenca", "metrica": metric, "modelo": reference,
                       "modelo_comparado": model, "n": len(wide), "estimativa": mean,
                       "ic_95_inferior": lower, "ic_95_superior": upper}
                row.update(paired_test(metric, ref_values, other_values))
                global_rows.append(row)

        for theme, group in themes:
            theme_values = wide.loc[wide.index.isin(group.index)]
            for model in models:
                mean, lower, upper = bootstrap_mean(theme_values[model].to_numpy(), rng)
                n = len(theme_values)
                theme_rows.append({"Tema": theme, "Modelo": model, "metrica": metric,
                                   "n_tema": len(group), "n": n, "aviso": "n<30" if n < 30 else "",
                                   "media": mean, "ic_95_inferior": lower, "ic_95_superior": upper})

    return (pd.DataFrame(global_rows, columns=STAT_COLUMNS),
            pd.DataFrame(theme_rows, columns=THEME_COLUMNS))


def export_statistical_tables(paired: pd.DataFrame, out: Path) -> None:
    """Guarda as estatísticas globais e por tema na pasta da análise."""
    global_table, theme_table = statistical_tables(paired)
    global_table.to_csv(out / "tabela_estatistica.csv", index=False, encoding="utf-8")
    theme_table.to_csv(out / "tabela_por_tema_ic.csv", index=False, encoding="utf-8")
    if config.DEFAULT_AMALIA_MODEL not in paired["Modelo"].unique():
        print("Aviso: AMALIA não está presente; não há comparações AMALIA − outro modelo.")
    print("Guardadas tabela_estatistica.csv e tabela_por_tema_ic.csv.")
