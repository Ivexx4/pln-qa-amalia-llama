"""Validação humana do juiz: amostra estratificada e concordância."""
import numpy as np
import pandas as pd

from . import cache

SAMPLE_SIZE = 100
SAMPLE_SEED = 42
SAMPLE_COLUMNS = ["id", "Modelo", "Tema", "Metodo", "Pergunta", "Gold", "Pred", "juiz", "humano"]
SUMMARY_COLUMNS = ["n", "n_juiz_correto", "n_humano_correto", "acordo_bruto", "kappa",
                   "precisao", "recall", "tp", "fp", "fn", "tn"]


def collect_pairs(df, models, judge_model, run_dir=None) -> pd.DataFrame:
    """Junta previsão, referência e veredito válido por modelo e pergunta."""
    rows = []
    for model in models:
        preds = cache.read_cache(cache.preds_path(model, run_dir))
        verdicts = cache.read_cache(cache.judge_path(model, judge_model, run_dir))
        for _, r in df.iterrows():
            identifier = r["id"]
            pred = preds.get(identifier)
            verdict = verdicts.get(identifier)
            if pred is None or verdict is None:
                continue
            cache.validate_prediction(pred, r["Pergunta"], identifier)
            cache.validate_judgement(verdict, pred["pred"], identifier)
            value = verdict.get("judge")
            if value not in (0, 1):
                continue
            rows.append({
                "id": identifier, "Modelo": model, "Tema": r["Tema"], "Metodo": r["Metodo"],
                "Pergunta": r["Pergunta"], "Gold": r["Resposta"], "Pred": pred["pred"],
                "juiz": int(value),
            })
    if not rows:
        raise ValueError("Não há vereditos válidos para amostrar; executar a fase judge primeiro.")
    return pd.DataFrame(rows, columns=SAMPLE_COLUMNS[:-1]).sort_values(["Modelo", "id"]).reset_index(drop=True)


def stratified_sample(pairs: pd.DataFrame, size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """Amostra reprodutível com metade de cada veredito; usa o máximo disponível por estrato."""
    pairs = pairs.sort_values(["Modelo", "id"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    per_stratum = size // 2
    parts = []
    for value in (1, 0):
        candidates = pairs[pairs["juiz"] == value]
        take = min(per_stratum, len(candidates))
        if take:
            positions = np.sort(rng.choice(len(candidates), size=take, replace=False))
            parts.append(candidates.iloc[positions])
    if not parts:
        raise ValueError("Não há vereditos CORRETO nem INCORRETO para amostrar.")
    return pd.concat(parts).sort_values(["Modelo", "id"]).reset_index(drop=True)


def export_sample(pairs: pd.DataFrame, path, size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """Guarda a amostra com a coluna humano vazia, pronta a preencher com 0 ou 1."""
    sample = stratified_sample(pairs, size, seed).copy()
    sample["humano"] = ""
    sample[SAMPLE_COLUMNS].to_csv(path, index=False, encoding="utf-8")
    counts = sample["juiz"].value_counts().to_dict()
    if len(sample) != size:
        print(f"Aviso: amostra com {len(sample)} pares em vez de {size} "
              f"(disponíveis: {counts}); vereditos em falta reduzem os estratos.")
    print(f"Guardada {path} ({len(sample)} pares; CORRETO={counts.get(1, 0)}, INCORRETO={counts.get(0, 0)}).")
    return sample


def agreement_summary(sample: pd.DataFrame) -> pd.Series:
    """Matriz de confusão e concordância entre o juiz e a validação humana."""
    for column in ("juiz", "humano"):
        values = pd.to_numeric(sample[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"A coluna {column} tem {int(values.isna().sum())} valores em falta; "
                             "a amostra só é válida com 0 ou 1 em todas as linhas.")
        if not values.isin([0, 1]).all():
            raise ValueError(f"A coluna {column} só aceita 0 ou 1.")
        sample = sample.assign(**{column: values.astype(int)})
    judge, human = sample["juiz"], sample["humano"]
    tp = int(((judge == 1) & (human == 1)).sum())
    fp = int(((judge == 1) & (human == 0)).sum())
    fn = int(((judge == 0) & (human == 1)).sum())
    tn = int(((judge == 0) & (human == 0)).sum())
    n = tp + fp + fn + tn
    if not n:
        raise ValueError("A amostra de validação está vazia.")

    observed = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (observed - expected) / (1 - expected) if expected != 1 else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return pd.Series({
        "n": n, "n_juiz_correto": tp + fp, "n_humano_correto": tp + fn,
        "acordo_bruto": observed, "kappa": kappa, "precisao": precision, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }, index=SUMMARY_COLUMNS)


def export_summary(run_dir) -> pd.Series:
    """Lê validacao_juiz.csv preenchido e grava validacao_juiz_resumo.csv."""
    source = run_dir / "validacao_juiz.csv"
    if not source.exists():
        raise ValueError(f"Falta {source}; executar primeiro a fase validate-judge.")
    sample = pd.read_csv(source, encoding="utf-8")
    missing = [column for column in SAMPLE_COLUMNS if column not in sample.columns]
    if missing:
        raise ValueError(f"{source} não tem as colunas {missing}.")
    summary = agreement_summary(sample)
    summary.to_frame().T.to_csv(run_dir / "validacao_juiz_resumo.csv", index=False, encoding="utf-8")
    print(f"Guardado {run_dir / 'validacao_juiz_resumo.csv'}.")
    return summary


def print_summary(summary: pd.Series) -> None:
    print(f"n={int(summary['n'])} | acordo bruto={summary['acordo_bruto']:.3f} | kappa={summary['kappa']:.3f}")
    print(f"precisão={summary['precisao']:.3f} | recall={summary['recall']:.3f}")
    print("matriz juiz (linhas) x humano (colunas):")
    print(f"  CORRETO: {int(summary['tp'])} {int(summary['fp'])}")
    print(f"  INCORRETO: {int(summary['fn'])} {int(summary['tn'])}")
