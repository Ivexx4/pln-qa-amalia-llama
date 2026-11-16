"""Fase 3: junta previsões, veredictos do juiz e métricas num único CSV."""
import pandas as pd

from . import cache, config, metrics


def build_results(df, models, judge_model, run_dir=None, allow_partial=False) -> pd.DataFrame:
    rows = []
    problems = []
    for model in models:
        preds = cache.read_cache(cache.preds_path(model, run_dir))
        verdicts = cache.read_cache(cache.judge_path(model, judge_model, run_dir))
        missing_preds = df[~df["id"].isin(preds)]
        missing_judge = df[df["id"].isin(preds) & ~df["id"].isin(verdicts)]
        if not missing_preds.empty or not missing_judge.empty:
            problems.append(f"{model}: {len(missing_preds)} previsões e {len(missing_judge)} vereditos em falta")
        for _, r in df.iterrows():
            identifier = r["id"]
            if identifier not in preds:
                continue
            cache.validate_prediction(preds[identifier], r["Pergunta"], identifier)
            pred = preds[identifier]["pred"]
            verdict = verdicts.get(identifier)
            if verdict is not None:
                cache.validate_judgement(verdict, pred, identifier)
                value = verdict.get("judge")
                if value not in (0, 1, None):
                    raise ValueError(f"Veredito inválido na cache para {identifier}.")
                status = "válido" if value is not None else "inválido"
            else:
                value, status = None, "em falta"
            rows.append({
                "id": identifier, "split": r["split"], "Autor": r["Autor"], "Tema": r["Tema"],
                "Metodo": r["Metodo"], "Pergunta": r["Pergunta"], "Gold": r["Resposta"],
                "Modelo": model, "Pred": pred,
                "EM": metrics.exact_match(pred, r["Resposta"]),
                "Contains": metrics.contains(pred, r["Resposta"]),
                "ROUGE_L": metrics.rouge_l(pred, r["Resposta"]),
                "Judge": value, "Judge_status": status,
                "Estado_experiencia": "parcial" if allow_partial else "completa",
            })
    if problems and not allow_partial:
        raise ValueError("Resultados incompletos: " + "; ".join(problems) + ".")
    if problems:
        print("Aviso: resultados parciais — " + "; ".join(problems))
    if not rows:
        raise ValueError("Não existem previsões para pontuar.")
    return pd.DataFrame(rows)


def bert_score_layers(bert_model: str, num_layers: int | None = None) -> int | None:
    if num_layers is None and bert_model == config.DEFAULT_BERT_MODEL:
        return config.DEFAULT_BERTSCORE_NUM_LAYERS
    return num_layers


def add_bertscore(res: pd.DataFrame, bert_model: str, batch_size: int = config.DEFAULT_BERTSCORE_BATCH_SIZE,
                  num_layers: int | None = None) -> pd.DataFrame:
    from bert_score import BERTScorer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_layers = bert_score_layers(bert_model, num_layers)
    options = {"num_layers": num_layers} if num_layers is not None else {}
    scorer = BERTScorer(model_type=bert_model, batch_size=batch_size, device=device, **options)
    # Albertina é distribuído em bfloat16; DeBERTa requer precisão uniforme na atenção.
    scorer._model.float()
    _, _, f1 = scorer.score(res["Pred"].tolist(), res["Gold"].tolist(),
                            batch_size=batch_size, verbose=False)
    res["BERTScore_F1"] = f1.detach().cpu().numpy()
    return res


def score(df, models, judge_model, bert_model, run_dir=None, allow_partial=False,
          batch_size=config.DEFAULT_BERTSCORE_BATCH_SIZE, output_dir=None, scope="both", num_layers=None):
    res = build_results(df, models, judge_model, run_dir, allow_partial)
    res["Escopo"] = scope
    invalid = res["Judge_status"].eq("inválido").sum()
    print(f"Escopo: {scope}. Vereditos inválidos: {invalid} (excluídos das médias do juiz).")
    res = add_bertscore(res, bert_model, batch_size, num_layers)
    out = (output_dir or run_dir or config.RESULTS_DIR) / "all_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False, encoding="utf-8")
    print(f"Guardado {out} ({len(res)} linhas)")