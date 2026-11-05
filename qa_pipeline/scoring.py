"""Fase 3: junta previsões, veredictos do juiz e métricas num único CSV."""
import pandas as pd

from . import cache, config, metrics


def build_results(df, models, judge_model) -> pd.DataFrame:
    rows = []
    for model in models:
        preds = cache.read_cache(cache.preds_path(model))
        verdicts = cache.read_cache(cache.judge_path(model, judge_model))
        for _, r in df.iterrows():
            if r["id"] not in preds:
                continue
            pred = preds[r["id"]]["pred"]
            rows.append({
                "id": r["id"], "split": r["split"], "Autor": r["Autor"], "Tema": r["Tema"],
                "Metodo": r["Metodo"], "Pergunta": r["Pergunta"], "Gold": r["Resposta"],
                "Modelo": model, "Pred": pred,
                "EM": metrics.exact_match(pred, r["Resposta"]),
                "Contains": metrics.contains(pred, r["Resposta"]),
                "ROUGE_L": metrics.rouge_l(pred, r["Resposta"]),
                "Judge": verdicts.get(r["id"], {}).get("judge"),
            })
    return pd.DataFrame(rows)


def add_bertscore(res: pd.DataFrame, bert_model: str) -> pd.DataFrame:
    from bert_score import score as bert_score  # import tardio: biblioteca pesada
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, f1 = bert_score(res["Pred"].tolist(), res["Gold"].tolist(),
                          model_type=bert_model, batch_size=32, device=device,
                          verbose=False)
    res["BERTScore_F1"] = f1.detach().cpu().numpy()
    return res


def score(df, models, judge_model, bert_model):
    res = add_bertscore(build_results(df, models, judge_model), bert_model)
    out = config.RESULTS_DIR / "all_results.csv"
    res.to_csv(out, index=False, encoding="utf-8")
    print(f"Guardado {out} ({len(res)} linhas)")
