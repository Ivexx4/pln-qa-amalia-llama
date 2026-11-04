"""Evaluate responses per question and model."""

from __future__ import annotations

import pandas as pd

from .bertscore import bertscore_f1
from .exact_match import exact_match
from .rouge import rouge_l

RESPONSE_COLUMNS = (
    "tema",
    "metodo",
    "pergunta",
    "resposta_correta",
    "resposta_amalia",
    "resposta_llama",
)
METRIC_COLUMNS = (
    "tema",
    "metodo",
    "modelo",
    "exact_match",
    "rougeL",
    "bertscore",
)
MODEL_OUTPUT_COLUMNS = {
    "AMALIA": "resposta_amalia",
    "Llama 3.1 8B": "resposta_llama",
}


def evaluate_responses(
    responses: pd.DataFrame,
    device: str = "auto",
    batch_size: int = 8,
) -> pd.DataFrame:
    """Calculate all requested metrics for each model and dataset question."""
    missing = set(RESPONSE_COLUMNS) - set(responses.columns)
    if missing:
        raise ValueError(f"Colunas em falta: {', '.join(sorted(missing))}.")
    if responses.empty:
        raise ValueError("Não existem respostas para avaliar.")

    references = responses["resposta_correta"].fillna("").astype(str).tolist()
    predictions_by_model = {
        model_name: responses[column].fillna("").astype(str).tolist()
        for model_name, column in MODEL_OUTPUT_COLUMNS.items()
    }
    all_predictions = [
        prediction
        for predictions in predictions_by_model.values()
        for prediction in predictions
    ]
    all_references = references * len(predictions_by_model)
    all_bert_scores = bertscore_f1(
        all_references,
        all_predictions,
        device=device,
        batch_size=batch_size,
    )
    count = len(references)
    results: list[dict[str, object]] = []
    for model_index, (model_name, predictions) in enumerate(
        predictions_by_model.items()
    ):
        bert_scores = all_bert_scores[
            model_index * count : (model_index + 1) * count
        ]
        for index, row in responses.reset_index(drop=True).iterrows():
            prediction = predictions[index]
            reference = references[index]
            results.append(
                {
                    "tema": row["tema"],
                    "metodo": row["metodo"],
                    "modelo": model_name,
                    "exact_match": exact_match(reference, prediction),
                    "rougeL": rouge_l(reference, prediction),
                    "bertscore": bert_scores[index],
                }
            )
    return pd.DataFrame(results, columns=METRIC_COLUMNS)
