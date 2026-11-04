"""Multilingual BERTScore F1 calculation for Portuguese."""

from __future__ import annotations

from collections.abc import Sequence


def bertscore_f1(
    references: Sequence[str],
    predictions: Sequence[str],
    device: str = "auto",
    batch_size: int = 8,
) -> list[float]:
    """Calculate XLM-R BERTScore F1, leaving empty predictions at score zero."""
    if batch_size < 1:
        raise ValueError("O batch size tem de ser um inteiro positivo.")

    import torch
    from bert_score import BERTScorer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "Foi pedido BERTScore em CUDA, mas CUDA não está disponível."
        )

    scores = [0.0] * len(predictions)
    nonempty = [i for i, prediction in enumerate(predictions) if prediction.strip()]
    if not nonempty:
        return scores

    scorer = BERTScorer(
        model_type="xlm-roberta-base",
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=False,
    )
    _, _, f1 = scorer.score(
        [predictions[i] for i in nonempty],
        [references[i] for i in nonempty],
    )
    for index, value in zip(nonempty, f1.tolist()):
        scores[index] = float(value)
    return scores
