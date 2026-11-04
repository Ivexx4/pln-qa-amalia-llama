"""ROUGE-L F-measure metric."""

from rouge_score import rouge_scorer


def rouge_l(reference: str, prediction: str) -> float:
    """Calculate ROUGE-L F-measure without stemming."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return scorer.score(reference, prediction)["rougeL"].fmeasure
