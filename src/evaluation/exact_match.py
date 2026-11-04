"""Exact string-match metric."""


def exact_match(reference: str, prediction: str) -> int:
    """Return 1 only when prediction and reference are literally identical."""
    return int(prediction == reference)
