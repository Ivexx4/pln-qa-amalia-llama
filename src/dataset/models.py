"""Canonical question and answer data model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QAExample:
    """A dataset example with its category, source and reference answer."""

    tema: str
    pergunta: str
    metodo: str
    resposta: str


DATASET_COLUMNS = ("tema", "pergunta", "metodo", "resposta")
