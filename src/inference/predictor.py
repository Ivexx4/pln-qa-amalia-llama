"""Generate and checkpoint answers for each configured model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DEFAULT_OLLAMA_URL, OLLAMA_TIMEOUT_SECONDS
from src.dataset.models import DATASET_COLUMNS
from src.inference.ollama_client import OllamaClient
from src.utils.io import write_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)
RESPONSE_COLUMNS = [
    "tema",
    "metodo",
    "pergunta",
    "resposta_correta",
    "resposta_amalia",
    "resposta_llama",
]


def generate_responses(
    questions: pd.DataFrame,
    output_path: str | Path,
    amalia_model: str,
    llama_model: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = OLLAMA_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Generate answers and persist progress after each question/model."""
    missing = set(DATASET_COLUMNS) - set(questions.columns)
    if missing:
        raise ValueError(f"Colunas em falta no dataset: {', '.join(sorted(missing))}.")
    if questions.empty:
        raise ValueError("Não existem perguntas para processar.")

    responses = pd.DataFrame(
        {
            "tema": questions["tema"].tolist(),
            "metodo": questions["metodo"].tolist(),
            "pergunta": questions["pergunta"].tolist(),
            "resposta_correta": questions["resposta"].tolist(),
            "resposta_amalia": [""] * len(questions),
            "resposta_llama": [""] * len(questions),
        },
        columns=RESPONSE_COLUMNS,
    )
    destination = Path(output_path)
    write_csv(responses, destination)

    model_specs = (
        ("AMALIA", amalia_model, "resposta_amalia"),
        ("Llama 3.1 8B", llama_model, "resposta_llama"),
    )
    for display_name, model_name, output_column in model_specs:
        client = OllamaClient(model_name, ollama_url, timeout)
        logger.info("A gerar respostas com %s (%s).", display_name, model_name)
        try:
            for index, question in questions.reset_index(drop=True).iterrows():
                responses.at[index, output_column] = client.generate(
                    str(question["pergunta"])
                )
                write_csv(responses, destination)
                logger.info("%s: pergunta %d/%d concluída.", display_name, index + 1, len(questions))
        finally:
            client.close()
    return responses
