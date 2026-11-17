"""Embeddings através da API do Ollama (/api/embed)."""
from __future__ import annotations

import logging
import time

import requests

from . import config, rag_config

logger = logging.getLogger(__name__)


def embed(model: str, texts: list[str]) -> list[list[float]]:
    """Devolve um vetor por texto, pela mesma ordem."""
    if not texts:
        return []
    for attempt in range(1, config.HTTP_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(f"{config.OLLAMA_URL}/api/embed",
                                     json={"model": model, "input": texts},
                                     timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            vectors = response.json()["embeddings"]
            if len(vectors) != len(texts):
                raise ValueError(f"O Ollama devolveu {len(vectors)} vetores para {len(texts)} textos.")
            return vectors
        except requests.exceptions.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status not in config.RETRYABLE_HTTP_STATUSES or attempt == config.HTTP_MAX_ATTEMPTS:
                raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if attempt == config.HTTP_MAX_ATTEMPTS:
                raise
        logger.warning("Falha temporária nos embeddings; nova tentativa %d.", attempt + 1)
        time.sleep(config.HTTP_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
    raise RuntimeError("Os embeddings terminaram sem resposta.")


def embed_all(model: str, texts: list[str], batch_size: int = rag_config.EMBED_BATCH_SIZE) -> list[list[float]]:
    from tqdm import tqdm
    vectors: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"embeddings({model})"):
        vectors.extend(embed(model, texts[start:start + batch_size]))
    return vectors
