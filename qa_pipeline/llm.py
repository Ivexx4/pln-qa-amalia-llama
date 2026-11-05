"""Cliente mínimo para a API do Ollama."""
import logging
import time

import requests

from . import config

logger = logging.getLogger(__name__)


def chat(model: str, messages: list, max_tokens: int = config.GEN_MAX_TOKENS) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": config.TEMPERATURE,
            "seed": config.SEED,
            "num_predict": max_tokens,
        },
    }

    for attempt in range(1, config.HTTP_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                f"{config.OLLAMA_URL}/api/chat",
                json=payload,
                timeout=config.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.exceptions.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if (
                status not in config.RETRYABLE_HTTP_STATUSES
                or attempt == config.HTTP_MAX_ATTEMPTS
            ):
                raise
            logger.warning(
                "Ollama respondeu HTTP %s; nova tentativa %d/%d.",
                status,
                attempt + 1,
                config.HTTP_MAX_ATTEMPTS,
            )
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as error:
            if attempt == config.HTTP_MAX_ATTEMPTS:
                raise
            logger.warning(
                "Falha temporária na comunicação com Ollama (%s); "
                "nova tentativa %d/%d.",
                error,
                attempt + 1,
                config.HTTP_MAX_ATTEMPTS,
            )

        time.sleep(config.HTTP_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))

    raise RuntimeError("A chamada ao Ollama terminou sem resposta.")
