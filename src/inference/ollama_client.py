"""Client for Ollama's local chat API."""

from __future__ import annotations

from typing import Protocol

import requests

from .prompts import build_system_prompt, build_user_prompt


class AnswerGenerator(Protocol):
    """Interface implemented by Ollama and future RAG-backed generators."""

    def generate(self, question: str, context: str | None = None) -> str: ...


class OllamaClient:
    """Send deterministic chat requests to a locally running Ollama server."""

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: int,
    ) -> None:
        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/api/chat"
        self.timeout = timeout
        self.session = requests.Session()

    def generate(self, question: str, context: str | None = None) -> str:
        """Generate one answer, raising an explicit error on API failure."""
        try:
            response = self.session.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": build_system_prompt(context)},
                        {"role": "user", "content": build_user_prompt(question, context)},
                    ],
                    "stream": False,
                    "options": {"temperature": 0, "seed": 42, "num_predict": 128},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(
                f"Falha ao consultar Ollama para '{self.model}': {error}"
            ) from error

        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"Ollama reportou um erro: {payload['error']}")
        try:
            answer = payload["message"]["content"].strip()
        except (KeyError, AttributeError, TypeError) as error:
            raise RuntimeError(
                f"Resposta inesperada do Ollama para '{self.model}'."
            ) from error
        if not answer:
            raise RuntimeError(f"O modelo '{self.model}' devolveu resposta vazia.")
        return answer

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self.session.close()
