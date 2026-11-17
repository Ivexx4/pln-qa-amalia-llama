"""Parâmetros do RAG num único objeto: ficheiro JSON -> sobreposições -> validação -> identificação.

Precedência (da menor para a maior): valores por omissão < ficheiro --config < opções da CLI < --set chave=valor.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from . import cache, config, rag_config
from .rag_prompts import PROMPTS

STORES = ("chroma", "memory")
_INT = {"chunk_chars", "chunk_overlap", "top_k"}
_FLOAT = {"min_score"}
# Parâmetros que determinam o índice vetorial; os restantes só afetam a recuperação ou a geração.
INDEX_FIELDS = ("embed_model", "chunk_chars", "chunk_overlap")


@dataclass(frozen=True)
class RagSettings:
    theme: str
    models: tuple = tuple(config.DEFAULT_MODELS)
    judge_model: str = config.DEFAULT_JUDGE
    embed_model: str = rag_config.DEFAULT_EMBED_MODEL
    chunk_chars: int = rag_config.CHUNK_CHARS
    chunk_overlap: int = rag_config.CHUNK_OVERLAP_CHARS
    top_k: int = rag_config.DEFAULT_TOP_K
    min_score: float | None = None  # descarta excertos com similaridade do cosseno inferior
    store: str = rag_config.DEFAULT_STORE
    prompt: str = "default"

    def __post_init__(self):
        if not self.theme:
            raise ValueError("O tema é obrigatório.")
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models tem de ter pelo menos um modelo, sem repetições.")
        if self.top_k < 1:
            raise ValueError("top_k deve ser >= 1.")
        if self.chunk_chars < 100 or not 0 <= self.chunk_overlap < self.chunk_chars:
            raise ValueError("chunk_chars deve ser >= 100 e 0 <= chunk_overlap < chunk_chars.")
        if self.min_score is not None and not -1.0 <= self.min_score <= 1.0:
            raise ValueError("min_score deve estar entre -1 e 1.")
        if self.store not in STORES:
            raise ValueError(f"store deve ser um de {STORES}.")
        if self.prompt not in PROMPTS:
            raise ValueError(f"prompt desconhecido: {self.prompt}; disponíveis: {sorted(PROMPTS)}.")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["models"] = list(self.models)
        return data

    def index_params(self) -> dict:
        return {name: getattr(self, name) for name in INDEX_FIELDS}

    def fingerprint(self) -> str:
        """Identificador curto e estável: mesmos parâmetros, mesmo identificador."""
        return cache.digest_text(json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False))[:8]


def coerce(name: str, value):
    """Converte valores vindos de JSON ou da CLI (texto) para o tipo do parâmetro."""
    if name not in {f.name for f in fields(RagSettings)}:
        raise ValueError(f"Parâmetro desconhecido: {name}")
    if name == "models":
        return tuple(value.split(",")) if isinstance(value, str) else tuple(value)
    if name in _FLOAT:
        return None if value is None or str(value).lower() in {"none", "null", ""} else float(value)
    if name in _INT:
        return int(value)
    return str(value)


def build_settings(*layers: dict) -> RagSettings:
    """Junta camadas por ordem de precedência crescente, ignorando valores None."""
    merged: dict = {}
    for layer in layers:
        for name, value in layer.items():
            if value is None and name not in _FLOAT:
                continue
            merged[name] = coerce(name, value)
    if "theme" not in merged:
        raise ValueError("Falta o tema (--theme ou 'theme' no ficheiro de configuração).")
    return RagSettings(**merged)


def load_config_file(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: esperado um objeto JSON.")
    return data


def parse_overrides(pairs: list[str] | None) -> dict:
    result = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set espera chave=valor, recebido: {pair}")
        key, value = pair.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def with_changes(settings: RagSettings, **changes) -> RagSettings:
    return replace(settings, **{name: coerce(name, value) for name, value in changes.items()})
