"""Recuperação lexical de contexto para geração aumentada por recuperação (RAG)."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence

try:  # pragma: no cover - depende do ambiente do projeto.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None


def normalize_text(text: str) -> str:
    """Normaliza texto para comparação simples, preservando sinais de acentos."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_retriever(df, top_k: int = 3):
    """Cria um recuperador ligeiro a partir do conjunto de perguntas."""
    data = df.reset_index(drop=True)
    if data.empty:
        return {"df": data, "vectorizer": None, "embeddings": None, "top_k": top_k}

    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        embeddings = vectorizer.fit_transform(data["Pergunta"].fillna("").astype(str))
        return {"df": data, "vectorizer": vectorizer, "embeddings": embeddings, "top_k": top_k}

    return {"df": data, "vectorizer": None, "embeddings": None, "top_k": top_k}


def _fallback_similarity(question: str, candidate: str) -> float:
    q_tokens = set(normalize_text(question).split())
    c_tokens = set(normalize_text(candidate).split())
    if not q_tokens and not c_tokens:
        return 0.0
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = q_tokens & c_tokens
    union = q_tokens | c_tokens
    return len(overlap) / len(union) if union else 0.0


def retrieve_context(question: str, retriever, top_k: int | None = None, exclude_id: str | None = None):
    """Devolve as entradas mais semelhantes à pergunta."""
    data = retriever["df"]
    top_k = top_k or retriever.get("top_k", 3)
    if data.empty:
        return []

    if retriever.get("vectorizer") is not None and retriever.get("embeddings") is not None:
        query = retriever["vectorizer"].transform([question])
        scores = cosine_similarity(query, retriever["embeddings"])[0]
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
    else:
        ranked = [
            (idx, _fallback_similarity(question, row["Pergunta"]))
            for idx, row in data.iterrows()
        ]
        ranked = sorted(ranked, key=lambda item: item[1], reverse=True)

    hits = []
    for idx, score in ranked:
        row = data.iloc[idx]
        if exclude_id and row.get("id") == exclude_id:
            continue
        if score <= 0:
            continue
        hits.append({
            "Tema": row.get("Tema", ""),
            "Pergunta": row.get("Pergunta", ""),
            "Resposta": row.get("Resposta", ""),
            "score": float(score),
        })
        if len(hits) >= top_k:
            break

    return hits


def format_context(context: Sequence[dict]) -> str:
    """Formata a recuperação para inserir no prompt do modelo."""
    if not context:
        return ""
    blocks = []
    for idx, item in enumerate(context, start=1):
        blocks.append(
            f"[Contexto {idx}] Tema: {item.get('Tema', '')}\n"
            f"Pergunta: {item.get('Pergunta', '')}\n"
            f"Resposta relevante: {item.get('Resposta', '')}"
        )
    return "\n\n".join(blocks)
