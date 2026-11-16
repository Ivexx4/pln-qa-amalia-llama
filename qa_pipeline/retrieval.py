"""Recuperação lexical de contexto para geração aumentada por recuperação (RAG)."""
from __future__ import annotations

from typing import Sequence

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError as exc:  # pragma: no cover - depende do ambiente do projeto.
    raise ModuleNotFoundError(
        "scikit-learn é necessário para a recuperação lexical; instalar as dependências com "
        "'python -m pip install -r requirements.txt'."
    ) from exc


def build_retriever(df, top_k: int = 3):
    """Cria um recuperador TF-IDF a partir do conjunto de perguntas."""
    data = df.reset_index(drop=True)
    if data.empty:
        return {"df": data, "vectorizer": None, "embeddings": None, "top_k": top_k}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    embeddings = vectorizer.fit_transform(data["Pergunta"].fillna("").astype(str))
    return {"df": data, "vectorizer": vectorizer, "embeddings": embeddings, "top_k": top_k}


def retrieve_context(question: str, retriever, top_k: int | None = None, exclude_id: str | None = None):
    """Devolve as entradas mais semelhantes à pergunta."""
    data = retriever["df"]
    top_k = top_k or retriever.get("top_k", 3)
    if data.empty:
        return []

    query = retriever["vectorizer"].transform([question])
    scores = cosine_similarity(query, retriever["embeddings"])[0]
    ranked = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )

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
