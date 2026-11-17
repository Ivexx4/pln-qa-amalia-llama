"""Bases de dados vetoriais: ChromaDB (persistente) e uma alternativa em memória (NumPy)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class MemoryStore:
    """Índice exato por similaridade do cosseno. Serve de alternativa sem dependências e para testes."""

    def __init__(self):
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.matrix: np.ndarray | None = None

    def count(self) -> int:
        return len(self.ids)

    def add(self, ids, embeddings, documents, metadatas) -> None:
        vectors = np.asarray(embeddings, dtype=float)
        vectors = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])

    def query(self, embedding, k: int) -> list[dict]:
        if self.matrix is None:
            return []
        vector = np.asarray(embedding, dtype=float)
        vector = vector / max(np.linalg.norm(vector), 1e-12)
        scores = self.matrix @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [{"chunk_id": self.ids[i], "text": self.documents[i], "score": float(scores[i]),
                 **self.metadatas[i]} for i in order]


class ChromaStore:
    """Base de dados vetorial ChromaDB com índice HNSW (distância do cosseno), guardada em disco."""

    def __init__(self, path: Path, collection: str):
        try:
            import chromadb
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "chromadb é necessário para a base de dados vetorial; instalar com "
                "'python -m pip install -r requirements.txt' ou usar --store memory."
            ) from exc
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(name=collection,
                                                               metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.collection.count()

    def add(self, ids, embeddings, documents, metadatas) -> None:
        batch = 500
        for start in range(0, len(ids), batch):
            end = start + batch
            self.collection.add(ids=list(ids[start:end]), embeddings=[list(map(float, e)) for e in embeddings[start:end]],
                                documents=list(documents[start:end]), metadatas=list(metadatas[start:end]))

    def query(self, embedding, k: int) -> list[dict]:
        result = self.collection.query(query_embeddings=[list(map(float, embedding))], n_results=k,
                                       include=["documents", "metadatas", "distances"])
        hits = []
        for chunk_id, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                                  result["metadatas"][0], result["distances"][0]):
            hits.append({"chunk_id": chunk_id, "text": text, "score": 1.0 - float(distance), **meta})
        return hits


def chunk_metadata(chunk: dict) -> dict:
    return {key: chunk[key] for key in ("doc_id", "title", "section", "url")}


def build_index(store, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Preenche um índice vazio; recusa índices já preenchidos com outro número de excertos."""
    if len(chunks) != len(embeddings):
        raise ValueError("Número de excertos e de embeddings diferente.")
    if store.count():
        if store.count() != len(chunks):
            raise ValueError(f"O índice existente tem {store.count()} excertos e o corpus tem {len(chunks)}; "
                             "apague a pasta do índice ou altere o corpus/parâmetros.")
        return
    store.add([c["chunk_id"] for c in chunks], embeddings, [c["text"] for c in chunks],
              [chunk_metadata(c) for c in chunks])
