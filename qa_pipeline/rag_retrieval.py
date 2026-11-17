"""Recuperação de excertos e métricas de recuperação (hit@k, MRR, cobertura da resposta)."""
from __future__ import annotations

import pandas as pd

from . import cache, metrics, rag_embed


class Retriever:
    def __init__(self, store, embed_model: str, top_k: int, embed_fn=None, min_score: float | None = None):
        self.store = store
        self.embed_model = embed_model
        self.top_k = top_k
        self.min_score = min_score
        self._embed = embed_fn or (lambda texts: rag_embed.embed(embed_model, texts))

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        vector = self._embed([question])[0]
        hits = self.store.query(vector, top_k or self.top_k)
        if self.min_score is not None:
            hits = [hit for hit in hits if hit["score"] >= self.min_score]
        return hits


def retrieve_all(df, retriever: Retriever, path) -> dict:
    """Recupera contexto para cada pergunta e guarda em JSONL (retomável)."""
    from tqdm import tqdm
    done = cache.read_cache(path)
    for _, row in df[df["id"].isin(done)].iterrows():
        if done[row["id"]].get("question_sha256") != cache.digest_text(row["Pergunta"]):
            raise ValueError(f"Recuperação incompatível para {row['id']}.")
    todo = df[~df["id"].isin(done)]
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="recuperação"):
        hits = retriever.retrieve(row["Pergunta"])
        record = {"id": row["id"], "question_sha256": cache.digest_text(row["Pergunta"]), "hits": hits}
        cache.append_cache(path, record)
        done[row["id"]] = record
    return done


def gold_rank(hits: list[dict], gold: str) -> int | None:
    """Posição (a partir de 1) do primeiro excerto que contém a resposta de referência."""
    for rank, hit in enumerate(hits, start=1):
        if metrics.contains(hit["text"], gold):
            return rank
    return None


def token_recall(hits: list[dict], gold: str) -> float:
    """Fração dos tokens distintos da referência presentes no contexto recuperado."""
    gold_tokens = set(metrics.tokens(gold))
    if not gold_tokens:
        return 0.0
    context_tokens = set(metrics.tokens(" ".join(hit["text"] for hit in hits)))
    return len(gold_tokens & context_tokens) / len(gold_tokens)


def retrieval_table(df, retrievals: dict, ks=(1, 3, 5)) -> pd.DataFrame:
    """Uma linha por pergunta. Mede se a resposta aparece no contexto, não a correção do LLM."""
    rows = []
    for _, row in df.iterrows():
        hits = retrievals[row["id"]]["hits"]
        rank = gold_rank(hits, row["Resposta"])
        record = {"id": row["id"], "Tema": row["Tema"], "Metodo": row["Metodo"], "Pergunta": row["Pergunta"],
                  "Gold": row["Resposta"], "rank_gold": rank,
                  "mrr": 1.0 / rank if rank else 0.0,
                  "token_recall": token_recall(hits, row["Resposta"]),
                  "score_top1": hits[0]["score"] if hits else float("nan"),
                  "fontes": " | ".join(dict.fromkeys(hit["title"] for hit in hits))}
        for k in ks:
            record[f"hit@{k}"] = int(rank is not None and rank <= k)
        rows.append(record)
    return pd.DataFrame(rows)


def retrieval_summary(table: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in table.columns if c.startswith("hit@")] + ["mrr", "token_recall"]
    overall = table[cols].mean().to_frame().T.reset_index(drop=True)
    overall.insert(0, "grupo", "todas")
    overall.insert(1, "n", len(table))
    by_method = table.groupby("Metodo")[cols].mean().reset_index().rename(columns={"Metodo": "grupo"})
    by_method.insert(1, "n", table.groupby("Metodo").size().values)
    return pd.concat([overall, by_method], ignore_index=True)
