"""Fase 1: geração de respostas com cada LLM."""
from tqdm import tqdm

from . import cache, llm
from .prompts import RAG_SYSTEM_PROMPT, SYSTEM_PROMPT
from .retrieval import build_retriever, format_context, retrieve_context


def _build_user_prompt(question, context_items=None):
    if not context_items:
        return question
    context_text = format_context(context_items)
    return (
        f"Pergunta: {question}\n\n"
        f"Contexto relevante:\n{context_text}\n\n"
        "Responde com base no contexto fornecido."
    )


def generate(df, models, use_rag=False, top_k=3):
    retriever = build_retriever(df, top_k=top_k) if use_rag else None
    for model in models:
        path = cache.preds_path(model)
        done = cache.read_cache(path)
        todo = df[~df["id"].isin(done)]
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc=model):
            context_items = []
            if use_rag and retriever is not None:
                context_items = retrieve_context(row["Pergunta"], retriever, top_k=top_k, exclude_id=row["id"])

            system_prompt = RAG_SYSTEM_PROMPT if use_rag and context_items else SYSTEM_PROMPT
            prompt_content = _build_user_prompt(row["Pergunta"], context_items) if use_rag and context_items else row["Pergunta"]
            pred = llm.chat(
                model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_content},
                ],
            )
            cache.append_cache(path, {"id": row["id"], "pred": pred})
