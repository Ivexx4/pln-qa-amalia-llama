"""Geração de respostas com contexto recuperado (usa o mesmo formato de cache que a Meta 2)."""
from tqdm import tqdm

from . import cache, llm
from .prompts import SYSTEM_PROMPT
from .rag_prompts import PROMPTS

RAG_SYSTEM_PROMPT = PROMPTS["default"]


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[Excerto {i}] {hit['text']}" for i, hit in enumerate(hits, start=1))


def build_user_prompt(question: str, hits: list[dict]) -> str:
    if not hits:
        return question
    return f"Excertos:\n{format_context(hits)}\n\nPergunta: {question}"


def generate_rag(df, models, retrievals: dict, run_dir, prompt: str = "default") -> None:
    rag_prompt = PROMPTS[prompt]
    for model in models:
        path = cache.preds_path(model, run_dir)
        done = cache.read_cache(path)
        for _, row in df[df["id"].isin(done)].iterrows():
            cache.validate_prediction(done[row["id"]], row["Pergunta"], row["id"])
        todo = df[~df["id"].isin(done)]
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"rag({model})"):
            hits = retrievals[row["id"]]["hits"]
            system = rag_prompt if hits else SYSTEM_PROMPT
            pred = llm.chat(model, [{"role": "system", "content": system},
                                    {"role": "user", "content": build_user_prompt(row["Pergunta"], hits)}])
            cache.append_cache(path, {"id": row["id"], "question_sha256": cache.digest_text(row["Pergunta"]),
                                      "pred": pred})
