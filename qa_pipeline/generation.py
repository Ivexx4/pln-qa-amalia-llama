"""Fase 1: geração de respostas com cada LLM."""
from tqdm import tqdm

from . import cache, llm
from .prompts import SYSTEM_PROMPT


def generate(df, models):
    for model in models:
        path = cache.preds_path(model)
        done = cache.read_cache(path)
        todo = df[~df["id"].isin(done)]
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc=model):
            pred = llm.chat(model, [{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": row["Pergunta"]}])
            cache.append_cache(path, {"id": row["id"], "pred": pred})
