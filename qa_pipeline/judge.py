"""Fase 2: LLM-as-Judge (veredito CORRETO/INCORRETO)."""
import json

from tqdm import tqdm

from . import cache, config, llm
from .prompts import JUDGE_PROMPT


def parse_verdict(text: str) -> int | None:
    """1 = correto, 0 = incorreto, None = veredito não reconhecido."""
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        verdict = text.strip().upper()
    else:
        if not isinstance(result, dict) or set(result) != {"veredito"}:
            return None
        verdict = result.get("veredito")
        if not isinstance(verdict, str):
            return None
        verdict = verdict.strip().upper()

    if verdict in {"INCORRETO", "INCORRECTO"}:
        return 0
    if verdict in {"CORRETO", "CORRECTO"}:
        return 1
    return None


def judge(df, models, judge_model, run_dir=None, allow_partial=False):
    for model in models:
        preds = cache.read_cache(cache.preds_path(model, run_dir))
        path = cache.judge_path(model, judge_model, run_dir)
        done = cache.read_cache(path)
        missing = df[~df["id"].isin(preds)]
        if not missing.empty and not allow_partial:
            raise ValueError(f"Faltam {len(missing)} previsões de {model}; execute generate primeiro.")
        for _, row in df[df["id"].isin(preds)].iterrows():
            identifier = row["id"]
            cache.validate_prediction(preds[identifier], row["Pergunta"], identifier)
            if identifier in done:
                cache.validate_judgement(done[identifier], preds[identifier]["pred"], identifier)
        todo = df[df["id"].isin(preds) & ~df["id"].isin(done)]
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"juiz({model})"):
            pred = preds[row["id"]]["pred"]
            prompt = JUDGE_PROMPT.format(q=row["Pergunta"], gold=row["Resposta"], pred=pred)
            out = llm.chat(judge_model, [{"role": "user", "content": prompt}],
                           max_tokens=config.JUDGE_MAX_TOKENS)
            cache.append_cache(path, {"id": row["id"], "pred_sha256": cache.digest_text(pred),
                                      "raw": out, "judge": parse_verdict(out)})
