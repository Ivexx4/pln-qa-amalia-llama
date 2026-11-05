"""Cache em JSONL, para retomar execuções interrompidas."""
import json
import re
from pathlib import Path

from . import config


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def preds_path(model: str) -> Path:
    return config.RESULTS_DIR / f"preds_{safe_name(model)}.jsonl"


def judge_path(model: str, judge_model: str) -> Path:
    return config.RESULTS_DIR / (
        f"judge_{safe_name(model)}__{safe_name(judge_model)}"
        f"__{config.JUDGE_CACHE_VERSION}.jsonl"
    )


def read_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cache[rec["id"]] = rec
    return cache


def append_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
