"""Interface de linha de comandos."""
import argparse
from pathlib import Path

from . import config
from .analysis import analyze
from .data import load_data
from .generation import generate
from .judge import judge
from .scoring import score

STAGES = ["generate", "judge", "score", "analyze"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PLN Meta 3: resposta a perguntas com LLMs e recuperação de contexto.")
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS)
    ap.add_argument("--judge-model", default=config.DEFAULT_JUDGE)
    ap.add_argument("--bert-model", default=config.DEFAULT_BERT_MODEL)
    ap.add_argument("--split", choices=["both", "train", "test"], default="both")
    ap.add_argument("--train", type=Path, default=config.TRAIN_PATH)
    ap.add_argument("--test", type=Path, default=config.TEST_PATH)
    ap.add_argument("--limit", type=int, default=0, help="só as N primeiras (teste rápido)")
    ap.add_argument("--rag", action="store_true", help="ativa recuperação de contexto a partir do conjunto de dados")
    ap.add_argument("--rag-top-k", type=int, default=3, help="número de itens de contexto a recuperar")
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    config.RESULTS_DIR.mkdir(exist_ok=True)

    df = load_data(a.train, a.test, a.split)
    if a.limit:
        df = df.head(a.limit)
    print(f"{len(df)} perguntas | modelos: {a.models} | juiz: {a.judge_model} | rag: {a.rag}")

    run = STAGES if a.stage == "all" else [a.stage]
    if "generate" in run:
        generate(df, a.models, use_rag=a.rag, top_k=a.rag_top_k)
    if "judge" in run:
        judge(df, a.models, a.judge_model)
    if "score" in run:
        score(df, a.models, a.judge_model, a.bert_model)
    if "analyze" in run:
        analyze()
