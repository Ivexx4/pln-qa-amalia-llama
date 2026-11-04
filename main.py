"""Command-line entry point for the Meta 2 evaluation project."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import requests

import config
from src.analysis.method_analysis import analyze_by_method
from src.analysis.overall_analysis import analyze_overall
from src.analysis.theme_analysis import analyze_by_theme
from src.dataset.loader import load_dataset
from src.evaluation.evaluator import evaluate_responses
from src.inference.predictor import generate_responses
from src.utils.io import read_responses, write_csv
from src.utils.logger import get_logger
from src.visualization.generate_plots import generate_plots

logger = get_logger(__name__)


def _generate(args: argparse.Namespace) -> None:
    questions = load_dataset(args.input)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit tem de ser um inteiro positivo.")
        questions = questions.head(args.limit).copy()

    responses = generate_responses(
        questions=questions,
        output_path=args.responses,
        amalia_model=args.amalia_model,
        llama_model=args.llama_model,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
    )
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(Path(args.input).resolve()),
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "models": {
            "AMALIA": args.amalia_model,
            "Llama 3.1 8B": args.llama_model,
        },
        "ollama_url": args.ollama_url,
        "seed": 42,
        "temperature": 0,
    }
    metadata_path = Path(args.responses).parent / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Respostas geradas para %d perguntas.", len(responses))


def _evaluate(args: argparse.Namespace) -> None:
    responses = read_responses(args.responses)
    metrics = evaluate_responses(
        responses=responses,
        device=args.device,
        batch_size=args.batch_size,
    )
    write_csv(metrics, args.metrics)
    logger.info("Métricas por pergunta guardadas em %s.", args.metrics)


def _analyze(args: argparse.Namespace) -> None:
    metrics = pd.read_csv(args.metrics, encoding="utf-8-sig")
    analyze_overall(metrics, config.METRICS_DIR / "summary_global.csv")
    analyze_by_theme(metrics, config.METRICS_DIR / "summary_tema.csv")
    analyze_by_method(metrics, config.METRICS_DIR / "summary_metodo.csv")
    logger.info("Tabelas de análise guardadas em %s.", config.METRICS_DIR)


def _plot(args: argparse.Namespace) -> None:
    metrics = pd.read_csv(args.metrics, encoding="utf-8-sig")
    generate_plots(metrics, config.PLOTS_DIR)
    logger.info("Gráficos guardados em %s.", config.PLOTS_DIR)


def _run_all(args: argparse.Namespace) -> None:
    _generate(args)
    _evaluate(args)
    _analyze(args)
    _plot(args)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser and pipeline stage commands."""
    parser = argparse.ArgumentParser(
        description="Executa e avalia modelos de Question Answering com Ollama."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text, handler in (
        ("run", "Executa o pipeline completo.", _run_all),
        ("responses", "Gera respostas com os dois modelos.", _generate),
        ("evaluate", "Calcula as métricas por pergunta.", _evaluate),
        ("analysis", "Gera tabelas agregadas para o relatório.", _analyze),
        ("plots", "Gera os gráficos a partir das métricas.", _plot),
    ):
        command = commands.add_parser(name, help=help_text)
        command.set_defaults(handler=handler)
        if name in {"run", "responses"}:
            command.add_argument("--input", type=Path, default=config.DEFAULT_DATASET)
            command.add_argument(
                "--responses", type=Path, default=config.DEFAULT_RESPONSES
            )
            command.add_argument(
                "--amalia-model", default=config.DEFAULT_AMALIA_MODEL
            )
            command.add_argument(
                "--llama-model", default=config.DEFAULT_LLAMA_MODEL
            )
            command.add_argument("--ollama-url", default=config.DEFAULT_OLLAMA_URL)
            command.add_argument(
                "--timeout", type=int, default=config.OLLAMA_TIMEOUT_SECONDS
            )
            command.add_argument(
                "--limit",
                type=int,
                help="Limita perguntas para um teste rápido.",
            )
        if name in {"run", "evaluate", "analysis", "plots"}:
            command.add_argument(
                "--metrics", type=Path, default=config.DEFAULT_METRICS
            )
        if name in {"run", "evaluate"}:
            command.add_argument(
                "--device",
                choices=("auto", "cpu", "cuda"),
                default=config.DEFAULT_BERTSCORE_DEVICE,
            )
            command.add_argument(
                "--batch-size",
                type=int,
                default=config.DEFAULT_BERTSCORE_BATCH_SIZE,
            )
        if name == "evaluate":
            command.add_argument(
                "--responses",
                type=Path,
                default=config.DEFAULT_RESPONSES,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one pipeline stage or the complete pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, RuntimeError, requests.RequestException) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
