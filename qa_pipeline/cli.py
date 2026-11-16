"""Interface de linha de comandos e proveniência das experiências."""
import argparse
import importlib.metadata
import json
import platform
import re
from pathlib import Path

from . import cache, config, llm, prompts
from .analysis import analyze, export_statistics
from .data import load_data
from .generation import generate
from .judge import judge
from .scoring import bert_score_layers, score

STAGES = ["generate", "judge", "score", "analyze"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PLN Meta 2: avaliação de resposta a perguntas com LLMs.")
    ap.add_argument("stage", choices=STAGES + ["all"])
    ap.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS)
    ap.add_argument("--judge-model", default=config.DEFAULT_JUDGE)
    ap.add_argument("--bert-model", default=config.DEFAULT_BERT_MODEL)
    ap.add_argument("--bert-num-layers", type=int, help="camada usada pelo BERTScore; por omissão, 12 para Albertina")
    ap.add_argument("--bert-batch-size", type=int, default=config.DEFAULT_BERTSCORE_BATCH_SIZE)
    ap.add_argument("--split", choices=["both", "train", "test"], default="both")
    ap.add_argument("--train", type=Path, default=config.TRAIN_PATH)
    ap.add_argument("--test", type=Path, default=config.TEST_PATH)
    ap.add_argument("--limit", type=int, default=0, help="usar apenas as N primeiras perguntas; 0 usa todas")
    ap.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR, help="pasta principal dos resultados")
    ap.add_argument("--run-id", default=config.DEFAULT_RUN_ID, help="nome da experiência retomável")
    ap.add_argument("--source-run", help="experiência com previsões e vereditos já gerados; apenas para score/analyze")
    ap.add_argument("--allow-partial", action="store_true", help="permitir resultados incompletos, identificados como parciais")
    ap.add_argument("--rag", action="store_true", help="modo experimental anterior, separado da Meta 2; não corresponde à Meta 3")
    ap.add_argument("--rag-top-k", type=int, default=3, help="número de itens de contexto no modo experimental")
    ap.add_argument("--stats", action="store_true",
                    help="apenas as tabelas estatísticas, a partir do all_results.csv já existente")
    return ap


def _versions() -> dict:
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "requests", "tqdm", "matplotlib", "torch", "bert-score", "transformers", "scipy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def build_manifest(args, digests: dict) -> dict:
    datasets = {}
    for name, path in (("train", args.train), ("test", args.test)):
        path = path.resolve()
        datasets[name] = {"path": str(path), "sha256": cache.digest_file(path) if path.is_file() else None}
    modules = ["config.py", "data.py", "cache.py", "cli.py", "generation.py", "judge.py",
               "llm.py", "prompts.py", "metrics.py", "scoring.py", "analysis.py", "stats.py"]
    if args.rag:
        modules.append("retrieval.py")
    code = {name: cache.digest_file(Path(__file__).parent / name) for name in modules}
    manifest = {
        "version": 1,
        "mode": "rag-experimental" if args.rag else "meta2",
        "datasets": datasets,
        "models": args.models,
        "judge_model": args.judge_model,
        "ollama_model_digests": digests,
        "ollama_url": config.OLLAMA_URL,
        "generation": {"system_prompt": prompts.RAG_SYSTEM_PROMPT if args.rag else prompts.SYSTEM_PROMPT,
                       "temperature": config.TEMPERATURE, "seed": config.SEED,
                       "max_tokens": config.GEN_MAX_TOKENS},
        "judgement": {"prompt": prompts.JUDGE_PROMPT, "cache_version": config.JUDGE_CACHE_VERSION,
                      "temperature": config.TEMPERATURE, "seed": config.SEED,
                      "max_tokens": config.JUDGE_MAX_TOKENS},
        "metrics": {"bert_model": args.bert_model,
                    "bert_num_layers": bert_score_layers(args.bert_model, args.bert_num_layers),
                    "bert_batch_size": args.bert_batch_size, "names": config.METRICS},
        "versions": _versions(),
        "source_sha256": code,
    }
    if args.rag:
        # O recuperador experimental depende do próprio subconjunto selecionado.
        manifest["rag"] = {"top_k": args.rag_top_k, "split": args.split, "limit": args.limit}
    return manifest


def validate_source_run(source_manifest: dict, manifest: dict) -> None:
    """Verifica que previsões e vereditos provêm da mesma experiência de geração."""
    fields = ("mode", "datasets", "models", "judge_model", "ollama_model_digests",
              "ollama_url", "generation", "judgement", "rag")
    differing = [name for name in fields if source_manifest.get(name) != manifest.get(name)]
    modules = ["data.py", "cache.py", "generation.py", "judge.py", "llm.py", "prompts.py"]
    if manifest["mode"] == "rag-experimental":
        modules.append("retrieval.py")
    old_code = source_manifest.get("source_sha256", {})
    new_code = manifest["source_sha256"]
    differing.extend(name for name in modules if old_code.get(name) != new_code.get(name))
    if differing:
        raise ValueError("A experiência de origem é incompatível: " + ", ".join(differing) + ".")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("--run-id deve conter apenas letras, números, pontos, hífen ou sublinhado")
    if args.source_run:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.source_run):
            parser.error("--source-run deve ser um nome de experiência válido")
        if args.source_run == args.run_id or args.stage not in ("score", "analyze"):
            parser.error("--source-run exige uma experiência diferente e a fase score ou analyze")
    if args.limit < 0 or args.rag_top_k < 1 or args.bert_batch_size < 1:
        parser.error("--limit deve ser não negativo; --rag-top-k e --bert-batch-size devem ser positivos")
    if args.bert_num_layers is not None and args.bert_num_layers < 1:
        parser.error("--bert-num-layers deve ser positivo")
    if len(set(args.models)) != len(args.models):
        parser.error("--models não pode conter modelos repetidos")
    if len(args.models) < 2 or config.DEFAULT_AMALIA_MODEL not in args.models:
        print("Aviso: a comparação exigida na Meta 2 requer AMALIA e pelo menos outro LLM aberto com menos de 21B parâmetros.")
    if args.judge_model in args.models:
        print("Aviso: o modelo de avaliação também consta entre os modelos avaliados.")
    if args.rag:
        print("Aviso: RAG experimental não cumpre os requisitos da Meta 3 e usa uma pasta separada.")
    if args.stats and args.stage != "analyze":
        parser.error("--stats só pode ser usado com a fase analyze")

    mode = "rag-experimental" if args.rag else "meta2"
    run_dir = args.output_dir.resolve() / mode / args.run_id
    scope = f"{args.split}_{'limite_' + str(args.limit) if args.limit else 'todas'}"
    output_dir = run_dir / "analises" / (scope + ("_parcial" if args.allow_partial else "_completa"))
    source_dir = args.output_dir.resolve() / mode / args.source_run if args.source_run else run_dir
    if args.stats:
        # Não reescreve o manifesto nem as restantes tabelas: apenas a análise estatística.
        export_statistics(output_dir)
        return

    df = load_data(args.train, args.test, args.split)
    if args.limit:
        df = df.head(args.limit)
    source_manifest_path = source_dir / "manifest.json"
    if args.stage in ("generate", "judge", "all"):
        digests = llm.model_digests(args.models + [args.judge_model])
        source_manifest = None
    else:
        if not source_manifest_path.exists():
            raise ValueError(f"Experiência inexistente em {source_dir}; execute generate primeiro.")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        digests = source_manifest["ollama_model_digests"]
    manifest = build_manifest(args, digests)
    if args.source_run:
        validate_source_run(source_manifest, manifest)
        manifest["source_run"] = {"id": args.source_run, "manifest_sha256": cache.digest_file(source_manifest_path)}
    cache.ensure_manifest(run_dir, manifest)
    print(f"{len(df)} perguntas | modelos: {args.models} | juiz: {args.judge_model}")
    print(f"Experiência: {run_dir} | escopo das análises: {scope}")
    if args.source_run:
        print(f"Previsões e vereditos de: {source_dir}")

    run = STAGES if args.stage == "all" else [args.stage]
    if "generate" in run:
        generate(df, args.models, use_rag=args.rag, top_k=args.rag_top_k, run_dir=run_dir)
    if "judge" in run:
        judge(df, args.models, args.judge_model, run_dir=run_dir, allow_partial=args.allow_partial)
    if "score" in run:
        score(df, args.models, args.judge_model, args.bert_model, run_dir=source_dir,
              allow_partial=args.allow_partial, batch_size=args.bert_batch_size,
              output_dir=output_dir, scope=scope, num_layers=args.bert_num_layers)
    if "analyze" in run:
        analyze(output_dir, args.models)