"""CLI da Meta 3 (RAG com base de dados vetorial): `python main.py rag <fase> --theme "<Tema>"`.

Todos os parâmetros do RAG vivem em RagSettings (rag_settings.py). Podem vir de um ficheiro
(--config), de opções da CLI ou de --set chave=valor, por esta ordem de precedência crescente.
"""
import argparse
import importlib.metadata
import re
from pathlib import Path

from . import (cache, config, llm, prompts, rag_config, rag_corpus, rag_embed, rag_settings, rag_store,
               rag_sweep)
from .analysis import analyze
from .data import load_data
from .judge import judge
from .rag_compare import export_comparison
from .rag_generation import generate_rag
from .rag_prompts import PROMPTS
from .rag_retrieval import Retriever, retrieval_summary, retrieval_table, retrieve_all
from .scoring import bert_score_layers, score

STAGES = ["corpus", "index", "retrieve", "generate", "judge", "score", "analyze", "compare"]
ALL_STAGES = ["index", "retrieve", "generate", "judge", "score", "analyze", "compare"]
CODE_MODULES = ["config.py", "data.py", "cache.py", "llm.py", "prompts.py", "judge.py", "scoring.py",
                "metrics.py", "analysis.py", "stats.py", "rag_config.py", "rag_corpus.py", "rag_embed.py",
                "rag_store.py", "rag_retrieval.py", "rag_generation.py", "rag_compare.py", "rag_cli.py",
                "rag_prompts.py", "rag_settings.py"]
BASELINE_CSV = (config.RESULTS_DIR / "meta2" / "principal-ptpt" / "analises" / "both_todas_completa"
                / "all_results.csv")
SETTING_FLAGS = ["theme", "models", "judge_model", "embed_model", "chunk_chars", "chunk_overlap", "top_k",
                 "min_score", "store", "prompt"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="main.py rag", description="PLN Meta 3: RAG com base de dados vetorial.")
    ap.add_argument("stage", choices=STAGES + ["all", "sweep"],
                    help="all = index..compare para uma configuração; sweep = all para cada variante de uma grelha "
                         "(o corpus é à parte)")
    group = ap.add_argument_group("parâmetros do RAG (também aceites em --config e --set)")
    group.add_argument("--config", type=Path, help="ficheiro JSON com parâmetros, p. ex. configs/rag/base.json")
    group.add_argument("--set", dest="overrides", action="append", metavar="CHAVE=VALOR",
                       help="sobrepõe um parâmetro; repetível, p. ex. --set top_k=6 --set min_score=0.4")
    group.add_argument("--theme", help='tema do dataset, p. ex. "Futebol"')
    group.add_argument("--models", nargs="+")
    group.add_argument("--judge-model")
    group.add_argument("--embed-model")
    group.add_argument("--chunk-chars", type=int)
    group.add_argument("--chunk-overlap", type=int)
    group.add_argument("--top-k", type=int)
    group.add_argument("--min-score", type=float, help="descarta excertos com similaridade inferior")
    group.add_argument("--store", choices=rag_settings.STORES)
    group.add_argument("--prompt", choices=sorted(PROMPTS))
    ap.add_argument("--sweep", type=Path, help="ficheiro JSON da grelha (fase sweep)")
    ap.add_argument("--seeds", type=Path, help="sementes do corpus (por omissão data/rag/<tema>/seeds.txt)")
    ap.add_argument("--bert-model", default=config.DEFAULT_BERT_MODEL)
    ap.add_argument("--bert-num-layers", type=int)
    ap.add_argument("--bert-batch-size", type=int, default=config.DEFAULT_BERTSCORE_BATCH_SIZE)
    ap.add_argument("--split", choices=["both", "train", "test"], default="both")
    ap.add_argument("--train", type=Path, default=config.TRAIN_PATH)
    ap.add_argument("--test", type=Path, default=config.TEST_PATH)
    ap.add_argument("--limit", type=int, default=0, help="usar apenas as N primeiras perguntas do tema")
    ap.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--run-id", help="por omissão rag-<fingerprint dos parâmetros>")
    ap.add_argument("--baseline", type=Path, default=BASELINE_CSV,
                    help="all_results.csv da Meta 2 usado na comparação")
    ap.add_argument("--allow-partial", action="store_true")
    return ap


def cli_layers(args) -> list[dict]:
    """Camadas de parâmetros, da menor para a maior precedência: ficheiro, opções da CLI, --set."""
    layers = []
    if args.config:
        layers.append(rag_settings.load_config_file(args.config))
    layers.append({name: getattr(args, name) for name in SETTING_FLAGS if getattr(args, name) is not None})
    layers.append(rag_settings.parse_overrides(args.overrides))
    return layers


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collection_name(settings, corpus_sha: str) -> str:
    key = cache.digest_text(f"{corpus_sha}|{sorted(settings.index_params().items())}")[:12]
    return f"{rag_corpus.slugify(settings.theme)}_{key}"


def prepare_store(settings):
    """Constrói (ou reabre) o índice vetorial do corpus do tema; variantes com o mesmo índice partilham-no."""
    folder = rag_corpus.theme_dir(settings.theme)
    docs = rag_corpus.load_corpus(folder)
    chunks = rag_corpus.chunk_corpus(docs, size=settings.chunk_chars, overlap=settings.chunk_overlap)
    name = collection_name(settings, rag_corpus.corpus_digest(folder))
    if settings.store == "chroma":
        store = rag_store.ChromaStore(folder / "chroma", name)
    else:
        print("Aviso: o índice em memória não é persistente; os embeddings são recalculados.")
        store = rag_store.MemoryStore()
    if not store.count():
        print(f"{len(docs)} documentos -> {len(chunks)} excertos; a calcular embeddings ({settings.embed_model}).")
        vectors = rag_embed.embed_all(settings.embed_model, [c["text"] for c in chunks])
        rag_store.build_index(store, chunks, vectors)
    else:
        rag_store.build_index(store, chunks, [None] * len(chunks))  # só valida a contagem
    print(f"Índice '{name}' com {store.count()} excertos.")
    return store, docs, chunks


def build_manifest(args, settings, digests: dict, folder: Path, n_chunks: int) -> dict:
    return {
        "version": 2, "mode": "meta3-rag", "settings": settings.to_dict(),
        "datasets": {name: {"path": str(p.resolve()), "sha256": cache.digest_file(p.resolve())}
                     for name, p in (("train", args.train), ("test", args.test))},
        "ollama_model_digests": digests,
        "corpus": {"sha256": rag_corpus.corpus_digest(folder), "chunks": n_chunks},
        "generation": {"system_prompt": PROMPTS[settings.prompt], "fallback_prompt": prompts.SYSTEM_PROMPT,
                       "temperature": config.TEMPERATURE, "seed": config.SEED,
                       "max_tokens": config.GEN_MAX_TOKENS},
        "judgement": {"prompt": prompts.JUDGE_PROMPT, "cache_version": config.JUDGE_CACHE_VERSION},
        "metrics": {"bert_model": args.bert_model,
                    "bert_num_layers": bert_score_layers(args.bert_model, args.bert_num_layers),
                    "names": config.METRICS},
        "versions": {name: _version(name) for name in ("numpy", "pandas", "requests", "chromadb", "scipy",
                                                       "torch", "bert-score")},
        "source_sha256": {name: cache.digest_file(Path(__file__).parent / name) for name in CODE_MODULES},
    }


def select_questions(args, settings):
    df = load_data(args.train, args.test, args.split)
    df = df[df["Tema"] == settings.theme]
    if df.empty:
        raise ValueError(f"Tema sem perguntas: {settings.theme}")
    return df.head(args.limit) if args.limit else df


def run_variant(args, settings, run_id: str, stages: list[str], store_cache: dict | None = None) -> Path:
    """Executa as fases pedidas para uma configuração; devolve a pasta das análises."""
    df = select_questions(args, settings)
    folder = rag_corpus.theme_dir(settings.theme)
    run_dir = args.output_dir.resolve() / "meta3" / run_id
    scope = (f"{args.split}_{rag_corpus.slugify(settings.theme)}_"
             + (f"limite_{args.limit}" if args.limit else "todas"))
    out = run_dir / "analises" / (scope + ("_parcial" if args.allow_partial else "_completa"))

    store = chunks = None
    if {"index", "retrieve"} & set(stages):
        key = tuple(sorted(settings.index_params().items())) + (settings.store, settings.theme)
        if store_cache is not None and key in store_cache:
            store, chunks = store_cache[key]
        else:
            store, _, chunks = prepare_store(settings)
            if store_cache is not None:
                store_cache[key] = (store, chunks)
    if stages == ["index"]:
        return out

    if {"retrieve", "generate", "judge"} & set(stages):
        digests = llm.model_digests(list(settings.models) + [settings.judge_model, settings.embed_model])
        if chunks is None:
            chunks = rag_corpus.chunk_corpus(rag_corpus.load_corpus(folder), size=settings.chunk_chars,
                                             overlap=settings.chunk_overlap)
        cache.ensure_manifest(run_dir, build_manifest(args, settings, digests, folder, len(chunks)))
    print(f"[{run_id}] {len(df)} perguntas de '{settings.theme}' | {settings.to_dict()}")

    retrieval_path = run_dir / "retrieval.jsonl"
    if "retrieve" in stages:
        retriever = Retriever(store, settings.embed_model, settings.top_k, min_score=settings.min_score)
        retrievals = retrieve_all(df, retriever, retrieval_path)
        table = retrieval_table(df, retrievals)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "recuperacao_por_pergunta.csv", index=False, encoding="utf-8")
        summary = retrieval_summary(table).round(3)
        summary.to_csv(out / "recuperacao_resumo.csv", index=False, encoding="utf-8")
        print("\n== Recuperação ==\n" + summary.to_string(index=False))
    if "generate" in stages:
        retrievals = cache.read_cache(retrieval_path)
        missing = df[~df["id"].isin(retrievals)]
        if not missing.empty:
            raise ValueError(f"Faltam {len(missing)} recuperações; execute 'retrieve' primeiro.")
        generate_rag(df, list(settings.models), retrievals, run_dir, prompt=settings.prompt)
    if "judge" in stages:
        judge(df, list(settings.models), settings.judge_model, run_dir=run_dir, allow_partial=args.allow_partial)
    if "score" in stages:
        score(df, list(settings.models), settings.judge_model, args.bert_model, run_dir=run_dir,
              allow_partial=args.allow_partial, batch_size=args.bert_batch_size, output_dir=out,
              scope=scope, num_layers=args.bert_num_layers)
    if "analyze" in stages:
        analyze(out, list(settings.models))
    if "compare" in stages:
        if not args.baseline.is_file():
            raise ValueError(f"Baseline inexistente: {args.baseline}")
        table = export_comparison(args.baseline, out / "all_results.csv", out / "recuperacao_por_pergunta.csv", out)
        print("\n== RAG vs. baseline ==\n" + table.round(3).to_string(index=False))
    return out


def run_sweep(args, layers: list[dict]) -> None:
    if not args.sweep:
        raise ValueError("A fase sweep exige --sweep <ficheiro.json>.")
    spec = rag_settings.load_config_file(args.sweep)
    name = spec.get("name") or args.sweep.stem
    variants = rag_sweep.expand_sweep(spec, lower=layers[:1] if args.config else [], upper=layers[1:])
    stages = [s for s in ALL_STAGES if s != "compare"]
    print(f"Varrimento '{name}': {len(variants)} variantes -> " + ", ".join(v for v, _ in variants))
    rows, store_cache = [], {}
    for variant, settings in variants:
        run_id = f"{name}-{variant}"
        out = run_variant(args, settings, run_id, stages, store_cache)
        rows.extend(rag_sweep.summarize_variant(variant, settings, out))
    summary_path = args.output_dir.resolve() / "meta3" / f"{name}-resumo.csv"
    table = rag_sweep.write_summary(rows, summary_path)
    print(f"\n== Resumo do varrimento ({summary_path}) ==\n" + table.round(3).to_string(index=False))


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("--run-id deve conter apenas letras, números, pontos, hífen ou sublinhado")
    if args.limit < 0:
        parser.error("--limit deve ser não negativo")
    try:
        layers = cli_layers(args)
        if args.stage == "sweep":
            run_sweep(args, layers)
            return
        settings = rag_settings.build_settings(*layers)
        if args.stage == "corpus":
            folder = rag_corpus.theme_dir(settings.theme)
            rag_corpus.build_corpus(settings.theme, rag_corpus.read_seeds(args.seeds or folder / "seeds.txt"),
                                    folder)
            return
        run_id = args.run_id or f"rag-{settings.fingerprint()}"
        stages = ALL_STAGES if args.stage == "all" else [args.stage]
        run_variant(args, settings, run_id, stages)
    except ValueError as error:
        parser.error(str(error))
