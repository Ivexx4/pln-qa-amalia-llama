"""CLI da Meta 3 (RAG com base de dados vetorial): `python main.py rag <fase> --theme "<Tema>"`."""
import argparse
import importlib.metadata
import re
from pathlib import Path

from . import cache, config, llm, prompts, rag_config, rag_corpus, rag_embed, rag_store
from .analysis import analyze
from .data import load_data
from .judge import judge
from .rag_compare import export_comparison
from .rag_generation import RAG_SYSTEM_PROMPT, generate_rag
from .rag_retrieval import Retriever, retrieval_summary, retrieval_table, retrieve_all
from .scoring import bert_score_layers, score

STAGES = ["corpus", "index", "retrieve", "generate", "judge", "score", "analyze", "compare"]
CODE_MODULES = ["config.py", "data.py", "cache.py", "llm.py", "prompts.py", "judge.py", "scoring.py",
                "metrics.py", "analysis.py", "stats.py", "rag_config.py", "rag_corpus.py", "rag_embed.py",
                "rag_store.py", "rag_retrieval.py", "rag_generation.py", "rag_compare.py", "rag_cli.py"]
BASELINE_CSV = (config.RESULTS_DIR / "meta2" / "principal-ptpt" / "analises" / "both_todas_completa"
                / "all_results.csv")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="main.py rag", description="PLN Meta 3: RAG com base de dados vetorial.")
    ap.add_argument("stage", choices=STAGES + ["all"],
                    help="all = index, retrieve, generate, judge, score, analyze e compare (o corpus é à parte)")
    ap.add_argument("--theme", required=True, help='tema do dataset, p. ex. "Futebol"')
    ap.add_argument("--seeds", type=Path, help="ficheiro de sementes (por omissão data/rag/<tema>/seeds.txt)")
    ap.add_argument("--models", nargs="+", default=config.DEFAULT_MODELS)
    ap.add_argument("--judge-model", default=config.DEFAULT_JUDGE)
    ap.add_argument("--embed-model", default=rag_config.DEFAULT_EMBED_MODEL)
    ap.add_argument("--bert-model", default=config.DEFAULT_BERT_MODEL)
    ap.add_argument("--bert-num-layers", type=int)
    ap.add_argument("--bert-batch-size", type=int, default=config.DEFAULT_BERTSCORE_BATCH_SIZE)
    ap.add_argument("--top-k", type=int, default=rag_config.DEFAULT_TOP_K)
    ap.add_argument("--chunk-chars", type=int, default=rag_config.CHUNK_CHARS)
    ap.add_argument("--chunk-overlap", type=int, default=rag_config.CHUNK_OVERLAP_CHARS)
    ap.add_argument("--store", choices=["chroma", "memory"], default=rag_config.DEFAULT_STORE)
    ap.add_argument("--split", choices=["both", "train", "test"], default="both")
    ap.add_argument("--train", type=Path, default=config.TRAIN_PATH)
    ap.add_argument("--test", type=Path, default=config.TEST_PATH)
    ap.add_argument("--limit", type=int, default=0, help="usar apenas as N primeiras perguntas do tema")
    ap.add_argument("--output-dir", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--run-id", default="rag-v1")
    ap.add_argument("--baseline", type=Path, default=BASELINE_CSV,
                    help="all_results.csv da Meta 2 usado na comparação")
    ap.add_argument("--allow-partial", action="store_true")
    return ap


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collection_name(args, corpus_sha: str) -> str:
    key = cache.digest_text(f"{corpus_sha}|{args.embed_model}|{args.chunk_chars}|{args.chunk_overlap}")[:12]
    return f"{rag_corpus.slugify(args.theme)}_{key}"


def prepare_store(args):
    """Constrói (ou reabre) o índice vetorial do corpus do tema."""
    folder = rag_corpus.theme_dir(args.theme)
    docs = rag_corpus.load_corpus(folder)
    chunks = rag_corpus.chunk_corpus(docs, size=args.chunk_chars, overlap=args.chunk_overlap)
    name = collection_name(args, rag_corpus.corpus_digest(folder))
    if args.store == "chroma":
        store = rag_store.ChromaStore(folder / "chroma", name)
    else:
        print("Aviso: o índice em memória não é persistente; os embeddings são recalculados.")
        store = rag_store.MemoryStore()
    if not store.count():
        print(f"{len(docs)} documentos -> {len(chunks)} excertos; a calcular embeddings ({args.embed_model}).")
        vectors = rag_embed.embed_all(args.embed_model, [c["text"] for c in chunks])
        rag_store.build_index(store, chunks, vectors)
    else:
        rag_store.build_index(store, chunks, [None] * len(chunks))  # só valida a contagem
    print(f"Índice '{name}' com {store.count()} excertos.")
    return store, docs, chunks


def build_manifest(args, digests: dict, folder: Path, n_chunks: int) -> dict:
    return {
        "version": 1, "mode": "meta3-rag", "theme": args.theme,
        "datasets": {name: {"path": str(p.resolve()), "sha256": cache.digest_file(p.resolve())}
                     for name, p in (("train", args.train), ("test", args.test))},
        "models": args.models, "judge_model": args.judge_model, "embed_model": args.embed_model,
        "ollama_model_digests": digests,
        "corpus": {"sha256": rag_corpus.corpus_digest(folder), "chunks": n_chunks,
                   "chunk_chars": args.chunk_chars, "chunk_overlap": args.chunk_overlap},
        "retrieval": {"store": args.store, "top_k": args.top_k, "distance": "cosine"},
        "generation": {"system_prompt": RAG_SYSTEM_PROMPT, "fallback_prompt": prompts.SYSTEM_PROMPT,
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


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        parser.error("--run-id deve conter apenas letras, números, pontos, hífen ou sublinhado")
    if args.top_k < 1 or args.limit < 0 or args.chunk_chars < 100 or args.chunk_overlap >= args.chunk_chars:
        parser.error("parâmetros inválidos: --top-k >= 1, --limit >= 0, --chunk-chars >= 100, "
                     "--chunk-overlap < --chunk-chars")

    folder = rag_corpus.theme_dir(args.theme)
    if args.stage == "corpus":
        seeds = rag_corpus.read_seeds(args.seeds or folder / "seeds.txt")
        rag_corpus.build_corpus(args.theme, seeds, folder)
        return

    df = load_data(args.train, args.test, args.split)
    df = df[df["Tema"] == args.theme]
    if df.empty:
        parser.error(f"Tema sem perguntas: {args.theme}")
    if args.limit:
        df = df.head(args.limit)
    run_dir = args.output_dir.resolve() / "meta3" / args.run_id
    scope = f"{args.split}_{rag_corpus.slugify(args.theme)}_" + (f"limite_{args.limit}" if args.limit else "todas")
    out = run_dir / "analises" / (scope + ("_parcial" if args.allow_partial else "_completa"))
    retrieval_path = run_dir / "retrieval.jsonl"
    run = ["index", "retrieve", "generate", "judge", "score", "analyze", "compare"] \
        if args.stage == "all" else [args.stage]

    store = chunks = None
    if {"index", "retrieve"} & set(run):
        store, _, chunks = prepare_store(args)
    if args.stage == "index":
        return

    if {"retrieve", "generate", "judge"} & set(run):
        digests = llm.model_digests(args.models + [args.judge_model, args.embed_model])
        if chunks is None:
            chunks = rag_corpus.chunk_corpus(rag_corpus.load_corpus(folder), size=args.chunk_chars,
                                             overlap=args.chunk_overlap)
        cache.ensure_manifest(run_dir, build_manifest(args, digests, folder, len(chunks)))
    print(f"{len(df)} perguntas do tema '{args.theme}' | modelos: {args.models} | run: {run_dir}")

    if "retrieve" in run:
        retriever = Retriever(store, args.embed_model, args.top_k)
        retrievals = retrieve_all(df, retriever, retrieval_path)
        table = retrieval_table(df, retrievals)
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "recuperacao_por_pergunta.csv", index=False, encoding="utf-8")
        summary = retrieval_summary(table).round(3)
        summary.to_csv(out / "recuperacao_resumo.csv", index=False, encoding="utf-8")
        print("\n== Recuperação ==\n" + summary.to_string(index=False))
    if "generate" in run:
        retrievals = cache.read_cache(retrieval_path)
        missing = df[~df["id"].isin(retrievals)]
        if not missing.empty:
            raise ValueError(f"Faltam {len(missing)} recuperações; execute 'retrieve' primeiro.")
        generate_rag(df, args.models, retrievals, run_dir)
    if "judge" in run:
        judge(df, args.models, args.judge_model, run_dir=run_dir, allow_partial=args.allow_partial)
    if "score" in run:
        score(df, args.models, args.judge_model, args.bert_model, run_dir=run_dir,
              allow_partial=args.allow_partial, batch_size=args.bert_batch_size, output_dir=out,
              scope=scope, num_layers=args.bert_num_layers)
    if "analyze" in run:
        analyze(out, args.models)
    if "compare" in run:
        if not args.baseline.is_file():
            raise ValueError(f"Baseline inexistente: {args.baseline}")
        table = export_comparison(args.baseline, out / "all_results.csv", out / "recuperacao_por_pergunta.csv", out)
        print("\n== RAG vs. baseline ==\n" + table.round(3).to_string(index=False))
