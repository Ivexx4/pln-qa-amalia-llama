import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from qa_pipeline import cache, rag_cli, rag_compare, rag_config, rag_corpus, rag_generation, rag_retrieval, rag_store

VOCAB = ["benfica", "porto", "sporting", "ronaldo", "messi", "madrid", "barcelona", "luz", "dragao", "alvalade"]


def fake_embed(texts):
    """Saco de palavras determinístico: suficiente para testar a recuperação sem Ollama."""
    out = []
    for text in texts:
        words = text.lower().split()
        out.append([float(sum(w.strip(".,?—") == v for w in words)) + 1e-3 for v in VOCAB])
    return out


DOC = {"doc_id": "d000", "title": "Estádio da Luz", "url": "https://pt.wikipedia.org/wiki/Est%C3%A1dio_da_Luz",
       "text": "O Estádio da Luz é a casa do Benfica.\n\n== História ==\n" + " ".join(["Frase longa sobre o estádio."] * 60)
       + "\n\n== Referências ==\nNada relevante aqui."}


class CorpusTests(unittest.TestCase):
    def test_slugify_removes_accents(self):
        self.assertEqual(rag_corpus.slugify("Tradições Académicas de Coimbra"), "tradicoes-academicas-de-coimbra")

    def test_strip_sections_drops_references(self):
        text = rag_corpus.strip_sections(DOC["text"])
        self.assertIn("História", text)
        self.assertNotIn("Nada relevante", text)

    def test_read_seeds_ignores_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.txt"
            path.write_text("# x\n\nFutebol\n?Taça de Portugal\n", encoding="utf-8")
            self.assertEqual(rag_corpus.read_seeds(path), ["Futebol", "?Taça de Portugal"])
            path.write_text("# só comentários\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                rag_corpus.read_seeds(path)

    def test_fetch_article_handles_missing_and_cleans(self):
        page = {"title": "Estádio da Luz", "fullurl": "u", "lastrevid": 7, "extract": DOC["text"]}
        with patch.object(rag_corpus, "_wiki_get", return_value={"query": {"pages": [page]}}):
            article = rag_corpus.fetch_article(None, "x")
        self.assertEqual(article["revision"], 7)
        self.assertNotIn("Referências", article["text"])
        with patch.object(rag_corpus, "_wiki_get", return_value={"query": {"pages": [{"title": "x", "missing": True}]}}):
            self.assertIsNone(rag_corpus.fetch_article(None, "x"))

    def test_chunks_respect_size_and_carry_title_and_section(self):
        doc = {**DOC, "text": rag_corpus.strip_sections(DOC["text"])}
        chunks = rag_corpus.chunk_document(doc, size=400, overlap=60)
        self.assertGreater(len(chunks), 3)
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
        for chunk in chunks:
            self.assertTrue(chunk["text"].startswith("Estádio da Luz"))
            self.assertLessEqual(len(chunk["text"]), 400 + len("Estádio da Luz — História\n") + 5)
        self.assertTrue(any(c["section"] == "História" for c in chunks))

    def test_chunk_rejects_bad_overlap(self):
        with self.assertRaises(ValueError):
            rag_corpus.chunk_document(DOC, size=100, overlap=100)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"chunk_id": f"c{i}", "doc_id": "d", "title": t, "section": "", "url": "", "text": t}
                       for i, t in enumerate(["benfica luz", "porto dragao", "sporting alvalade"])]
        self.vectors = fake_embed([c["text"] for c in self.chunks])

    def test_memory_store_ranks_by_cosine(self):
        store = rag_store.MemoryStore()
        rag_store.build_index(store, self.chunks, self.vectors)
        hits = store.query(fake_embed(["benfica"])[0], 2)
        self.assertEqual(hits[0]["chunk_id"], "c0")
        self.assertEqual(len(hits), 2)
        self.assertGreaterEqual(hits[0]["score"], hits[1]["score"])

    def test_build_index_is_idempotent_and_detects_mismatch(self):
        store = rag_store.MemoryStore()
        rag_store.build_index(store, self.chunks, self.vectors)
        rag_store.build_index(store, self.chunks, self.vectors)
        self.assertEqual(store.count(), 3)
        with self.assertRaises(ValueError):
            rag_store.build_index(store, self.chunks[:2], self.vectors[:2])


class RetrievalMetricTests(unittest.TestCase):
    hits = [{"text": "O Benfica joga na Luz", "title": "A", "score": 0.9},
            {"text": "Ronaldo ganhou a Bola de Ouro em 2008", "title": "B", "score": 0.8}]

    def test_gold_rank(self):
        self.assertEqual(rag_retrieval.gold_rank(self.hits, "Bola de Ouro"), 2)
        self.assertEqual(rag_retrieval.gold_rank(self.hits, "Messi"), None)

    def test_token_recall(self):
        self.assertEqual(rag_retrieval.token_recall(self.hits, "Ronaldo 2008"), 1.0)
        self.assertEqual(rag_retrieval.token_recall(self.hits, "Ronaldo Messi"), 0.5)

    def test_table_and_summary(self):
        df = pd.DataFrame([{"id": "a", "Tema": "F", "Metodo": "HUM", "Pergunta": "q1", "Resposta": "Luz"},
                           {"id": "b", "Tema": "F", "Metodo": "LLM", "Pergunta": "q2", "Resposta": "Messi"}])
        table = rag_retrieval.retrieval_table(df, {"a": {"hits": self.hits}, "b": {"hits": self.hits}})
        self.assertEqual(table["hit@1"].tolist(), [1, 0])
        self.assertEqual(table["mrr"].tolist(), [1.0, 0.0])
        summary = rag_retrieval.retrieval_summary(table)
        self.assertEqual(summary["grupo"].tolist(), ["todas", "HUM", "LLM"])
        self.assertAlmostEqual(summary.loc[0, "hit@3"], 0.5)

    def test_retrieve_all_resumes_and_detects_changed_question(self):
        store = rag_store.MemoryStore()
        chunks = [{"chunk_id": "c0", "doc_id": "d", "title": "T", "section": "", "url": "", "text": "benfica"}]
        rag_store.build_index(store, chunks, fake_embed(["benfica"]))
        retriever = rag_retrieval.Retriever(store, "m", 1, embed_fn=fake_embed)
        df = pd.DataFrame([{"id": "a", "Pergunta": "benfica?"}])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            first = rag_retrieval.retrieve_all(df, retriever, path)
            again = rag_retrieval.retrieve_all(df, retriever, path)
            self.assertEqual(first, again)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            with self.assertRaises(ValueError):
                rag_retrieval.retrieve_all(pd.DataFrame([{"id": "a", "Pergunta": "outra"}]), retriever, path)


class GenerationTests(unittest.TestCase):
    def test_prompt_contains_numbered_chunks_and_question(self):
        prompt = rag_generation.build_user_prompt("Quem?", [{"text": "T1"}, {"text": "T2"}])
        self.assertIn("[Excerto 1] T1", prompt)
        self.assertTrue(prompt.endswith("Pergunta: Quem?"))
        self.assertEqual(rag_generation.build_user_prompt("Quem?", []), "Quem?")

    def test_generate_writes_same_cache_format_as_meta2(self):
        df = pd.DataFrame([{"id": "a", "Pergunta": "Quem?"}])
        retrievals = {"a": {"hits": [{"text": "T1"}]}}
        with tempfile.TemporaryDirectory() as tmp, patch("qa_pipeline.llm.chat", return_value="Benfica") as chat:
            rag_generation.generate_rag(df, ["m"], retrievals, Path(tmp))
            rag_generation.generate_rag(df, ["m"], retrievals, Path(tmp))  # retoma: não repete
            self.assertEqual(chat.call_count, 1)
            record = cache.read_cache(cache.preds_path("m", Path(tmp)))["a"]
            self.assertEqual(record["pred"], "Benfica")
            cache.validate_prediction(record, "Quem?", "a")
            messages = chat.call_args.args[1]
            self.assertEqual(messages[0]["content"], rag_generation.RAG_SYSTEM_PROMPT)


class CompareTests(unittest.TestCase):
    @staticmethod
    def frame(judge, contains, rouge):
        n = len(judge)
        return pd.DataFrame({"id": [f"q{i}" for i in range(n)], "Modelo": ["m"] * n, "Judge": judge,
                             "Contains": contains, "ROUGE_L": rouge, "EM": [0] * n, "BERTScore_F1": rouge})

    def test_compare_signs_and_counts(self):
        base = self.frame([0, 0, 1, 0], [0, 0, 1, 0], [0.1, 0.2, 0.9, 0.0])
        rag = self.frame([1, 1, 1, 0], [1, 1, 1, 0], [0.8, 0.7, 0.9, 0.0])
        table = rag_compare.compare_runs(base, rag).set_index("metrica")
        self.assertAlmostEqual(table.loc["Judge", "diferenca"], 0.5)
        self.assertEqual(table.loc["Judge", "so_rag_certo"], 2)
        self.assertEqual(table.loc["Judge", "so_baseline_certo"], 0)
        self.assertEqual(table.loc["Judge", "n"], 4)

    def test_compare_ignores_missing_judge_values(self):
        base = self.frame([0, None, 1], [0, 0, 1], [0.1, 0.2, 0.9])
        rag = self.frame([1, 1, 1], [1, 1, 1], [0.8, 0.7, 0.9])
        table = rag_compare.compare_runs(base, rag).set_index("metrica")
        self.assertEqual(table.loc["Judge", "n"], 2)


class CliTests(unittest.TestCase):
    def test_pipeline_index_retrieve_generate_with_mocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            corpus_dir = tmp / "rag" / "futebol"
            corpus_dir.mkdir(parents=True)
            docs = [{**DOC, "text": rag_corpus.strip_sections(DOC["text"])}]
            (corpus_dir / "corpus.jsonl").write_text(json.dumps(docs[0], ensure_ascii=False) + "\n", encoding="utf-8")
            digests = {"m": "x", "j": "y", "e": "z"}
            args = ["all", "--theme", "Futebol", "--store", "memory", "--models", "m", "--judge-model", "j",
                    "--embed-model", "e", "--limit", "2", "--output-dir", str(tmp / "res")]
            with patch.object(rag_config, "RAG_DIR", tmp / "rag"), \
                 patch("qa_pipeline.rag_embed.embed", side_effect=lambda model, texts: fake_embed(texts)), \
                 patch("qa_pipeline.llm.model_digests", return_value=digests), \
                 patch("qa_pipeline.llm.chat", return_value="Benfica"), \
                 patch("qa_pipeline.rag_cli.judge"), patch("qa_pipeline.rag_cli.score"), \
                 patch("qa_pipeline.rag_cli.analyze"), patch("qa_pipeline.rag_cli.export_comparison"), \
                 patch.object(rag_cli.Path, "is_file", return_value=True):
                rag_cli.main(args)
            run = tmp / "res" / "meta3" / "rag-v1"
            self.assertTrue((run / "manifest.json").is_file())
            self.assertEqual(len((run / "retrieval.jsonl").read_text().splitlines()), 2)
            preds = cache.read_cache(cache.preds_path("m", run))
            self.assertEqual(len(preds), 2)

    def test_unknown_theme_is_rejected(self):
        with self.assertRaises(SystemExit):
            rag_cli.main(["retrieve", "--theme", "Tema Inexistente"])


if __name__ == "__main__":
    unittest.main()
