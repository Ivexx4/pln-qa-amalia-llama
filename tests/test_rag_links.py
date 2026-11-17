"""Testes da expansão do corpus por ligações (rag_corpus.fetch_links / rank_link_candidates / build_corpus)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qa_pipeline import rag_corpus


def page(title, text="Texto do artigo. " * 5):
    """Artigo no formato devolvido por rag_corpus.fetch_article."""
    return {"title": title, "url": f"https://pt.wikipedia.org/wiki/{title}", "revision": 1, "text": text}


class RankLinkCandidatesTest(unittest.TestCase):
    def test_ranks_by_number_of_seeds_and_applies_filters(self):
        link_map = {
            "A": ["Rootkit", "Worm", "2020", "Século XXI", "Lista de vírus", "A", "Rootkit"],
            "B": ["Rootkit", "Worm", "Keylogger", "17 de março"],
            "C": ["Rootkit", "keylogger"],
        }
        ranked = rag_corpus.rank_link_candidates(link_map, exclude={"A"}, min_links=2)
        # "Rootkit" aparece em 3 sementes (duplicados dentro de uma página contam uma vez); "Worm" em 2.
        self.assertEqual(ranked, [("Rootkit", 3), ("Worm", 2)])

    def test_min_links_one_keeps_singletons_in_stable_order(self):
        ranked = rag_corpus.rank_link_candidates({"A": ["Zeta", "Alfa"], "B": ["Alfa"]}, set(), min_links=1)
        self.assertEqual(ranked, [("Alfa", 2), ("Zeta", 1)])


class FetchLinksTest(unittest.TestCase):
    def test_follows_continuation(self):
        replies = [
            {"continue": {"plcontinue": "x|1", "continue": "||"},
             "query": {"pages": [{"title": "A", "links": [{"ns": 0, "title": "B"}]}]}},
            {"query": {"pages": [{"title": "A", "links": [{"ns": 0, "title": "C"}]}]}},
        ]
        with patch.object(rag_corpus, "_wiki_get", side_effect=replies) as get:
            self.assertEqual(rag_corpus.fetch_links(None, "A"), ["B", "C"])
        self.assertEqual(get.call_args_list[1].args[1]["plcontinue"], "x|1")

    def test_page_without_links(self):
        with patch.object(rag_corpus, "_wiki_get", return_value={"query": {"pages": [{"title": "A"}]}}):
            self.assertEqual(rag_corpus.fetch_links(None, "A"), [])


class BuildCorpusExpansionTest(unittest.TestCase):
    LINKS = {"Seed1": ["Rootkit", "Worm", "Seed2", "Inexistente"], "Seed2": ["Rootkit", "Worm", "Inexistente"]}

    def run_build(self, out, **kwargs):
        articles = {"Seed1": page("Seed1"), "Seed2": page("Seed2"), "Rootkit": page("Rootkit"),
                    "Worm": page("Worm")}

        def fake_article(session, title):
            return articles.get(title)

        def fake_links(session, title, pause=0.0):
            return self.LINKS[title]

        with patch.object(rag_corpus, "fetch_article", side_effect=fake_article), \
             patch.object(rag_corpus, "fetch_links", side_effect=fake_links), \
             patch.object(rag_corpus.time, "sleep"):
            return rag_corpus.build_corpus("Tema", ["Seed1", "Seed2"], out, pause=0, **kwargs)

    def test_without_expansion_only_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = self.run_build(Path(tmp))
            self.assertEqual([d["title"] for d in docs], ["Seed1", "Seed2"])
            self.assertTrue(all("origin" not in d for d in docs))
            self.assertFalse((Path(tmp) / "links_expandidos.txt").exists())

    def test_expansion_adds_top_linked_after_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            docs = self.run_build(out, expand_links=True, max_links=1)
            self.assertEqual([d["title"] for d in docs], ["Seed1", "Seed2", "Rootkit"])
            self.assertEqual([d["doc_id"] for d in docs], ["d000", "d001", "d002"])  # ids das sementes intactos
            # "Inexistente" é o 1.º candidato por ordem alfabética mas não tem artigo: é saltado.
            self.assertEqual(docs[2]["origin"], "ligacao")
            self.assertEqual(docs[2]["linked_from"], 2)
            listing = (out / "links_expandidos.txt").read_text(encoding="utf-8")
            self.assertIn("Rootkit\t2", listing)
            self.assertNotIn("Worm", listing)
            self.assertIn("ligacao", (out / "fontes.csv").read_text(encoding="utf-8"))
            lines = (out / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[2])["title"], "Rootkit")

    def test_expansion_skips_missing_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = self.run_build(Path(tmp), expand_links=True, max_links=5)
            self.assertEqual([d["title"] for d in docs], ["Seed1", "Seed2", "Rootkit", "Worm"])

    def test_cache_avoids_refetching(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.run_build(out, expand_links=True, max_links=2)
            with patch.object(rag_corpus, "fetch_article", side_effect=AssertionError("sem rede")), \
                 patch.object(rag_corpus, "fetch_links", side_effect=AssertionError("sem rede")), \
                 patch.object(rag_corpus.time, "sleep"):
                docs = rag_corpus.build_corpus("Tema", ["Seed1", "Seed2"], out, pause=0,
                                               expand_links=True, max_links=2)
            self.assertEqual(len(docs), 4)


if __name__ == "__main__":
    unittest.main()
