import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qa_pipeline import analysis, cli, config, data, generation, judge, metrics

import pandas as pd
import requests

from qa_pipeline import cache, llm, scoring
from qa_pipeline.judge import parse_verdict


class ParseVerdictTests(unittest.TestCase):
    def test_accepts_exact_json_verdicts(self):
        self.assertEqual(parse_verdict('{"veredito": "CORRETO"}'), 1)
        self.assertEqual(parse_verdict('{"veredito": "INCORRETO"}'), 0)

    def test_accepts_exact_legacy_token(self):
        self.assertEqual(parse_verdict("  CORRETO\n"), 1)
        self.assertEqual(parse_verdict("INCORRETO"), 0)

    def test_rejects_prose_and_malformed_json(self):
        for text in (
            "A resposta está correta",
            "A resposta não está incorreta",
            "Não está totalmente correto",
            '{"veredito": "CORRETO", "explicacao": "texto"}',
            '{"veredito": "talvez"}',
            "```json\n{\"veredito\": \"CORRETO\"}\n```",
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_verdict(text))


class LLMRetryTests(unittest.TestCase):
    @patch("qa_pipeline.llm.time.sleep")
    @patch("qa_pipeline.llm.requests.post")
    def test_retries_timeout_then_returns_response(self, post, sleep):
        response = Mock()
        response.json.return_value = {"message": {"content": "Lisboa"}}
        post.side_effect = [
            requests.Timeout("temporary"),
            requests.Timeout("temporary"),
            response,
        ]

        self.assertEqual(llm.chat("model", []), "Lisboa")
        self.assertEqual(post.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [1, 2],
        )

    @patch("qa_pipeline.llm.time.sleep")
    @patch("qa_pipeline.llm.requests.post")
    def test_retries_transient_http_error(self, post, sleep):
        response = Mock()
        response.raise_for_status.side_effect = [
            requests.HTTPError(
                "server error", response=Mock(status_code=503)
            ),
            None,
        ]
        response.json.return_value = {"message": {"content": "Lisboa"}}
        post.return_value = response

        self.assertEqual(llm.chat("model", []), "Lisboa")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(1)

    @patch("qa_pipeline.llm.time.sleep")
    @patch("qa_pipeline.llm.requests.post")
    def test_does_not_retry_permanent_http_error(self, post, sleep):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError(
            "bad request", response=Mock(status_code=400)
        )
        post.return_value = response

        with self.assertRaises(requests.HTTPError):
            llm.chat("model", [])

        post.assert_called_once()
        sleep.assert_not_called()


class ScoringDeviceTests(unittest.TestCase):
    @patch("bert_score.BERTScorer")
    @patch("torch.cuda.is_available", return_value=True)
    def test_uses_cuda_when_available(self, is_available, bert_score):
        scores = Mock()
        scores.detach.return_value.cpu.return_value.numpy.return_value = [0.8]
        bert_score.return_value.score.return_value = (None, None, scores)
        results = pd.DataFrame({"Pred": ["Lisboa"], "Gold": ["Lisboa"]})

        scored = scoring.add_bertscore(results, "bert-base-multilingual-cased")

        self.assertEqual(scored["BERTScore_F1"].tolist(), [0.8])
        self.assertEqual(bert_score.call_args.kwargs["device"], "cuda")

    @patch("bert_score.BERTScorer")
    @patch("torch.cuda.is_available", return_value=False)
    def test_portuguese_model_uses_its_last_layer(self, is_available, bert_score):
        scores = Mock()
        scores.detach.return_value.cpu.return_value.numpy.return_value = [0.8]
        bert_score.return_value.score.return_value = (None, None, scores)
        results = pd.DataFrame({"Pred": ["Lisboa"], "Gold": ["Lisboa"]})

        scoring.add_bertscore(results, config.DEFAULT_BERT_MODEL)
        self.assertEqual(bert_score.call_args.kwargs["model_type"], config.DEFAULT_BERT_MODEL)
        self.assertEqual(bert_score.call_args.kwargs["num_layers"], 12)
        bert_score.return_value._model.float.assert_called_once_with()
        self.assertEqual(bert_score.return_value.score.call_args.kwargs["batch_size"], 8)
        scoring.add_bertscore(results, config.DEFAULT_BERT_MODEL, num_layers=9)
        self.assertEqual(bert_score.call_args.kwargs["num_layers"], 9)

    @patch("bert_score.BERTScorer")
    @patch("torch.cuda.is_available", return_value=False)
    def test_uses_cpu_when_cuda_is_unavailable(self, is_available, bert_score):
        scores = Mock()
        scores.detach.return_value.cpu.return_value.numpy.return_value = [0.8]
        bert_score.return_value.score.return_value = (None, None, scores)
        results = pd.DataFrame({"Pred": ["Lisboa"], "Gold": ["Lisboa"]})

        scoring.add_bertscore(results, "bert-base-multilingual-cased")

        self.assertEqual(bert_score.call_args.kwargs["device"], "cpu")


class JudgeCacheTests(unittest.TestCase):
    def test_cache_filename_versions_judge_prompt_results(self):
        path = cache.judge_path("model", "judge")
        self.assertIn(cache.config.JUDGE_CACHE_VERSION, path.name)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = self.root / "train.tsv"
        self.test = self.root / "test.tsv"
        header = "Autor\tTema\tPergunta\tMétodo\tResposta\n"
        self.train.write_text(header + "A\tGeografia\tCapital de Portugal?\tHUM\tLisboa\n", encoding="utf-8")
        self.test.write_text(header + "B\tGeografia\tCapital de França?\tLLM\tParis\n", encoding="utf-8")
        self.addCleanup(self.temp.cleanup)

    def test_ids_are_stable_across_splits(self):
        both = data.load_data(self.train, self.test, "both")
        test = data.load_data(self.train, self.test, "test")
        self.assertEqual(both.loc[both["split"] == "test", "id"].tolist(), test["id"].tolist())
        self.assertEqual(test["id"].tolist(), ["test_0"])

    def test_rejects_invalid_method_and_empty_answer(self):
        self.test.write_text("Tema\tPergunta\tMétodo\tResposta\nX\tQ\tOUTRO\tR\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "métodos inválidos"):
            data.load_data(self.train, self.test, "test")
        self.test.write_text("Tema\tPergunta\tMétodo\tResposta\nX\tQ\tLLM\t\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Resposta está vazio"):
            data.load_data(self.train, self.test, "test")

    def test_preserves_unknown_escapes_and_invalid_dates(self):
        self.assertEqual(data.fix_text("_x0096_ _x0081_ _x1234_"), "– _x0081_ _x1234_")
        self.assertEqual(data.fix_text("2020-13-01 00:00:00"), "2020-13-01 00:00:00")
        self.assertEqual(data.fix_text("2020-02-01 00:00:00"), "1 de fevereiro de 2020")


class CacheIsolationTests(unittest.TestCase):
    def test_manifest_detects_changes_and_paths_do_not_collide(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache.ensure_manifest(root, {"prompt": "primeiro"})
            cache.ensure_manifest(root, {"prompt": "primeiro"})
            with self.assertRaisesRegex(ValueError, "outro --run-id"):
                cache.ensure_manifest(root, {"prompt": "outro"})
            self.assertNotEqual(cache.preds_path("a/b", root), cache.preds_path("a_b", root))

    def test_conflicting_duplicate_records_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.jsonl"
            cache.append_cache(path, {"id": "test_0", "pred": "A"})
            cache.append_cache(path, {"id": "test_0", "pred": "B"})
            with self.assertRaisesRegex(ValueError, "contraditórios"):
                cache.read_cache(path)

    def test_verdict_must_match_prediction(self):
        with self.assertRaisesRegex(ValueError, "incompatível"):
            cache.validate_judgement({"pred_sha256": cache.digest_text("outra")}, "Lisboa", "test_0")


class MetricAndAnalysisTests(unittest.TestCase):
    def test_lexical_metrics(self):
        self.assertEqual(metrics.exact_match("Lisboa!", "LISBOA"), 1)
        self.assertEqual(metrics.contains("Capital: Lisboa.", "Lisboa"), 1)
        self.assertEqual(metrics.contains("Lisboeta", "Lisboa"), 0)
        self.assertAlmostEqual(metrics.rouge_l("um dois três", "um três"), 0.8)

    def test_empty_gold_never_matches(self):
        self.assertEqual(metrics.exact_match("", ""), 0)
        self.assertEqual(metrics.exact_match("!!!", "???"), 0)
        self.assertEqual(metrics.exact_match("Lisboa", ""), 0)
        self.assertEqual(metrics.exact_match("中国", "日本"), 0)

    def test_tokens_keep_non_latin_letters(self):
        self.assertEqual(metrics.tokens("Αθήνα"), ["αθηνα"])
        self.assertEqual(metrics.exact_match("Αθήνα", "ΑΘΗΝΑ"), 1)
        self.assertEqual(metrics.tokens("Москва"), ["москва"])

    def test_exact_match_is_stricter_than_contains(self):
        self.assertEqual(metrics.exact_match("O Tejo", "Tejo"), 0)
        self.assertEqual(metrics.contains("O Tejo", "Tejo"), 1)

    def test_paired_analysis_excludes_incomplete_metric_pairs(self):
        base = {"Pergunta": "Q", "Gold": "R", "Tema": "T", "Metodo": "HUM", "Autor": "A",
                "split": "test", "EM": 1, "Contains": 1, "ROUGE_L": 1, "BERTScore_F1": 1}
        rows = [dict(base, id="test_0", Modelo="m1", Judge=1),
                dict(base, id="test_0", Modelo="m2", Judge=None),
                dict(base, id="test_1", Modelo="m1", Judge=1)]
        paired = analysis.paired_results(pd.DataFrame(rows), ["m1", "m2"])
        self.assertEqual(paired["id"].tolist(), ["test_0", "test_0"])
        self.assertTrue(paired["Judge"].isna().all())
        tables = analysis.summary_tables(paired)
        self.assertIn("por_autor", tables)
        self.assertEqual(tables["por_modelo"].loc["m1", "n_Judge"], 0)


class PipelineIntegrationTests(DataTests):
    def test_incomplete_results_are_not_silent(self):
        df = data.load_data(self.train, self.test, "both")
        with self.assertRaisesRegex(ValueError, "Resultados incompletos"):
            scoring.build_results(df, ["modelo"], "juiz", self.root)
        row = df.iloc[0]
        cache.append_cache(cache.preds_path("modelo", self.root), {
            "id": row["id"], "pred": "Lisboa", "question_sha256": cache.digest_text(row["Pergunta"]),
        })
        partial = scoring.build_results(df, ["modelo"], "juiz", self.root, allow_partial=True)
        self.assertEqual(partial["Judge_status"].tolist(), ["em falta"])
        self.assertEqual(partial["Estado_experiencia"].tolist(), ["parcial"])
        cache.append_cache(cache.judge_path("modelo", "juiz", self.root), {
            "id": row["id"], "pred_sha256": cache.digest_text("Lisboa"), "raw": "não sei", "judge": None,
        })
        partial = scoring.build_results(df, ["modelo"], "juiz", self.root, allow_partial=True)
        self.assertEqual(partial["Judge_status"].tolist(), ["inválido"])
        self.assertTrue(partial["Judge"].isna().all())

    def test_end_to_end_and_resume_without_reusing_history(self):
        args = ["all", "--train", str(self.train), "--test", str(self.test),
                "--output-dir", str(self.root), "--run-id", "teste"]
        digests = {name: "digest-" + name for name in config.DEFAULT_MODELS + [config.DEFAULT_JUDGE]}

        def reply(model, messages, **kwargs):
            return '{"veredito": "CORRETO"}' if model == config.DEFAULT_JUDGE else "Lisboa"

        def fake_bert(frame, model, batch_size, num_layers):
            return frame.assign(BERTScore_F1=0.8)

        with patch("qa_pipeline.llm.model_digests", return_value=digests), \
             patch("qa_pipeline.llm.chat", side_effect=reply) as chat, \
             patch("qa_pipeline.scoring.add_bertscore", side_effect=fake_bert), \
             patch("qa_pipeline.analysis.plot_bars"):
            cli.main(args)
            self.assertEqual(chat.call_count, 8)
            cli.main(args)
            self.assertEqual(chat.call_count, 8)
            cli.main(["generate", "--train", str(self.train), "--test", str(self.test),
                      "--output-dir", str(self.root), "--run-id", "teste", "--split", "test"])
            self.assertEqual(chat.call_count, 8)
            source_args = ["--train", str(self.train), "--test", str(self.test),
                           "--output-dir", str(self.root), "--run-id", "outra-metrica",
                           "--source-run", "teste", "--bert-num-layers", "9"]
            cli.main(["score"] + source_args)
            cli.main(["analyze"] + source_args)
            self.assertEqual(chat.call_count, 8)
            source_manifest = json.loads((self.root / "meta2" / "outra-metrica" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(source_manifest["source_run"]["id"], "teste")
            self.assertEqual(source_manifest["metrics"]["bert_num_layers"], 9)
            self.assertEqual(source_manifest["source_run"]["manifest_sha256"],
                             cache.digest_file(self.root / "meta2" / "teste" / "manifest.json"))
        run_dir = self.root / "meta2" / "teste"
        result = pd.read_csv(run_dir / "analises" / "both_todas_completa" / "all_results.csv")
        self.assertEqual(len(result), 4)
        self.assertTrue(result["Judge_status"].eq("válido").all())
        self.assertTrue((run_dir / "analises" / "both_todas_completa" / "tabela_por_autor.csv").exists())
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["mode"], "meta2")
        self.assertEqual(manifest["metrics"]["bert_model"], config.DEFAULT_BERT_MODEL)
        self.assertEqual(manifest["metrics"]["bert_num_layers"], 12)

    def test_source_run_rejects_incompatible_generation(self):
        args = cli.build_parser().parse_args(["score", "--source-run", "origem"])
        manifest = cli.build_manifest(args, {name: "digest" for name in config.DEFAULT_MODELS + [config.DEFAULT_JUDGE]})
        different = dict(manifest, generation={**manifest["generation"], "seed": 999})
        with self.assertRaisesRegex(ValueError, "incompatível.*generation"):
            cli.validate_source_run(different, manifest)
        different = dict(manifest, source_sha256={**manifest["source_sha256"], "judge.py": "alterado"})
        with self.assertRaisesRegex(ValueError, r"incompatível.*judge\.py"):
            cli.validate_source_run(different, manifest)


if __name__ == "__main__":
    unittest.main()
