import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qa_pipeline import analysis, cli, config, data, generation, judge, metrics, stats, validation

import numpy as np
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


class StatisticalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.reference = config.DEFAULT_AMALIA_MODEL
        self.other = "outro"
        rows = []
        for i in range(33):
            theme = "Grande" if i < 30 else "Pequeno" if i < 32 else "Unico"
            common = {"id": f"q{i:02}", "Pergunta": f"Q{i}", "Gold": "R", "Tema": theme,
                      "Metodo": "HUM", "Autor": "A", "split": "test"}
            base = 0.2 + (i % 2) * 0.2
            for model, bonus in ((self.reference, 0.1), (self.other, 0.0)):
                rows.append(dict(common, Modelo=model, EM=i % 2, Contains=i % 2,
                                 ROUGE_L=base + bonus, BERTScore_F1=base + bonus,
                                 Judge=None if model == self.other and i in (0, 32) else i % 2))
        self.rows = pd.DataFrame(rows)
        self.paired = analysis.paired_results(self.rows, [self.reference, self.other])

    def test_bootstrap_is_deterministic_and_differences_are_paired(self):
        self.assertEqual(stats.BOOTSTRAP_SAMPLES, 10_000)
        overall, themes = stats.statistical_tables(self.paired)
        shuffled = analysis.paired_results(self.rows.sample(frac=1, random_state=77),
                                           [self.reference, self.other])
        overall_again, themes_again = stats.statistical_tables(shuffled)
        pd.testing.assert_frame_equal(overall, overall_again)
        pd.testing.assert_frame_equal(themes, themes_again)
        row = overall[(overall["tipo"] == "diferenca") & (overall["metrica"] == "ROUGE_L")].iloc[0]
        self.assertEqual(row["n"], 33)
        self.assertAlmostEqual(row["estimativa"], 0.1)
        self.assertAlmostEqual(row["ic_95_inferior"], 0.1)
        self.assertAlmostEqual(row["ic_95_superior"], 0.1)
        for metric in config.METRICS:
            self.assertEqual(len(overall[(overall["tipo"] == "media") & (overall["metrica"] == metric)]), 2)
            self.assertEqual(len(overall[(overall["tipo"] == "diferenca") & (overall["metrica"] == metric)]), 1)

    def test_bootstrap_percentiles_match_seeded_resampling(self):
        values = np.array([0.0, 0.2, 0.5, 0.7, 0.9, 1.0])
        mean, lower, upper = stats.bootstrap_mean(values, np.random.default_rng(42))
        generator = np.random.default_rng(42)
        samples = generator.choice(values, size=(10_000, len(values)), replace=True)
        expected_lower, expected_upper = np.quantile(samples.mean(axis=1), [0.025, 0.975])
        self.assertAlmostEqual(mean, values.mean())
        self.assertAlmostEqual(lower, expected_lower)
        self.assertAlmostEqual(upper, expected_upper)

    def test_exact_mcnemar_and_wilcoxon(self):
        reference = np.array([1, 1, 1, 1, 1, 1, 0, 0])
        other = 1 - reference
        for metric in ("Judge", "Contains"):
            result = stats.paired_test(metric, reference, other)
            self.assertEqual((result["discordantes_amalia"], result["discordantes_outro"]), (6, 2))
            self.assertAlmostEqual(result["p_valor"], 0.2890625)
        reference = np.arange(1, 9, dtype=float)
        for metric in ("ROUGE_L", "BERTScore_F1"):
            result = stats.paired_test(metric, reference, np.zeros(8))
            self.assertEqual(result["estatistica"], 0)
            self.assertAlmostEqual(result["p_valor"], 0.0078125)
            self.assertEqual(stats.paired_test(metric, reference, reference)["p_valor"], 1)
        self.assertEqual(stats.paired_test("Judge", other, other)["p_valor"], 1)
        self.assertTrue(pd.isna(stats.paired_test("Judge", reference[:0], other[:0])["p_valor"]))
        with self.assertRaisesRegex(ValueError, "binária"):
            stats.paired_test("Contains", np.array([2]), np.array([0]))

    def test_theme_counts_missing_values_and_small_samples(self):
        overall, themes = stats.statistical_tables(self.paired)
        row = overall[(overall["tipo"] == "diferenca") & (overall["metrica"] == "Judge")].iloc[0]
        self.assertEqual(row["n"], 31)
        large = themes[(themes["Tema"] == "Grande") & (themes["Modelo"] == self.reference)]
        judge_row = large[large["metrica"] == "Judge"].iloc[0]
        self.assertEqual((judge_row["n_tema"], judge_row["n"], judge_row["aviso"]), (30, 29, "n<30"))
        em_row = large[large["metrica"] == "EM"].iloc[0]
        self.assertEqual((em_row["n"], em_row["aviso"]), (30, ""))
        single = themes[(themes["Tema"] == "Unico") & (themes["Modelo"] == self.reference)]
        self.assertTrue(pd.isna(single[single["metrica"] == "EM"].iloc[0]["ic_95_inferior"]))
        no_judge = single[single["metrica"] == "Judge"].iloc[0]
        self.assertEqual(no_judge["n"], 0)
        self.assertTrue(pd.isna(no_judge["media"]))

    def test_metric_with_no_valid_pairs_has_no_estimate_or_p_value(self):
        rows = self.rows.copy()
        rows["Judge"] = None
        paired = analysis.paired_results(rows, [self.reference, self.other])
        overall, themes = stats.statistical_tables(paired)
        judge_rows = overall[overall["metrica"] == "Judge"]
        self.assertTrue(judge_rows["n"].eq(0).all())
        self.assertTrue(judge_rows["estimativa"].isna().all())
        self.assertTrue(judge_rows["p_valor"].isna().all())
        self.assertTrue(themes[themes["metrica"] == "Judge"]["aviso"].eq("n<30").all())

    def test_multiple_models_and_no_amalia(self):
        third = self.rows[self.rows["Modelo"] == self.other].copy()
        third["Modelo"] = "terceiro"
        data = pd.concat([self.rows, third], ignore_index=True)
        overall, _ = stats.statistical_tables(analysis.paired_results(data, [self.reference, self.other, "terceiro"]))
        comparisons = overall[(overall["tipo"] == "diferenca") & (overall["metrica"] == "Judge")]
        self.assertEqual(set(comparisons["modelo_comparado"]), {self.other, "terceiro"})
        only_other = analysis.paired_results(data, [self.other, "terceiro"])
        without_amalia, _ = stats.statistical_tables(only_other)
        self.assertFalse(without_amalia["tipo"].eq("diferenca").any())


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.df = pd.DataFrame([
            {"id": f"q{i:03}", "split": "test", "Autor": "A", "Tema": f"T{i % 3}",
             "Metodo": "HUM", "Pergunta": f"Pergunta {i}", "Resposta": f"R{i}"}
            for i in range(120)
        ])
        preds = cache.preds_path("modelo-a", self.root)
        verdicts = cache.judge_path("modelo-a", "juiz", self.root)
        for i, row in self.df.iterrows():
            cache.append_cache(preds, {"id": row["id"], "pred": f"R{i}",
                                       "question_sha256": cache.digest_text(row["Pergunta"])})
            cache.append_cache(verdicts, {"id": row["id"], "pred_sha256": cache.digest_text(f"R{i}"),
                                          "raw": "{}", "judge": int(i % 2 == 0)})
        self.pairs = validation.collect_pairs(self.df, ["modelo-a"], "juiz", self.root)

    def test_sample_is_stratified_and_reproducible(self):
        self.assertEqual(len(self.pairs), 120)
        sample = validation.stratified_sample(self.pairs)
        self.assertEqual(len(sample), 100)
        self.assertEqual(sample["juiz"].value_counts().to_dict(), {1: 50, 0: 50})
        self.assertEqual(sample["id"].nunique(), 100)
        shuffled = validation.collect_pairs(self.df.sample(frac=1, random_state=3), ["modelo-a"], "juiz", self.root)
        pd.testing.assert_frame_equal(sample, validation.stratified_sample(shuffled))
        small = validation.stratified_sample(self.pairs[self.pairs["juiz"] == 1], size=100)
        self.assertEqual(len(small), 50)

    def test_export_leaves_human_column_empty(self):
        path = self.root / "validacao_juiz.csv"
        exported = validation.export_sample(self.pairs, path)
        saved = pd.read_csv(path, encoding="utf-8")
        self.assertEqual(list(saved.columns), validation.SAMPLE_COLUMNS)
        self.assertTrue(saved["humano"].isna().all())
        self.assertEqual(len(saved), len(exported))
        self.assertEqual(set(saved["Modelo"]), {"modelo-a"})
        self.assertTrue(saved["Pred"].str.startswith("R").all())

    def test_agreement_metrics_match_hand_computation(self):
        sample = pd.DataFrame({"juiz": [1, 1, 1, 1, 0, 0, 0, 0],
                               "humano": [1, 1, 1, 0, 1, 0, 0, 0]})
        summary = validation.agreement_summary(sample)
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"], summary["tn"]), (3, 1, 1, 3))
        self.assertAlmostEqual(summary["acordo_bruto"], 0.75)
        self.assertAlmostEqual(summary["kappa"], 0.5)
        self.assertAlmostEqual(summary["precisao"], 0.75)
        self.assertAlmostEqual(summary["recall"], 0.75)
        perfect = validation.agreement_summary(sample.assign(humano=sample["juiz"]))
        self.assertAlmostEqual(perfect["kappa"], 1.0)
        self.assertTrue(np.isnan(validation.agreement_summary(
            sample.assign(juiz=1, humano=1))["kappa"]))

    def test_missing_or_invalid_human_values_raise(self):
        sample = pd.DataFrame({"juiz": [1, 0], "humano": [1, None]})
        with self.assertRaisesRegex(ValueError, "1 valores em falta"):
            validation.agreement_summary(sample)
        with self.assertRaisesRegex(ValueError, "só aceita 0 ou 1"):
            validation.agreement_summary(pd.DataFrame({"juiz": [1, 0], "humano": [1, 2]}))

    def test_summary_is_written_from_filled_csv(self):
        filled = validation.stratified_sample(self.pairs).assign(humano=lambda d: d["juiz"])
        run_dir = self.root / "run"
        run_dir.mkdir()
        filled[validation.SAMPLE_COLUMNS].to_csv(run_dir / "validacao_juiz.csv", index=False)
        summary = validation.export_summary(run_dir)
        self.assertAlmostEqual(summary["kappa"], 1.0)
        saved = pd.read_csv(run_dir / "validacao_juiz_resumo.csv")
        self.assertEqual(list(saved.columns), validation.SUMMARY_COLUMNS)
        self.assertEqual(int(saved.loc[0, "n"]), 100)
        with self.assertRaisesRegex(ValueError, "validate-judge"):
            validation.export_summary(self.root / "vazio")


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
        out = run_dir / "analises" / "both_todas_completa"
        self.assertTrue((out / "tabela_por_autor.csv").exists())
        global_stats = pd.read_csv(out / "tabela_estatistica.csv")
        theme_stats = pd.read_csv(out / "tabela_por_tema_ic.csv")
        self.assertEqual(len(global_stats), 15)
        self.assertEqual(len(theme_stats), 10)
        self.assertTrue(theme_stats["aviso"].eq("n<30").all())
        self.assertEqual(global_stats[global_stats["metrica"] == "Judge"]["n"].tolist(), [2, 2, 2])
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("stats.py", manifest["source_sha256"])
        self.assertIn("scipy", manifest["versions"])
        self.assertEqual(manifest["mode"], "meta2")
        self.assertEqual(manifest["metrics"]["bert_model"], config.DEFAULT_BERT_MODEL)
        self.assertEqual(manifest["metrics"]["bert_num_layers"], 12)

    def test_stats_only_regenerates_statistical_tables(self):
        args = ["all", "--train", str(self.train), "--test", str(self.test),
                "--output-dir", str(self.root), "--run-id", "teste"]
        digests = {name: "digest-" + name for name in config.DEFAULT_MODELS + [config.DEFAULT_JUDGE]}

        def reply(model, messages, **kwargs):
            return '{"veredito": "CORRETO"}' if model == config.DEFAULT_JUDGE else "Lisboa"

        def fake_bert(frame, model, batch_size, num_layers):
            return frame.assign(BERTScore_F1=0.8)

        with patch("qa_pipeline.llm.model_digests", return_value=digests), \
             patch("qa_pipeline.llm.chat", side_effect=reply), \
             patch("qa_pipeline.scoring.add_bertscore", side_effect=fake_bert), \
             patch("qa_pipeline.analysis.plot_bars"):
            cli.main(args)
        out = self.root / "meta2" / "teste" / "analises" / "both_todas_completa"
        manifest_before = (self.root / "meta2" / "teste" / "manifest.json").read_text(encoding="utf-8")
        for name in ("tabela_estatistica.csv", "tabela_por_tema_ic.csv"):
            (out / name).unlink()
        cli.main(["analyze", "--train", str(self.train), "--test", str(self.test),
                  "--output-dir", str(self.root), "--run-id", "teste", "--stats"])
        for name in ("tabela_estatistica.csv", "tabela_por_tema_ic.csv"):
            self.assertTrue((out / name).exists())
        self.assertEqual((self.root / "meta2" / "teste" / "manifest.json").read_text(encoding="utf-8"),
                         manifest_before)
        with self.assertRaisesRegex(ValueError, "Análise inexistente"):
            cli.main(["analyze", "--train", str(self.train), "--test", str(self.test),
                      "--output-dir", str(self.root), "--run-id", "sem-analise", "--stats"])

    def test_validate_judge_and_agreement_leave_caches_untouched(self):
        df = data.load_data(self.train, self.test, "both")
        run_dir = self.root / "meta2" / "teste"
        run_dir.mkdir(parents=True)
        for i, row in df.iterrows():
            cache.append_cache(cache.preds_path("modelo", run_dir), {
                "id": row["id"], "pred": "Lisboa", "question_sha256": cache.digest_text(row["Pergunta"])})
            cache.append_cache(cache.judge_path("modelo", "juiz", run_dir), {
                "id": row["id"], "pred_sha256": cache.digest_text("Lisboa"),
                "raw": "{}", "judge": 1 - i})
        caches = (cache.preds_path("modelo", run_dir), cache.judge_path("modelo", "juiz", run_dir))
        before = {path: path.read_bytes() for path in caches}
        args = ["--train", str(self.train), "--test", str(self.test), "--output-dir", str(self.root),
                "--run-id", "teste", "--models", "modelo", "--judge-model", "juiz"]

        cli.main(["validate-judge"] + args)
        sample_path = run_dir / "validacao_juiz.csv"
        saved = pd.read_csv(sample_path)
        self.assertEqual(len(saved), 2)
        self.assertEqual(set(saved["juiz"]), {0, 1})
        self.assertTrue(saved["humano"].isna().all())
        self.assertFalse((run_dir / "manifest.json").exists())
        self.assertFalse((run_dir / "validacao_juiz_resumo.csv").exists())
        with self.assertRaisesRegex(ValueError, "em falta"):
            cli.main(["judge-agreement"] + args)

        saved["humano"] = saved["juiz"]
        saved.to_csv(sample_path, index=False)
        cli.main(["judge-agreement"] + args)
        summary = pd.read_csv(run_dir / "validacao_juiz_resumo.csv")
        self.assertEqual(list(summary.columns), validation.SUMMARY_COLUMNS)
        self.assertEqual(int(summary.loc[0, "n"]), 2)
        self.assertAlmostEqual(summary.loc[0, "acordo_bruto"], 1.0)
        self.assertAlmostEqual(summary.loc[0, "kappa"], 1.0)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

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
