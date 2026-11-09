import unittest
from unittest.mock import Mock, patch

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
    @patch("bert_score.score")
    @patch("torch.cuda.is_available", return_value=True)
    def test_uses_cuda_when_available(self, is_available, bert_score):
        scores = Mock()
        scores.detach.return_value.cpu.return_value.numpy.return_value = [0.8]
        bert_score.return_value = (None, None, scores)
        results = pd.DataFrame({"Pred": ["Lisboa"], "Gold": ["Lisboa"]})

        scored = scoring.add_bertscore(results, "bert-base-multilingual-cased")

        self.assertEqual(scored["BERTScore_F1"].tolist(), [0.8])
        self.assertEqual(bert_score.call_args.kwargs["device"], "cuda")

    @patch("bert_score.score")
    @patch("torch.cuda.is_available", return_value=False)
    def test_uses_cpu_when_cuda_is_unavailable(self, is_available, bert_score):
        scores = Mock()
        scores.detach.return_value.cpu.return_value.numpy.return_value = [0.8]
        bert_score.return_value = (None, None, scores)
        results = pd.DataFrame({"Pred": ["Lisboa"], "Gold": ["Lisboa"]})

        scoring.add_bertscore(results, "bert-base-multilingual-cased")

        self.assertEqual(bert_score.call_args.kwargs["device"], "cpu")


class JudgeCacheTests(unittest.TestCase):
    def test_cache_filename_versions_judge_prompt_results(self):
        path = cache.judge_path("model", "judge")
        self.assertIn(cache.config.JUDGE_CACHE_VERSION, path.name)


if __name__ == "__main__":
    unittest.main()
