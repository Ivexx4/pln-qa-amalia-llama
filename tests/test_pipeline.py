"""Tests for dataset loading, evaluation, analysis and plotting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.analysis.method_analysis import analyze_by_method
from src.analysis.overall_analysis import analyze_overall
from src.analysis.theme_analysis import analyze_by_theme
from src.dataset.loader import load_dataset
from src.evaluation.evaluator import evaluate_responses
from src.inference.predictor import generate_responses
from src.visualization.generate_plots import generate_plots


ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_loads_meta1_tsv_and_ignores_author(self) -> None:
        dataset = load_dataset(ROOT / "data" / "dataset_PLN_test.tsv")
        self.assertEqual(
            list(dataset.columns), ["tema", "pergunta", "metodo", "resposta"]
        )
        self.assertEqual(len(dataset), 91)
        self.assertEqual(set(dataset["metodo"]), {"HUM", "LLM"})

    def test_loads_csv_and_preserves_reference_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            path.write_text(
                "tema,pergunta,metodo,resposta\n"
                "Cinema,Quem realizou Titanic?,HUM, James Cameron \n",
                encoding="utf-8",
            )
            dataset = load_dataset(path)
        self.assertEqual(dataset.iloc[0]["metodo"], "HUM")
        self.assertEqual(dataset.iloc[0]["resposta"], " James Cameron ")


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.responses = pd.DataFrame(
            [
                {
                    "tema": "Cinema",
                    "metodo": "HUM",
                    "pergunta": "Quem realizou Titanic?",
                    "resposta_correta": "James Cameron",
                    "resposta_amalia": "James Cameron",
                    "resposta_llama": " James Cameron ",
                }
            ]
        )

    @patch("src.evaluation.evaluator.bertscore_f1", return_value=[0.75, 0.5])
    def test_exact_match_is_literal_and_bertscore_is_called_once(self, score) -> None:
        metrics = evaluate_responses(self.responses, device="cpu")
        self.assertEqual(metrics["exact_match"].tolist(), [1, 0])
        self.assertEqual(metrics["bertscore"].tolist(), [0.75, 0.5])
        score.assert_called_once()

    @patch("src.evaluation.evaluator.bertscore_f1", return_value=[0.75, 0.5])
    def test_creates_analysis_tables_and_charts(self, _score) -> None:
        metrics = evaluate_responses(self.responses, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            analyze_overall(metrics, output / "summary_global.csv")
            analyze_by_theme(metrics, output / "summary_tema.csv")
            analyze_by_method(metrics, output / "summary_metodo.csv")
            generate_plots(metrics, output)

            for name in (
                "summary_global.csv",
                "summary_tema.csv",
                "summary_metodo.csv",
                "performance_global.png",
                "performance_tema.png",
                "performance_metodo.png",
            ):
                self.assertTrue((output / name).is_file(), name)


class InferenceTests(unittest.TestCase):
    @patch("src.inference.predictor.OllamaClient")
    def test_generates_answers_and_saves_checkpoint(self, client_class) -> None:
        client_class.return_value.generate.side_effect = [
            "Resposta AMALIA",
            "Resposta Llama",
        ]
        questions = pd.DataFrame(
            [
                {
                    "tema": "Cinema",
                    "pergunta": "Quem realizou Titanic?",
                    "metodo": "HUM",
                    "resposta": "James Cameron",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.csv"
            responses = generate_responses(
                questions, path, "amalia", "llama"
            )
            saved = pd.read_csv(path, encoding="utf-8-sig")

        self.assertEqual(responses.loc[0, "resposta_amalia"], "Resposta AMALIA")
        self.assertEqual(responses.loc[0, "resposta_llama"], "Resposta Llama")
        self.assertEqual(saved.loc[0, "resposta_llama"], "Resposta Llama")
        self.assertEqual(client_class.return_value.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
