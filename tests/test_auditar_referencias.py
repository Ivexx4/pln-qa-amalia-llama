import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qa_pipeline import data
from scripts import auditar_referencias as audit

HEADER = "Autor\tTema\tPergunta\tMétodo\tResposta\n"
DEFAULT_TEST = ("Autor 9", "Arte", "Quem pintou a Mona Lisa?", "HUM", "Leonardo da Vinci")


class ReferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "train.tsv"
        self.test = self.root / "test.tsv"

    def write(self, path, rows):
        path.write_text(HEADER + "".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")

    def dataset(self, rows, test_rows=None):
        self.write(self.train, rows)
        self.write(self.test, test_rows if test_rows is not None else [DEFAULT_TEST])
        return data.load_data(self.train, self.test, "both")

    def results(self, rows):
        return pd.DataFrame(rows, columns=["id", "Tema", "Pergunta", "Gold", "Modelo", "Pred", "ROUGE_L", "Judge"])

    def test_similarity_threshold_is_inclusive(self):
        self.assertAlmostEqual(audit.similarity("a b c d e f g h i", "a b c d e f g h i j"), 0.9)
        self.assertLess(audit.similarity("a b c d e f g h", "a b c d e f g h i j"), 0.9)
        self.assertEqual(audit.similarity("!!!", "?"), 0.0)

    def test_duplicate_and_near_duplicate_questions(self):
        df = self.dataset([
            ("A", "Geografia", "Qual é a capital de Portugal?", "HUM", "Lisboa"),
            ("B", "Geografia", "Qual é a capital de Portugal", "HUM", "Lisboa"),
            ("C", "Testes", "a b c d e f g h i", "HUM", "Mercúrio"),
            ("D", "Testes", "a b c d e f g h i j", "HUM", "Mercúrio"),
            ("E", "Literatura", "Quem escreveu Os Lusíadas?", "HUM", "Luís de Camões"),
        ])
        found = audit.audit(df)
        claims = found[found["motivo"].str.contains("duplicada")]
        self.assertEqual(len(claims), 4)
        self.assertEqual(claims[claims["motivo"].str.contains("quase")]["gravidade"].tolist(), ["média", "média"])
        exact = claims[claims["motivo"].str.contains("pergunta duplicada")]
        self.assertEqual(sorted(exact["id"]), ["train_0", "train_1"])
        self.assertTrue(exact["gravidade"].eq("alta").all())
        self.assertIn("similaridade 0.90", claims[claims["id"] == "train_2"]["motivo"].iloc[0])
        self.assertFalse(claims["id"].eq("train_4").any())

    def test_short_answers_and_strange_characters(self):
        df = self.dataset([
            ("A", "Saúde", "O tratamento é eficaz?", "HUM", "Sim"),
            ("B", "Cultura", "Onde fica o museu?", "HUM", "Lisboa 😀"),
            ("C", "Geografia", "Qual é a capital?", "HUM", "Lisboa"),
            ("D", "Matemática", "Quanto é dois mais dois?", "HUM", "!!!"),
        ])
        found = audit.audit(df)
        short = found[found["motivo"].str.contains("único token curto")]
        self.assertEqual(short["id"].tolist(), ["train_0"])
        self.assertEqual(short["gravidade"].tolist(), ["baixa"])
        strange = found[found["motivo"].str.contains("caracteres inválidos")]
        self.assertEqual(strange["id"].tolist(), ["train_1"])
        self.assertIn("U+1F600", strange["motivo"].iloc[0])
        self.assertEqual(strange["gravidade"].tolist(), ["média"])
        empty = found[found["motivo"].str.contains("sem tokens")]
        self.assertEqual(empty["id"].tolist(), ["train_3"])
        self.assertFalse(found["id"].eq("train_2").any())

    def test_recent_years_and_date_dependent_words(self):
        df = self.dataset([
            ("A", "Desporto", "Quem venceu o campeonato em 2024?", "LLM", "Benfica"),
            ("B", "Política", "Qual é a capital atual de França?", "LLM", "Paris"),
            ("C", "Política", "Quem é atualmente o primeiro-ministro?", "LLM", "Luís Montenegro"),
            ("D", "História", "Em que ano começou a guerra colonial?", "HUM", "1961"),
            ("E", "Ciência", "Qual é o planeta mais próximo do Sol?", "HUM", "Mercúrio"),
        ])
        found = audit.audit(df)
        claims = found[found["motivo"].str.contains("dependente da data|ano recente")]
        self.assertEqual(sorted(claims["id"]), ["train_0", "train_1", "train_2"])
        self.assertTrue(claims["gravidade"].eq("média").all())
        self.assertIn("ano recente 2024", claims[claims["id"] == "train_0"]["motivo"].iloc[0])
        self.assertIn("'atual'", claims[claims["id"] == "train_1"]["motivo"].iloc[0])
        self.assertIn("'atualmente'", claims[claims["id"] == "train_2"]["motivo"].iloc[0])

    def test_judge_and_rouge_disagreement_and_models_against_gold(self):
        df = self.dataset([
            ("A", "Geografia", "Qual é a capital de França?", "HUM", "Paris"),
            ("B", "Geografia", "Qual é a capital de Itália?", "HUM", "Roma"),
            ("C", "História", "Quem descobriu o Brasil?", "HUM", "Pedro Álvares Cabral"),
            ("D", "Vinhos", "Qual é a casta mais emblemática do Douro?", "HUM", "Touriga Nacional"),
        ])
        results = self.results([
            ("train_0", "Geografia", "Qual é a capital de França?", "Paris", "modelo-a", "Paris", 0.9, 1),
            ("train_0", "Geografia", "Qual é a capital de França?", "Paris", "modelo-b", "Paris", 0.9, 1),
            ("train_1", "Geografia", "Qual é a capital de Itália?", "Roma", "modelo-a", "Roma é a capital.", 0.05, 1),
            ("train_1", "Geografia", "Qual é a capital de Itália?", "Roma", "modelo-b", "Milão", 0.5, 0),
            ("train_2", "História", "Quem descobriu o Brasil?", "Pedro Álvares Cabral", "modelo-a", "Cabral", 0.4, 0),
            ("train_2", "História", "Quem descobriu o Brasil?", "Pedro Álvares Cabral", "modelo-b", "Cabral", 0.4, 0),
            ("train_3", "Vinhos", "Qual é a casta mais emblemática do Douro?", "Touriga Nacional",
             "modelo-a", "A casta é a Touriga Nacional.", 0.4, 1),
            ("train_3", "Vinhos", "Qual é a casta mais emblemática do Douro?", "Touriga Nacional",
             "modelo-b", "A casta é a Touriga Nacional.", 0.4, 1),
        ])
        found = audit.audit(df, results)
        judge = found[found["motivo"].str.contains("juiz=1")]
        self.assertEqual(len(judge), 1)
        self.assertIn("ROUGE-L=0.0500", judge["motivo"].iloc[0])
        self.assertIn("modelo-a", judge["motivo"].iloc[0])
        agree = found[found["motivo"].str.contains("divergem da referência")]
        self.assertEqual(agree["id"].tolist(), ["train_2"])
        self.assertEqual(agree["gravidade"].tolist(), ["alta"])
        self.assertIn("modelo-a, modelo-b", agree["motivo"].iloc[0])
        self.assertFalse(found["id"].isin(["train_0", "train_3"]).any())

    def test_main_writes_csv_and_does_not_touch_dataset(self):
        df_rows = [
            ("A", "Desporto", "Quem venceu o campeonato em 2024?", "LLM", "Benfica"),
            ("B", "Saúde", "O tratamento é eficaz?", "HUM", "Sim"),
            ("C", "Geografia", "Qual é a capital?", "HUM", "Lisboa"),
        ]
        self.dataset(df_rows)
        before = (self.train.read_bytes(), self.test.read_bytes())
        results_path = self.root / "all_results.csv"
        self.results([
            ("train_2", "Geografia", "Qual é a capital?", "Lisboa", "modelo-a", "Lisboa", 0.9, 1),
            ("train_2", "Geografia", "Qual é a capital?", "Lisboa", "modelo-b", "Lisboa", 0.9, 1),
        ]).to_csv(results_path, index=False)
        output = self.root / "auditoria_referencias.csv"

        table = audit.main(["--train", str(self.train), "--test", str(self.test),
                            "--results", str(results_path), "--output", str(output)])

        saved = pd.read_csv(output, encoding="utf-8-sig")
        self.assertEqual(list(saved.columns), audit.COLUMNS)
        self.assertEqual(len(saved), len(table))
        self.assertEqual(saved["gravidade"].tolist(),
                         sorted(saved["gravidade"], key=lambda value: audit.SEVERITY_ORDER[value]))
        self.assertEqual(set(saved["id"]), {"train_0", "train_1"})
        self.assertEqual(before, (self.train.read_bytes(), self.test.read_bytes()))

    def test_main_rejects_missing_results(self):
        self.dataset([("A", "Geografia", "Qual é a capital?", "HUM", "Lisboa")])
        with self.assertRaisesRegex(ValueError, "Resultados inexistentes"):
            audit.main(["--train", str(self.train), "--test", str(self.test),
                        "--results", str(self.root / "nao-existe.csv"),
                        "--output", str(self.root / "saida.csv")])


if __name__ == "__main__":
    unittest.main()
