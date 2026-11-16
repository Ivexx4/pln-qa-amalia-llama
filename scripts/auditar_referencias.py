"""Auditoria das referências do conjunto de dados: deteta indícios, não corrige nada."""
import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa_pipeline import config, data, metrics

COLUMNS = ["id", "Tema", "Pergunta", "Resposta", "motivo", "gravidade"]
SEVERITY_ORDER = {"alta": 0, "média": 1, "baixa": 2}
SIMILARITY_THRESHOLD = 0.9
RECENT_YEAR = 2022
SHORT_TOKEN_LIMIT = 3
ALLOWED_SYMBOLS = set("$€£¥%+<>≤≥±×÷°§©®™=~^|")
TEMPORAL_WORDS = re.compile(r"\b(?:atual|atuais|atualmente|mais\s+recentes?)\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(20[2-9]\d)\b")


def similarity(first: str, second: str) -> float:
    """Similaridade de Jaccard entre os tokens normalizados das duas perguntas."""
    left, right = set(metrics.tokens(first)), set(metrics.tokens(second))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def dataset_finding(row, motivo: str, gravidade: str) -> dict:
    return {"id": row.id, "Tema": row.Tema, "Pergunta": row.Pergunta,
            "Resposta": row.Resposta, "motivo": motivo, "gravidade": gravidade}


def results_finding(identifier, tema, pergunta, gold, motivo: str, gravidade: str, lookup: dict) -> dict:
    """Usa o dataset como fonte preferida e o all_results.csv como reserva."""
    known = lookup.get(identifier)
    if known is not None:
        tema, pergunta, gold = known["Tema"], known["Pergunta"], known["Resposta"]
    return {"id": identifier, "Tema": tema, "Pergunta": pergunta,
            "Resposta": gold, "motivo": motivo, "gravidade": gravidade}


def duplicate_findings(df: pd.DataFrame) -> list:
    findings = []
    rows = list(df.itertuples(index=False))
    for position, first in enumerate(rows):
        for second in rows[position + 1:]:
            score = similarity(first.Pergunta, second.Pergunta)
            if score < SIMILARITY_THRESHOLD:
                continue
            exact = score == 1.0
            kind = "duplicada" if exact else "quase duplicada"
            severity = "alta" if exact else "média"
            findings.append(dataset_finding(first, f"pergunta {kind} de {second.id} (similaridade {score:.2f})", severity))
            findings.append(dataset_finding(second, f"pergunta {kind} de {first.id} (similaridade {score:.2f})", severity))
    return findings


def strange_characters(text: str) -> list:
    """Caracteres que não são letras, dígitos, espaços, pontuação ou símbolos comuns."""
    return sorted({c for c in text
                   if not (c.isalnum() or c.isspace() or unicodedata.category(c)[0] == "P"
                           or c in ALLOWED_SYMBOLS)})


def answer_findings(df: pd.DataFrame) -> list:
    findings = []
    for row in df.itertuples(index=False):
        tokens = metrics.tokens(row.Resposta)
        if not tokens:
            findings.append(dataset_finding(row, "resposta sem tokens utilizáveis", "média"))
        elif len(tokens) == 1 and len(tokens[0]) <= SHORT_TOKEN_LIMIT:
            findings.append(dataset_finding(
                row, f"resposta de um único token curto: '{row.Resposta.strip()}'", "baixa"))
        strange = strange_characters(row.Resposta)
        if strange:
            listed = ", ".join(f"'{c}' (U+{ord(c):04X})" for c in strange[:5])
            severity = "alta" if "\ufffd" in strange else "média"
            findings.append(dataset_finding(row, f"caracteres inválidos na resposta: {listed}", severity))
    return findings


def temporal_findings(df: pd.DataFrame) -> list:
    findings = []
    for row in df.itertuples(index=False):
        years = sorted({year for year in YEAR_PATTERN.findall(row.Pergunta) if int(year) >= RECENT_YEAR})
        words = sorted({match.group(0) for match in TEMPORAL_WORDS.finditer(row.Pergunta)})
        if not years and not words:
            continue
        reasons = []
        if years:
            reasons.append("ano recente " + ", ".join(years))
        if words:
            reasons.append("expressão dependente da data: " + ", ".join(f"'{word}'" for word in words))
        findings.append(dataset_finding(row, "pergunta com " + "; ".join(reasons), "média"))
    return findings


def judge_rouge_findings(results: pd.DataFrame, lookup: dict) -> list:
    findings = []
    for row in results.itertuples(index=False):
        if row.Judge != 1 or not row.ROUGE_L < 0.1:
            continue
        findings.append(results_finding(row.id, row.Tema, row.Pergunta, row.Gold,
                                        f"juiz=1 com ROUGE-L={row.ROUGE_L:.4f} ({row.Modelo})", "média", lookup))
    return findings


def models_agree_findings(results: pd.DataFrame, lookup: dict) -> list:
    """Todos os modelos produzem a mesma resposta, que não contém a referência."""
    findings = []
    for identifier, group in results.groupby("id", sort=True):
        predictions = group["Pred"].dropna()
        if len(predictions) < 2:
            continue
        normalized = {tuple(metrics.tokens(str(prediction))) for prediction in predictions}
        if len(normalized) != 1:
            continue
        gold = str(group["Gold"].iloc[0])
        if not metrics.tokens(gold) or metrics.contains(str(predictions.iloc[0]), gold):
            continue
        models = ", ".join(sorted(set(group["Modelo"].dropna())))
        first = group.iloc[0]
        findings.append(results_finding(identifier, first["Tema"], first["Pergunta"], first["Gold"],
                                        f"respostas de {models} coincidem entre si e divergem da referência",
                                        "alta", lookup))
    return findings


def latest_results(root: Path = ROOT / "results"):
    """all_results.csv mais recente das experiências, com recurso ao histórico."""
    candidates = list(root.glob("meta2/*/analises/*/all_results.csv"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    legacy = root / "all_results.csv"
    return legacy if legacy.exists() else None


def audit(df: pd.DataFrame, results=None) -> pd.DataFrame:
    findings = duplicate_findings(df) + answer_findings(df) + temporal_findings(df)
    if results is not None:
        lookup = {row.id: {"Tema": row.Tema, "Pergunta": row.Pergunta, "Resposta": row.Resposta}
                  for row in df[["id", "Tema", "Pergunta", "Resposta"]].itertuples(index=False)}
        findings += judge_rouge_findings(results, lookup) + models_agree_findings(results, lookup)
    table = pd.DataFrame(findings, columns=COLUMNS)
    if table.empty:
        return table
    table = table.drop_duplicates(["id", "motivo", "gravidade"])
    order = table["gravidade"].map(SEVERITY_ORDER).fillna(len(SEVERITY_ORDER))
    return table.assign(_ordem=order).sort_values(["_ordem", "id", "motivo"]).drop(columns="_ordem")


def main(argv=None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Auditoria manual das referências; não altera o dataset.")
    parser.add_argument("--train", type=Path, default=config.TRAIN_PATH)
    parser.add_argument("--test", type=Path, default=config.TEST_PATH)
    parser.add_argument("--split", choices=["both", "train", "test"], default="both")
    parser.add_argument("--results", type=Path, help="all_results.csv a auditar; por omissão, o mais recente")
    parser.add_argument("--output", type=Path, default=Path("auditoria_referencias.csv"))
    args = parser.parse_args(argv)

    df = data.load_data(args.train, args.test, args.split)
    results_path = args.results if args.results is not None else latest_results()
    results = None
    if results_path is not None:
        if not results_path.exists():
            raise ValueError(f"Resultados inexistentes: {results_path}")
        results = pd.read_csv(results_path, dtype={"id": str})
        missing = {"id", "Modelo", "Pred", "Gold", "ROUGE_L", "Judge", "Tema", "Pergunta"} - set(results.columns)
        if missing:
            raise ValueError(f"{results_path} não tem as colunas {', '.join(sorted(missing))}")
        results["ROUGE_L"] = pd.to_numeric(results["ROUGE_L"], errors="coerce")
        results["Judge"] = pd.to_numeric(results["Judge"], errors="coerce")
        print(f"Resultados auditados: {results_path}")
    else:
        print("Aviso: sem all_results.csv; a auditoria cobre apenas o dataset.")

    table = audit(df, results)
    table.to_csv(args.output, index=False, encoding="utf-8-sig")
    counts = table["gravidade"].value_counts().to_dict() if not table.empty else {}
    print(f"Guardada {args.output} ({len(table)} achados; {counts}).")
    return table


if __name__ == "__main__":
    main()
