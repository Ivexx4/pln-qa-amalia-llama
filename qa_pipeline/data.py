"""Carregamento, validação e limpeza dos conjuntos de dados TSV."""
import csv
import re
import warnings
from datetime import date

import pandas as pd

from . import config

MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
         "agosto", "setembro", "outubro", "novembro", "dezembro"]


def fix_text(s: str) -> str:
    """Corrige artefactos conhecidos sem eliminar sequências desconhecidas."""
    def escape(match):
        value = int(match.group(1), 16)
        if 128 <= value <= 255:
            try:
                return bytes([value]).decode("cp1252")
            except UnicodeDecodeError:
                pass
        return match.group(0)

    def format_date(match):
        year, month, day = map(int, match.groups())
        try:
            date(year, month, day)
        except ValueError:
            return match.group(0)
        return f"{day} de {MESES[month - 1]} de {year}"

    s = re.sub(r"_x([0-9A-Fa-f]{4})_", escape, s)
    return re.sub(r"(\d{4})-(\d{2})-(\d{2}) 00:00:00", format_date, s).strip()


def load_data(train=config.TRAIN_PATH, test=config.TEST_PATH, split="both") -> pd.DataFrame:
    """Carrega os dados e atribui identificadores estáveis dentro de cada divisão."""
    if split not in {"both", "train", "test"}:
        raise ValueError(f"Divisão desconhecida: {split}")
    frames = []
    for name, path in (("train", train), ("test", test)):
        if split not in ("both", name):
            continue
        df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                         encoding="utf-8-sig", keep_default_na=False)
        df = df.rename(columns={"Método": "Metodo"})
        required = {"Tema", "Pergunta", "Metodo", "Resposta"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: faltam colunas obrigatórias: {', '.join(sorted(missing))}")
        if df.empty:
            raise ValueError(f"{path}: o conjunto de dados está vazio")
        if "Autor" not in df:
            df["Autor"] = ""
        for col in ("Pergunta", "Resposta", "Tema", "Autor", "Metodo"):
            df[col] = df[col].map(fix_text)
        for col in ("Tema", "Pergunta", "Metodo", "Resposta"):
            invalid = df[col].str.strip().eq("")
            if invalid.any():
                raise ValueError(f"{path}: {col} está vazio nas linhas {list(df.index[invalid] + 2)[:5]}")
        invalid_method = ~df["Metodo"].isin(["HUM", "LLM"])
        if invalid_method.any():
            values = df.loc[invalid_method, "Metodo"].unique().tolist()
            raise ValueError(f"{path}: métodos inválidos: {values}; esperados HUM ou LLM")
        df["split"] = name
        df["id"] = name + "_" + df.index.astype(str)
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    duplicated = result["Pergunta"].str.casefold().duplicated(keep=False)
    if duplicated.any():
        warnings.warn(f"Existem {int(duplicated.sum())} entradas com perguntas repetidas.", stacklevel=2)
    return result