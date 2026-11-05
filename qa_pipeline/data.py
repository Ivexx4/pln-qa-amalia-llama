"""Carregamento e limpeza dos conjuntos de dados (TSV)."""
import csv
import re

import pandas as pd

from . import config

MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
         "agosto", "setembro", "outubro", "novembro", "dezembro"]


def fix_text(s: str) -> str:
    """Corrige artefactos de exportação do Excel (_x0096_, datas 'YYYY-MM-DD 00:00:00')."""
    s = re.sub(
        r"_x([0-9A-Fa-f]{4})_",
        lambda m: bytes([int(m.group(1), 16)]).decode("cp1252", errors="ignore")
        if int(m.group(1), 16) < 256 else "",
        s,
    )

    def date(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{d} de {MESES[mo - 1]} de {y}"

    return re.sub(r"(\d{4})-(\d{2})-(\d{2}) 00:00:00", date, s).strip()


def load_data(train=config.TRAIN_PATH, test=config.TEST_PATH, split="both") -> pd.DataFrame:
    """Devolve um DataFrame com colunas Autor, Tema, Pergunta, Metodo, Resposta, split, id."""
    frames = []
    for name, path in (("train", train), ("test", test)):
        if split in ("both", name):
            df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                             encoding="utf-8", keep_default_na=False)
            df["split"] = name
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"Método": "Metodo"})
    for col in ("Pergunta", "Resposta", "Tema", "Autor"):
        df[col] = df[col].map(fix_text)
    df["id"] = df["split"] + "_" + df.index.astype(str)
    return df
