"""Recolha de documentos da Wikipédia em português e divisão em excertos (chunks)."""
from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from datetime import date
from pathlib import Path

import requests

from . import cache, rag_config


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "tema"


def theme_dir(theme: str, base: Path | None = None) -> Path:
    return (base or rag_config.RAG_DIR) / slugify(theme)


def read_seeds(path: Path) -> list[str]:
    """Uma entrada por linha: título de artigo, ou '?consulta' para pesquisar na Wikipédia.

    Linhas vazias e comentários ('#') são ignorados.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Ficheiro de sementes inexistente: {path}")
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            seeds.append(line)
    if not seeds:
        raise ValueError(f"{path}: sem sementes")
    return seeds


def _wiki_get(session: requests.Session, params: dict, retries: int = rag_config.WIKI_RETRIES) -> dict:
    """GET à API da Wikipédia; perante 429/5xx espera (respeitando 'Retry-After') e repete."""
    full = {**params, "format": "json", "formatversion": 2}
    for attempt in range(retries + 1):
        response = session.get(rag_config.WIKI_API, params=full, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == retries:
                break
            retry_after = response.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.isdigit() else rag_config.WIKI_BACKOFF_SECONDS * 2 ** attempt
            wait = min(wait, 300)
            print(f"  Wikipédia respondeu {response.status_code}; a esperar {wait:.0f}s "
                  f"(tentativa {attempt + 1}/{retries})...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return response.json()


def search_titles(session: requests.Session, query: str, limit: int) -> list[str]:
    data = _wiki_get(session, {"action": "query", "list": "search", "srsearch": query,
                               "srlimit": limit, "srprop": ""})
    return [item["title"] for item in data.get("query", {}).get("search", [])]


def strip_sections(text: str) -> str:
    """Remove as secções finais sem conteúdo (referências, ligações externas, ...)."""
    out, skipping = [], False
    for line in text.splitlines():
        heading = re.fullmatch(r"\s*(=+)\s*(.*?)\s*\1\s*", line)
        if heading:
            skipping = heading.group(2).strip().lower() in rag_config.DROP_SECTIONS
        if not skipping:
            out.append(line)
    return "\n".join(out).strip()


def fetch_article(session: requests.Session, title: str) -> dict | None:
    data = _wiki_get(session, {"action": "query", "prop": "extracts|info", "explaintext": 1,
                               "redirects": 1, "inprop": "url", "titles": title})
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or not pages[0].get("extract"):
        return None
    page = pages[0]
    text = strip_sections(page["extract"])
    if not text:
        return None
    return {"title": page["title"], "url": page.get("fullurl", ""), "revision": page.get("lastrevid"),
            "text": text}


def build_corpus(theme: str, seeds: list[str], out_dir: Path, pause: float | None = None) -> list[dict]:
    """Descarrega os artigos, guarda corpus.jsonl e fontes.csv e devolve os documentos."""
    pause = rag_config.WIKI_PAUSE_SECONDS if pause is None else pause
    session = requests.Session()
    session.headers["User-Agent"] = rag_config.WIKI_USER_AGENT
    titles: list[str] = []
    for seed in seeds:
        found = (search_titles(session, seed[1:].strip(), rag_config.WIKI_SEARCH_RESULTS)
                 if seed.startswith("?") else [seed])
        if not found:
            print(f"Aviso: a consulta '{seed}' não devolveu artigos.")
        titles.extend(found)
        time.sleep(pause)

    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs, seen, missing = [], set(), []
    for title in titles:
        cached = cache_dir / f"{slugify(title)}.json"
        if cached.is_file():  # já descarregado numa execução anterior
            article = json.loads(cached.read_text(encoding="utf-8")) or None
        else:
            article = fetch_article(session, title)
            cached.write_text(json.dumps(article, ensure_ascii=False), encoding="utf-8")
            time.sleep(pause)
        if article is None:
            missing.append(title)
            continue
        if article["title"] in seen:
            continue
        seen.add(article["title"])
        article["doc_id"] = f"d{len(docs):03d}"
        article["theme"] = theme
        article["retrieved"] = date.today().isoformat()
        docs.append(article)
    if missing:
        print("Aviso: artigos não encontrados (corrigir o ficheiro de sementes): " + "; ".join(missing))
    if not docs:
        raise ValueError("Nenhum documento foi recolhido.")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as stream:
        for doc in docs:
            stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with (out_dir / "fontes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["doc_id", "titulo", "url", "revisao", "recolhido_em", "caracteres"])
        for doc in docs:
            writer.writerow([doc["doc_id"], doc["title"], doc["url"], doc["revision"],
                             doc["retrieved"], len(doc["text"])])
    print(f"{len(docs)} documentos guardados em {out_dir}")
    return docs


def load_corpus(out_dir: Path) -> list[dict]:
    path = out_dir / "corpus.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Corpus inexistente em {path}; execute 'rag corpus' primeiro.")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def corpus_digest(out_dir: Path) -> str:
    return cache.digest_file(out_dir / "corpus.jsonl")


def _split_long(paragraph: str, size: int) -> list[str]:
    """Divide um parágrafo demasiado longo por frases, sem cortar palavras."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts, current = [], ""
    for sentence in sentences:
        while len(sentence) > size:  # frase gigante: corta por espaços
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > size:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_document(doc: dict, size: int = rag_config.CHUNK_CHARS,
                   overlap: int = rag_config.CHUNK_OVERLAP_CHARS,
                   min_chars: int = rag_config.MIN_CHUNK_CHARS) -> list[dict]:
    """Excertos por parágrafos, com o título e a secção prefixados e sobreposição entre excertos."""
    if overlap >= size:
        raise ValueError("A sobreposição tem de ser menor do que o tamanho do excerto.")
    units, section = [], ""
    for block in re.split(r"\n\s*\n|\n(?==)|(?<==)\n", doc["text"]):
        block = block.strip()
        if not block:
            continue
        heading = re.fullmatch(r"(=+)\s*(.*?)\s*\1", block)
        if heading:
            section = heading.group(2)
            continue
        for part in (_split_long(block, size) if len(block) > size else [block]):
            units.append((section, part))

    chunks, current, current_section = [], "", ""

    def flush():
        nonlocal current
        text = current.strip()
        if len(text) >= min_chars:
            chunks.append((current_section, text))
        current = current[-overlap:].split(" ", 1)[-1] if overlap and len(current) > overlap else ""

    for section, part in units:
        if current and (section != current_section or len(current) + len(part) + 1 > size):
            flush()
            if section != current_section or len(current) + len(part) + 1 > size:
                current = ""  # sem espaço para a sobreposição: o excerto nunca excede o tamanho máximo
        if not current:
            current_section = section
        current = f"{current} {part}".strip()
    flush()

    result = []
    for number, (section, text) in enumerate(chunks):
        header = doc["title"] + (f" — {section}" if section else "")
        result.append({"chunk_id": f"{doc['doc_id']}_c{number:03d}", "doc_id": doc["doc_id"],
                       "title": doc["title"], "section": section, "url": doc.get("url", ""),
                       "text": f"{header}\n{text}"})
    return result


def chunk_corpus(docs: list[dict], **kwargs) -> list[dict]:
    chunks = [chunk for doc in docs for chunk in chunk_document(doc, **kwargs)]
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Identificadores de excertos duplicados.")
    return chunks