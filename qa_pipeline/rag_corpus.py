"""Recolha de documentos da Wikipédia em português e divisão em excertos (chunks)."""
from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from collections import Counter
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


# Títulos de ligações sem prosa útil: anos, séculos, listas e anexos.
LINK_SKIP = re.compile(r"\d{1,4}( a\.C\.)?|séc(ulo|\.) .*|lista de .*|anexo:.*|\d{1,2} de [a-zç]+", re.IGNORECASE)


def fetch_links(session: requests.Session, title: str, pause: float = 0.0) -> list[str]:
    """Títulos dos artigos (espaço principal) para onde a página `title` tem ligações."""
    links: list[str] = []
    cont: dict = {}
    while True:
        data = _wiki_get(session, {"action": "query", "prop": "links", "pllimit": "max",
                                   "plnamespace": 0, "redirects": 1, "titles": title, **cont})
        pages = data.get("query", {}).get("pages", [])
        if pages:
            links.extend(item["title"] for item in pages[0].get("links", []))
        cont = data.get("continue") or {}
        if not cont:
            return links
        time.sleep(pause)


def rank_link_candidates(link_map: dict[str, list[str]], exclude: set[str],
                         min_links: int = 2) -> list[tuple[str, int]]:
    """Ordena os artigos ligados por quantas páginas-semente os referem (desc.) e depois por título.

    Só entram os referidos por pelo menos `min_links` sementes, fora os já presentes e os títulos
    sem prosa (anos, séculos, listas). A ordenação é determinística.
    """
    excluded = {title.casefold() for title in exclude}
    counts: Counter = Counter()
    for links in link_map.values():
        counts.update(set(links))
    ranked = [(title, n) for title, n in counts.items()
              if n >= min_links and title.casefold() not in excluded and not LINK_SKIP.fullmatch(title)]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))


def build_corpus(theme: str, seeds: list[str], out_dir: Path, pause: float | None = None,
                 expand_links: bool = False, max_links: int = 10, min_links: int = 2) -> list[dict]:
    """Descarrega os artigos, guarda corpus.jsonl e fontes.csv e devolve os documentos.

    Com `expand_links`, acrescenta até `max_links` artigos que as páginas-semente referem por ligação
    (os mais referidos, com pelo menos `min_links` sementes a ligar para eles). Esses documentos recebem
    os identificadores seguintes aos das sementes e ficam marcados com origin="ligacao"; a lista fica em
    links_expandidos.txt para poderes revê-la e promovê-la a sementes.
    """
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

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_cache"
    (cache_dir / "links").mkdir(parents=True, exist_ok=True)

    def load_article(title: str) -> dict | None:
        cached = cache_dir / f"{slugify(title)}.json"
        if cached.is_file():  # já descarregado numa execução anterior
            return json.loads(cached.read_text(encoding="utf-8")) or None
        article = fetch_article(session, title)
        cached.write_text(json.dumps(article, ensure_ascii=False), encoding="utf-8")
        time.sleep(pause)
        return article

    def add_doc(article: dict) -> None:
        seen.add(article["title"])
        article["doc_id"] = f"d{len(docs):03d}"
        article["theme"] = theme
        article["retrieved"] = date.today().isoformat()
        docs.append(article)

    docs, seen, missing = [], set(), []
    for title in titles:
        article = load_article(title)
        if article is None:
            missing.append(title)
            continue
        if article["title"] in seen:
            continue
        add_doc(article)
    if missing:
        print("Aviso: artigos não encontrados (corrigir o ficheiro de sementes): " + "; ".join(missing))
    if not docs:
        raise ValueError("Nenhum documento foi recolhido.")

    if expand_links and max_links > 0:
        n_seeds = len(docs)
        link_map: dict[str, list[str]] = {}
        for doc in docs[:n_seeds]:
            cached = cache_dir / "links" / f"{slugify(doc['title'])}.json"
            if cached.is_file():
                link_map[doc["title"]] = json.loads(cached.read_text(encoding="utf-8"))
            else:
                link_map[doc["title"]] = fetch_links(session, doc["title"], pause)
                cached.write_text(json.dumps(link_map[doc["title"]], ensure_ascii=False), encoding="utf-8")
                time.sleep(pause)
        candidates = rank_link_candidates(link_map, set(titles) | seen, min_links)
        added: list[tuple[str, int]] = []
        for title, count in candidates[:max_links * 5]:  # margem para redirecionamentos/inexistentes
            if len(added) >= max_links:
                break
            article = load_article(title)
            if article is None or article["title"] in seen:
                continue
            article["origin"] = "ligacao"
            article["linked_from"] = count
            add_doc(article)
            added.append((article["title"], count))
        with (out_dir / "links_expandidos.txt").open("w", encoding="utf-8") as stream:
            stream.write("# Artigos acrescentados por --expand-links (título <TAB> nº de sementes que ligam para ele).\n"
                         "# Para os fixar, copiar os títulos para seeds.txt.\n")
            for title, count in added:
                stream.write(f"{title}\t{count}\n")
        print(f"Expansão por ligações: +{len(added)} artigos "
              f"(candidatos com >= {min_links} sementes a ligar: {len(candidates)}).")
        for title, count in added:
            print(f"  + {title} ({count} sementes)")

    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as stream:
        for doc in docs:
            stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with (out_dir / "fontes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["doc_id", "titulo", "url", "revisao", "recolhido_em", "caracteres", "origem"])
        for doc in docs:
            writer.writerow([doc["doc_id"], doc["title"], doc["url"], doc["revision"],
                             doc["retrieved"], len(doc["text"]), doc.get("origin", "semente")])
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
