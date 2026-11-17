"""Configuração do RAG com base de dados vetorial (Meta 3)."""
from . import config

RAG_DIR = config.DATA_DIR / "rag"
META3_RESULTS_DIR = config.RESULTS_DIR / "meta3"

WIKI_API = "https://pt.wikipedia.org/w/api.php"
WIKI_USER_AGENT = "PLN-UC-projeto/1.0 (trabalho universitario; contacto via repositorio Ivexx4/pln-qa-amalia-llama)"
WIKI_SEARCH_RESULTS = 3  # artigos por linha "?consulta" do ficheiro de sementes
WIKI_PAUSE_SECONDS = 0.5

# Secções finais da Wikipédia sem conteúdo informativo.
DROP_SECTIONS = {"referências", "ligações externas", "ver também", "notas", "bibliografia",
                 "notas e referências", "leitura adicional", "fontes"}

DEFAULT_EMBED_MODEL = "bge-m3:latest"
EMBED_BATCH_SIZE = 16

CHUNK_CHARS = 900
CHUNK_OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 80

DEFAULT_TOP_K = 4
DEFAULT_STORE = "chroma"  # "chroma" ou "memory"
