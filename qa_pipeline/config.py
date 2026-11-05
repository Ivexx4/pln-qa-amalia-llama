"""Configuração central: caminhos, modelos e parâmetros por omissão."""
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"

DEFAULT_MODELS = ["teex/amalia:q4_K_M", "llama3.1:8b"]
DEFAULT_JUDGE = "qwen2.5:7b"
DEFAULT_BERT_MODEL = "bert-base-multilingual-cased"

TRAIN_PATH = Path("data/dataset_PLN_train.tsv")
TEST_PATH = Path("data/dataset_PLN_test.tsv")
RESULTS_DIR = Path("results")

# Decodificação determinística
SEED = 42
TEMPERATURE = 0
GEN_MAX_TOKENS = 120
JUDGE_MAX_TOKENS = 32
JUDGE_CACHE_VERSION = "json-v1"
REQUEST_TIMEOUT = 600
HTTP_MAX_ATTEMPTS = 3
HTTP_RETRY_BACKOFF_SECONDS = 1
RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}

METRICS = ["EM", "Contains", "ROUGE_L", "BERTScore_F1", "Judge"]
