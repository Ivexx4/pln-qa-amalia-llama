"""Project paths and default model configuration."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESPONSES_DIR = RESULTS_DIR / "responses"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

DEFAULT_DATASET = DATA_DIR / "dataset_PLN_test.tsv"
DEFAULT_RESPONSES = RESPONSES_DIR / "responses.csv"
DEFAULT_METRICS = METRICS_DIR / "metrics.csv"
DEFAULT_AMALIA_MODEL = "teex/amalia:q4_K_M"
DEFAULT_LLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_BERTSCORE_DEVICE = "auto"
DEFAULT_BERTSCORE_BATCH_SIZE = 8
OLLAMA_TIMEOUT_SECONDS = 180
