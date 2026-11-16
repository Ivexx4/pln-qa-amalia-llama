"""Cache JSONL e manifesto de proveniência das experiências."""
import hashlib
import json
import re
from pathlib import Path

from . import config


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _model_name(name: str) -> str:
    return f"{safe_name(name)}_{digest_text(name)[:12]}"


def preds_path(model: str, run_dir: Path | None = None) -> Path:
    return (run_dir or config.RESULTS_DIR) / f"preds_{_model_name(model)}.jsonl"


def judge_path(model: str, judge_model: str, run_dir: Path | None = None) -> Path:
    return (run_dir or config.RESULTS_DIR) / (
        f"judge_{_model_name(model)}__{_model_name(judge_model)}"
        f"__{config.JUDGE_CACHE_VERSION}.jsonl"
    )


def ensure_manifest(run_dir: Path, manifest: dict) -> None:
    """Recusa misturar experiências incompatíveis na mesma pasta."""
    path = run_dir / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            differing = sorted(key for key in previous.keys() | manifest.keys()
                               if previous.get(key) != manifest.get(key))
            raise ValueError(
                "A execução existente tem uma configuração diferente "
                f"({', '.join(differing)}). Escolha outro --run-id."
            )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_cache(path: Path) -> dict:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                key = record["id"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Cache inválida em {path}, linha {number}.") from error
            if key in records and records[key] != record:
                raise ValueError(f"Registos contraditórios para {key} em {path}.")
            records[key] = record
    return records


def append_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate_prediction(record: dict, question: str, identifier: str) -> None:
    if (not isinstance(record.get("pred"), str)
            or record.get("question_sha256") != digest_text(question)):
        raise ValueError(f"Previsão incompatível ou incompleta para {identifier}.")


def validate_judgement(record: dict, prediction: str, identifier: str) -> None:
    if record.get("pred_sha256") != digest_text(prediction):
        raise ValueError(f"Veredito incompatível com a previsão de {identifier}.")