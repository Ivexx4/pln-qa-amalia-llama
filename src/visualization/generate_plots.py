"""Generate global, per-theme and per-source metric charts."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METRICS = ["exact_match", "rougeL", "bertscore"]
METRIC_LABELS = ["Exact Match", "ROUGE-L", "BERTScore"]
MODEL_ORDER = ["AMALIA", "Llama 3.1 8B"]


def _plot_grouped(
    metrics: pd.DataFrame,
    group: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(12, 12))
    for axis, metric, label in zip(axes, METRICS, METRIC_LABELS):
        pivot = metrics.pivot(index=group, columns="modelo", values=metric)
        pivot = pivot.reindex(columns=MODEL_ORDER)
        pivot.plot(kind="bar", ax=axis, rot=35, ylim=(0, 1))
        axis.set_title(label)
        axis.set_ylabel("Média")
        axis.set_xlabel("")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Modelo")
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_plots(metrics: pd.DataFrame, output_dir: str | Path) -> None:
    """Save three PNG charts comparing models overall and by dataset group."""
    required = {"tema", "metodo", "modelo", *METRICS}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Colunas de métricas em falta: {', '.join(sorted(missing))}.")
    if metrics.empty:
        raise ValueError("Não existem métricas para criar gráficos.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    overall = metrics.groupby("modelo")[METRICS].mean().reindex(MODEL_ORDER)
    axis = overall.plot(kind="bar", figsize=(9, 5), ylim=(0, 1), rot=0)
    axis.set_title("Desempenho global por modelo")
    axis.set_ylabel("Média da métrica")
    axis.set_xlabel("")
    axis.legend(METRIC_LABELS, loc="best")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(destination / "performance_global.png", dpi=180)
    plt.close()

    theme = metrics.groupby(["tema", "modelo"], as_index=False)[METRICS].mean()
    _plot_grouped(
        theme,
        "tema",
        "Desempenho por tema",
        destination / "performance_tema.png",
    )
    method = metrics.groupby(["metodo", "modelo"], as_index=False)[METRICS].mean()
    _plot_grouped(
        method,
        "metodo",
        "Desempenho por origem da pergunta (HUM vs LLM)",
        destination / "performance_metodo.png",
    )
