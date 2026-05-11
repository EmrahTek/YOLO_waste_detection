"""Create a matplotlib comparison plot for YOLO model metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["precision", "recall", "f1", "map50", "map50_95"]
METRIC_LABELS = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "model" not in data:
        data["model"] = path.stem.replace("_test", "")

    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot YOLO model comparison.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/compare/yolo_model_comparison_metrics.png"),
    )
    args = parser.parse_args()

    models = [load_json(path) for path in args.inputs]

    x = np.arange(len(METRICS))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 8))

    for index, model in enumerate(models):
        values = [float(model[metric]) for metric in METRICS]
        offset = (index - (len(models) - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model["model"],
        )

        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    explanations = (
        "Precision: Anteil korrekter positiver Vorhersagen.\n"
        "Recall: Anteil der tatsächlich gefundenen Objekte.\n"
        "F1-Score: Gleichgewicht zwischen Precision und Recall.\n"
        "mAP50: Erkennungsqualität bei IoU = 0.50.\n"
        "mAP50-95: Strengere Bewertung über mehrere IoU-Grenzen."
    )

    ax.set_title("YOLO Model Comparison - Test Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.text(
        0.08,
        0.02,
        explanations,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.22, 1, 1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.show()

    print(f"Plot saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())