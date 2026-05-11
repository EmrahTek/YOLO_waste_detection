"""Create a matplotlib comparison plot for YOLO11n and YOLO26n metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("reports/compare")
OUTPUT_PATH = OUTPUT_DIR / "yolo11n_vs_yolo26n_metrics.png"


def main() -> int:
    metrics = ["Precision", "Recall", "F1", "mAP50", "mAP50-95"]

    yolo11n = np.array([0.8080, 0.7292, 0.7666, 0.8516, 0.7113])
    yolo26n = np.array([0.7672, 0.7617, 0.7645, 0.7442, 0.6286])

    explanations = (
        "Precision: Anteil korrekter positiver Vorhersagen.\n"
        "Recall: Anteil der tatsächlich gefundenen Objekte.\n"
        "F1-Score: Gleichgewicht zwischen Precision und Recall.\n"
        "mAP50: Erkennungsqualität bei IoU = 0.50.\n"
        "mAP50-95: Strengere Bewertung über mehrere IoU-Grenzen."
    )

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 8))

    bars_11 = ax.bar(x - width / 2, yolo11n, width, label="YOLO11n")
    bars_26 = ax.bar(x + width / 2, yolo26n, width, label="YOLO26n")

    ax.set_title("YOLO11n vs YOLO26n - Test Metrics Müllabfalltrennung")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.bar_label(bars_11, fmt="%.3f", padding=3)
    ax.bar_label(bars_26, fmt="%.3f", padding=3)

    fig.text(
        0.1,
        0.04,
        explanations,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.22, 1, 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.show()

    print(f"Plot saved to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())