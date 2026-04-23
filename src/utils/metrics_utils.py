"""Helpers for summarizing Ultralytics validation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    """Convert scalar-like values to floats when possible."""
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


def _harmonic_mean(precision: float | None, recall: float | None) -> float | None:
    """Compute F1 from precision and recall when both are available."""
    if precision is None or recall is None:
        return None
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def extract_validation_metrics(metrics: Any) -> dict[str, float | None]:
    """Extract the common object-detection metrics from Ultralytics results."""
    box = getattr(metrics, "box", None)

    precision = _safe_float(getattr(box, "mp", None))
    recall = _safe_float(getattr(box, "mr", None))
    map50 = _safe_float(getattr(box, "map50", None))
    map50_95 = _safe_float(getattr(box, "map", None))

    return {
        "precision": precision,
        "recall": recall,
        "f1": _harmonic_mean(precision, recall),
        "map50": map50,
        "map50_95": map50_95,
    }


def write_metrics_json(metrics: dict[str, float | None], output_path: Path) -> Path:
    """Write metrics as formatted JSON and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return output_path


def format_metrics(metrics: dict[str, float | None]) -> str:
    """Format metrics for human-readable logging."""
    parts: list[str] = []
    for key, value in metrics.items():
        if value is None:
            parts.append(f"{key}=n/a")
        else:
            parts.append(f"{key}={value:.4f}")
    return ", ".join(parts)
