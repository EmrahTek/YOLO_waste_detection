"""Compare multiple YOLO validation result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ["precision", "recall", "f1", "map50", "map50_95"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if "model" not in data:
        data["model"] = path.stem.replace("_test", "")

    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare multiple YOLO models.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = [load_json(path) for path in args.inputs]

    lines = []
    lines.append("# Model Comparison")
    lines.append("")
    lines.append("| Metric | " + " | ".join(model["model"] for model in models) + " | Best |")
    lines.append("|---" + "|---:" * len(models) + "|---|")

    for metric in METRICS:
        values = [float(model[metric]) for model in models]
        best_index = max(range(len(values)), key=values.__getitem__)
        best_model = models[best_index]["model"]

        row = [metric]
        row.extend(f"{value:.4f}" for value in values)
        row.append(best_model)

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Precision zeigt, wie viele erkannte Objekte wirklich korrekt sind."
    )
    lines.append(
        "- Recall zeigt, wie viele echte Objekte vom Modell gefunden wurden."
    )
    lines.append(
        "- Der F1-Score kombiniert Precision und Recall."
    )
    lines.append(
        "- mAP50 bewertet die Detektionsqualität bei IoU = 0.50."
    )
    lines.append(
        "- mAP50-95 ist strenger und bewertet die Modellqualität über mehrere IoU-Grenzen."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nComparison report written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())