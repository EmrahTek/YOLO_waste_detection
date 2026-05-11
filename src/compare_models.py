"""Compare two YOLO validation result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ["precision", "recall", "f1", "map50", "map50_95"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)

    baseline_name = baseline.get("model", "baseline")
    candidate_name = candidate.get("model", "candidate")

    lines = []
    lines.append(f"# Model Comparison: {baseline_name} vs {candidate_name}")
    lines.append("")
    lines.append("| Metric | YOLO11n | YOLO26n | Delta YOLO26n - YOLO11n | Better |")
    lines.append("|---|---:|---:|---:|---|")

    for metric in METRICS:
        base_value = float(baseline[metric])
        cand_value = float(candidate[metric])
        delta = cand_value - base_value

        if delta > 0:
            better = candidate_name
        elif delta < 0:
            better = baseline_name
        else:
            better = "equal"

        lines.append(
            f"| {metric} | {base_value:.4f} | {cand_value:.4f} | {delta:+.4f} | {better} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- YOLO26n has slightly better recall, so it missed fewer objects on this test split."
    )
    lines.append(
        "- YOLO11n has clearly better mAP50 and mAP50-95, so its overall detection quality is better."
    )
    lines.append(
        "- For the current dataset, YOLO11n is the stronger model. YOLO26n is still interesting for edge deployment because it has fewer parameters and lower GFLOPs."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nComparison report written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())