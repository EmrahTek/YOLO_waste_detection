"""Validate a trained Ultralytics YOLO model on val or test data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src import config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.metrics_utils import extract_validation_metrics, format_metrics, write_metrics_json
from src.utils.paths import resolve_model_reference, resolve_project_path


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_model = config.DEFAULT_TRAIN_PROJECT / config.DEFAULT_TRAIN_NAME / "weights" / "best.pt"
    parser = argparse.ArgumentParser(description="Validate a trained YOLO model.")
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--data", type=Path, default=config.DEFAULT_DATASET_YAML)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", type=Path, default=config.DEFAULT_VAL_PROJECT)
    parser.add_argument("--name", default="validation")
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save Ultralytics plots such as confusion matrix and PR curves.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for a compact JSON metrics summary.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_yolo_class() -> Any:
    """Import Ultralytics lazily."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed.") from exc
    return YOLO


def _validate_paths(args: argparse.Namespace) -> tuple[str, Path]:
    """Validate required local paths and return resolved references."""
    data_path = resolve_project_path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model_reference = resolve_model_reference(args.model)
    model_path = Path(model_reference)
    if len(model_path.parts) > 1 and not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model_reference, data_path


def main() -> int:
    """Run YOLO validation and write a compact metrics summary."""
    args = parse_args()
    setup_logging(args.log_level)

    model_reference, data_path = _validate_paths(args)
    summary_json = (
        resolve_project_path(args.summary_json)
        if args.summary_json
        else config.REPORTS_DIR / f"validation_metrics_{args.split}.json"
    )

    YOLO = _load_yolo_class()
    model = YOLO(model_reference)
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        plots=args.plots,
        project=str(resolve_project_path(args.project)),
        name=args.name,
    )

    summary = extract_validation_metrics(metrics)
    output_path = write_metrics_json(summary, summary_json)

    LOGGER.info("Validation metrics: %s", format_metrics(summary))
    LOGGER.info("Metrics summary written to: %s", output_path)
    LOGGER.info("Ultralytics output directory: %s", getattr(metrics, "save_dir", "unknown"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
