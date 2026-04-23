"""Run local YOLO predictions on images, folders, video files, or webcam."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src import config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.paths import resolve_model_reference, resolve_project_path


LOGGER = get_logger(__name__)
DEFAULT_CONFIDENCE = 0.5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_model = config.DEFAULT_TRAIN_PROJECT / config.DEFAULT_TRAIN_NAME / "weights" / "best.pt"
    parser = argparse.ArgumentParser(description="Run YOLO predictions.")
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, folder path, video path, or webcam index such as 0.",
    )
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Confidence threshold. Current desktop testing works best around 0.5.",
    )
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    parser.add_argument("--project", type=Path, default=config.DEFAULT_PREDICT_PROJECT)
    parser.add_argument("--name", default="predictions")
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save annotated predictions. Use --no-save for webcam preview-only tests.",
    )
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display predictions in a window when supported. Press q to quit webcam/video preview.",
    )
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the output directory name.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_yolo_class() -> Any:
    """Import Ultralytics lazily."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed.") from exc
    return YOLO


def _resolve_source(source: str) -> str | int:
    """Resolve local source paths while preserving webcam indexes."""
    if source.isdigit():
        return int(source)
    source_path = resolve_project_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Prediction source not found: {source_path}")
    return str(source_path)


def _is_webcam_source(source: str | int) -> bool:
    """Return True when the source is a local camera index."""
    return isinstance(source, int)


def _validate_thresholds(confidence: float, iou: float) -> None:
    """Validate detection thresholds before running inference."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence threshold must be between 0 and 1, got {confidence}.")
    if not 0.0 <= iou <= 1.0:
        raise ValueError(f"IoU threshold must be between 0 and 1, got {iou}.")


def main() -> int:
    """Run prediction and save annotated outputs."""
    args = parse_args()
    setup_logging(args.log_level)
    _validate_thresholds(args.conf, args.iou)

    source = _resolve_source(args.source)
    model_reference = resolve_model_reference(args.model)
    model_path = Path(model_reference)
    if len(model_path.parts) > 1 and not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_project = resolve_project_path(args.project)
    webcam_source = _is_webcam_source(source)

    LOGGER.info("Model: %s", model_reference)
    LOGGER.info("Source: %s", source)
    LOGGER.info("Confidence threshold: %.2f", args.conf)
    if webcam_source:
        LOGGER.info("Webcam/video preview can be closed by pressing q in the display window.")
        if args.save:
            LOGGER.info("Webcam output saving is enabled. Use --no-save for preview-only testing.")

    YOLO = _load_yolo_class()
    model = YOLO(model_reference)
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=args.show,
        project=str(output_project),
        name=args.name,
        exist_ok=args.exist_ok,
    )

    LOGGER.info("Prediction completed for %s input item(s).", len(results))
    if args.save:
        LOGGER.info("Annotated outputs were saved under: %s", output_project / args.name)
    else:
        LOGGER.info("Saving was disabled for this prediction run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
