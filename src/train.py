"""Train an Ultralytics YOLO object detector.

This script prepares and launches training when the user runs it manually. It is
not executed automatically by the project setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src import config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.paths import resolve_model_reference, resolve_project_path


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a YOLO detector manually.")
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DEFAULT_DATASET_YAML,
        help="Path to the prepared Ultralytics data.yaml.",
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL_WEIGHTS,
        help="Model weights or YAML. Example: yolo11n.pt.",
    )
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    parser.add_argument("--project", type=Path, default=config.DEFAULT_TRAIN_PROJECT)
    parser.add_argument("--name", default=config.DEFAULT_TRAIN_NAME)
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained weights when supported by the selected model.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable mixed precision training when supported.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument(
        "--cache",
        choices=("none", "ram", "disk"),
        default="none",
        help="Optional Ultralytics image cache mode.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the resolved training configuration without starting training.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_yolo_class() -> Any:
    """Import Ultralytics lazily so missing dependencies produce a clear error."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install dependencies with requirements.txt first."
        ) from exc
    return YOLO


def _cache_argument(cache: str) -> bool | str:
    """Convert the CLI cache option to the value expected by Ultralytics."""
    return False if cache == "none" else cache


def build_train_kwargs(args: argparse.Namespace, allow_missing_data: bool = False) -> dict[str, Any]:
    """Build keyword arguments for YOLO.train."""
    data_path = resolve_project_path(args.data)
    if not data_path.exists() and not allow_missing_data:
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Run prepare_dataset.py first."
        )
    if not data_path.exists() and allow_missing_data:
        LOGGER.warning("Dataset YAML does not exist yet: %s", data_path)

    return {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "project": str(resolve_project_path(args.project)),
        "name": args.name,
        "patience": args.patience,
        "pretrained": args.pretrained,
        "amp": args.amp,
        "workers": args.workers,
        "seed": args.seed,
        "cache": _cache_argument(args.cache),
        "exist_ok": args.exist_ok,
    }


def main() -> int:
    """Run manual YOLO training."""
    args = parse_args()
    setup_logging(args.log_level)

    model_reference = resolve_model_reference(args.model)
    train_kwargs = build_train_kwargs(args, allow_missing_data=args.dry_run)

    LOGGER.info("Model: %s", model_reference)
    for key, value in train_kwargs.items():
        LOGGER.info("%s: %s", key, value)

    if args.dry_run:
        LOGGER.info("Dry run requested. Training was not started.")
        return 0

    YOLO = _load_yolo_class()
    model = YOLO(model_reference)
    results = model.train(**train_kwargs)
    LOGGER.info("Training completed. Results: %s", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
