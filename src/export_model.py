"""Export trained YOLO models for desktop and Raspberry Pi deployment tests."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from src import config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.paths import resolve_model_reference, resolve_project_path


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_model = config.DEFAULT_TRAIN_PROJECT / config.DEFAULT_TRAIN_NAME / "weights" / "best.pt"
    parser = argparse.ArgumentParser(description="Export a trained YOLO model.")
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("onnx", "ncnn"),
        default=["onnx"],
        help="Export format(s) to produce.",
    )
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=config.EXPORTED_MODELS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_yolo_class() -> Any:
    """Import Ultralytics lazily."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed.") from exc
    return YOLO


def _copy_artifact(artifact: Path, target_dir: Path, overwrite: bool) -> Path:
    """Copy an exported artifact into the project export folder."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / artifact.name

    if artifact.is_dir():
        if target.exists() and not overwrite:
            LOGGER.warning("Export artifact already exists, leaving it unchanged: %s", target)
            return target
        shutil.copytree(artifact, target, dirs_exist_ok=overwrite)
        return target

    if target.exists() and not overwrite:
        LOGGER.warning("Export artifact already exists, leaving it unchanged: %s", target)
        return target

    shutil.copy2(artifact, target)
    return target


def export_format(model: Any, args: argparse.Namespace, format_name: str) -> Path:
    """Export one format and copy it into models/exported."""
    export_kwargs: dict[str, Any] = {
        "format": format_name,
        "imgsz": args.imgsz,
        "device": args.device,
        "half": args.half,
        "int8": args.int8,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
    }
    if args.opset is not None:
        export_kwargs["opset"] = args.opset

    LOGGER.info("Exporting format=%s", format_name)
    artifact = Path(model.export(**export_kwargs))
    if not artifact.exists():
        raise FileNotFoundError(f"Ultralytics reported an export path that does not exist: {artifact}")

    target = _copy_artifact(
        artifact=artifact,
        target_dir=resolve_project_path(args.output_root) / format_name,
        overwrite=args.overwrite,
    )
    LOGGER.info("Copied %s export to: %s", format_name, target)
    return target


def main() -> int:
    """Run model export."""
    args = parse_args()
    setup_logging(args.log_level)

    model_reference = resolve_model_reference(args.model)
    model_path = Path(model_reference)
    if len(model_path.parts) > 1 and not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    YOLO = _load_yolo_class()
    model = YOLO(model_reference)

    exported_paths = [export_format(model, args, format_name) for format_name in args.formats]
    LOGGER.info("Export completed: %s", ", ".join(path.as_posix() for path in exported_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
