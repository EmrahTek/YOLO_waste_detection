"""Raspberry Pi inference starter script.

The default backend uses Ultralytics directly and can run on a local image,
folder, video, or camera index. The Picamera2 backend is structured for later
hardware testing on Raspberry Pi 5 and is not executed during project setup.
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
    parser = argparse.ArgumentParser(description="Raspberry Pi inference starter.")
    parser.add_argument("--model", required=True, help="Exported model path or Ultralytics model path.")
    parser.add_argument(
        "--source",
        default="0",
        help="Image, folder, video, or camera index. Used by the Ultralytics backend.",
    )
    parser.add_argument("--backend", choices=("ultralytics", "picamera2"), default="ultralytics")
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold. The current desktop baseline works well around 0.5.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=config.RUNS_DIR / "pi_inference")
    parser.add_argument("--name", default="pi_demo")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Picamera2 backend only. 0 means run until interrupted.",
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


def _resolve_source(source: str) -> str:
    """Resolve local source paths while preserving camera indexes."""
    if source.isdigit():
        return source
    source_path = resolve_project_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Inference source not found: {source_path}")
    return str(source_path)


def run_ultralytics_backend(args: argparse.Namespace) -> None:
    """Run inference through Ultralytics predict."""
    YOLO = _load_yolo_class()
    model = YOLO(resolve_model_reference(args.model))
    model.predict(
        source=_resolve_source(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
        show=args.show,
        project=str(resolve_project_path(args.project)),
        name=args.name,
    )


def run_picamera2_backend(args: argparse.Namespace) -> None:
    """Run a simple Picamera2 capture loop for later Raspberry Pi testing."""
    try:
        import cv2
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError(
            "Picamera2 backend requires picamera2 and opencv on Raspberry Pi OS."
        ) from exc

    YOLO = _load_yolo_class()
    model = YOLO(resolve_model_reference(args.model))
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"format": "RGB888"}))
    camera.start()

    frame_count = 0
    try:
        while True:
            frame = camera.capture_array()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            annotated = results[0].plot()
            if args.show:
                cv2.imshow("YOLO Pi inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()


def main() -> int:
    """Run the selected Raspberry Pi inference backend."""
    args = parse_args()
    setup_logging(args.log_level)

    LOGGER.info("Backend: %s", args.backend)
    LOGGER.info("Model: %s", args.model)

    if args.backend == "ultralytics":
        run_ultralytics_backend(args)
    else:
        run_picamera2_backend(args)

    LOGGER.info("Inference finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
