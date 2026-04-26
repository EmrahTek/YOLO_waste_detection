"""Run Raspberry Pi inference with Ultralytics or Picamera2."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src import config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.paths import resolve_model_reference, resolve_project_path


LOGGER = get_logger(__name__)
DEFAULT_PI_MODEL = config.EXPORTED_MODELS_DIR / "ncnn" / "best_ncnn_model"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run YOLO inference on Raspberry Pi.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_PI_MODEL,
        help="Exported model path or Ultralytics model path.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Image, folder, video, or camera index. Used by the Ultralytics backend.",
    )
    parser.add_argument("--backend", choices=("ultralytics", "picamera2"), default="picamera2")
    parser.add_argument("--imgsz", type=int, default=config.DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold. The current desktop baseline works well around 0.5.",
    )
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=config.RUNS_DIR / "pi_inference")
    parser.add_argument("--name", default="pi_demo")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the output directory name.")
    parser.add_argument("--camera-num", type=int, default=0, help="Picamera2 camera number.")
    parser.add_argument("--width", type=int, default=1280, help="Picamera2 capture width.")
    parser.add_argument("--height", type=int, default=720, help="Picamera2 capture height.")
    parser.add_argument("--fps", type=float, default=15.0, help="Picamera2 capture frame rate.")
    parser.add_argument(
        "--camera-format",
        default="RGB888",
        help="Picamera2 main stream format. RGB888 is converted to BGR for Ultralytics/OpenCV.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=1.0,
        help="Seconds to wait after starting the camera.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Picamera2 backend only. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Picamera2 backend only. Log detections every N frames; 0 disables periodic logs.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_yolo_class() -> Any:
    """Import Ultralytics lazily."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Install the Raspberry Pi runtime first.") from exc
    return YOLO


def _resolve_source(source: str) -> str | int:
    """Resolve local source paths while preserving camera indexes."""
    if source.isdigit():
        return int(source)
    source_path = resolve_project_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Inference source not found: {source_path}")
    return str(source_path)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate common runtime settings before touching hardware."""
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError(f"Confidence threshold must be between 0 and 1, got {args.conf}.")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError(f"IoU threshold must be between 0 and 1, got {args.iou}.")
    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"Camera size must be positive, got {args.width}x{args.height}.")
    if args.fps <= 0:
        raise ValueError(f"FPS must be positive, got {args.fps}.")
    if args.max_frames < 0:
        raise ValueError(f"max-frames must be zero or positive, got {args.max_frames}.")
    if args.log_every < 0:
        raise ValueError(f"log-every must be zero or positive, got {args.log_every}.")
    if args.warmup_seconds < 0:
        raise ValueError(f"warmup-seconds must be zero or positive, got {args.warmup_seconds}.")


def _resolve_run_dir(project: Path, name: str, exist_ok: bool) -> Path:
    """Create a run directory, incrementing the name when needed."""
    project_dir = resolve_project_path(project)
    run_dir = project_dir / name
    if exist_ok or not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    for index in range(2, 10_000):
        candidate = project_dir / f"{name}{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate

    raise RuntimeError(f"Could not create a unique run directory under {project_dir}.")


def _format_detections(result: Any) -> str:
    """Build a compact text summary for terminal logs."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return "no detections"

    names = getattr(result, "names", {}) or {}
    classes = boxes.cls.tolist()
    confidences = boxes.conf.tolist()
    entries = []
    for class_id, confidence in zip(classes, confidences):
        class_index = int(class_id)
        if isinstance(names, dict):
            label = names.get(class_index, str(class_index))
        else:
            label = str(class_index)
        entries.append(f"{label}:{confidence:.2f}")

    if len(entries) > 6:
        return ", ".join(entries[:6]) + f", +{len(entries) - 6} more"
    return ", ".join(entries)


def _open_video_writer(cv2: Any, output_dir: Path, frame: Any, fps: float) -> tuple[Any, Path]:
    """Open an MJPEG writer once the first annotated frame size is known."""
    height, width = frame.shape[:2]
    video_path = output_dir / "picamera2_inference.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")
    LOGGER.info("Saving annotated video to: %s", video_path)
    return writer, video_path


def _to_ultralytics_frame(cv2: Any, frame: Any, camera_format: str) -> Any:
    """Convert Picamera2 RGB frames into the BGR layout expected by OpenCV inputs."""
    if camera_format.upper() == "RGB888" and len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def run_ultralytics_backend(args: argparse.Namespace) -> None:
    """Run inference through Ultralytics predict."""
    YOLO = _load_yolo_class()
    model = YOLO(resolve_model_reference(args.model), task="detect")
    model.predict(
        source=_resolve_source(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save,
        show=args.show,
        project=str(resolve_project_path(args.project)),
        name=args.name,
        exist_ok=args.exist_ok,
    )


def run_picamera2_backend(args: argparse.Namespace) -> None:
    """Run a Picamera2 capture loop on Raspberry Pi camera hardware."""
    try:
        import cv2
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError(
            "Picamera2 backend requires Raspberry Pi OS packages: "
            "python3-picamera2 and python3-opencv. Create the venv with --system-site-packages."
        ) from exc

    YOLO = _load_yolo_class()
    model = YOLO(resolve_model_reference(args.model), task="detect")
    output_dir = _resolve_run_dir(args.project, args.name, args.exist_ok) if args.save else None
    video_writer = None

    camera = Picamera2(camera_num=args.camera_num)
    camera_config = camera.create_preview_configuration(
        main={"format": args.camera_format, "size": (args.width, args.height)},
        controls={"FrameRate": args.fps},
    )
    camera.configure(camera_config)

    LOGGER.info(
        "Starting Picamera2 camera %s at %sx%s, %.1f FPS.",
        args.camera_num,
        args.width,
        args.height,
        args.fps,
    )
    camera.start()
    if args.warmup_seconds:
        time.sleep(args.warmup_seconds)

    frame_count = 0
    try:
        while True:
            frame = _to_ultralytics_frame(cv2, camera.capture_array(), args.camera_format)
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            annotated = results[0].plot()

            if output_dir is not None and video_writer is None:
                video_writer, _ = _open_video_writer(cv2, output_dir, annotated, args.fps)
            if video_writer is not None:
                video_writer.write(annotated)

            if args.show:
                cv2.imshow("YOLO Pi inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
            if args.log_every and frame_count % args.log_every == 0:
                LOGGER.info("Frame %s: %s", frame_count, _format_detections(results[0]))
            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        camera.stop()
        if video_writer is not None:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()


def main() -> int:
    """Run the selected Raspberry Pi inference backend."""
    args = parse_args()
    setup_logging(args.log_level)
    _validate_args(args)

    LOGGER.info("Backend: %s", args.backend)
    LOGGER.info("Model: %s", args.model)
    LOGGER.info("Confidence threshold: %.2f", args.conf)

    if args.backend == "ultralytics":
        run_ultralytics_backend(args)
    else:
        run_picamera2_backend(args)

    LOGGER.info("Inference finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
