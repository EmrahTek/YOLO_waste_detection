"""Central configuration for the YOLO waste-sorting project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET_DIR = PROJECT_ROOT / "yolo_dataset"
PREPARED_DATASET_DIR = RAW_DATASET_DIR / "prepared"
DEFAULT_DATASET_YAML = PREPARED_DATASET_DIR / "data.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
EXPORTED_MODELS_DIR = MODELS_DIR / "exported"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"

SPLITS = ("train", "val", "test")
CLASS_NAMES = ("tetrapak", "dose")
CLASS_ID_TO_NAME = {index: name for index, name in enumerate(CLASS_NAMES)}
VALID_CLASS_IDS = set(CLASS_ID_TO_NAME)

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

DEFAULT_MODEL_WEIGHTS = "yolo11n.pt"
DEFAULT_IMAGE_SIZE = 640
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_PATIENCE = 30
DEFAULT_DEVICE = "0"
DEFAULT_SEED = 42

DEFAULT_TRAIN_PROJECT = RUNS_DIR / "train"
DEFAULT_TRAIN_NAME = "tetrapak_dose_yolo11n"
DEFAULT_VAL_PROJECT = RUNS_DIR / "val"
DEFAULT_PREDICT_PROJECT = RUNS_DIR / "predict"
DEFAULT_EXPORT_FORMATS = ("onnx", "ncnn")
