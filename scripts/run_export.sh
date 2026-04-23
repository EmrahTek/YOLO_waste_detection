#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" -m src.export_model \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --formats onnx ncnn \
  --imgsz 640 \
  --device cpu \
  --output-root models/exported
