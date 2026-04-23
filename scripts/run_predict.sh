#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-.venv/bin/python}"
SOURCE="${1:-yolo_dataset/prepared/images/val}"
CONF="${CONF:-0.5}"

"${PYTHON}" -m src.predict \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --source "${SOURCE}" \
  --imgsz 640 \
  --conf "${CONF}" \
  --device 0 \
  --project runs/predict \
  --name predictions
