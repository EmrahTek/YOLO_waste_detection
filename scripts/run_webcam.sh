#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-.venv/bin/python}"
CAMERA="${CAMERA:-0}"
CONF="${CONF:-0.5}"
DEVICE="${DEVICE:-0}"

"${PYTHON}" -m src.predict \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --source "${CAMERA}" \
  --imgsz 640 \
  --conf "${CONF}" \
  --device "${DEVICE}" \
  --show \
  --no-save \
  --project runs/predict \
  --name webcam_test
