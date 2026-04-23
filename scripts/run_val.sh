#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-.venv/bin/python}"
SPLIT="${SPLIT:-val}"
NAME="${NAME:-${SPLIT}}"

"${PYTHON}" -m src.validate \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --data yolo_dataset/prepared/data.yaml \
  --split "${SPLIT}" \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --project runs/val \
  --name "${NAME}"
