#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" -m src.train \
  --data yolo_dataset/prepared/data.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --project runs/train \
  --name tetrapak_dose_yolo11n \
  --patience 30
