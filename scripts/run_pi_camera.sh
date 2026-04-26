#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="${MODEL:-models/exported/ncnn/best_ncnn_model}"
CONF="${CONF:-0.5}"
DEVICE="${DEVICE:-cpu}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-15}"
MAX_FRAMES="${MAX_FRAMES:-0}"
LOG_EVERY="${LOG_EVERY:-30}"
SHOW="${SHOW:-1}"
SAVE="${SAVE:-0}"

ARGS=(
  -m src.pi_inference
  --backend picamera2
  --model "${MODEL}"
  --imgsz 640
  --conf "${CONF}"
  --device "${DEVICE}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --fps "${FPS}"
  --max-frames "${MAX_FRAMES}"
  --log-every "${LOG_EVERY}"
  --project runs/pi_inference
  --name pi_camera
)

if [[ "${SHOW}" == "1" || "${SHOW}" == "true" ]]; then
  ARGS+=(--show)
fi

if [[ "${SAVE}" == "1" || "${SAVE}" == "true" ]]; then
  ARGS+=(--save)
else
  ARGS+=(--no-save)
fi

ARGS+=("$@")

"${PYTHON}" "${ARGS[@]}"
