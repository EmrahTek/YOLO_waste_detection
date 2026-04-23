# YOLO_Mulltrennung

Python project for detecting recyclable waste objects with Ultralytics YOLO.

Target classes:

| Class ID | Name | Meaning |
| ---: | --- | --- |
| 0 | `tetrapak` | Beverage carton / Tetra Pak style package |
| 1 | `dose` | Metal can |

The desktop training and export workflow is complete. Raspberry Pi 5 and Pi AI Camera deployment is the next project step.

## Current Status

- Cleaned YOLO dataset: `yolo_dataset/prepared`
- Trained model: `runs/train/tetrapak_dose_yolo11n/weights/best.pt`
- Training run: `runs/train/tetrapak_dose_yolo11n`
- ONNX export: `models/exported/onnx/best.onnx`
- NCNN export: `models/exported/ncnn/best_ncnn_model`
- Test metrics summary: `reports/validation_metrics_test.json`
- Recommended demo confidence threshold: about `0.5`

The baseline works on prepared test images. A confidence threshold of `0.5` produced cleaner desktop predictions than `0.25`. The model has not yet been tested with the local laptop/desktop webcam.

## Project Structure

```text
.
├── models/
│   ├── exported/
│   │   ├── onnx/best.onnx
│   │   └── ncnn/best_ncnn_model/
│   └── pretrained/
├── notebooks/
├── reports/
│   ├── evaluation_summary.md
│   └── validation_metrics_test.json
├── runs/
│   ├── train/tetrapak_dose_yolo11n/
│   ├── val/test/
│   └── predict/
├── scripts/
│   ├── run_check_dataset.sh
│   ├── run_export.sh
│   ├── run_predict.sh
│   ├── run_train.sh
│   ├── run_val.sh
│   └── run_webcam.sh
├── src/
│   ├── check_dataset.py
│   ├── export_model.py
│   ├── pi_inference.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── validate.py
│   └── utils/
└── yolo_dataset/
    ├── images/ and labels/      # Raw CVAT-derived source data
    └── prepared/                # Clean Ultralytics YOLO dataset
```

## Dataset

The raw CVAT export had nested label folders and missing labels in the source split folders. The cleaned dataset was created non-destructively under `yolo_dataset/prepared`.

Prepared dataset layout:

```text
yolo_dataset/prepared/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Prepared dataset counts:

| Split | Images | Labels | Objects |
| --- | ---: | ---: | ---: |
| train | 247 | 247 | 431 |
| val | 87 | 87 | 175 |
| test | 57 | 57 | 65 |

The prepared dataset passes the project checker with no blocking issues.

## Setup

Use the existing virtual environment when available:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

For a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

For GPU training, PyTorch must match the CUDA setup on the training computer.

## Dataset Commands

Check the prepared dataset:

```bash
.venv/bin/python -m src.check_dataset \
  --dataset-root yolo_dataset/prepared \
  --layout standard \
  --strict
```

Preview dataset preparation from the raw CVAT-derived source:

```bash
.venv/bin/python -m src.prepare_dataset \
  --overlap-policy drop-from-train
```

Recreate the prepared dataset only if needed:

```bash
.venv/bin/python -m src.prepare_dataset \
  --overlap-policy drop-from-train \
  --apply
```

This copies data into `yolo_dataset/prepared`; it does not delete or move source files.

## Training Summary

Training was completed successfully with:

- Model: `yolo11n.pt`
- Epochs: `100`
- Image size: `640`
- Batch size: `16`
- Device: GPU device `0`
- Patience: `30`
- Output: `runs/train/tetrapak_dose_yolo11n`

Training command used by the project:

```bash
.venv/bin/python -m src.train \
  --data yolo_dataset/prepared/data.yaml \
  --model yolo11n.pt \
  --imgsz 640 \
  --epochs 100 \
  --batch 16 \
  --device 0 \
  --project runs/train \
  --name tetrapak_dose_yolo11n \
  --patience 30
```

Training is already done. Do not rerun it unless you intentionally want a new model.

## Evaluation Summary

Final test metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.808 |
| Recall | 0.729 |
| F1 | 0.767 |
| mAP50 | 0.852 |
| mAP50-95 | 0.711 |

Relevant outputs:

- Training curves: `runs/train/tetrapak_dose_yolo11n/results.png`
- Training metrics CSV: `runs/train/tetrapak_dose_yolo11n/results.csv`
- Test validation plots: `runs/val/test/`
- Test metrics JSON: `reports/validation_metrics_test.json`
- Evaluation notes: `reports/evaluation_summary.md`

Validate on the validation split:

```bash
.venv/bin/python -m src.validate \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --data yolo_dataset/prepared/data.yaml \
  --split val \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --project runs/val \
  --name val
```

Validate on the test split:

```bash
.venv/bin/python -m src.validate \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --data yolo_dataset/prepared/data.yaml \
  --split test \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --project runs/val \
  --name test
```

## Prediction

Run prediction on the prepared test folder:

```bash
.venv/bin/python -m src.predict \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --source yolo_dataset/prepared/images/test \
  --imgsz 640 \
  --conf 0.5 \
  --device 0 \
  --project runs/predict \
  --name test_conf_05
```

Run prediction on any folder:

```bash
.venv/bin/python -m src.predict \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --source path/to/images \
  --conf 0.5 \
  --device 0
```

Run prediction on one image:

```bash
.venv/bin/python -m src.predict \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --source path/to/image.jpg \
  --conf 0.5 \
  --device 0
```

Annotated outputs are saved under `runs/predict/<name>/` unless `--no-save` is used.

## Local Webcam Test

The model has not yet been tested with the laptop/desktop webcam. Use this command tonight for a quick desktop check:

```bash
.venv/bin/python -m src.predict \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --source 0 \
  --imgsz 640 \
  --conf 0.5 \
  --device 0 \
  --show \
  --no-save \
  --name webcam_test
```

Press `q` in the preview window to quit. If GPU webcam inference has display issues, try CPU:

```bash
.venv/bin/python -m src.predict \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --source 0 \
  --imgsz 640 \
  --conf 0.5 \
  --device cpu \
  --show \
  --no-save \
  --name webcam_test_cpu
```

Convenience wrapper:

```bash
./scripts/run_webcam.sh
```

Optional environment variables:

```bash
CAMERA=0 CONF=0.5 DEVICE=cpu ./scripts/run_webcam.sh
```

If camera access is blocked by the operating system or desktop session, run the command locally from a normal terminal where the webcam is available.

## Export Summary

Export to ONNX:

```bash
.venv/bin/python -m src.export_model \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --formats onnx \
  --imgsz 640 \
  --device cpu \
  --output-root models/exported
```

Export to NCNN:

```bash
.venv/bin/python -m src.export_model \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --formats ncnn \
  --imgsz 640 \
  --device cpu \
  --output-root models/exported
```

Export both formats:

```bash
.venv/bin/python -m src.export_model \
  --model runs/train/tetrapak_dose_yolo11n/weights/best.pt \
  --formats onnx ncnn \
  --imgsz 640 \
  --device cpu \
  --output-root models/exported
```

Current export artifacts:

- `models/exported/onnx/best.onnx`
- `models/exported/ncnn/best_ncnn_model/model.ncnn.param`
- `models/exported/ncnn/best_ncnn_model/model.ncnn.bin`
- `models/exported/ncnn/best_ncnn_model/metadata.yaml`

## Raspberry Pi Next Step

Deployment is intentionally not done yet. Tomorrow's Raspberry Pi work should start from the existing exports:

```bash
python -m src.pi_inference \
  --model models/exported/ncnn/best_ncnn_model \
  --source 0 \
  --backend ultralytics \
  --device cpu \
  --conf 0.5
```

For Picamera2 testing on Raspberry Pi OS, install Pi camera dependencies on the Pi and then adapt/run:

```bash
python -m src.pi_inference \
  --model models/exported/ncnn/best_ncnn_model \
  --backend picamera2 \
  --device cpu \
  --conf 0.5 \
  --show
```

## Helper Scripts

```bash
./scripts/run_check_dataset.sh
./scripts/run_predict.sh yolo_dataset/prepared/images/test
SPLIT=test NAME=test ./scripts/run_val.sh
./scripts/run_export.sh
./scripts/run_webcam.sh
```

`./scripts/run_train.sh` is available for reproducibility, but training is already complete.

## Known Limitations

- The dataset is small, especially the prepared test split with `57` images.
- Recall is lower than precision, so missed detections should be reviewed in real scenes.
- The model has not yet been tested with the local webcam.
- The model has not yet been tested on Raspberry Pi 5 or the Pi AI Camera.
- Real lighting, motion blur, camera angle, object scale, and clutter may require more training data.
- The raw CVAT-derived dataset still contains excluded unlabeled images; source data was preserved.

## Recommended Next Improvements

- Run the local webcam test at `conf=0.5`.
- Save a few false positives and false negatives from webcam testing.
- Test ONNX and NCNN inference speed on Raspberry Pi 5.
- Add Pi camera examples to the dataset if deployment reveals weak cases.
- Consider a second training run only after collecting new real-world failure cases.
