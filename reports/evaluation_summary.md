# Evaluation Summary

## Model

- Model path: `runs/train/tetrapak_dose_yolo11n/weights/best.pt`
- Training run: `runs/train/tetrapak_dose_yolo11n`
- Dataset YAML: `yolo_dataset/prepared/data.yaml`
- Evaluation split: `test`
- Image size: `640`
- Confidence threshold: `0.001` for metric calculation, about `0.5` recommended for cleaner demos
- IoU threshold: `0.7`

## Core Metrics

| Metric | Value |
| --- | ---: |
| Precision | 0.808 |
| Recall | 0.729 |
| F1 | 0.767 |
| mAP50 | 0.852 |
| mAP50-95 | 0.711 |

## Per-Class Notes

| Class | Strengths | Weaknesses |
| --- | --- | --- |
| tetrapak | Baseline detects the class successfully in the prepared test workflow. | Needs more real-world webcam and Raspberry Pi camera testing. |
| dose | Baseline detects the class successfully in the prepared test workflow. | Needs more real-world webcam and Raspberry Pi camera testing. |

## Confusion Matrix Notes

- Main confusion patterns: Review `runs/val/test/confusion_matrix.png` and `runs/val/test/confusion_matrix_normalized.png`.
- Classes with low recall: Test recall is usable but lower than precision, so missed detections should be reviewed.
- Classes with low precision: Precision is acceptable for the baseline; confidence `0.5` gave cleaner predictions than `0.25` during desktop testing.

## Error Analysis

- Common false positives: Review saved predictions under `runs/predict/`.
- Common false negatives: Review low-confidence and missed examples from the test split.
- Difficult lighting/background cases: Still needs webcam and Raspberry Pi camera testing.
- Small-object or occlusion issues: Still needs targeted real-world testing.
- Annotation issues noticed during review: The prepared dataset is structurally valid; the raw CVAT export contained unlabeled images that were excluded from training.

## Example Predictions To Save

- Good detections: Add selected examples from `runs/predict/`.
- False positives: Add selected examples after manual review.
- False negatives: Add selected examples after manual review.
- Ambiguous examples: Add examples from webcam/Pi tests.

## Next Actions

- Dataset fixes: Add more real-world images if webcam or Pi testing reveals weak cases.
- Training changes: Keep `yolo11n.pt` baseline for Pi deployment; consider longer training or more data only after deployment testing.
- Deployment checks: Test `models/exported/onnx/best.onnx` and `models/exported/ncnn/best_ncnn_model` on Raspberry Pi 5.
