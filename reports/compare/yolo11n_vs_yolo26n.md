# Model Comparison: yolo11n vs yolo26n

| Metric | YOLO11n | YOLO26n | Delta YOLO26n - YOLO11n | Better |
|---|---:|---:|---:|---|
| precision | 0.8080 | 0.7672 | -0.0408 | yolo11n |
| recall | 0.7292 | 0.7617 | +0.0325 | yolo26n |
| f1 | 0.7666 | 0.7645 | -0.0021 | yolo11n |
| map50 | 0.8516 | 0.7442 | -0.1074 | yolo11n |
| map50_95 | 0.7113 | 0.6286 | -0.0827 | yolo11n |

## Interpretation

- YOLO26n has slightly better recall, so it missed fewer objects on this test split.
- YOLO11n has clearly better mAP50 and mAP50-95, so its overall detection quality is better.
- For the current dataset, YOLO11n is the stronger model. YOLO26n is still interesting for edge deployment because it has fewer parameters and lower GFLOPs.