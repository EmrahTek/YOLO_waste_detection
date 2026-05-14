# Model Comparison

| Metric | baseline_yolo11n_640 | yolo11n_800 | Best |
|---|---:|---:|---|
| precision | 0.8080 | 0.7760 | baseline_yolo11n_640 |
| recall | 0.7292 | 0.6737 | baseline_yolo11n_640 |
| f1 | 0.7666 | 0.7212 | baseline_yolo11n_640 |
| map50 | 0.8516 | 0.8003 | baseline_yolo11n_640 |
| map50_95 | 0.7113 | 0.6412 | baseline_yolo11n_640 |

## Interpretation

- Precision zeigt, wie viele erkannte Objekte wirklich korrekt sind.
- Recall zeigt, wie viele echte Objekte vom Modell gefunden wurden.
- Der F1-Score kombiniert Precision und Recall.
- mAP50 bewertet die Detektionsqualität bei IoU = 0.50.
- mAP50-95 ist strenger und bewertet die Modellqualität über mehrere IoU-Grenzen.