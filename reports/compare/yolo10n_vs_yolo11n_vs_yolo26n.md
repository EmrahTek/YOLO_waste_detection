# Model Comparison

| Metric | YOLO10n | yolo11n | yolo26n | Best |
|---|---:|---:|---:|---|
| precision | 0.6809 | 0.8080 | 0.7672 | yolo11n |
| recall | 0.8000 | 0.7292 | 0.7617 | YOLO10n |
| f1 | 0.7356 | 0.7666 | 0.7645 | yolo11n |
| map50 | 0.8158 | 0.8516 | 0.7442 | yolo11n |
| map50_95 | 0.6798 | 0.7113 | 0.6286 | yolo11n |

## Interpretation

- Precision zeigt, wie viele erkannte Objekte wirklich korrekt sind.
- Recall zeigt, wie viele echte Objekte vom Modell gefunden wurden.
- Der F1-Score kombiniert Precision und Recall.
- mAP50 bewertet die Detektionsqualität bei IoU = 0.50.
- mAP50-95 ist strenger und bewertet die Modellqualität über mehrere IoU-Grenzen.