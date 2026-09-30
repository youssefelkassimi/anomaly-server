# Anomaly Detection Server (Random Forest)

A standalone Flask service that receives system metrics from a backend
app and returns anomaly detection results. Uses a scikit-learn Random
Forest classifier.

## Install

```bash
cd anomaly_server
pip install -r requirements.txt
```

## Run

```bash
python app.py                    # serves on 0.0.0.0:5001
```

Optional env vars:
| Env | Default | Purpose |
|-----|---------|---------|
| `PORT` | `5001` | HTTP port |
| `ANOMALY_MODEL_PATH` | `anomaly_rf.pkl` | where the model is saved/loaded |
| `ANOMALY_CONTAMINATION` | `0.05` | anomaly aggressiveness (lower = fewer false alarms) |
| `ANOMALY_N_ESTIMATORS` | `200` | number of trees |
| `ANOMALY_MAX_DEPTH` | (none) | tree depth limit |
| `ANOMALY_API_KEY` | (empty) | if set, requires `Authorization: Bearer <key>` |

## Endpoints

### `GET /health`
Service status and whether the model is trained.

### `POST /train`
Teach the model with historical metrics (supervised or semi-supervised).

Body:
```json
{
  "data": [
    {"cpu_percent": 45, "memory_percent": 60, "disk_percent": 30, "network_errin": 5, "process_count": 150},
    {"cpu_percent": 98, "memory_percent": 97, "disk_percent": 95, "network_errin": 500, "process_count": 800}
  ],
  "labels": [0, 1]
}
```
- `labels` is **optional**. Omit it to treat all samples as *normal*;
  synthetic anomalies are generated automatically.
- Returns accuracy, confusion matrix, and feature importances.

### `POST /predict`
Return anomaly results for new metrics.

Body:
```json
{
  "data": [
    {"cpu_percent": 45, "memory_percent": 60, "disk_percent": 30, "network_errin": 5, "process_count": 150}
  ]
}
```

Response:
```json
{
  "status": "ok",
  "count": 1,
  "results": [
    {
      "prediction": 0,
      "label": "normal",
      "probability": 0.02,
      "feature_contributions": {}
    }
  ]
}
```
`probability` is the anomaly score (0.0 = normal, 1.0 = anomaly).
`feature_contributions` lists which metrics drove the decision, but only
for anomalies.

### `POST /score`
Only the raw anomaly probability, no other fields.

### `GET /importances`
Which features matter most to the current model.

### `DELETE /model`
Reset the model (only if `ANOMALY_API_KEY` is set).

## Notes

- A single sample `{"data": {...}}` (dict) is also accepted.
- The model persists to `anomaly_rf.pkl` and is auto-loaded on restart.
- `contamination` tunes sensitivity: lower (0.01-0.05) = fewer false
  positives; higher (0.1-0.2) = catches more anomalies.
