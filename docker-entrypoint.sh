#!/bin/sh
set -e

CSV_PATH="${ANOMALY_CSV_PATH:-/data/metrics.csv}"
MODEL_PATH="${ANOMALY_MODEL_PATH:-/app/data/anomaly_rf.pkl}"

# Train on startup if a CSV is provided and no model exists yet.
if [ ! -f "$MODEL_PATH" ] && [ -f "$CSV_PATH" ]; then
    echo "No model at $MODEL_PATH; training from $CSV_PATH"
    python train_csv.py "$CSV_PATH" --out "$MODEL_PATH" \
        --counter-mode "${ANOMALY_COUNTER_MODE:-delta}" \
        ${ANOMALY_LABEL_COL:+--label-col "$ANOMALY_LABEL_COL"} \
        ${ANOMALY_HOST_COL:+--host-col "$ANOMALY_HOST_COL"} \
        ${ANOMALY_TIME_COL:+--time-col "$ANOMALY_TIME_COL"}
elif [ ! -f "$MODEL_PATH" ]; then
    echo "No model at $MODEL_PATH and no CSV at $CSV_PATH; starting untrained."
    echo "Train later via: curl -X POST localhost:5001/train_csv -F file=@metrics.csv"
fi

exec python app.py