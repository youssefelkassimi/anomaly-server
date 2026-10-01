"""
Train the anomaly detector from a CSV file.

CSV format: one row per metric snapshot, one column per feature, named exactly:
  system.cpu.cpu_percent, system.memory.percent, system.disk_usage.0.percent,
  system.network.total_errin, system.processes.total_count,
  system.cpu.load_avg.0, system.memory.swap_percent
Optional columns: host_id, timestamp, label (0=normal, 1=anomaly).

Usage:
  python train_csv.py metrics.csv
  python train_csv.py metrics.csv --counter-mode cumulative --host-col host_id --time-col timestamp
  python train_csv.py metrics.csv --label-col label --out models/anomaly_rf.pkl
"""
import os
import argparse
import json
import logging
import sys
from typing import Optional

import numpy as np
import pandas as pd

from core import AnomalyDetector, DEFAULT_MODEL_PATH

logger = logging.getLogger("anomaly.train")

FEATURE_KEYS = [
    "system.cpu.cpu_percent",
    "system.memory.percent",
    "system.disk_usage.0.percent",
    "system.network.total_errin",
    "system.processes.total_count",
    "system.cpu.load_avg.0",
    "system.memory.swap_percent",
]
COUNTER_KEYS = ["system.network.total_errin"]
MIN_ROWS = 100


def load_training_frame(
    source,                                   # file path or file-like object
    counter_mode: str = "delta",              # "delta" | "cumulative"
    host_col: Optional[str] = None,
    time_col: Optional[str] = None,
    label_col: Optional[str] = None,
):
    """Read + clean the CSV. Returns (X_df, labels_or_None, info_dict)."""
    df = pd.read_csv(source)
    info = {"rows_read": int(len(df))}

    missing = [c for c in FEATURE_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Numeric coercion (bad cells become NaN and get dropped below)
    for c in FEATURE_KEYS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Cumulative counters -> per-cycle delta (per host, in time order)
    if counter_mode == "cumulative":
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col)
        for c in COUNTER_KEYS:
            if host_col and host_col in df.columns:
                diff = df.groupby(host_col)[c].diff()
            else:
                diff = df[c].diff()
            df[c] = diff.clip(lower=0)        # counter reset (reboot) -> 0
    elif counter_mode != "delta":
        raise ValueError("counter_mode must be 'delta' or 'cumulative'")

    cols = FEATURE_KEYS + ([label_col] if label_col else [])
    before = len(df)
    df = df.dropna(subset=cols).replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    info["rows_dropped_invalid"] = int(before - len(df))

    if len(df) < MIN_ROWS:
        raise ValueError(f"Only {len(df)} valid rows; need at least {MIN_ROWS}.")

    labels = None
    if label_col:
        labels = df[label_col].astype(int).to_numpy()
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("label column must contain both 0 (normal) and 1 (anomaly).")

    info["rows_used"] = int(len(df))
    return df[FEATURE_KEYS].reset_index(drop=True), labels, info


def sanity_check(detector: AnomalyDetector, X: pd.DataFrame) -> dict:
    """Score a typical row, then the same row with each feature spiked by +5 std."""
    base = X.median().to_dict()
    stds = X.std().replace(0, 1.0).to_dict()
    out = {"typical_row_score": round(float(detector.predict_score(base)), 4), "spikes": {}}
    for k in FEATURE_KEYS:
        row = dict(base)
        row[k] = base[k] + 5 * stds[k]
        out["spikes"][k] = round(float(detector.predict_score(row)), 4)
    return out


def train_from_csv(source, model_path=DEFAULT_MODEL_PATH, contamination=0.05,
                   n_estimators=200, max_depth=None, **csv_kwargs) -> dict:
    X_df, labels, info = load_training_frame(source, **csv_kwargs)

    detector = AnomalyDetector(
        n_estimators=n_estimators, max_depth=max_depth, contamination=contamination
    )
    report = detector.train(X_df.to_dict("records"), labels=labels)
    detector.save(model_path)
    return {
        "model_path": model_path,
        "data": info,
        "mode": "supervised" if labels is not None else "semi-supervised (synthetic anomalies)",
        "n_samples_after_augmentation": report["n_samples"],
        "n_anomalies": report["n_anomalies"],
        "accuracy": report["accuracy"],
        "confusion_matrix": report["confusion_matrix"],
        "feature_importances": report["feature_importances"],
        "sanity_check": sanity_check(detector, X_df),
    }, detector


def main():
    p = argparse.ArgumentParser(description="Train anomaly model from CSV")
    p.add_argument("csv")
    p.add_argument("--out", default=DEFAULT_MODEL_PATH)
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--counter-mode", choices=["delta", "cumulative"], default="delta")
    p.add_argument("--host-col", default=None)
    p.add_argument("--time-col", default=None)
    p.add_argument("--label-col", default=None)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary, _ = train_from_csv(
            a.csv, model_path=a.out, contamination=a.contamination,
            n_estimators=a.n_estimators, max_depth=a.max_depth,
            counter_mode=a.counter_mode, host_col=a.host_col,
            time_col=a.time_col, label_col=a.label_col,
        )
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
