"""
Flask API for anomaly detection.

Backend apps POST metric data here; this service trains a Random Forest
model and returns anomaly predictions.

Run:
    python app.py
or:
    flask --app app run --port 5001
"""
import os
import json
import logging
import threading

from flask import Flask, request, jsonify

from core import AnomalyDetector, DEFAULT_MODEL_PATH

from train_csv import train_from_csv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("anomaly.server")

app = Flask(__name__)

MODEL_PATH = os.environ.get("ANOMALY_MODEL_PATH", DEFAULT_MODEL_PATH)
CONTAMINATION = float(os.environ.get("ANOMALY_CONTAMINATION", 0.05))
N_ESTIMATORS = int(os.environ.get("ANOMALY_N_ESTIMATORS", 200))
MAX_DEPTH = os.environ.get("ANOMALY_MAX_DEPTH")
MAX_DEPTH = int(MAX_DEPTH) if MAX_DEPTH else None
API_KEY = os.environ.get("ANOMALY_API_KEY", "")


_detector = AnomalyDetector(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    contamination=CONTAMINATION,
)
_lock = threading.Lock()

if os.path.exists(MODEL_PATH):
    try:
        _detector.load(MODEL_PATH)
        logger.info("Loaded existing model from %s", MODEL_PATH)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load model: %s", exc)



def _require_auth():
    if not API_KEY:
        return None
    token = request.headers.get("Authorization", "")
    if token != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401
    return None


def _unpack_data(payload):
    """Accept either {"data": [...]} or a bare list / dict."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"], payload.get("labels")
    return payload, None


FEATURE_KEYS = [
    "system.cpu.cpu_percent",
    "system.memory.percent",
    "system.disk_usage.0.percent",
    "system.network.total_errin",
    "system.processes.total_count",
    "system.cpu.load_avg.0",
    "system.memory.swap_percent",
]
COUNTER_KEYS = {"system.network.total_errin"}
_prev = {}  # host_id -> {counter_key: last_value}


def _get(d, path, default=0.0):
    """Accepts flat {"a.b": 1} or nested {"a": {"b": 1}} input."""
    if path in d:
        v = d[path]
    else:
        v = d
        for part in path.split("."):
            if isinstance(v, list):
                try:
                    v = v[int(part)]
                except (ValueError, IndexError):
                    return default
            elif isinstance(v, dict) and part in v:
                v = v[part]
            else:
                return default
    return float(v) if isinstance(v, (int, float)) else default


@app.route("/check", methods=["POST"])
def check():
    auth_error = _require_auth()
    if auth_error:
        return auth_error
    if not _detector.is_trained:
        return jsonify({"error": "Model is not trained. POST /train first."}), 409

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400

    host_id = request.args.get("host_id", "default")
    with _lock:
        prev = _prev.setdefault(host_id, {})
        row = {}
        for key in FEATURE_KEYS:
            v = _get(payload, key)
            if key in COUNTER_KEYS:
                last = prev.get(key, v)
                prev[key] = v
                v = max(v - last, 0.0)
            row[key] = v
        result = _detector.predict(row)

    return jsonify({
        "host_id": host_id,
        "is_anomaly": result["label"] == "anomaly",
        "score": round(result["probability"], 4),
        "top_features": result["feature_contributions"],
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "trained": _detector.is_trained,
            "features": _detector.feature_names,
            "model": os.path.basename(MODEL_PATH),
        }
    )



@app.route("/train", methods=["POST"])
def train():
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON body"}), 400

    data, labels = _unpack_data(payload)
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "data must be a non-empty list of metric dicts"}), 400

    try:
        with _lock:
            result = _detector.train(data, labels=labels)
            _detector.save(MODEL_PATH)
        return jsonify({"status": "ok", "trained": True, "model_path": MODEL_PATH, **result})
    except Exception as exc:  # pragma: no cover
        logger.exception("Training failed")
        return jsonify({"error": str(exc)}), 500

from train_csv import train_from_csv

@app.route("/train_csv", methods=["POST"])
def train_csv_endpoint():
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    f = request.files.get("file")
    if not f:
        return jsonify({"error": "Send a multipart form with a 'file' field"}), 400

    try:
        with _lock:
            summary, new_detector = train_from_csv(
                f.stream,
                model_path=MODEL_PATH,
                contamination=CONTAMINATION,
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                counter_mode=request.form.get("counter_mode", "delta"),
                host_col=request.form.get("host_col"),
                time_col=request.form.get("time_col"),
                label_col=request.form.get("label_col"),
            )
            global _detector
            _detector = new_detector          # hot-swap the live model
        return jsonify({"status": "ok", **summary})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("CSV training failed")
        return jsonify({"error": str(exc)}), 500



@app.route("/predict", methods=["POST"])
def predict():
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    if not _detector.is_trained:
        return jsonify({"error": "Model is not trained. POST /train first."}), 409

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON body"}), 400

    data, _ = _unpack_data(payload)

    single = False
    if isinstance(data, dict):
        single = True
        data = [data]
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "data must be a metric dict or a list of metric dicts"}), 400

    try:
        with _lock:
            results = _detector.predict(data)
        if not isinstance(results, list):
            results = [results]
        return jsonify({"status": "ok", "count": len(results), "results": results})
    except Exception as exc:  # pragma: no cover
        logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500



@app.route("/score", methods=["POST"])
def score():
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    if not _detector.is_trained:
        return jsonify({"error": "Model is not trained. POST /train first."}), 409

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON body"}), 400

    data, _ = _unpack_data(payload)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "data must be a metric dict or list"}), 400

    try:
        with _lock:
            scores = _detector.predict_score(data)
        if not isinstance(scores, list):
            scores = [scores]
        return jsonify({"status": "ok", "scores": scores})
    except Exception as exc:  # pragma: no cover
        logger.exception("Scoring failed")
        return jsonify({"error": str(exc)}), 500



@app.route("/importances", methods=["GET"])
def importances():
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    if not _detector.is_trained:
        return jsonify({"error": "Model is not trained yet"}), 409
    return jsonify({"status": "ok", "importances": _detector.feature_importances()})



@app.route("/model", methods=["DELETE"])
def delete_model():
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    global _detector
    with _lock:
        _detector = AnomalyDetector(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            contamination=CONTAMINATION,
        )
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
    return jsonify({"status": "ok", "trained": False})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logger.info("Starting anomaly server on port %s", port)
    app.run(host="0.0.0.0", port=port)
