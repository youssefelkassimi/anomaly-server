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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Score (probability only)
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Feature importances
# ----------------------------------------------------------------------

@app.route("/importances", methods=["GET"])
def importances():
    auth_error = _require_auth()
    if auth_error:
        return auth_error

    if not _detector.is_trained:
        return jsonify({"error": "Model is not trained yet"}), 409
    return jsonify({"status": "ok", "importances": _detector.feature_importances()})


# ----------------------------------------------------------------------
# Model management
# ----------------------------------------------------------------------

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
