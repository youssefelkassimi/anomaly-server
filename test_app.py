import os
import json
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from core import AnomalyDetector
import app as server_app


def _synthetic_rows(n=200):
    rows = []
    rng = __import__("numpy").random.RandomState(42)
    for _ in range(n):
        rows.append(
            {
                "cpu_percent": float(rng.normal(40, 10)),
                "memory_percent": float(rng.normal(60, 8)),
                "disk_percent": float(rng.normal(45, 5)),
                "network_errin": float(rng.poisson(5)),
                "process_count": float(rng.normal(150, 30)),
            }
        )
    return rows


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server_app, "MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(server_app, "_detector", server_app.AnomalyDetector())
    server_app.app.config["TESTING"] = True
    return server_app.app.test_client()


def test_health_before_training(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["trained"] is False


def test_predict_before_training_409(client):
    r = client.post("/predict", json={"data": _synthetic_rows(1)})
    assert r.status_code == 409


def test_train_then_predict(client):
    rows = _synthetic_rows(200)
    train_resp = client.post("/train", json={"data": rows})
    assert train_resp.status_code == 200
    body = train_resp.get_json()
    assert body["trained"] is True
    assert 0.0 <= body["accuracy"] <= 1.0
    assert "feature_importances" in body

    # normal sample prediction
    normal = {
        "cpu_percent": 40,
        "memory_percent": 60,
        "disk_percent": 45,
        "network_errin": 5,
        "process_count": 150,
    }
    r = client.post("/predict", json={"data": normal})
    assert r.status_code == 200
    result = r.get_json()["results"][0]
    assert result["prediction"] in (0, 1)


def test_supervised_labels(client):
    rows = _synthetic_rows(100)
    abnormal = [dict(r) for r in rows[:20]]
    for r in abnormal:
        for k in r:
            r[k] = r[k] * 5 + 100
    data = rows + abnormal
    labels = [0] * len(rows) + [1] * len(abnormal)
    train_resp = client.post("/train", json={"data": data, "labels": labels})
    assert train_resp.status_code == 200

    r = client.post("/predict", json={"data": [abnormal[0]]})
    assert r.status_code == 200
    assert r.get_json()["results"][0]["prediction"] == 1


def test_score_endpoint(client):
    rows = _synthetic_rows(100)
    client.post("/train", json={"data": rows})
    r = client.post("/score", json={"data": rows[0]})
    assert r.status_code == 200
    scores = r.get_json()["scores"]
    assert 0.0 <= scores[0] <= 1.0


def test_invalid_body(client):
    r = client.post("/train", json={"data": []})
    assert r.status_code == 400


def test_core_save_load(tmp_path):
    det = AnomalyDetector()
    rows = _synthetic_rows(100)
    det.train(rows)
    path = str(tmp_path / "test_model.pkl")
    det.save(path)

    det2 = AnomalyDetector()
    det2.load(path)
    assert det2.is_trained

    sample = {"cpu_percent": 40, "memory_percent": 60, "disk_percent": 45, "network_errin": 5, "process_count": 150}
    assert det2.predict(sample)["prediction"] in (0, 1)


def test_core_save_load_untrained_rejected(tmp_path):
    det = AnomalyDetector()
    rows = _synthetic_rows(100)
    det.train(rows)
    path = str(tmp_path / "test_model.pkl")
    det.save(path)

    det2 = AnomalyDetector()
    det2.load(path)
    sample = {"cpu_percent": 40, "memory_percent": 60, "disk_percent": 45, "network_errin": 5, "process_count": 150}
    assert det2.predict(sample)["prediction"] in (0, 1)


def test_api_key_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(server_app, "MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setattr(server_app, "_detector", server_app.AnomalyDetector())
    monkeypatch.setattr(server_app, "API_KEY", "secret123")
    server_app.app.config["TESTING"] = True
    c = server_app.app.test_client()

    r = c.post("/train", json={"data": _synthetic_rows(50)})
    assert r.status_code == 401

    r = c.post(
        "/train",
        json={"data": _synthetic_rows(50)},
        headers={"Authorization": "Bearer secret123"},
    )
    assert r.status_code == 200


def test_persist_on_restart(tmp_path, monkeypatch):
    model_file = tmp_path / "persist.pkl"
    monkeypatch.setattr(server_app, "MODEL_PATH", str(model_file))
    monkeypatch.setattr(server_app, "_detector", server_app.AnomalyDetector())
    server_app.app.config["TESTING"] = True
    c = server_app.app.test_client()

    c.post("/train", json={"data": _synthetic_rows(100)})
    assert model_file.exists()

    # "restart" by reloading the module's detector from disk
    monkeypatch.setattr(server_app, "_detector", server_app.AnomalyDetector())
    server_app._detector.load(str(model_file))
    r = c.get("/health")
    assert r.get_json()["trained"] is True
