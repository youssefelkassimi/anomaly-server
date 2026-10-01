"""Core Random Forest anomaly detection logic (framework-agnostic)."""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "anomaly_rf.pkl"


class AnomalyDetector:
    """
    Random Forest-based anomaly detector.

    Modes:
      - Supervised:        provide X (features) and y (0=normal, 1=anomaly).
      - Semi-supervised:   provide only normal samples; synthetic anomalies
                           are generated via feature-range perturbation.

    `contamination` controls aggressiveness. Lower values (0.01-0.05) give
    fewer false positives; higher values (0.1-0.2) catch more anomalies.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        data: Union[List[Dict], np.ndarray],
        labels: Optional[Union[List[int], np.ndarray]] = None,
        test_size: float = 0.2,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Train the Random Forest classifier.

        data: list of dicts OR 2-d array-like.
          If dicts, keys are feature names (must be consistent).
        labels: optional array of ints (0=normal, 1=anomaly).
          When None, all samples are treated as normal and synthetic
          anomalies are generated automatically.
        """
        X, y, feature_names = self._prepare_data(data, labels, feature_names)
        self.feature_names = feature_names

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        self.model.fit(X_train, y_train)
        self._is_trained = True

        y_pred = self.model.predict(X_test)
        acc = float(np.mean(y_pred == y_test))
        cm = confusion_matrix(y_test, y_pred).tolist()

        logger.info("Training complete - accuracy: %.4f", acc)

        return {
            "accuracy": acc,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": cm,
            "n_samples": int(len(y)),
            "n_anomalies": int(y.sum()),
            "feature_importances": dict(
                zip(self.feature_names, self.model.feature_importances_.tolist())
            ),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, data: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
        """Predict whether data points are anomalous."""
        self._ensure_trained()
        X, _ = self._vectorize(data)
        X_scaled = self.scaler.transform(X)

        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        normal_idx = list(self.model.classes_).index(0) if 0 in self.model.classes_ else 0
        anomaly_idx = 1 - normal_idx

        results = []
        for x_scaled, pred, prob_row in zip(X_scaled, preds, probs):
            anomaly_prob = float(prob_row[anomaly_idx])
            results.append(
                {
                    "prediction": int(pred),
                    "label": "anomaly" if pred == 1 else "normal",
                    "probability": anomaly_prob,
                    "feature_contributions": self._top_contributions(x_scaled, anomaly_prob),
                }
            )

        return results[0] if len(results) == 1 else results

    def predict_score(self, data: Union[Dict, List[Dict]]) -> Union[float, List[float]]:
        """Return raw anomaly probability scores (0.0 normal -> 1.0 anomaly)."""
        self._ensure_trained()
        X, _ = self._vectorize(data)
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)
        normal_idx = list(self.model.classes_).index(0) if 0 in self.model.classes_ else 0
        anomaly_idx = 1 - normal_idx
        scores = probs[:, anomaly_idx]

        return float(scores[0]) if scores.shape[0] == 1 else scores.tolist()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        path = path or DEFAULT_MODEL_PATH
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "is_trained": self._is_trained,
            },
            path,
        )
        logger.info("Model saved to %s", path)
        return path

    def load(self, path: Optional[str] = None) -> None:
        path = path or DEFAULT_MODEL_PATH
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle.get("feature_names", [])
        self._is_trained = bundle.get("is_trained", True)
        logger.info("Model loaded from %s", path)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self) -> Dict[str, float]:
        self._ensure_trained()
        importances = self.model.feature_importances_
        return dict(
            sorted(
                zip(self.feature_names, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_data(self, data, labels, feature_names):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            feature_names = list(data[0].keys())
            X = np.array([[row.get(k, 0.0) for k in feature_names] for row in data], dtype=float)
        else:
            X = np.array(data, dtype=float)
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(X.shape[1])]

        if labels is None:
            X, y = self._generate_synthetic_anomalies(X)
        else:
            y = np.array(labels, dtype=int)

        return X, y, feature_names

    def _generate_synthetic_anomalies(self, X_normal):
        n_normal, n_feat = X_normal.shape
        n_anom = max(int(n_normal * self.contamination / (1 - self.contamination)), 1)

        stds = X_normal.std(axis=0)
        stds[stds == 0] = 1.0
        rng = np.random.RandomState(self.random_state)

        rows = []
        for _ in range(n_anom):
            row = X_normal[rng.randint(n_normal)].copy()
            k = rng.randint(1, min(3, n_feat) + 1)
            for j in rng.choice(n_feat, k, replace=False):
                row[j] += rng.choice([-1, 1]) * (3 + 3 * rng.rand()) * stds[j]
            rows.append(np.maximum(row, 0.0))

        X = np.vstack([X_normal, np.array(rows)])
        y = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anom, dtype=int)])
        idx = rng.permutation(len(y))
        return X[idx], y[idx]

    def _vectorize(self, data):
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            keys = self.feature_names or list(data[0].keys())
            X = np.array([[row.get(k, 0.0) for k in keys] for row in data], dtype=float)
            return X, keys
        return np.array(data, dtype=float), self.feature_names

    def _top_contributions(self, x_scaled, anomaly_prob, top_k: int = 5):
        if anomaly_prob < 0.5:
            return {}
        importances = self.model.feature_importances_
        deviations = np.abs(x_scaled)
        weighted = importances * deviations
        top_idx = np.argsort(weighted)[::-1][:top_k]
        return {
            self.feature_names[i]: float(weighted[i])
            for i in top_idx
            if i < len(self.feature_names)
        }

    def _ensure_trained(self):
        if not self._is_trained:
            raise RuntimeError("Model is not trained yet. Call train() or load() first.")
