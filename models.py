from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from .data_pipeline import generate_synthetic_dataset
from .feature_engineering import create_feature_matrix

if TF_AVAILABLE:
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
LABEL_MAP = {"healthy": 0, "warning": 1, "fail": 2}


def _numeric_matrix(features: pd.DataFrame) -> np.ndarray:
    return features.select_dtypes(include=[np.number]).to_numpy(dtype=float)


def _create_sequences(array: np.ndarray, lookback: int) -> np.ndarray:
    sequences = []
    for idx in range(len(array)):
        start = max(0, idx - lookback + 1)
        window = array[start : idx + 1]
        if window.shape[0] < lookback:
            pad = np.zeros((lookback - window.shape[0], array.shape[1]), dtype=float)
            window = np.vstack([pad, window])
        sequences.append(window)
    return np.stack(sequences)


def train_xgboost(features: np.ndarray, targets: np.ndarray, **params: Any) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_jobs=4,
        **params,
    )
    model.fit(features, targets)
    return model


def train_random_forest(features: np.ndarray, targets: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=200, max_depth=8, n_jobs=4, random_state=42)
    model.fit(features, targets)
    return model


def train_isolation_forest(features: np.ndarray) -> IsolationForest:
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(features)
    return model


def train_lstm(
    sequences: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    feature_dim: int,
):
    if not TF_AVAILABLE:
        # Return a simple dummy model when TensorFlow is not available
        class DummyLSTM:
            def predict(self, data, verbose=0):
                # Return simple RUL predictions based on sequence mean
                return np.mean(data, axis=(1, 2)).reshape(-1, 1)
        return DummyLSTM()
    
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, feature_dim)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(sequences, targets, epochs=10, batch_size=32, verbose=0)
    return model


def tune_xgboost(
    features: np.ndarray,
    targets: np.ndarray,
    trials: int = 20,
) -> Dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        clf = XGBClassifier(
            objective="multi:softprob",
            use_label_encoder=False,
            eval_metric="mlogloss",
            **params,
        )
        clf.fit(features, targets)
        preds = clf.predict(features)
        return float(np.mean(preds == targets))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study.best_params


def build_demo_models(
    df: pd.DataFrame | None = None,
    lookback: int = 7,
    optuna_trials: int = 20,
) -> Dict[str, Any]:
    if df is None:
        df = generate_synthetic_dataset(samples=600)
    features, _ = create_feature_matrix(df)
    numeric_features = _numeric_matrix(features)
    labels_source = df.get("label")
    if labels_source is None:
        labels_source = pd.Series("healthy", index=df.index)
    labels = labels_source.map(LABEL_MAP).fillna(0).astype(int).to_numpy()
    rul_source = df.get("rul_days")
    if rul_source is None:
        rul_source = pd.Series(30.0, index=df.index)
    rul_targets = pd.to_numeric(rul_source, errors="coerce").fillna(30.0).to_numpy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_features)
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        dummy_label = int(unique_labels[0]) if unique_labels.size else 0
        xgb_model = DummyClassifier(strategy="constant", constant=dummy_label)
        xgb_model.fit(scaled_features, labels)
    else:
        xgb_params = tune_xgboost(scaled_features, labels, trials=optuna_trials)
        xgb_model = train_xgboost(scaled_features, labels, **xgb_params)
    rf_model = train_random_forest(scaled_features, labels)
    iso_model = train_isolation_forest(scaled_features)
    sequences = _create_sequences(scaled_features, lookback)
    lstm_model = train_lstm(sequences, rul_targets, lookback, scaled_features.shape[1])

    iso_scores = iso_model.score_samples(scaled_features)
    anomaly_threshold = float(np.percentile(iso_scores, 5))

    models = {
        "xgb": xgb_model,
        "rf": rf_model,
        "iso": iso_model,
        "lstm": lstm_model,
        "scaler": scaler,
        "lookback": lookback,
        "anomaly_threshold": anomaly_threshold,
        "rul_reference": max(1.0, float(np.median(rul_targets))),
    }
    return models


def _class_probability(
    estimator: Any, scaled_features: np.ndarray, target_class: int
) -> np.ndarray:
    if not hasattr(estimator, "predict_proba") or not hasattr(estimator, "classes_"):
        return np.zeros(scaled_features.shape[0])
    proba = estimator.predict_proba(scaled_features)
    classes = estimator.classes_
    if target_class in classes:
        idx = int(np.where(classes == target_class)[0][0])
        return proba[:, idx]
    return np.zeros(scaled_features.shape[0])


def ensemble_predict(
    models: Dict[str, Any],
    features: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    numeric_features = features.select_dtypes(include=[np.number]).fillna(0.0)
    scaled = models["scaler"].transform(numeric_features)
    lookback = int(models.get("lookback", 7))

    fail_class = LABEL_MAP["fail"]
    xgb_prob = _class_probability(models["xgb"], scaled, fail_class)
    rf_prob = _class_probability(models["rf"], scaled, fail_class)
    sequences = _create_sequences(scaled, lookback)
    rul_pred = models["lstm"].predict(sequences, verbose=0).flatten()
    rul_norm = np.clip(rul_pred / max(1e-3, models.get("rul_reference", 30.0)), 0.0, 1.0)

    anomaly_scores = models["iso"].score_samples(scaled)
    anomaly_boost = np.where(anomaly_scores < models["anomaly_threshold"], 0.15, 0.0)

    risk_score = 0.6 * xgb_prob + 0.25 * rf_prob + 0.15 * (1.0 - rul_norm)
    risk_score = np.clip(risk_score + anomaly_boost, 0.0, 1.0)

    risk_status = np.select(
        [risk_score >= 0.8, risk_score >= 0.5, risk_score >= 0.2],
        ["FAIL", "WARN", "MONITOR"],
        default="HEALTHY",
    )

    result = metadata.reset_index(drop=True).copy()
    result["risk_score"] = risk_score
    result["status"] = risk_status
    result["xgb_fail_prob"] = xgb_prob
    result["rf_fail_prob"] = rf_prob
    result["rul_days"] = np.clip(rul_pred, 0, None)
    result["anomaly_score"] = anomaly_scores
    return result
