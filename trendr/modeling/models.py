"""
Model pipelines for Trendr.

Three classifiers are supported:

  lgb   — LightGBM (primary, recommended)
  gbc   — sklearn GradientBoostingClassifier (comparison)
  dummy — DummyClassifier (random-chance baseline)

Tree models do not require feature scaling, so no StandardScaler is used.
This makes SHAP TreeExplainer work directly on the input feature space.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

_PACKAGE_ROOT  = os.path.dirname(os.path.dirname(__file__))
ARTIFACT_DIR   = os.path.abspath(os.path.join(_PACKAGE_ROOT, "..", "artifacts", "models"))
REPORT_DIR     = os.path.abspath(os.path.join(_PACKAGE_ROOT, "..", "artifacts", "reports"))


def _ensure_dirs() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_pipeline(model_name: str = "lgb") -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Return a (pipeline, param_grid) tuple for ``GridSearchCV``.

    Tree models do not require scaling — features enter the classifier raw.
    """
    if model_name == "lgb":
        pipe = Pipeline(
            [("clf", LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1))]
        )
        grid: Dict[str, Any] = {
            "clf__n_estimators":  [200, 400],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth":     [4, 6],
            "clf__num_leaves":    [15, 31],
            "clf__reg_lambda":    [0.1, 1.0],
        }

    elif model_name == "gbc":
        pipe = Pipeline(
            [("clf", GradientBoostingClassifier(random_state=42))]
        )
        grid = {
            "clf__n_estimators":  [150, 250],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth":     [2, 3],
        }

    elif model_name == "dummy":
        pipe = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
        grid = {}

    else:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose: lgb, gbc, dummy.")

    return pipe, grid


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_with_cv(
    X,
    y,
    cv,
    model_name: str = "lgb",
    scoring: str = "roc_auc",
) -> GridSearchCV:
    """
    Fit a model via GridSearchCV with time-series cross-validation.

    Returns the fitted ``GridSearchCV`` object (best estimator is accessible
    via ``.best_estimator_``).
    """
    pipe, grid = build_pipeline(model_name)

    if not grid:
        # DummyClassifier — no grid, just fit directly
        pipe.fit(X, y)
        return pipe  # type: ignore[return-value]

    gs = GridSearchCV(
        pipe,
        grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    gs.fit(X, y)
    logger.info("Best params (%s): %s  |  CV score: %.4f", model_name, gs.best_params_, gs.best_score_)
    return gs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Return a dictionary of classification metrics for train and test splits.

    Parameters
    ----------
    threshold : float
        Decision threshold applied to predicted probabilities.
    """
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test  = model.predict_proba(X_test)[:, 1]
    pred_train  = (proba_train >= threshold).astype(int)
    pred_test   = (proba_test  >= threshold).astype(int)

    return {
        "acc_train":     float(accuracy_score(y_train, pred_train)),
        "acc_test":      float(accuracy_score(y_test,  pred_test)),
        "prec_test":     float(precision_score(y_test, pred_test,  zero_division=0)),
        "recall_test":   float(recall_score(y_test,    pred_test,  zero_division=0)),
        "roc_auc_train": float(roc_auc_score(y_train,  proba_train)),
        "roc_auc_test":  float(roc_auc_score(y_test,   proba_test)),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, symbol: str, interval: str = "1d") -> str:
    _ensure_dirs()
    path = os.path.join(ARTIFACT_DIR, f"model_{symbol}_{interval}.joblib")
    joblib.dump(model, path)
    logger.info("Model saved: %s", path)
    return path


def load_model(symbol: str, interval: str = "1d") -> Any:
    path = os.path.join(ARTIFACT_DIR, f"model_{symbol}_{interval}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at {path}. Run: trendr train --symbol {symbol}"
        )
    return joblib.load(path)


def save_metrics(
    metrics: Dict[str, Any],
    symbol: str,
    interval: str = "1d",
    model_name: Optional[str] = None,
) -> str:
    _ensure_dirs()
    suffix = f"_{model_name}" if model_name else ""
    path = os.path.join(REPORT_DIR, f"metrics_{symbol}_{interval}{suffix}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_metrics(
    symbol: str,
    interval: str = "1d",
    model_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    suffix = f"_{model_name}" if model_name else ""
    path = os.path.join(REPORT_DIR, f"metrics_{symbol}_{interval}{suffix}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
