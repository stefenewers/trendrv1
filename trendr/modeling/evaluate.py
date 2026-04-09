"""
Extended evaluation utilities for Trendr.

Covers both classification metrics (confusion matrix, calibration, ROC)
and trading-strategy metrics (Sharpe, Calmar, win rate, profit factor).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def make_confusion_df(y_true, y_proba, thresh: float = 0.5) -> pd.DataFrame:
    """Return a labelled confusion-matrix DataFrame."""
    y_pred = (y_proba >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return pd.DataFrame(
        {"metric": ["TN", "FP", "FN", "TP"], "value": [tn, fp, fn, tp]}
    )


def text_report(y_true, y_proba, thresh: float = 0.5) -> str:
    y_pred = (y_proba >= thresh).astype(int)
    return classification_report(y_true, y_pred, digits=3)


def get_roc_curve(y_true, y_proba) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) arrays for a ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return fpr, tpr


def get_calibration_curve(
    y_true, y_proba, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean_predicted_prob, fraction_positives) for a reliability diagram."""
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return mean_pred, frac_pos


# ---------------------------------------------------------------------------
# Trading-strategy metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio from a series of daily strategy returns."""
    std = daily_returns.std()
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * daily_returns.mean() / std)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Peak-to-trough maximum drawdown (negative number)."""
    roll_max = equity_curve.cummax()
    dd = equity_curve / roll_max - 1.0
    return float(dd.min())


def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar = CAGR / |max drawdown|.
    Returns 0 if max drawdown is negligible.
    """
    n_years = max(len(equity_curve) / periods_per_year, 1e-6)
    cagr = equity_curve.iloc[-1] ** (1 / n_years) - 1
    mdd  = abs(max_drawdown(equity_curve))
    if mdd < 1e-6:
        return 0.0
    return float(cagr / mdd)


def win_rate(position: pd.Series, ret: pd.Series) -> float:
    """
    Fraction of trades that were profitable.
    A 'trade' is any day with a non-zero position.
    """
    active = position != 0
    if active.sum() == 0:
        return 0.0
    trade_ret = (position * ret)[active]
    return float((trade_ret > 0).mean())


def profit_factor(position: pd.Series, ret: pd.Series) -> float:
    """
    Gross profit / gross loss.  >1.0 means the strategy makes money overall.
    """
    trade_ret = position * ret
    gross_profit = trade_ret[trade_ret > 0].sum()
    gross_loss   = abs(trade_ret[trade_ret < 0].sum())
    if gross_loss < 1e-12:
        return float("inf")
    return float(gross_profit / gross_loss)


def full_strategy_metrics(
    equity_curve: pd.Series,
    position: pd.Series,
    ret: pd.Series,
    bh_curve: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Compute a complete set of strategy performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series   Cumulative equity (starts at 1.0).
    position     : pd.Series   Daily position {-1, 0, 1}.
    ret          : pd.Series   Raw daily returns of the underlying.
    bh_curve     : pd.Series   Buy-and-hold cumulative equity (optional).
    """
    daily_strat_ret = equity_curve.pct_change().dropna()
    n_years = max(len(equity_curve) / 252, 1e-6)

    metrics: Dict[str, float] = {
        "cagr_strategy": float(equity_curve.iloc[-1] ** (1 / n_years) - 1),
        "sharpe":        sharpe_ratio(daily_strat_ret),
        "max_drawdown":  max_drawdown(equity_curve),
        "calmar":        calmar_ratio(equity_curve),
        "win_rate":      win_rate(position, ret),
        "profit_factor": profit_factor(position, ret),
    }

    if bh_curve is not None:
        metrics["cagr_buyhold"] = float(bh_curve.iloc[-1] ** (1 / n_years) - 1)

    return metrics


# ---------------------------------------------------------------------------
# SHAP feature importance (tree models only)
# ---------------------------------------------------------------------------

def shap_feature_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with mean |SHAP value| per feature, sorted descending.

    Works with LightGBM and GradientBoostingClassifier pipelines that expose
    a ``clf`` step directly (no scaler in front of the tree).

    Parameters
    ----------
    model : fitted sklearn Pipeline or GridSearchCV
    X     : feature matrix (the same X used during evaluation)
    """
    try:
        import shap  # optional dependency guard
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    # Unwrap GridSearchCV → Pipeline → classifier
    estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
    clf = estimator.named_steps.get("clf", estimator)

    explainer   = shap.TreeExplainer(clf)
    shap_matrix = explainer.shap_values(X)

    # For binary classification some libraries return a list [neg, pos]
    if isinstance(shap_matrix, list):
        shap_matrix = shap_matrix[1]

    importance = np.abs(shap_matrix).mean(axis=0)
    return (
        pd.DataFrame({"feature": X.columns.tolist(), "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
