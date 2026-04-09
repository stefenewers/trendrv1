"""
Backtesting engine for Trendr.

Provides:
  backtest()            — vectorised single-period backtest
  performance()         — legacy performance dict (kept for CLI compat)
  walk_forward()        — k-fold walk-forward evaluation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..data.features import FEATURE_COLUMNS, TARGET_COLUMN
from .evaluate import full_strategy_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

@dataclass
class StrategyParams:
    threshold_long:  float = 0.55   # go long above this probability
    threshold_short: float = 0.45   # go short below this probability
    tx_cost_bps:     float = 5.0    # round-trip transaction cost in bps


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def _positions_from_proba(proba: np.ndarray, p: StrategyParams) -> np.ndarray:
    pos = np.zeros_like(proba)
    pos[proba >  p.threshold_long]  =  1.0
    pos[proba <  p.threshold_short] = -1.0
    return pos


def backtest(
    prices: pd.Series,
    proba: np.ndarray,
    params: StrategyParams,
) -> pd.DataFrame:
    """
    Run a vectorised backtest on a price series with model probabilities.

    Positions are entered at the next bar (no look-ahead):
        position[t] = signal derived from close[t-1]

    Transaction costs are deducted on every position change.

    Parameters
    ----------
    prices : pd.Series   Daily close prices (DatetimeIndex).
    proba  : np.ndarray  Model probability of up-move for each bar.
    params : StrategyParams

    Returns
    -------
    pd.DataFrame with columns: close, ret, position, strat_cum, bh_cum
    """
    df = pd.DataFrame({"close": prices.values}, index=prices.index).copy()
    df["ret"] = df["close"].pct_change().fillna(0.0)

    raw_pos = _positions_from_proba(proba, params)
    df["position"] = np.concatenate([[0.0], raw_pos[:-1]])  # enter next bar

    changes  = df["position"].diff().abs().fillna(0.0)
    tc       = (changes > 0).astype(float) * (params.tx_cost_bps / 10_000.0)

    strat_ret      = df["position"] * df["ret"] - tc
    df["strat_cum"] = (1.0 + strat_ret).cumprod()
    df["bh_cum"]    = (1.0 + df["ret"]).cumprod()
    return df


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------

def performance(df_bt: pd.DataFrame) -> Dict[str, float]:
    """
    Compute a full set of strategy performance metrics.

    Returns a flat dict for JSON serialisation.
    """
    pos   = df_bt["position"]
    ret   = df_bt["ret"]
    strat = df_bt["strat_cum"]
    bh    = df_bt["bh_cum"]

    metrics = full_strategy_metrics(strat, pos, ret, bh_curve=bh)
    return metrics


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward(
    df_feat: pd.DataFrame,
    model_fn,
    params: StrategyParams,
    n_splits: int = 5,
    min_train_rows: int = 200,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation.

    At each fold:
      1. Train ``model_fn(X_train, y_train)`` on all data up to the split.
      2. Generate signals on the next out-of-sample window.
      3. Run ``backtest()`` and record performance.

    Parameters
    ----------
    df_feat    : pd.DataFrame   Features DataFrame (sorted by date).
    model_fn   : callable       ``model_fn(X_train, y_train) → fitted model``
    params     : StrategyParams
    n_splits   : int            Number of walk-forward folds.
    min_train_rows : int        Skip folds where training data is too sparse.

    Returns
    -------
    pd.DataFrame  One row per fold with columns:
        fold, train_start, train_end, test_start, test_end,
        cagr_strategy, cagr_buyhold, sharpe, max_drawdown,
        calmar, win_rate, profit_factor, n_train, n_test
    """
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    X_all  = df_feat[FEATURE_COLUMNS]
    y_all  = df_feat[TARGET_COLUMN]
    dates  = df_feat["date"].values
    closes = df_feat["close"].values

    rows: List[Dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        if len(train_idx) < min_train_rows:
            logger.info("Fold %d: skipping (only %d train rows)", fold_idx, len(train_idx))
            continue

        X_tr, y_tr = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_te, y_te = X_all.iloc[test_idx],  y_all.iloc[test_idx]

        try:
            fitted = model_fn(X_tr, y_tr)
            proba  = fitted.predict_proba(X_te)[:, 1]
        except Exception as exc:
            logger.warning("Fold %d: training/prediction failed — %s", fold_idx, exc)
            continue

        price_series = pd.Series(
            closes[test_idx],
            index=pd.to_datetime(dates[test_idx]),
            name="close",
        )
        bt_df = backtest(price_series, proba, params)
        perf  = performance(bt_df)

        rows.append({
            "fold":          fold_idx + 1,
            "train_start":   pd.Timestamp(dates[train_idx[0]]).date(),
            "train_end":     pd.Timestamp(dates[train_idx[-1]]).date(),
            "test_start":    pd.Timestamp(dates[test_idx[0]]).date(),
            "test_end":      pd.Timestamp(dates[test_idx[-1]]).date(),
            "n_train":       len(train_idx),
            "n_test":        len(test_idx),
            **perf,
        })

    return pd.DataFrame(rows)
