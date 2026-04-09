"""
Dataset splitting utilities for Trendr.

Enforces strict temporal ordering:
    train  →  val  →  test
                      ↑
                 touch once, at the very end

The validation window is used for threshold tuning and model selection.
The test window is the uncontaminated, final hold-out.
"""

from __future__ import annotations

from typing import Tuple
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..data.features import FEATURE_COLUMNS, TARGET_COLUMN


def train_val_test_split(
    df: pd.DataFrame,
    val_size_days: int = 365,
    test_size_days: int = 365,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a features DataFrame into train / val / test by date.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column (datetime).
    val_size_days : int
        Size of the validation window in calendar days.
    test_size_days : int
        Size of the test (hold-out) window in calendar days.

    Returns
    -------
    train, val, test : DataFrames
    """
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_size_days)
    val_start  = test_start - pd.Timedelta(days=val_size_days)

    train = df[df["date"] < val_start].copy()
    val   = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test  = df[df["date"] >= test_start].copy()
    return train, val, test


def train_test_split_time(
    df: pd.DataFrame,
    test_size_days: int = 365,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Legacy two-way split (kept for CLI / backtest compatibility).
    For model evaluation, prefer ``train_val_test_split``.
    """
    cutoff = df["date"].max() - pd.Timedelta(days=test_size_days)
    train  = df[df["date"] <= cutoff].copy()
    test   = df[df["date"] > cutoff].copy()
    return train, test


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix and target vector from a split DataFrame."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def get_timeseries_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Return a ``TimeSeriesSplit`` for in-sample cross-validation."""
    return TimeSeriesSplit(n_splits=n_splits)
