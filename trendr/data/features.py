"""
Feature engineering for Trendr.

All features are computed from OHLCV data with strict look-ahead discipline:
every value at row t uses only information available at close of day t.
The target (next-day positive return) uses t+1 close, so it is naturally
shifted one row forward — the final row is always dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down + 1e-10)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range — normalised by prior close."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean() / (close + 1e-10)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume, z-score normalised over a 20-day rolling window."""
    direction = np.sign(close.diff()).fillna(0)
    obv_raw = (direction * volume).cumsum()
    obv_z = (obv_raw - obv_raw.rolling(20).mean()) / (obv_raw.rolling(20).std() + 1e-10)
    return obv_z


def _volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    return (volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-10)


def _cyclical(series: pd.Series, max_val: int) -> tuple[pd.Series, pd.Series]:
    """Encode a cyclic integer feature as (sin, cos) to preserve continuity."""
    rad = 2 * np.pi * series / max_val
    return np.sin(rad), np.cos(rad)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features from a raw OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date, open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with feature columns and a ``target`` column.
        Rows with NaN / inf values and the final row (no target available) are dropped.
    """
    df = df.copy()

    # Defensive coercion
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (
        df.dropna(subset=["date", "close", "high", "low", "volume"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # ------------------------------------------------------------------
    # Returns & volatility
    # ------------------------------------------------------------------
    df["ret_1"]  = close.pct_change()
    df["ret_5"]  = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["vol_10"] = close.pct_change().rolling(10).std()
    df["vol_20"] = close.pct_change().rolling(20).std()

    # ------------------------------------------------------------------
    # Moving averages & momentum
    # ------------------------------------------------------------------
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    df["sma_10"]          = sma10
    df["sma_20"]          = sma20
    df["ema_12"]          = _ema(close, 12)
    df["ema_26"]          = _ema(close, 26)
    df["sma_ratio_10_20"] = sma10 / (sma20 + 1e-10)
    df["roc_10"]          = close.pct_change(10)

    # Price position relative to 20-day SMA (normalised)
    df["price_vs_sma20"]  = (close - sma20) / (sma20 + 1e-10)

    # ------------------------------------------------------------------
    # RSI & MACD
    # ------------------------------------------------------------------
    df["rsi_14"]   = _rsi(close, 14)
    macd, sig, hist = _macd(close, 12, 26, 9)
    df["macd"]     = macd
    df["macd_sig"] = sig
    df["macd_hist"] = hist

    # ------------------------------------------------------------------
    # Bollinger Bands
    # ------------------------------------------------------------------
    sd20  = close.rolling(20).std()
    upper = sma20 + 2 * sd20
    lower = sma20 - 2 * sd20
    df["bb_width"] = (upper - lower) / (sma20 + 1e-10)
    df["bb_pct_b"] = (close - lower) / (upper - lower + 1e-10)  # 0=lower, 1=upper

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------
    df["atr_14"]    = _atr(high, low, close, 14)
    df["true_range"] = (high - low) / (close.shift(1) + 1e-10)  # kept for compat

    # Daily candle: where did close land in the high-low range?
    df["daily_range_pct"] = (close - low) / (high - low + 1e-10)

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------
    df["volume_zscore"] = _volume_zscore(volume, 20)
    df["obv_zscore"]    = _obv(close, volume)

    # ------------------------------------------------------------------
    # Calendar — cyclical encoding avoids discontinuities
    # ------------------------------------------------------------------
    dow = df["date"].dt.dayofweek   # 0=Mon … 6=Sun
    mth = df["date"].dt.month       # 1 … 12
    df["dow_sin"], df["dow_cos"] = _cyclical(dow, 7)
    df["mth_sin"], df["mth_cos"] = _cyclical(mth, 12)

    # ------------------------------------------------------------------
    # Target: next-day return positive?
    # ------------------------------------------------------------------
    df["target"] = (close.shift(-1) / close - 1.0 > 0.0).astype(int)

    # Final cleanup
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS: list[str] = [
    # Returns & vol
    "ret_1", "ret_5", "ret_10", "vol_10", "vol_20",
    # Averages & momentum
    "sma_10", "sma_20", "ema_12", "ema_26",
    "sma_ratio_10_20", "roc_10", "price_vs_sma20",
    # Oscillators
    "rsi_14", "macd", "macd_sig", "macd_hist",
    # Bollinger
    "bb_width", "bb_pct_b",
    # Volatility
    "atr_14", "true_range", "daily_range_pct",
    # Volume
    "volume_zscore", "obv_zscore",
    # Calendar (cyclical)
    "dow_sin", "dow_cos", "mth_sin", "mth_cos",
]

TARGET_COLUMN: str = "target"
