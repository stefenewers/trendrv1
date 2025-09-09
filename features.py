import numpy as np
import pandas as pd

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down + 1e-10)
    return 100 - (100 / (1 + rs))

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Defensive typing + cleanup
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close", "high", "low", "volume"]).sort_values("date").reset_index(drop=True)

    close, high, low = df["close"], df["high"], df["low"]

    # Returns & volatility
    df["ret_1"] = close.pct_change()
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["vol_10"] = close.pct_change().rolling(10).std()
    df["vol_20"] = close.pct_change().rolling(20).std()

    # Averages & momentum
    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["ema_12"] = _ema(close, 12)
    df["ema_26"] = _ema(close, 26)
    df["sma_ratio_10_20"] = df["sma_10"] / (df["sma_20"] + 1e-10)
    df["roc_10"] = close.pct_change(10)

    # RSI & MACD
    df["rsi_14"] = _rsi(close, 14)
    macd, sig, hist = _macd(close, 12, 26, 9)
    df["macd"] = macd
    df["macd_sig"] = sig
    df["macd_hist"] = hist

    # Bollinger width
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    upper, lower = ma20 + 2*sd20, ma20 - 2*sd20
    df["bb_width"] = (upper - lower) / (ma20 + 1e-10)

    # True range proxy
    df["true_range"] = (high - low) / (close.shift(1) + 1e-10)

    # Calendar features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Target: next-day positive return?
    df["target"] = (df["close"].shift(-1) / df["close"] - 1.0 > 0.0).astype(int)

    # Final cleanup
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df

FEATURE_COLUMNS = [
    "ret_1","ret_5","ret_10","vol_10","vol_20",
    "sma_10","sma_20","ema_12","ema_26","sma_ratio_10_20",
    "roc_10","rsi_14","macd","macd_sig","macd_hist",
    "bb_width","true_range","dayofweek","month"
]
TARGET_COLUMN = "target"
