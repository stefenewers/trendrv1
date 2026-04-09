import os
import pandas as pd
import yfinance as yf

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "raw")

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if present (e.g., ('Open','ETH-USD') -> 'Open_ETH-USD')."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join([str(x) for x in tup if x is not None]).strip()
                      for tup in df.columns.to_list()]
    return df

def _find_col(df: pd.DataFrame, base: str):
    """
    Find a column that either matches `base` (case-insensitive) or
    starts/ends with it separated by underscores (e.g., 'Open_ETH-USD').
    Returns the actual column name or None.
    """
    base_l = base.lower()
    # exact match
    for c in df.columns:
        if str(c).lower() == base_l:
            return c
    # startswith base_
    for c in df.columns:
        cl = str(c).lower()
        if cl.startswith(base_l + "_"):
            return c
    # endswith _base
    for c in df.columns:
        cl = str(c).lower()
        if cl.endswith("_" + base_l):
            return c
    return None

def _sanitize_download(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, coerce dtypes, drop junk rows, and dedupe by date."""
    # Expose datetime index if needed
    if isinstance(df.index, (pd.DatetimeIndex, pd.MultiIndex)) or df.index.name is not None:
        df = df.reset_index()

    # Flatten odd headers (MultiIndex etc.)
    df = _flatten_columns(df)

    # Rename common standard headers first
    df = df.rename(columns={
        "Date": "date",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # Robust DATE selection — pick any column that *begins with* "date"
    date_col = None
    for c in df.columns:
        if str(c).lower().startswith("date"):
            date_col = c
            break
    if date_col is None:
        # fallback to common alternatives
        for cand in ("datetime", "index", "time"):
            if cand in df.columns:
                date_col = cand
                break
    if date_col is None:
        # last resort: the first column if it looks like dates
        cand = df.columns[0]
        if pd.api.types.is_datetime64_any_dtype(df[cand]) or str(cand).lower().startswith(("date","time")):
            date_col = cand
    if date_col is None:
        raise RuntimeError(f"Could not locate a date column in columns={list(df.columns)}")

    # Build normalized frame
    out = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce")})

    # Find price/volume columns (handles 'Open_ETH-USD' etc.)
    for src, tgt in (("open","open"), ("high","high"), ("low","low"),
                     ("close","close"), ("adj close","adj_close"),
                     ("adj_close","adj_close"), ("volume","volume")):
        c = _find_col(df, src)
        if c is not None:
            out[tgt] = df[c]

    # Coerce numerics
    for c in ("open","high","low","close","adj_close","volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Final clean
    out = out[out["date"].notna()]
    if "close" in out.columns and "volume" in out.columns:
        out = out.dropna(subset=["close","volume"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out

def download_prices(symbol: str, start: str = "2016-01-01", interval: str = "1d") -> str:
    """Download OHLCV data and cache to CSV. Returns file path."""
    ensure_dirs()
    path = os.path.abspath(os.path.join(RAW_DIR, f"{symbol}_{interval}.csv"))

    df = yf.download(
        symbol, start=start, interval=interval,
        auto_adjust=False, progress=False, group_by="column", threads=False
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for {symbol}.")

    df = _sanitize_download(df)
    if df.empty:
        raise RuntimeError(f"Sanitized dataframe is empty for {symbol}.")
    df.to_csv(path, index=False)
    return path

def load_prices(symbol: str, interval: str = "1d") -> pd.DataFrame:
    """Load cached CSV into a DataFrame."""
    path = os.path.abspath(os.path.join(RAW_DIR, f"{symbol}_{interval}.csv"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run download first.")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df
