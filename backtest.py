from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class StrategyParams:
    threshold_long: float = 0.55
    threshold_short: float = 0.45
    tx_cost_bps: float = 5.0  # round-trip bps

def _positions_from_proba(proba: np.ndarray, p: StrategyParams) -> np.ndarray:
    pos = np.zeros_like(proba)
    pos[proba > p.threshold_long] = 1
    pos[proba < p.threshold_short] = -1
    return pos

def backtest(prices: pd.Series, proba: np.ndarray, params: StrategyParams) -> pd.DataFrame:
    df = pd.DataFrame({"close": prices.values}, index=prices.index).copy()
    df["ret"] = df["close"].pct_change().fillna(0.0)

    pos = _positions_from_proba(proba, params)
    df["position"] = np.concatenate([[0], pos[:-1]])  # enter next bar

    changes = np.abs(df["position"].diff().fillna(0.0))
    tc = (changes > 0).astype(float) * (params.tx_cost_bps / 10000.0)

    strat_ret = df["position"] * df["ret"] - tc
    df["strat_cum"] = (1 + strat_ret).cumprod()
    df["bh_cum"] = (1 + df["ret"]).cumprod()
    return df

def performance(df_bt: pd.DataFrame) -> dict:
    strat_curve, bh_curve = df_bt["strat_cum"], df_bt["bh_cum"]
    n_years = max((df_bt.index[-1] - df_bt.index[0]).days, 1) / 365.25
    strat_cagr = strat_curve.iloc[-1] ** (1/n_years) - 1 if n_years > 0 else 0.0
    bh_cagr = bh_curve.iloc[-1] ** (1/n_years) - 1 if n_years > 0 else 0.0
    ret = strat_curve.pct_change().dropna()
    sharpe = np.sqrt(252) * ret.mean() / (ret.std() + 1e-12)
    roll_max = strat_curve.cummax()
    mdd = (strat_curve / roll_max - 1.0).min()
    return {"cagr_strategy": float(strat_cagr), "cagr_buyhold": float(bh_cagr), "sharpe": float(sharpe), "max_drawdown": float(mdd)}
