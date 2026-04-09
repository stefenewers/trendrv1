from dataclasses import dataclass
from typing import List

SUPPORTED_SYMBOLS: List[str] = ["ETH-USD", "BTC-USD", "SOL-USD"]

SYMBOL_META: dict = {
    "ETH-USD": {"name": "Ethereum", "start": "2017-11-01", "color": "#627EEA"},
    "BTC-USD": {"name": "Bitcoin",  "start": "2017-01-01", "color": "#F7931A"},
    "SOL-USD": {"name": "Solana",   "start": "2020-04-01", "color": "#9945FF"},
}


@dataclass
class TrendrConfig:
    symbol: str = "ETH-USD"
    interval: str = "1d"
    horizon: int = 1                  # predict next-day move

    # Train / val / test split
    val_size_days: int = 365          # validation window (threshold tuning)
    test_size_days: int = 365         # held-out test window — touch once

    # Cross-validation
    n_cv_splits: int = 5

    # Signal thresholds (tuned on val set)
    threshold_long: float = 0.55     # enter long above this probability
    threshold_short: float = 0.45    # enter short below this probability

    # Backtest
    tx_cost_bps: float = 5.0         # round-trip transaction cost in bps

    # Walk-forward
    wf_n_splits: int = 5             # number of walk-forward folds
