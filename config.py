from dataclasses import dataclass

@dataclass
class TrendrConfig:
    symbol: str = "ETH-USD"
    start: str = "2016-01-01"
    interval: str = "1d"
    horizon: int = 1               # predict next-day move
    threshold_long: float = 0.55   # probability to go long
    threshold_short: float = 0.45  # probability to go short
    tx_cost_bps: float = 5.0       # 5 basis points per trade round trip
    test_size_days: int = 365      # last year as holdout
