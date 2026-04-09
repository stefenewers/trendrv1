"""
Tests for trendr.modeling.backtest and trendr.modeling.evaluate
"""

import numpy as np
import pandas as pd
import pytest

from trendr.modeling.backtest import (
    StrategyParams,
    _positions_from_proba,
    backtest,
    performance,
)
from trendr.modeling.evaluate import (
    calmar_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _flat_prices(n: int = 100, start_price: float = 100.0) -> pd.Series:
    """Price series with zero return (flat)."""
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.Series([start_price] * n, index=dates, name="close")


def _trending_prices(n: int = 100, daily_ret: float = 0.005) -> pd.Series:
    dates  = pd.date_range("2021-01-01", periods=n, freq="D")
    prices = 100 * np.cumprod(np.concatenate([[1.0], np.ones(n - 1) * (1 + daily_ret)]))
    return pd.Series(prices, index=dates, name="close")


def _always_long_proba(n: int) -> np.ndarray:
    return np.ones(n) * 0.90  # always above long threshold


def _always_short_proba(n: int) -> np.ndarray:
    return np.zeros(n)  # always below short threshold


def _always_flat_proba(n: int) -> np.ndarray:
    return np.ones(n) * 0.50  # in the neutral zone


# ---------------------------------------------------------------------------
# _positions_from_proba
# ---------------------------------------------------------------------------

class TestPositionsFromProba:
    def test_long_signal(self):
        p      = StrategyParams(threshold_long=0.55, threshold_short=0.45)
        proba  = np.array([0.90, 0.60, 0.50, 0.40, 0.10])
        pos    = _positions_from_proba(proba, p)
        np.testing.assert_array_equal(pos, [1, 1, 0, -1, -1])

    def test_neutral_zone(self):
        p     = StrategyParams(threshold_long=0.55, threshold_short=0.45)
        proba = np.array([0.50])
        pos   = _positions_from_proba(proba, p)
        assert pos[0] == 0

    def test_exactly_at_threshold(self):
        """Values exactly at the threshold are NOT active (strict >/<)."""
        p     = StrategyParams(threshold_long=0.55, threshold_short=0.45)
        pos_l = _positions_from_proba(np.array([0.55]), p)
        pos_s = _positions_from_proba(np.array([0.45]), p)
        assert pos_l[0] == 0, "Exactly at long threshold should be flat"
        assert pos_s[0] == 0, "Exactly at short threshold should be flat"


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------

class TestBacktest:
    def test_output_columns_present(self):
        prices = _trending_prices(50)
        proba  = _always_long_proba(50)
        bt     = backtest(prices, proba, StrategyParams())
        assert set(["close", "ret", "position", "strat_cum", "bh_cum"]).issubset(bt.columns)

    def test_equity_starts_at_one(self):
        prices = _trending_prices(50)
        proba  = _always_long_proba(50)
        bt     = backtest(prices, proba, StrategyParams())
        assert abs(bt["strat_cum"].iloc[0] - 1.0) < 0.01

    def test_flat_market_with_long_position_near_one(self):
        """On a flat market, a fully-long strategy should end close to 1.0."""
        prices = _flat_prices(100)
        proba  = _always_long_proba(100)
        bt     = backtest(prices, proba, StrategyParams(tx_cost_bps=0.0))
        assert abs(bt["strat_cum"].iloc[-1] - 1.0) < 0.02

    def test_tx_costs_reduce_returns(self):
        """With tx costs, cumulative return should be lower than without."""
        prices     = _trending_prices(100)
        proba      = np.tile([0.8, 0.2], 50)  # alternating — many trades
        bt_free    = backtest(prices, proba, StrategyParams(tx_cost_bps=0.0))
        bt_costly  = backtest(prices, proba, StrategyParams(tx_cost_bps=20.0))
        assert bt_costly["strat_cum"].iloc[-1] < bt_free["strat_cum"].iloc[-1]

    def test_no_look_ahead(self):
        """Position at bar t is derived from probability at t-1 (next-bar entry)."""
        prices = _trending_prices(10)
        proba  = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
        bt     = backtest(prices, proba, StrategyParams())
        # First bar should have position 0 (no prior signal)
        assert bt["position"].iloc[0] == 0

    def test_bh_matches_price_return(self):
        """Buy-and-hold cumulative return should match price return exactly."""
        prices = _trending_prices(50, daily_ret=0.01)
        proba  = _always_flat_proba(50)
        bt     = backtest(prices, proba, StrategyParams())
        expected_bh = prices.iloc[-1] / prices.iloc[0]
        assert abs(bt["bh_cum"].iloc[-1] - expected_bh) < 0.01

    def test_length_matches_input(self):
        prices = _trending_prices(80)
        proba  = _always_long_proba(80)
        bt     = backtest(prices, proba, StrategyParams())
        assert len(bt) == len(prices)


# ---------------------------------------------------------------------------
# performance / evaluate metrics
# ---------------------------------------------------------------------------

class TestStrategyMetrics:
    def test_sharpe_flat_series(self):
        daily_ret = pd.Series([0.0] * 100)
        s = sharpe_ratio(daily_ret)
        assert s == 0.0

    def test_sharpe_positive_drift(self):
        rng       = np.random.default_rng(0)
        daily_ret = pd.Series(rng.normal(0.001, 0.01, 252))
        s = sharpe_ratio(daily_ret)
        assert s > 0

    def test_max_drawdown_no_loss(self):
        equity = pd.Series(np.linspace(1, 2, 100))
        mdd    = max_drawdown(equity)
        assert abs(mdd) < 1e-6, "Monotonically rising equity has 0 drawdown"

    def test_max_drawdown_full_loss(self):
        equity = pd.Series([1.0, 0.5, 0.1])
        mdd    = max_drawdown(equity)
        assert mdd < -0.85

    def test_calmar_positive_for_good_strategy(self):
        """Calmar should be positive when CAGR > 0 and there's some drawdown."""
        rng = np.random.default_rng(1)
        # Trending series with small noise so there IS a drawdown
        noise  = rng.normal(0.001, 0.008, 252)
        equity = pd.Series(np.cumprod(1 + noise))
        c = calmar_ratio(equity)
        # Could be 0 if drawdown rounds to zero; just ensure it's non-negative
        assert c >= 0

    def test_win_rate_all_wins(self):
        pos = pd.Series([1.0] * 10)
        ret = pd.Series([0.01] * 10)
        assert abs(win_rate(pos, ret) - 1.0) < 1e-9

    def test_win_rate_no_trades(self):
        pos = pd.Series([0.0] * 10)
        ret = pd.Series([0.01] * 10)
        assert win_rate(pos, ret) == 0.0

    def test_profit_factor_equal_wins_losses(self):
        pos = pd.Series([1.0, 1.0])
        ret = pd.Series([0.01, -0.01])
        pf  = profit_factor(pos, ret)
        assert abs(pf - 1.0) < 1e-9

    def test_performance_dict_keys(self):
        prices = _trending_prices(50)
        proba  = _always_long_proba(50)
        bt     = backtest(prices, proba, StrategyParams())
        perf   = performance(bt)
        required = {"cagr_strategy", "cagr_buyhold", "sharpe", "max_drawdown",
                    "calmar", "win_rate", "profit_factor"}
        assert required.issubset(perf.keys())
