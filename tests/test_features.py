"""
Tests for trendr.data.features
"""

import numpy as np
import pandas as pd
import pytest

from trendr.data.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    _rsi,
    _ema,
    _atr,
    _obv,
    _volume_zscore,
    _cyclical,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, size=n))
    high  = close * (1 + rng.uniform(0, 0.02, size=n))
    low   = close * (1 - rng.uniform(0, 0.02, size=n))
    open_ = close * (1 + rng.normal(0, 0.01, size=n))
    vol   = rng.integers(1_000_000, 10_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low,
         "close": close, "adj_close": close, "volume": vol}
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv(120)


@pytest.fixture
def features_df(ohlcv) -> pd.DataFrame:
    return build_features(ohlcv)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_rsi_range(self, ohlcv):
        rsi = _rsi(ohlcv["close"], 14).dropna()
        assert rsi.between(0, 100).all(), "RSI must be in [0, 100]"

    def test_ema_convergence(self, ohlcv):
        """EMA of a flat series should equal that constant."""
        flat = pd.Series([10.0] * 50)
        ema  = _ema(flat, span=5)
        assert abs(ema.iloc[-1] - 10.0) < 1e-6

    def test_atr_positive(self, ohlcv):
        atr = _atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14).dropna()
        assert (atr >= 0).all(), "ATR must be non-negative"

    def test_obv_shape(self, ohlcv):
        obv = _obv(ohlcv["close"], ohlcv["volume"])
        assert len(obv) == len(ohlcv)

    def test_volume_zscore_mean_approx_zero(self, ohlcv):
        vz = _volume_zscore(ohlcv["volume"], window=20).dropna()
        assert abs(vz.mean()) < 1.0  # not exactly 0 but roughly centred

    def test_cyclical_encoding_unit_circle(self):
        values = pd.Series(range(7))
        sin_, cos_ = _cyclical(values, 7)
        norms = (sin_ ** 2 + cos_ ** 2).values
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)

    def test_cyclical_continuity(self):
        """
        Cyclical encoding should be continuous at the boundary.
        For a period-7 cycle, value 7 should encode identically to value 0.
        """
        v0 = pd.Series([0])
        v7 = pd.Series([7])
        sin0, cos0 = _cyclical(v0, 7)
        sin7, cos7 = _cyclical(v7, 7)
        np.testing.assert_allclose(sin0.values, sin7.values, atol=1e-9)
        np.testing.assert_allclose(cos0.values, cos7.values, atol=1e-9)


# ---------------------------------------------------------------------------
# build_features tests
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_all_feature_columns_present(self, features_df):
        missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
        assert not missing, f"Missing feature columns: {missing}"

    def test_target_column_present(self, features_df):
        assert TARGET_COLUMN in features_df.columns

    def test_target_is_binary(self, features_df):
        unique = set(features_df[TARGET_COLUMN].unique())
        assert unique <= {0, 1}, f"Target must be binary, got: {unique}"

    def test_no_nans_after_build(self, features_df):
        assert not features_df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any()

    def test_no_infs_after_build(self, features_df):
        nums = features_df[FEATURE_COLUMNS].select_dtypes(include=[np.number])
        assert not np.isinf(nums.values).any()

    def test_output_shorter_than_input(self, ohlcv, features_df):
        """Feature engineering drops warm-up rows and the last (target) row."""
        assert len(features_df) < len(ohlcv)

    def test_date_column_preserved(self, features_df):
        assert "date" in features_df.columns
        assert pd.api.types.is_datetime64_any_dtype(features_df["date"])

    def test_sorted_by_date(self, features_df):
        dates = features_df["date"].values
        assert (dates[1:] >= dates[:-1]).all(), "Output must be sorted by date"

    def test_bb_pct_b_bounded(self, features_df):
        """Bollinger %B is unbounded but should be roughly in [-1, 3] for normal data."""
        bb = features_df["bb_pct_b"]
        assert bb.between(-2, 4).mean() > 0.95

    def test_rsi_in_range(self, features_df):
        rsi = features_df["rsi_14"]
        assert rsi.between(0, 100).all()

    def test_feature_count(self):
        assert len(FEATURE_COLUMNS) == 27, (
            f"Expected 27 features, got {len(FEATURE_COLUMNS)}. "
            "Update this test if you intentionally add/remove features."
        )

    def test_idempotent(self, ohlcv):
        """Calling build_features twice on the same input should give the same result."""
        df1 = build_features(ohlcv)
        df2 = build_features(ohlcv)
        pd.testing.assert_frame_equal(df1, df2)

    def test_no_lookahead_in_features(self, ohlcv):
        """
        Verify that features at row t do NOT use future close prices.
        Strategy: shift the future close prices by 1, rebuild, and check
        that the feature matrix is identical — if it changes, there's a leak.
        """
        original_features = build_features(ohlcv)[FEATURE_COLUMNS].values

        perturbed = ohlcv.copy()
        # Shift ONLY the last 10 close values (which are future w.r.t. earlier rows)
        perturbed.loc[perturbed.index[:-10], "close"] = ohlcv["close"].iloc[:-10].values
        perturbed_features = build_features(perturbed)[FEATURE_COLUMNS]

        # The features for the first (n-10) rows should be identical
        n = min(len(original_features), len(perturbed_features)) - 10
        np.testing.assert_allclose(
            original_features[:n],
            perturbed_features.values[:n],
            rtol=1e-9,
            err_msg="Features for early rows changed when only future closes were perturbed — possible lookahead!",
        )
