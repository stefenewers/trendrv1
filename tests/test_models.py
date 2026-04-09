"""
Tests for trendr.modeling.models and trendr.modeling.dataset
"""

import numpy as np
import pandas as pd
import pytest

from trendr.data.features import build_features, FEATURE_COLUMNS, TARGET_COLUMN
from trendr.modeling.dataset import (
    train_val_test_split,
    train_test_split_time,
    get_X_y,
    get_timeseries_cv,
)
from trendr.modeling.models import build_pipeline, fit_with_cv, evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_features(n: int = 250, seed: int = 7) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
    high  = close * (1 + rng.uniform(0, 0.02, n))
    low   = close * (1 - rng.uniform(0, 0.02, n))
    vol   = rng.integers(500_000, 5_000_000, n).astype(float)
    raw   = pd.DataFrame(
        {"date": dates, "open": close, "high": high, "low": low,
         "close": close, "adj_close": close, "volume": vol}
    )
    return build_features(raw)


@pytest.fixture(scope="module")
def feature_df():
    return _make_features(250)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDatasetSplits:
    def test_three_way_split_non_overlapping(self, feature_df):
        train, val, test = train_val_test_split(feature_df, val_size_days=60, test_size_days=60)
        assert train["date"].max() < val["date"].min()
        assert val["date"].max()   < test["date"].min()

    def test_three_way_split_covers_all_data(self, feature_df):
        train, val, test = train_val_test_split(feature_df, val_size_days=60, test_size_days=60)
        total = len(train) + len(val) + len(test)
        assert total == len(feature_df)

    def test_two_way_split_non_overlapping(self, feature_df):
        train, test = train_test_split_time(feature_df, test_size_days=60)
        assert train["date"].max() < test["date"].min()

    def test_get_X_y_shapes(self, feature_df):
        X, y = get_X_y(feature_df)
        assert X.shape == (len(feature_df), len(FEATURE_COLUMNS))
        assert y.shape == (len(feature_df),)

    def test_get_X_y_columns(self, feature_df):
        X, _ = get_X_y(feature_df)
        assert list(X.columns) == FEATURE_COLUMNS

    def test_timeseries_cv_n_splits(self):
        tscv = get_timeseries_cv(n_splits=4)
        assert tscv.n_splits == 4


# ---------------------------------------------------------------------------
# Pipeline / model tests
# ---------------------------------------------------------------------------

class TestPipelines:
    @pytest.mark.parametrize("model_name", ["lgb", "gbc", "dummy"])
    def test_pipeline_builds(self, model_name):
        pipe, grid = build_pipeline(model_name)
        assert pipe is not None

    @pytest.mark.parametrize("model_name", ["lgb", "gbc"])
    def test_pipeline_has_clf_step(self, model_name):
        pipe, _ = build_pipeline(model_name)
        assert "clf" in pipe.named_steps

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model_name"):
            build_pipeline("xgboost_future")


class TestFitAndEvaluate:
    """Integration tests: fit on small synthetic data, check metrics."""

    @pytest.fixture(scope="class")
    def fitted_lgb(self, feature_df):
        train, val, test = train_val_test_split(
            feature_df, val_size_days=60, test_size_days=60
        )
        X_tr, y_tr = get_X_y(train)
        cv = get_timeseries_cv(3)
        return fit_with_cv(X_tr, y_tr, cv, model_name="lgb"), train, val, test

    def test_lgb_predict_proba_shape(self, fitted_lgb, feature_df):
        gs, train, val, test = fitted_lgb
        X_te, _ = get_X_y(test)
        proba = gs.predict_proba(X_te)
        assert proba.shape == (len(test), 2)

    def test_lgb_proba_sums_to_one(self, fitted_lgb, feature_df):
        gs, train, val, test = fitted_lgb
        X_te, _ = get_X_y(test)
        proba = gs.predict_proba(X_te)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_evaluate_returns_required_keys(self, fitted_lgb, feature_df):
        gs, train, val, test = fitted_lgb
        X_tr, y_tr = get_X_y(train)
        X_te, y_te = get_X_y(test)
        metrics = evaluate(gs, X_tr, y_tr, X_te, y_te)
        required = {"acc_train", "acc_test", "prec_test", "recall_test",
                    "roc_auc_train", "roc_auc_test"}
        assert required.issubset(metrics.keys())

    def test_evaluate_roc_auc_above_zero(self, fitted_lgb, feature_df):
        gs, train, val, test = fitted_lgb
        X_tr, y_tr = get_X_y(train)
        X_te, y_te = get_X_y(test)
        metrics = evaluate(gs, X_tr, y_tr, X_te, y_te)
        assert metrics["roc_auc_test"] > 0

    def test_dummy_baseline_roc_near_half(self, feature_df):
        """DummyClassifier should have ROC-AUC close to 0.5."""
        train, val, test = train_val_test_split(
            feature_df, val_size_days=60, test_size_days=60
        )
        X_tr, y_tr = get_X_y(train)
        X_te, y_te = get_X_y(test)

        from trendr.modeling.models import build_pipeline
        pipe, _ = build_pipeline("dummy")
        pipe.fit(X_tr, y_tr)
        metrics = evaluate(pipe, X_tr, y_tr, X_te, y_te)
        # Dummy should not be dramatically better or worse than chance
        assert 0.3 < metrics["roc_auc_test"] < 0.7
