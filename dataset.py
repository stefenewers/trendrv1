from typing import Tuple
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from ..data.features import FEATURE_COLUMNS, TARGET_COLUMN

def train_test_split_time(df: pd.DataFrame, test_size_days: int = 365) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].max() - pd.Timedelta(days=test_size_days)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test

def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y

def get_timeseries_cv(n_splits: int = 5):
    return TimeSeriesSplit(n_splits=n_splits)
