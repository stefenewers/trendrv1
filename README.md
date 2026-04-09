# Trendr — ML Crypto Price Direction Signals

**Live demo → [trendr.streamlit.app](https://trendr.streamlit.app)**

A production-quality ML pipeline for predicting next-day crypto price direction
(ETH, BTC, SOL) with a fully interactive Streamlit dashboard — including
SHAP explainability, walk-forward validation, and an embedded technical
case study.

---

## What it does

1. **Downloads** daily OHLCV data via yfinance
2. **Engineers 27 features** — momentum, volatility, volume divergence, oscillators, and cyclical calendar encodings
3. **Trains a LightGBM classifier** (benchmarked against GBC and a random baseline) using time-series cross-validation
4. **Evaluates honestly** — proper train/val/test split, walk-forward validation across 5 folds, SHAP feature importance, ROC curve, calibration curve
5. **Backtests** a long/short/flat strategy with transaction costs, Sharpe, Calmar, win rate, and profit factor
6. **Deploys** as a 4-tab Streamlit app: Dashboard, Model Analysis, Walk-Forward, Case Study

---

## Project structure

```
trendr/
├── trendr/
│   ├── config.py          # TrendrConfig dataclass, SUPPORTED_SYMBOLS
│   ├── cli.py             # Typer CLI (download / train / train-all / backtest)
│   ├── app.py             # Streamlit app (4 tabs)
│   ├── data/
│   │   ├── downloader.py  # yfinance integration with sanitisation
│   │   └── features.py    # 27-feature engineering pipeline
│   └── modeling/
│       ├── dataset.py     # train/val/test splits, TimeSeriesSplit CV
│       ├── models.py      # LightGBM, GBC, Dummy pipelines + GridSearchCV
│       ├── evaluate.py    # SHAP, ROC, calibration, Sharpe, Calmar, etc.
│       └── backtest.py    # vectorised backtest + walk-forward engine
├── tests/
│   ├── test_features.py   # Feature correctness, look-ahead audit
│   ├── test_backtest.py   # Backtest math, metric edge cases
│   └── test_models.py     # Pipeline build, train/evaluate integration
├── data/
│   ├── raw/               # Cached OHLCV CSVs
│   └── processed/         # Feature matrices
├── artifacts/
│   ├── models/            # Serialised joblib models
│   └── reports/           # JSON metrics, backtest CSVs
├── pyproject.toml
└── requirements.txt
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/stefenewers/trendr.git
cd trendr
pip install -r requirements.txt

# 2. Train all three assets (ETH, BTC, SOL)
python -m trendr.cli train-all

# 3. Launch the app
streamlit run trendr/app.py
```

Or train a single asset:

```bash
python -m trendr.cli download --symbol ETH-USD
python -m trendr.cli train    --symbol ETH-USD --model lgb
python -m trendr.cli backtest --symbol ETH-USD
```

---

## Features engineered

| Family          | Features |
|-----------------|----------|
| Returns         | ret_1, ret_5, ret_10 |
| Volatility      | vol_10, vol_20, atr_14, true_range, daily_range_pct |
| Trend / MA      | sma_10, sma_20, ema_12, ema_26, sma_ratio_10_20, roc_10, price_vs_sma20 |
| Oscillators     | rsi_14, macd, macd_sig, macd_hist |
| Bollinger       | bb_width, bb_pct_b |
| Volume          | volume_zscore, obv_zscore |
| Calendar        | dow_sin, dow_cos, mth_sin, mth_cos |

All features are computed from information available at close of day *t*.
Calendar features use (sin, cos) cyclical encoding to preserve continuity.

---

## Model & evaluation design

**Train / Val / Test split (temporal)**

```
|←──── Train ────→|←── Val ──→|←── Test ──→|
                              ↑
                        touch once only
```

- **Train** — model fitting and CV-based hyperparameter search
- **Val** — threshold tuning and model selection (never used for final reporting)
- **Test** — single honest hold-out evaluation

**Walk-forward validation**

5 expanding folds, each re-training on all prior data. This is the most
accurate simulation of live performance, accounting for regime changes
and distribution shift over time.

---

## Running tests

```bash
pytest tests/ -v
```

The test suite covers feature correctness, anti-lookahead auditing,
backtest math edge cases, and model pipeline integration.

---

## Deployment (Streamlit Community Cloud)

1. Push to GitHub (include trained `artifacts/` or add a startup script).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Set **Main file path** to `trendr/app.py`.
4. Deploy — live URL is shareable and embeddable on your portfolio.

---

## Built by

**Stefen Ewers** · [stefenewers.com](https://stefenewers.com)
