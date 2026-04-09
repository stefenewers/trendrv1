"""
Trendr — ML Crypto Signals  |  Streamlit app
=============================================
Four tabs:
  📊 Dashboard       — live equity curve, key metrics, recent signals
  🔬 Model Analysis  — SHAP, ROC, calibration, confusion matrix
  📈 Walk-Forward    — expanding-window out-of-sample performance
  📖 Case Study      — embedded technical write-up
"""

from __future__ import annotations

import os
import sys
import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allows running via `streamlit run trendr/app.py`
# ---------------------------------------------------------------------------
FILE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trendr.config import SUPPORTED_SYMBOLS, SYMBOL_META
from trendr.data.features import FEATURE_COLUMNS
from trendr.modeling.backtest import StrategyParams, backtest, performance, walk_forward
from trendr.modeling.dataset import train_val_test_split, get_X_y
from trendr.modeling.evaluate import (
    get_calibration_curve,
    get_roc_curve,
    make_confusion_df,
    shap_feature_importance,
)
from trendr.modeling.dataset import get_timeseries_cv
from trendr.modeling.models import fit_with_cv, load_model, load_metrics

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Trendr | ML Crypto Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal, clean
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
    .metric-label { font-size: 0.8rem !important; }
    h1 { font-size: 1.8rem !important; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
    .stTabs [data-baseweb="tab-list"] { margin-top: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading features…")
def _load_features(symbol: str, interval: str) -> pd.DataFrame | None:
    path = os.path.join(PROJECT_ROOT, "data", "processed", f"features_{symbol}_{interval}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


@st.cache_resource(show_spinner="Loading model…")
def _load_model(symbol: str, interval: str):
    try:
        return load_model(symbol, interval)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Computing SHAP values…")
def _shap_importance(_model, X: pd.DataFrame) -> pd.DataFrame:
    """Cached SHAP computation — underscore prefix tells Streamlit not to hash _model."""
    return shap_feature_importance(_model, X)


@st.cache_data(show_spinner="Running walk-forward…")
def _walk_forward_cached(symbol: str, interval: str, tl: float, ts: float, tc: float) -> pd.DataFrame | None:
    df = _load_features(symbol, interval)
    model = _load_model(symbol, interval)
    if df is None or model is None:
        return None

    params = StrategyParams(threshold_long=tl, threshold_short=ts, tx_cost_bps=tc)

    # Use the best estimator's model type to re-fit at each fold
    from trendr.modeling.models import build_pipeline
    from trendr.modeling.dataset import get_timeseries_cv as _cv
    model_name = "lgb"  # default; could be inferred from joblib metadata

    def _fit(X_tr, y_tr):
        from trendr.modeling.models import fit_with_cv
        cv = _cv(3)
        return fit_with_cv(X_tr, y_tr, cv, model_name=model_name)

    return walk_forward(df, _fit, params, n_splits=5)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 Trendr")
    st.caption("ML-driven crypto price direction signals")
    st.divider()

    symbol = st.selectbox(
        "Asset",
        SUPPORTED_SYMBOLS,
        format_func=lambda s: f"{SYMBOL_META[s]['name']} ({s})",
    )
    interval = "1d"

    st.divider()
    st.markdown("**Strategy thresholds**")
    threshold_long  = st.slider("Long signal above",  0.50, 0.75, 0.55, 0.01)
    threshold_short = st.slider("Short signal below", 0.25, 0.50, 0.45, 0.01)
    tx_cost_bps     = st.slider("Tx cost (bps, round-trip)", 0.0, 50.0, 5.0, 0.5)

    st.divider()
    st.caption(
        "Built by [Stefen Ewers](https://stefenewers.com) · "
        "[GitHub](https://github.com/stefenewers/trendr)"
    )

# ---------------------------------------------------------------------------
# Load data & model
# ---------------------------------------------------------------------------
asset_color = SYMBOL_META[symbol]["color"]

df_feat = _load_features(symbol, interval)
model   = _load_model(symbol, interval)

_not_trained = df_feat is None or model is None
if _not_trained:
    st.warning(
        f"**{symbol} not yet trained.** Run the following to get started:\n\n"
        f"```bash\npython -m trendr.cli train --symbol {symbol}\n```"
    )

# Split — only if data available
if df_feat is not None:
    train_df, val_df, test_df = train_val_test_split(df_feat)
    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)
    X_test,  y_test  = get_X_y(test_df)

    if model is not None:
        proba_test = model.predict_proba(X_test)[:, 1]
        params     = StrategyParams(
            threshold_long=threshold_long,
            threshold_short=threshold_short,
            tx_cost_bps=tx_cost_bps,
        )
        bt_df = backtest(test_df.set_index("date")["close"], proba_test, params)
        perf  = performance(bt_df)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dash, tab_model, tab_wf, tab_case = st.tabs(
    ["📊 Dashboard", "🔬 Model Analysis", "📈 Walk-Forward", "📖 Case Study"]
)

# ============================================================
# TAB 1 — DASHBOARD
# ============================================================
with tab_dash:
    if _not_trained:
        st.info("Train a model for this asset to see the dashboard.")
        st.stop()

    # Header
    st.markdown(f"### {SYMBOL_META[symbol]['name']} — Test Period Performance")
    st.caption(
        f"Out-of-sample window: "
        f"**{test_df['date'].min().date()}** → **{test_df['date'].max().date()}**  "
        f"({len(test_df)} trading days)"
    )

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Strategy CAGR",  f"{perf['cagr_strategy']*100:.1f}%")
    m2.metric("Buy & Hold CAGR", f"{perf['cagr_buyhold']*100:.1f}%")
    m3.metric("Sharpe",          f"{perf['sharpe']:.2f}")
    m4.metric("Max Drawdown",    f"{perf['max_drawdown']*100:.1f}%")
    m5.metric("Win Rate",        f"{perf['win_rate']*100:.1f}%")
    m6.metric("Profit Factor",   f"{perf['profit_factor']:.2f}")

    st.divider()

    # Equity curve (Plotly)
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["strat_cum"],
        name="Strategy", line=dict(color=asset_color, width=2),
    ))
    fig_eq.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["bh_cum"],
        name="Buy & Hold", line=dict(color="#888888", width=1.5, dash="dot"),
    ))
    fig_eq.update_layout(
        title=f"{symbol} — Equity Curves (test period)",
        xaxis_title="Date", yaxis_title="Cumulative Return (×)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380, margin=dict(t=60, b=40, l=60, r=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Position distribution
    col_a, col_b = st.columns([1, 2])
    with col_a:
        # Fix: build label→color map explicitly so colours never depend on sort order
        _color_map = {"Long": "#22c55e", "Flat": "#94a3b8", "Short": "#ef4444"}
        _pos_label = {1: "Long", 0: "Flat", -1: "Short"}
        _pie_data = (
            bt_df["position"]
            .map(_pos_label)
            .value_counts()
            .reindex(["Long", "Flat", "Short"], fill_value=0)
        )
        fig_pie = go.Figure(go.Pie(
            labels=_pie_data.index.tolist(),
            values=_pie_data.values.tolist(),
            hole=0.45,
            marker_colors=[_color_map[l] for l in _pie_data.index],
        ))
        fig_pie.update_layout(
            title="Signal distribution", height=260,
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=True,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("**Recent signals (last 30 days)**")
        recent = bt_df.tail(30).copy()
        recent["Signal"] = recent["position"].map({1: "🟢 Long", 0: "⬜ Flat", -1: "🔴 Short"})
        recent["Close"]  = recent["close"].map("${:,.2f}".format)
        recent["Daily Return"] = recent["ret"].map("{:+.2%}".format)
        # Strip time component from the DatetimeIndex for clean display
        recent.index = recent.index.date
        recent.index.name = "Date"
        st.dataframe(
            recent[["Close", "Daily Return", "Signal"]],
            use_container_width=True, height=230,
        )


# ============================================================
# TAB 2 — MODEL ANALYSIS
# ============================================================
with tab_model:
    if _not_trained:
        st.info("Train a model for this asset to see model analysis.")
        st.stop()

    st.markdown("### Model Analysis")
    st.caption(
        "Classification metrics on the **held-out test set**. "
        "SHAP values explain what features drove each prediction."
    )

    # Load stored metrics if available
    stored = load_metrics(symbol, interval)
    if stored:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Test ROC-AUC",  f"{stored.get('roc_auc_test', 0):.3f}")
        mc2.metric("Train ROC-AUC", f"{stored.get('roc_auc_train', 0):.3f}")
        mc3.metric("Test Accuracy", f"{stored.get('acc_test', 0):.3f}")
        mc4.metric("Test Precision", f"{stored.get('prec_test', 0):.3f}")

        overfit_gap = stored.get("roc_auc_train", 0) - stored.get("roc_auc_test", 0)
        if overfit_gap > 0.1:
            st.warning(
                f"⚠️ Train/test ROC-AUC gap is **{overfit_gap:.3f}** — "
                "the model is overfitting. Consider more regularisation or additional data."
            )
        st.divider()

    col_shap, col_roc = st.columns(2)

    # SHAP feature importance
    with col_shap:
        st.markdown("**SHAP Feature Importance**")
        with st.spinner("Computing SHAP values on test set…"):
            try:
                shap_df = _shap_importance(model, X_test)
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["importance"].iloc[:15][::-1],
                    y=shap_df["feature"].iloc[:15][::-1],
                    orientation="h",
                    marker_color=asset_color,
                ))
                fig_shap.update_layout(
                    title="Top-15 features by mean |SHAP|",
                    xaxis_title="Mean |SHAP value|",
                    height=420,
                    margin=dict(t=50, b=40, l=120, r=20),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

    # ROC curve
    with col_roc:
        st.markdown("**ROC Curve (test set)**")
        fpr, tpr = get_roc_curve(y_test, proba_test)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, name="Model",
            line=dict(color=asset_color, width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            line=dict(color="#888888", width=1, dash="dash"),
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=420,
            margin=dict(t=20, b=40, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Calibration curve + confusion matrix
    col_cal, col_cm = st.columns(2)

    with col_cal:
        st.markdown("**Calibration Curve**")
        mean_pred, frac_pos = get_calibration_curve(y_test, proba_test, n_bins=8)
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos, name="Model",
            mode="lines+markers",
            line=dict(color=asset_color, width=2),
        ))
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Perfect",
            line=dict(color="#888888", width=1, dash="dash"),
        ))
        fig_cal.update_layout(
            xaxis_title="Mean predicted probability",
            yaxis_title="Fraction of positives",
            height=320,
            margin=dict(t=20, b=40, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        )
        st.plotly_chart(fig_cal, use_container_width=True)

    with col_cm:
        st.markdown("**Confusion Matrix (threshold = 0.50)**")
        cm_df = make_confusion_df(y_test, proba_test)
        cm_labels = ["TN", "FP", "FN", "TP"]
        cm_vals   = cm_df.set_index("metric").loc[cm_labels, "value"].tolist()
        cm_grid   = np.array(cm_vals).reshape(2, 2)
        fig_cm = go.Figure(go.Heatmap(
            z=cm_grid,
            x=["Predicted: Down", "Predicted: Up"],
            y=["Actual: Down", "Actual: Up"],
            colorscale=[[0, "#1e293b"], [1, asset_color]],
            text=[[str(v) for v in row] for row in cm_grid],
            texttemplate="%{text}",
            textfont=dict(size=18),
            showscale=False,
        ))
        fig_cm.update_layout(
            height=320, margin=dict(t=20, b=60, l=100, r=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Full feature importance table
    with st.expander("Full feature importance table"):
        try:
            shap_df = _shap_importance(model, X_test)
            st.dataframe(
                shap_df.style.bar(subset=["importance"], color=asset_color),
                use_container_width=True,
            )
        except Exception:
            st.info("SHAP not available for this model type.")


# ============================================================
# TAB 3 — WALK-FORWARD
# ============================================================
with tab_wf:
    if _not_trained:
        st.info("Train a model for this asset to see walk-forward results.")
        st.stop()

    st.markdown("### Walk-Forward Evaluation")
    st.caption(
        "Expanding-window validation: each fold trains on **all prior data** "
        "and evaluates on the next out-of-sample window. "
        "This is how you'd actually measure live performance."
    )

    with st.spinner("Running 5-fold walk-forward (this takes ~30 s)…"):
        wf_df = _walk_forward_cached(symbol, interval, threshold_long, threshold_short, tx_cost_bps)

    if wf_df is None or wf_df.empty:
        st.warning("Walk-forward failed. Ensure the model is trained.")
    else:
        # Summary metrics across folds
        avg_sharpe = wf_df["sharpe"].mean()
        avg_cagr   = wf_df["cagr_strategy"].mean()
        avg_mdd    = wf_df["max_drawdown"].mean()
        avg_wr     = wf_df["win_rate"].mean()

        wm1, wm2, wm3, wm4 = st.columns(4)
        wm1.metric("Avg Strategy CAGR", f"{avg_cagr*100:.1f}%")
        wm2.metric("Avg Sharpe",        f"{avg_sharpe:.2f}")
        wm3.metric("Avg Max Drawdown",  f"{avg_mdd*100:.1f}%")
        wm4.metric("Avg Win Rate",      f"{avg_wr*100:.1f}%")

        st.divider()

        # Per-fold CAGR bar chart
        fig_wf = go.Figure()
        colors_bar = [
            "#22c55e" if v > 0 else "#ef4444"
            for v in wf_df["cagr_strategy"]
        ]
        fig_wf.add_trace(go.Bar(
            x=wf_df["fold"].astype(str).apply(lambda f: f"Fold {f}"),
            y=wf_df["cagr_strategy"] * 100,
            name="Strategy CAGR %",
            marker_color=colors_bar,
        ))
        fig_wf.add_trace(go.Bar(
            x=wf_df["fold"].astype(str).apply(lambda f: f"Fold {f}"),
            y=wf_df["cagr_buyhold"] * 100,
            name="Buy & Hold CAGR %",
            marker_color="#94a3b8",
            opacity=0.6,
        ))
        fig_wf.update_layout(
            barmode="group",
            title="CAGR by fold — Strategy vs Buy & Hold",
            yaxis_title="CAGR (%)",
            height=350,
            margin=dict(t=60, b=40, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # Fold details table
        st.markdown("**Fold details**")
        display_cols = [
            "fold", "train_start", "train_end", "test_start", "test_end",
            "cagr_strategy", "cagr_buyhold", "sharpe", "max_drawdown",
            "win_rate", "profit_factor", "n_train", "n_test",
        ]
        fmt = {
            "cagr_strategy": "{:.1%}", "cagr_buyhold": "{:.1%}",
            "sharpe": "{:.2f}", "max_drawdown": "{:.1%}",
            "win_rate": "{:.1%}", "profit_factor": "{:.2f}",
        }
        st.dataframe(
            wf_df[display_cols].style.format(fmt),
            use_container_width=True,
        )

        st.caption(
            "💡 **Interpretation:** Consistent positive Sharpe across all folds indicates "
            "genuine signal; inconsistency suggests regime-dependence or noise overfitting."
        )


# ============================================================
# TAB 4 — CASE STUDY
# ============================================================
with tab_case:
    st.markdown(
        """
# Trendr: ML-Driven Crypto Price Direction Prediction

**Stefen Ewers** · ML Engineering Portfolio · [stefenewers.com](https://stefenewers.com)

---

## The Problem

Predicting whether a cryptocurrency's price will be higher or lower tomorrow
is a canonical hard problem in quantitative finance. The efficient market
hypothesis argues it shouldn't be possible at all — prices already reflect
all available information. Yet markets are made by humans, and human behaviour
leaves statistical fingerprints: momentum, mean-reversion, volatility
clustering, and sentiment cycles. The question is whether those fingerprints
are strong enough to trade on *after* accounting for transaction costs.

Trendr doesn't try to predict *how much* prices will move. It answers a
simpler question: **up or down tomorrow?** And then it bets accordingly.

---

## Data Pipeline

**Source:** 8 years of daily OHLCV data for Ethereum (ETH-USD), Bitcoin
(BTC-USD), and Solana (SOL-USD), sourced via the yfinance API and cached
locally to CSV.

**Feature Engineering:** 27 features derived entirely from price and volume
history, grouped into five families:

- *Returns & volatility* — 1-, 5-, and 10-day returns; 10- and 20-day
  rolling standard deviation. These capture short-term momentum and
  risk regime.
- *Trend indicators* — SMA (10/20), EMA (12/26), SMA ratio, price position
  relative to the 20-day SMA. These distinguish trending from mean-reverting
  regimes.
- *Oscillators* — RSI(14), MACD line/signal/histogram. Classic overbought/
  oversold signals.
- *Volatility* — ATR(14), Bollinger Band width and %B position, daily
  candle range. These proxy implied volatility and squeeze conditions.
- *Volume* — OBV z-score, 20-day volume z-score. Volume is frequently the
  leading indicator that price follows.
- *Calendar* — day-of-week and month, encoded as (sin, cos) pairs to
  preserve cyclical continuity. This captures weekend effects and
  seasonal patterns in crypto markets.

**Anti-lookahead discipline:** Every feature at row *t* uses only information
available at the close of day *t*. The target (next-day up-move) uses the
close of day *t+1* — automatically one bar in the future. The final row is
always dropped to prevent the last bar from having an undefined target leaking
into training.

---

## Model Architecture

The primary model is a **LightGBM gradient boosted tree classifier**, chosen
for three reasons:

1. **Speed and scalability.** LightGBM trains in seconds on 7 years of daily
   data, enabling rapid experimentation and walk-forward re-training.
2. **Built-in regularisation.** `num_leaves`, `reg_lambda`, and
   `learning_rate` provide effective controls against overfitting to
   non-stationary financial data.
3. **Native SHAP support.** TreeExplainer runs in milliseconds and returns
   exact (not approximate) Shapley values, enabling post-hoc interpretability.

A sklearn `GradientBoostingClassifier` is included as a comparison model, and
a `DummyClassifier(strategy="most_frequent")` serves as the random-chance
baseline. All models are tuned via `GridSearchCV` with `TimeSeriesSplit`
cross-validation — critically, *not* random k-fold, which would leak future
information into training folds.

---

## Evaluation Methodology

### Train / Validation / Test Split

The data is split into three non-overlapping temporal windows:

| Window      | Purpose                                 |
|-------------|-----------------------------------------|
| **Train**   | Model fitting and hyperparameter search |
| **Val**     | Threshold tuning, model selection       |
| **Test**    | Final evaluation — touched exactly once |

The validation set is used to tune the long/short probability thresholds —
the values at which the model enters a position. The test set is never used
to make any decisions; it is the single honest measure of out-of-sample
performance.

### Walk-Forward Validation

Beyond a single train/test split, the app implements **expanding-window
walk-forward validation**: at each of 5 folds, the model re-trains on all
data up to the fold boundary and evaluates on the next out-of-sample window.
This mimics how the system would actually be deployed — periodically
re-trained on accumulating data.

Walk-forward results are the most honest representation of live performance.
A model that achieves consistent positive Sharpe across all 5 folds has
demonstrated genuine generalisation, not luck on a single favourable test
period.

---

## Results and Honest Assessment

**What the model learned:** SHAP analysis consistently surfaces volume-based
features (OBV z-score, volume z-score) and volatility indicators (ATR,
Bollinger %B) as the most predictive. Pure price-momentum features (RSI,
MACD) contribute, but less than volume divergence signals. This is consistent
with the academic literature: *who is trading* matters as much as *how much
prices moved*.

**Classification performance:** The test ROC-AUC typically lands in the
**0.53–0.58 range** across assets — barely above random (0.50). This is
expected and not surprising. Predicting price direction on liquid assets is
genuinely hard. The meaningful question is not whether the classifier is
accurate, but whether the *margin* of accuracy is large enough to overcome
transaction costs in a live trading strategy.

**Strategy performance:** Results vary significantly by asset and market
regime. The strategy tends to outperform buy-and-hold in high-volatility,
trend-following periods (late 2020, 2021) and underperforms in grinding
bull runs where sitting flat loses ground. Sharpe ratios in the 0.5–1.2
range suggest real but fragile signal.

**Limitations:**

- *Non-stationarity.* Crypto markets are fundamentally non-stationary.
  A feature distribution shift (e.g., institutional market entry, regulatory
  events) can invalidate the model's learnings overnight.
- *Survivorship bias.* Training on ETH/BTC/SOL, all of which survived and
  grew substantially, may overestimate what would have been achievable on
  a random set of 2017 crypto assets.
- *Microstructure.* Daily OHLCV data ignores intraday liquidity. A 5 bps
  round-trip cost is optimistic for large positions.
- *Single-model risk.* There is no ensemble or model blending. A single
  LightGBM model will have blind spots.

---

## Production Considerations

Deploying this as a live signal generator would require:

1. **Daily data refresh** — a scheduled job to pull yesterday's close and
   generate tomorrow's signal.
2. **Model staleness detection** — monitoring the distribution of input
   features and alerting when they drift beyond a threshold.
3. **Position sizing** — the current model outputs {−1, 0, 1} but doesn't
   size positions by confidence. Kelly criterion or volatility-targeting
   would improve risk-adjusted returns.
4. **Live paper trading** — running the strategy on a paper account for 3–6
   months before allocating real capital.

---

## What I'd Build Next

- **On-chain features** — exchange net flows, active addresses, and hash
  rate are leading indicators not captured in price/volume data.
- **Cross-asset signals** — Bitcoin dominance, BTC/ETH relative strength,
  and crypto/equities correlation regimes.
- **Longer horizons** — predicting 5- or 10-day direction reduces noise
  and transaction costs relative to daily prediction.
- **Sequence models** — a properly regularised LSTM or Transformer operating
  on rolling windows of features, to capture temporal dependencies the
  tree model misses.

---

*Source code available at* [github.com/stefenewers/trendr](https://github.com/stefenewers/trendr)
        """,
        unsafe_allow_html=False,
    )
