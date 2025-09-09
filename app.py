import os, sys
import streamlit as st
import pandas as pd

# --- Make sure the project root is on sys.path so "import trendr.*" works ---
FILE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../trendr/trendr
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, os.pardir))  # .../trendr
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trendr.modeling.models import load_model
from trendr.modeling.backtest import StrategyParams, backtest, performance
from trendr.modeling.dataset import train_test_split_time, get_X_y

st.set_page_config(page_title="Trendr — ML Signals", layout="wide")

st.title("📈 Trendr")
st.caption("Signals & backtest for a simple ML strategy on crypto prices.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Symbol", value="ETH-USD")
    interval = st.selectbox("Interval", options=["1d"], index=0)
    threshold_long = st.slider("Long threshold", 0.50, 0.70, 0.55, 0.01)
    threshold_short = st.slider("Short threshold", 0.30, 0.50, 0.45, 0.01)
    tx_cost_bps = st.slider("Tx cost (bps, round-trip)", 0.0, 50.0, 5.0, 0.5)

    data_path = os.path.join(PROJECT_ROOT, "data", "processed", f"features_{symbol}_{interval}.csv")
    model_info = f"artifacts/models/model_{symbol}_{interval}.joblib"

# Check files exist
if not os.path.exists(data_path):
    st.warning(f"Features not found at `{data_path}`. Run:\n\n`python -m trendr.cli train --symbol {symbol}`")
    st.stop()

try:
    model = load_model(symbol, interval)
except FileNotFoundError:
    st.warning(f"Model not found at `{model_info}`. Run:\n\n`python -m trendr.cli train --symbol {symbol}`")
    st.stop()

# Load features and split
df_feat = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date")
train_df, test_df = train_test_split_time(df_feat)
X_train, y_train = get_X_y(train_df)
X_test, y_test = get_X_y(test_df)

# Predict & backtest
proba = model.predict_proba(X_test)[:, 1]
params = StrategyParams(threshold_long=threshold_long, threshold_short=threshold_short, tx_cost_bps=tx_cost_bps)
bt = backtest(test_df.set_index("date")["close"], proba, params)
perf = performance(bt)

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategy CAGR", f"{perf['cagr_strategy']*100:.2f}%")
c2.metric("Buy & Hold CAGR", f"{perf['cagr_buyhold']*100:.2f}%")
c3.metric("Sharpe (daily→252)", f"{perf['sharpe']:.2f}")
c4.metric("Max Drawdown", f"{perf['max_drawdown']*100:.2f}%")

# Chart
st.subheader("Equity Curves")
plot_df = bt[["strat_cum", "bh_cum"]].rename(columns={"strat_cum": "Strategy", "bh_cum": "Buy & Hold"})
st.line_chart(plot_df)

# Table of recent signals
st.subheader("Recent Signals")
recent = bt.tail(20).copy()
recent = recent.rename(columns={"strat_cum": "Equity (Strat)", "bh_cum": "Equity (B&H)"})
st.dataframe(recent, use_container_width=True)

st.caption("Tip: tweak thresholds in the sidebar and watch the curve update.")
