"""
Trendr CLI — download, featurize, train, backtest

Usage:
    python -m trendr.cli download --symbol ETH-USD
    python -m trendr.cli train --symbol ETH-USD --model lgb
    python -m trendr.cli backtest --symbol ETH-USD
    python -m trendr.cli train-all          # ETH + BTC + SOL
"""

import json
import logging
import os

import typer
import pandas as pd

from trendr.config import SUPPORTED_SYMBOLS, SYMBOL_META, TrendrConfig
from trendr.data.downloader import download_prices, load_prices
from trendr.data.features import build_features
from trendr.modeling.backtest import StrategyParams, backtest, performance
from trendr.modeling.dataset import train_val_test_split, get_X_y, get_timeseries_cv
from trendr.modeling.models import fit_with_cv, evaluate, save_model, save_metrics, load_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Trendr CLI: download data, train model, backtest strategy.",
    add_completion=False,
)

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
ART_DIR   = os.path.join(BASE_DIR, "artifacts", "reports")

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)


@app.command()
def download(
    symbol:   str = typer.Option("ETH-USD", help="Ticker, e.g. ETH-USD"),
    start:    str = typer.Option("",        help="Start date YYYY-MM-DD (default: asset-specific)"),
    interval: str = typer.Option("1d",      help="OHLCV interval"),
) -> None:
    """Download raw OHLCV data from yfinance and cache to CSV."""
    if not start:
        start = SYMBOL_META.get(symbol, {}).get("start", "2017-01-01")
    path = download_prices(symbol, start, interval)
    typer.echo(f"Saved: {path}")


@app.command()
def featurize(
    symbol:   str = typer.Option("ETH-USD"),
    interval: str = typer.Option("1d"),
) -> None:
    """Build feature matrix from raw OHLCV and save to data/processed/."""
    df      = load_prices(symbol, interval)
    df_feat = build_features(df)
    out     = os.path.join(PROC_DIR, f"features_{symbol}_{interval}.csv")
    df_feat.to_csv(out, index=False)
    typer.echo(f"Saved features ({len(df_feat)} rows, {len(df_feat.columns)} cols): {out}")


@app.command()
def train(
    symbol:   str = typer.Option("ETH-USD"),
    interval: str = typer.Option("1d"),
    model:    str = typer.Option("lgb", help="Model type: lgb, gbc, dummy"),
) -> None:
    """Train a model with time-series CV and save artefacts."""
    cfg     = TrendrConfig(symbol=symbol, interval=interval)
    df      = load_prices(symbol, interval)
    df_feat = build_features(df)

    train_df, val_df, test_df = train_val_test_split(
        df_feat,
        val_size_days=cfg.val_size_days,
        test_size_days=cfg.test_size_days,
    )

    typer.echo(
        f"Split: train={len(train_df)} / val={len(val_df)} / test={len(test_df)} rows"
    )

    X_train, y_train = get_X_y(train_df)
    X_val,   y_val   = get_X_y(val_df)
    X_test,  y_test  = get_X_y(test_df)

    cv = get_timeseries_cv(cfg.n_cv_splits)
    gs = fit_with_cv(X_train, y_train, cv, model_name=model)

    # Evaluate on both val and test (report test as the honest number)
    metrics = evaluate(gs, X_train, y_train, X_test, y_test)
    val_metrics = evaluate(gs, X_train, y_train, X_val, y_val)
    metrics["roc_auc_val"]  = val_metrics["roc_auc_test"]
    metrics["acc_val"]      = val_metrics["acc_test"]
    if hasattr(gs, "best_params_"):
        metrics["best_params"] = gs.best_params_
    metrics["model_type"]   = model

    metrics_path = save_metrics(metrics, symbol, interval)
    model_path   = save_model(gs, symbol, interval)

    # Save features
    feat_path = os.path.join(PROC_DIR, f"features_{symbol}_{interval}.csv")
    df_feat.to_csv(feat_path, index=False)

    typer.echo(json.dumps(
        {"model": model_path, "metrics": metrics_path, "features": feat_path, **metrics},
        indent=2,
    ))


@app.command(name="train-all")
def train_all(
    model:    str = typer.Option("lgb"),
    interval: str = typer.Option("1d"),
) -> None:
    """Download, featurize, and train for all supported assets (ETH, BTC, SOL)."""
    for sym in SUPPORTED_SYMBOLS:
        typer.echo(f"\n{'='*50}\nTraining {sym}…\n{'='*50}")
        start = SYMBOL_META[sym]["start"]
        try:
            download_prices(sym, start, interval)
            df      = load_prices(sym, interval)
            df_feat = build_features(df)
            cfg     = TrendrConfig(symbol=sym, interval=interval)
            train_df, val_df, test_df = train_val_test_split(
                df_feat, cfg.val_size_days, cfg.test_size_days
            )
            X_train, y_train = get_X_y(train_df)
            X_test,  y_test  = get_X_y(test_df)
            cv  = get_timeseries_cv(cfg.n_cv_splits)
            gs  = fit_with_cv(X_train, y_train, cv, model_name=model)
            metrics = evaluate(gs, X_train, y_train, X_test, y_test)
            if hasattr(gs, "best_params_"):
                metrics["best_params"] = gs.best_params_
            metrics["model_type"] = model
            save_metrics(metrics, sym, interval)
            save_model(gs, sym, interval)
            feat_path = os.path.join(PROC_DIR, f"features_{sym}_{interval}.csv")
            df_feat.to_csv(feat_path, index=False)
            typer.echo(f"✓ {sym}  ROC-AUC test={metrics['roc_auc_test']:.4f}")
        except Exception as exc:
            typer.echo(f"✗ {sym}  FAILED: {exc}", err=True)


@app.command(name="backtest")
def backtest_cmd(
    symbol:          str   = typer.Option("ETH-USD"),
    interval:        str   = typer.Option("1d"),
    threshold_long:  float = typer.Option(0.55),
    threshold_short: float = typer.Option(0.45),
    tx_cost_bps:     float = typer.Option(5.0),
) -> None:
    """Run backtest on the test-set window and print performance metrics."""
    feat_path = os.path.join(PROC_DIR, f"features_{symbol}_{interval}.csv")
    df_feat   = pd.read_csv(feat_path, parse_dates=["date"]).sort_values("date")

    cfg = TrendrConfig(symbol=symbol, interval=interval)
    _, _, test_df = train_val_test_split(df_feat, cfg.val_size_days, cfg.test_size_days)
    X_test, y_test = get_X_y(test_df)

    mdl    = load_model(symbol, interval)
    proba  = mdl.predict_proba(X_test)[:, 1]
    params = StrategyParams(
        threshold_long=threshold_long,
        threshold_short=threshold_short,
        tx_cost_bps=tx_cost_bps,
    )
    bt   = backtest(test_df.set_index("date")["close"], proba, params)
    perf = performance(bt)

    out_path = os.path.join(ART_DIR, f"backtest_{symbol}_{interval}.csv")
    bt.to_csv(out_path)
    typer.echo(json.dumps({"backtest_csv": out_path, **perf}, indent=2))


if __name__ == "__main__":
    app()
