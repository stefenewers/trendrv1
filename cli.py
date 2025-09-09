import os, json
import typer
import pandas as pd
from trendr.data.downloader import download_prices, load_prices
from trendr.data.features import build_features
from trendr.modeling.dataset import train_test_split_time, get_X_y, get_timeseries_cv
from trendr.modeling.models import fit_with_cv, evaluate, save_model, save_metrics, load_model
from trendr.modeling.backtest import StrategyParams, backtest, performance

app = typer.Typer(help="Trendr CLI: download data, train model, backtest strategy.")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
ART_DIR = os.path.join(BASE_DIR, 'artifacts', 'reports')
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

@app.command()
def download(symbol: str = "ETH-USD", start: str = "2016-01-01", interval: str = "1d"):
    path = download_prices(symbol, start, interval)
    typer.echo(f"Saved: {path}")

@app.command()
def featurize(symbol: str = "ETH-USD", interval: str = "1d"):
    df = load_prices(symbol, interval)
    df_feat = build_features(df)
    out = os.path.join(PROC_DIR, f"features_{symbol}_{interval}.csv")
    df_feat.to_csv(out, index=False)
    typer.echo(f"Saved features: {out}")

@app.command()
def train(symbol: str = "ETH-USD", interval: str = "1d", model: str = "gbc"):
    df = load_prices(symbol, interval)
    df_feat = build_features(df)
    train_df, test_df = train_test_split_time(df_feat)
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    cv = get_timeseries_cv(5)
    gs = fit_with_cv(X_train, y_train, cv, model_name=model)

    from sklearn.metrics import roc_auc_score
    proba_train = gs.predict_proba(X_train)[:, 1]
    proba_test = gs.predict_proba(X_test)[:, 1]
    metrics = evaluate(gs, X_train, y_train, X_test, y_test)
    metrics["best_params"] = gs.best_params_
    metrics["roc_auc_train"] = roc_auc_score(y_train, proba_train)
    metrics_path = save_metrics(metrics, symbol, interval)

    model_path = save_model(gs, symbol, interval)

    out = os.path.join(PROC_DIR, f"features_{symbol}_{interval}.csv")
    df_feat.to_csv(out, index=False)

    typer.echo(json.dumps({"model": model_path, "metrics": metrics_path, "features": out}, indent=2))

@app.command(name="backtest")
def backtest_cmd(symbol: str = "ETH-USD", interval: str = "1d",
                 threshold_long: float = 0.55, threshold_short: float = 0.45, tx_cost_bps: float = 5.0):
    feat_path = os.path.join(PROC_DIR, f"features_{symbol}_{interval}.csv")
    df_feat = pd.read_csv(feat_path, parse_dates=["date"]).sort_values("date")
    _, test_df = train_test_split_time(df_feat)
    X_test, y_test = get_X_y(test_df)

    model = load_model(symbol, interval)
    proba = model.predict_proba(X_test)[:, 1]
    params = StrategyParams(threshold_long=threshold_long, threshold_short=threshold_short, tx_cost_bps=tx_cost_bps)

    bt = backtest(test_df.set_index("date")["close"], proba, params)
    perf = performance(bt)

    out_path = os.path.join(ART_DIR, f"backtest_{symbol}_{interval}.csv")
    bt.to_csv(out_path)
    typer.echo(json.dumps({"backtest_csv": out_path, **perf}, indent=2))

if __name__ == "__main__":
    app()
