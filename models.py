from typing import Dict, Any, Tuple
import os, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

ARTIFACT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'artifacts', 'models'))
REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'artifacts', 'reports'))

def ensure_dirs():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

def build_pipeline(model_name: str = "gbc") -> Tuple[Pipeline, Dict[str, Any]]:
    if model_name == "rf":
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=42))])
        grid = {"clf__n_estimators": [200, 400], "clf__max_depth": [4, 6, 8]}
    else:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(random_state=42))])
        grid = {"clf__n_estimators": [150, 250], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]}
    return pipe, grid

def fit_with_cv(X, y, cv, model_name: str = "gbc"):
    pipe, grid = build_pipeline(model_name)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(X, y)
    return gs

def evaluate(model, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test  = model.predict_proba(X_test)[:, 1]
    preds_train = (proba_train > 0.5).astype(int)
    preds_test  = (proba_test  > 0.5).astype(int)
    return {
        "acc_train": accuracy_score(y_train, preds_train),
        "acc_test": accuracy_score(y_test, preds_test),
        "prec_test": precision_score(y_test, preds_test, zero_division=0),
        "recall_test": recall_score(y_test, preds_test, zero_division=0),
        "roc_auc_test": roc_auc_score(y_test, proba_test),
    }

def save_model(model, symbol: str, interval: str = "1d"):
    ensure_dirs()
    path = os.path.join(ARTIFACT_DIR, f"model_{symbol}_{interval}.joblib")
    joblib.dump(model, path)
    return path

def load_model(symbol: str, interval: str = "1d"):
    path = os.path.join(ARTIFACT_DIR, f"model_{symbol}_{interval}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Train first.")
    return joblib.load(path)

def save_metrics(metrics: Dict[str, float], symbol: str, interval: str = "1d") -> str:
    ensure_dirs()
    path = os.path.join(REPORT_DIR, f"metrics_{symbol}_{interval}.json")
    import json
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path
