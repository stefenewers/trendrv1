import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def make_eval_df(y_true, y_proba, thresh=0.5) -> pd.DataFrame:
    y_pred = (y_proba > thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return pd.DataFrame({"metric": ["TN","FP","FN","TP"], "value": [tn, fp, fn, tp]})

def text_report(y_true, y_proba, thresh=0.5) -> str:
    y_pred = (y_proba > thresh).astype(int)
    return classification_report(y_true, y_pred, digits=3)
