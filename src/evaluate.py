import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score,
                              RocCurveDisplay, ConfusionMatrixDisplay,
                              classification_report)
import sys; sys.path.insert(0, str(Path(__file__).parent))
from config import REPORTS_DIR

def compute_metrics(model, X_te, y_te) -> dict:
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    return {
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall":    recall_score(y_te, y_pred),
        "f1":        f1_score(y_te, y_pred),
        "auc_roc":   roc_auc_score(y_te, y_proba),
    }

def save_report(model, name, X_te, y_te):
    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(X_te)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax, name=name)
    ax.plot([0,1],[0,1],"k--"); ax.set_title(f"ROC — {name}", fontweight="bold")
    plt.tight_layout(); fig.savefig(Path(REPORTS_DIR) / f"roc_{name.lower()}.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, y_pred, ax=ax,
        display_labels=["Pas AVC","AVC"], cmap="Blues")
    ax.set_title(f"Confusion — {name}", fontweight="bold")
    plt.tight_layout(); fig.savefig(Path(REPORTS_DIR) / f"cm_{name.lower()}.png"); plt.close(fig)
    print(classification_report(y_te, y_pred, target_names=["Pas AVC","AVC"]))
