# =============================================================
# ÉTAPE 5 : TRACKING MLFLOW
# Compatible : VS Code Local | CPU only
# Lancer l'UI : mlflow ui  →  http://localhost:5000
# =============================================================

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (f1_score, recall_score, roc_auc_score,
                              accuracy_score, precision_score,
                              RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score, StratifiedKFold

PROCESSED_DIR   = Path("data/processed")
MODELS_DIR      = Path("models/trained")
REPORTS_DIR     = Path("reports")
MLFLOW_URI      = "mlruns"
EXPERIMENT_NAME = "stroke_prediction"
SEED            = 42


# ── Setup ─────────────────────────────────────────────────────
def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✅ MLflow prêt")
    print(f"   Tracking URI  : {mlflow.get_tracking_uri()}")
    print(f"   Expérience    : {EXPERIMENT_NAME}")
    print(f"   UI            : mlflow ui  →  http://localhost:5000\n")


# ── Chargement données ────────────────────────────────────────
def load_data():
    train   = pd.read_csv(PROCESSED_DIR / "train.csv")
    test    = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train = train.drop(columns=["stroke"]).values
    y_train = train["stroke"].values
    X_test  = test.drop(columns=["stroke"]).values
    y_test  = test["stroke"].values
    return X_train, X_test, y_train, y_test


# ── Artefacts visuels ─────────────────────────────────────────
def make_artifacts(model, name, X_te, y_te):
    paths = []

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_te, y_te, ax=ax, name=name, color="#4C9BE8")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"ROC — {name}", fontweight="bold")
    plt.tight_layout()
    p = REPORTS_DIR / f"roc_{name.lower()}.png"
    fig.savefig(p); plt.close(fig); paths.append(str(p))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, model.predict(X_te), ax=ax,
        display_labels=["Pas AVC", "AVC"], cmap="Blues")
    ax.set_title(f"Confusion — {name}", fontweight="bold")
    plt.tight_layout()
    p = REPORTS_DIR / f"cm_{name.lower()}.png"
    fig.savefig(p); plt.close(fig); paths.append(str(p))

    return paths


# ── Logger un run MLflow ──────────────────────────────────────
def log_run(model, name, X_tr, y_tr, X_te, y_te):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    with mlflow.start_run(run_name=f"{name}_{datetime.now():%H%M%S}") as run:
        # Tags
        mlflow.set_tags({
            "model_type":    name,
            "training_date": datetime.now().isoformat(),
            "dataset":       "Stroke Prediction (synthetic)",
        })

        # Paramètres
        params = {k: str(v) for k, v in model.get_params().items()}
        mlflow.log_params(params)

        # Métriques test
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        test_metrics = {
            "test_accuracy":  accuracy_score(y_te, y_pred),
            "test_precision": precision_score(y_te, y_pred, zero_division=0),
            "test_recall":    recall_score(y_te, y_pred),
            "test_f1":        f1_score(y_te, y_pred),
            "test_auc_roc":   roc_auc_score(y_te, y_proba),
        }
        mlflow.log_metrics(test_metrics)

        # Métriques cross-val
        for metric in ["recall", "f1", "roc_auc"]:
            scores = cross_val_score(model, X_tr, y_tr, cv=cv,
                                     scoring=metric, n_jobs=-1)
            mlflow.log_metric(f"cv_{metric}_mean", scores.mean())
            mlflow.log_metric(f"cv_{metric}_std",  scores.std())

        # Artefacts
        for p in make_artifacts(model, name, X_te, y_te):
            mlflow.log_artifact(p)

        # Modèle
        if name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id

    print(f"  [{name}] run_id={run_id[:8]}  "
          f"Recall={test_metrics['test_recall']:.4f}  "
          f"F1={test_metrics['test_f1']:.4f}  "
          f"AUC={test_metrics['test_auc_roc']:.4f}")

    return run_id, test_metrics


# ── Enregistrer le meilleur modèle ────────────────────────────
def register_best(run_results):
    best = max(run_results, key=lambda x: x["test_f1"])
    model_uri = f"runs:/{best['run_id']}/model"

    print(f"\n🏆 Meilleur modèle : {best['name']}  (F1={best['test_f1']:.4f})")

    try:
        registered = mlflow.register_model(model_uri, "StrokePredictor")
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="StrokePredictor",
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"  ✅ Modèle v{registered.version} → stage 'Production'")
    except Exception as e:
        print(f"  ⚠️  Registry non disponible en local (normal sans serveur MLflow) : {e}")

    # Sauvegarder les infos JSON pour l'API
    info = {
        "model_name":    best["name"],
        "run_id":        best["run_id"],
        "test_f1":       best["test_f1"],
        "test_recall":   best["test_recall"],
        "test_auc_roc":  best["test_auc_roc"],
        "registered_at": datetime.now().isoformat(),
    }
    Path("models").mkdir(exist_ok=True)
    with open("models/best_model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("  ✅ Infos sauvegardées → models/best_model_info.json")


# ── Pipeline MLflow complet ───────────────────────────────────
def run_mlflow_tracking():
    print("\n📊 DÉMARRAGE TRACKING MLFLOW\n")
    setup_mlflow()

    X_tr, X_te, y_tr, y_te = load_data()

    model_files = list(MODELS_DIR.glob("best_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError("Aucun modèle trouvé. Exécutez d'abord étape 4.")

    run_results = []

    for path in model_files:
        raw_name = path.stem.replace("best_model_", "")
        # Exclure le fichier "best_model_name.joblib"
        if raw_name == "name":
            continue
        name = {"xgboost": "XGBoost", "mlp": "MLP", "svm": "SVM"}.get(raw_name, raw_name.upper())

        model  = joblib.load(path)
        run_id, metrics = log_run(model, name, X_tr, y_tr, X_te, y_te)
        run_results.append({
            "run_id":      run_id,
            "name":        name,
            "test_f1":     metrics["test_f1"],
            "test_recall": metrics["test_recall"],
            "test_auc_roc": metrics["test_auc_roc"],
        })

    register_best(run_results)

    print("\n✅ Tracking MLflow terminé.")
    print("   Pour visualiser : mlflow ui  →  http://localhost:5000")


if __name__ == "__main__":
    run_mlflow_tracking()
