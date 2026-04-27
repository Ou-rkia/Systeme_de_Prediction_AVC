# =============================================================
# ÉTAPE 4 : MODÉLISATION — MLP / SVM / XGBoost
# Compatible : VS Code Local | CPU only | Optimisé pour CPU
# =============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import time
from pathlib import Path

from sklearn.neural_network  import MLPClassifier
from sklearn.svm             import SVC
from xgboost                 import XGBClassifier

from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV)
from sklearn.metrics         import (accuracy_score, precision_score,
                                      recall_score, f1_score, roc_auc_score,
                                      confusion_matrix, classification_report,
                                      RocCurveDisplay)
warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models/trained")
REPORTS_DIR   = Path("reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEED     = 42
CV_FOLDS = 5


# ── Chargement ────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test  = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_tr  = train.drop(columns=["stroke"]).values
    y_tr  = train["stroke"].values
    X_te  = test.drop(columns=["stroke"]).values
    y_te  = test["stroke"].values
    print(f"  Train : {X_tr.shape}  |  Test : {X_te.shape}")
    return X_tr, X_te, y_tr, y_te


# ── Configs modèles (optimisées CPU — pas de GPU) ─────────────
def get_configs():
    return {
        # MLP : peu de couches, iterations réduites → rapide sur CPU
        "MLP": {
            "model": MLPClassifier(
                random_state=SEED,
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=15,
                validation_fraction=0.1,
            ),
            "params": {
                "hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "activation":         ["relu", "tanh"],
                "alpha":              [1e-4, 1e-3],
                "learning_rate_init": [1e-3, 5e-4],
            },
            "n_iter": 6,   # Peu d'itérations RandomSearch → rapide
        },

        # SVM : C et kernel limités pour ne pas exploser sur CPU
        "SVM": {
            "model": SVC(
                probability=True,
                random_state=SEED,
                cache_size=500,     # Cache mémoire (Mo) pour accélérer
            ),
            "params": {
                "C":      [0.1, 1.0, 10.0],
                "kernel": ["rbf", "linear"],
                "gamma":  ["scale", "auto"],
            },
            "n_iter": 6,
        },

        # XGBoost : tree_method="hist" = algorithme rapide sur CPU
        "XGBoost": {
            "model": XGBClassifier(
                random_state=SEED,
                eval_metric="logloss",
                verbosity=0,
                tree_method="hist",   # ← Rapide sur CPU (pas de GPU)
                device="cpu",         # ← Forcer CPU explicitement
                n_jobs=-1,            # ← Utilise tous les cœurs CPU
            ),
            "params": {
                "n_estimators":      [100, 200],
                "max_depth":         [3, 5],
                "learning_rate":     [0.1, 0.2],
                "subsample":         [0.8, 1.0],
                "colsample_bytree":  [0.8, 1.0],
                "scale_pos_weight":  [1, 3],
            },
            "n_iter": 8,
        },
    }


# ── Tuning ────────────────────────────────────────────────────
def tune(name, model, params, n_iter, X_tr, y_tr):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        model, params, n_iter=n_iter,
        scoring="recall",
        cv=cv, n_jobs=-1,
        random_state=SEED,
        verbose=0,
    )
    search.fit(X_tr, y_tr)
    print(f"  [{name}] Meilleurs params : {search.best_params_}")
    print(f"  [{name}] Recall CV        : {search.best_score_:.4f}")
    return search.best_estimator_


# ── Évaluation ────────────────────────────────────────────────
def evaluate(name, model, X_te, y_te):
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    m = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_te, y_pred),          4),
        "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_te, y_pred),             4),
        "f1":        round(f1_score(y_te, y_pred),                 4),
        "auc_roc":   round(roc_auc_score(y_te, y_proba),           4),
    }
    print(f"\n  ── {name} — Test Set ──")
    for k, v in m.items():
        if k != "model":
            flag = " ← métrique principale" if k == "recall" else ""
            print(f"    {k:12s}: {v:.4f}{flag}")

    print(f"\n{classification_report(y_te, y_pred, target_names=['Pas AVC','AVC'])}")
    return m


# ── Visualisations ────────────────────────────────────────────
def plot_conf_matrices(models_dict, X_te, y_te):
    n   = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    for ax, (name, model) in zip(axes, models_dict.items()):
        cm = confusion_matrix(y_te, model.predict(X_te))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pas AVC", "AVC"],
                    yticklabels=["Pas AVC", "AVC"])
        ax.set_title(f"Confusion — {name}", fontweight="bold")
        ax.set_ylabel("Réel")
        ax.set_xlabel("Prédit")

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "07_confusion_matrices.png")
    plt.close(fig)
    print("✅ Graphique sauvegardé → reports/07_confusion_matrices.png")


def plot_roc(models_dict, X_te, y_te):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#4C9BE8", "#E8574C", "#4CAF50"]

    for (name, model), color in zip(models_dict.items(), colors):
        RocCurveDisplay.from_estimator(model, X_te, y_te,
                                        ax=ax, name=name, color=color)
    ax.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    ax.set_title("Courbes ROC", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "08_roc_curves.png")
    plt.close(fig)
    print("✅ Graphique sauvegardé → reports/08_roc_curves.png")


def plot_metrics(results):
    df = pd.DataFrame(results)
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    melt = df.melt(id_vars="model", value_vars=metrics,
                   var_name="Métrique", value_name="Score")

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=melt, x="Métrique", y="Score", hue="model",
                palette=["#4C9BE8", "#E8574C", "#4CAF50"], ax=ax)
    ax.set_ylim(0, 1.15)
    ax.set_title("Comparaison des modèles", fontweight="bold")
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "09_metrics_comparison.png")
    plt.close(fig)
    print("✅ Graphique sauvegardé → reports/09_metrics_comparison.png")


# ── Pipeline complet ──────────────────────────────────────────
def run_modeling():
    print("\n🧠 DÉMARRAGE MODÉLISATION\n")
    X_tr, X_te, y_tr, y_te = load_data()

    configs  = get_configs()
    trained  = {}
    results  = []

    for name, cfg in configs.items():
        print(f"\n{'='*50}\n  Modèle : {name}\n{'='*50}")
        t0    = time.time()
        model = tune(name, cfg["model"], cfg["params"],
                     cfg["n_iter"], X_tr, y_tr)
        elapsed = round(time.time() - t0, 1)

        m = evaluate(name, model, X_te, y_te)
        m["temps_s"] = elapsed
        results.append(m)
        trained[name] = model

        joblib.dump(model, MODELS_DIR / f"best_model_{name.lower()}.joblib")
        print(f"  Modèle sauvegardé → models/trained/best_model_{name.lower()}.joblib")
        print(f"  Temps d'entraînement : {elapsed}s")

    # Tableau comparatif
    print("\n" + "="*65)
    print("  TABLEAU COMPARATIF")
    print("="*65)
    df_cmp = pd.DataFrame(results).sort_values("f1", ascending=False)
    print(df_cmp.to_string(index=False))

    # Meilleur modèle
    best = df_cmp.iloc[0]
    print(f"\n🏆 Meilleur modèle : {best['model']}  (F1={best['f1']:.4f}, Recall={best['recall']:.4f})")

    # Sauvegarde du nom du meilleur modèle
    joblib.dump(best["model"], MODELS_DIR / "best_model_name.joblib")

    # Graphiques
    plot_conf_matrices(trained, X_te, y_te)
    plot_roc(trained, X_te, y_te)
    plot_metrics(results)

    print("\n✅ Modélisation terminée.")
    return trained, results, best["model"]


if __name__ == "__main__":
    run_modeling()
