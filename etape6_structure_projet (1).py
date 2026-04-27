# =============================================================
# ÉTAPE 6 : GÉNÉRATION DES MODULES src/
# Compatible : VS Code Local | CPU only
# Exécuter : python etape6_structure_projet.py
# =============================================================

import os
from pathlib import Path

SRC = Path("src")
SRC.mkdir(exist_ok=True)

# ── config.py ─────────────────────────────────────────────────
(SRC / "config.py").write_text('''\
from pathlib import Path

BASE_DIR          = Path(__file__).resolve().parent.parent
DATA_RAW          = BASE_DIR / "data" / "raw"       / "stroke_data.csv"
DATA_TRAIN        = BASE_DIR / "data" / "processed" / "train.csv"
DATA_TEST         = BASE_DIR / "data" / "processed" / "test.csv"
MODEL_DIR         = BASE_DIR / "models" / "trained"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessors" / "stroke_preprocessor.joblib"
FEATURE_NAMES_PATH= BASE_DIR / "models" / "preprocessors" / "feature_names.joblib"
BEST_MODEL_INFO   = BASE_DIR / "models" / "best_model_info.json"
REPORTS_DIR       = BASE_DIR / "reports"

TARGET_COL        = "stroke"
DROP_COLS         = ["id"]
NUMERIC_COLS      = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_COLS  = ["gender", "work_type", "Residence_type",
                     "smoking_status", "ever_married"]
BINARY_COLS       = ["hypertension", "heart_disease"]

TEST_SIZE         = 0.20
SEED              = 42
CV_FOLDS          = 5
KNN_NEIGHBORS     = 5
SMOTE_RATIO       = 1.0

MLFLOW_URI        = "mlruns"
MLFLOW_EXPERIMENT = "stroke_prediction"
MLFLOW_MODEL_NAME = "StrokePredictor"

API_HOST          = "0.0.0.0"
API_PORT          = 8000
MODEL_VERSION     = "1.0.0"
''')

# ── data_loader.py ────────────────────────────────────────────
(SRC / "data_loader.py").write_text('''\
import pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_RAW, NUMERIC_COLS, CATEGORICAL_COLS, BINARY_COLS, TARGET_COL

EXPECTED = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS + [TARGET_COL]

def load_raw_data(path=None):
    path = path or str(DATA_RAW)
    if not Path(path).exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    for col in NUMERIC_COLS + BINARY_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Colonne \'{col}\' doit être numérique.")
    return df

def load_patient_dict(patient: dict) -> pd.DataFrame:
    required = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS
    missing  = [f for f in required if f not in patient]
    if missing:
        raise ValueError(f"Champs manquants : {missing}")
    return pd.DataFrame([patient])
''')

# ── preprocessing.py ──────────────────────────────────────────
(SRC / "preprocessing.py").write_text('''\
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline      import Pipeline
from sklearn.compose       import ColumnTransformer
from sklearn.impute        import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling  import SMOTE
import sys; sys.path.insert(0, str(Path(__file__).parent))
from config import (NUMERIC_COLS, CATEGORICAL_COLS, BINARY_COLS, TARGET_COL,
                    DROP_COLS, TEST_SIZE, SEED, KNN_NEIGHBORS, SMOTE_RATIO,
                    PREPROCESSOR_PATH, FEATURE_NAMES_PATH, DATA_RAW,
                    DATA_TRAIN, DATA_TEST)

def clean(df):
    df = df.drop(columns=DROP_COLS, errors="ignore").copy()
    df = df.drop_duplicates()
    df.loc[df["bmi"] == 0, "bmi"]  = np.nan
    df.loc[df["age"] < 0,  "age"]  = np.nan
    df["gender"]         = df["gender"].replace("Other", "Unknown")
    df["smoking_status"] = df["smoking_status"].replace("Unknown", np.nan)
    return df

def build_preprocessor():
    num = Pipeline([("imp", KNNImputer(n_neighbors=KNN_NEIGHBORS)),
                    ("sc",  RobustScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUMERIC_COLS),
                               ("cat", cat, CATEGORICAL_COLS),
                               ("bin", "passthrough", BINARY_COLS)], remainder="drop")

def run_preprocessing(path=None):
    from data_loader import load_raw_data
    df = clean(load_raw_data(path))
    X  = df.drop(columns=[TARGET_COL])
    y  = df[TARGET_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                               stratify=y, random_state=SEED)
    prep     = build_preprocessor()
    X_tr_p   = prep.fit_transform(X_tr)
    X_te_p   = prep.transform(X_te)
    X_tr_p, y_tr = SMOTE(sampling_strategy=SMOTE_RATIO,
                          random_state=SEED).fit_resample(X_tr_p, y_tr)
    ohe_names    = prep.named_transformers_["cat"]["ohe"].get_feature_names_out(CATEGORICAL_COLS)
    feat_names   = NUMERIC_COLS + list(ohe_names) + BINARY_COLS
    Path(PREPROCESSOR_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prep,       PREPROCESSOR_PATH)
    joblib.dump(feat_names, FEATURE_NAMES_PATH)
    tr = pd.DataFrame(X_tr_p, columns=feat_names); tr[TARGET_COL] = y_tr.values
    te = pd.DataFrame(X_te_p, columns=feat_names); te[TARGET_COL] = y_te.values
    Path(DATA_TRAIN).parent.mkdir(parents=True, exist_ok=True)
    tr.to_csv(DATA_TRAIN, index=False); te.to_csv(DATA_TEST, index=False)
    return X_tr_p, X_te_p, y_tr, y_te, feat_names
''')

# ── train.py ──────────────────────────────────────────────────
(SRC / "train.py").write_text('''\
import mlflow, mlflow.sklearn, mlflow.xgboost
import joblib, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import sys; sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_TRAIN, DATA_TEST, MODEL_DIR, TARGET_COL, MLFLOW_URI, MLFLOW_EXPERIMENT, SEED, CV_FOLDS
from evaluate import compute_metrics

def train_and_log(model, name, param_grid, X_tr, y_tr, X_te, y_te, n_iter=10):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(model, param_grid, n_iter=n_iter,
                                scoring="recall", cv=cv, n_jobs=-1, random_state=SEED)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    with mlflow.start_run(run_name=f"{name}_{datetime.now():%H%M%S}") as run:
        mlflow.set_tag("model_type", name)
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
        metrics = compute_metrics(best, X_te, y_te)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        if name == "XGBoost": mlflow.xgboost.log_model(best, "model")
        else:                  mlflow.sklearn.log_model(best, "model")
        run_id = run.info.run_id
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(best, Path(MODEL_DIR) / f"best_model_{name.lower()}.joblib")
    return best, run_id, search.best_params_
''')

# ── evaluate.py ───────────────────────────────────────────────
(SRC / "evaluate.py").write_text('''\
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
''')

# ── predict.py ────────────────────────────────────────────────
(SRC / "predict.py").write_text('''\
import numpy as np, json, joblib
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent))
from config import PREPROCESSOR_PATH, MODEL_DIR, BEST_MODEL_INFO

def load_artifacts():
    with open(BEST_MODEL_INFO) as f:
        info = json.load(f)
    name  = info["model_name"].lower()
    path  = Path(MODEL_DIR) / f"best_model_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path), joblib.load(PREPROCESSOR_PATH), info

def predict_single(patient_dict: dict) -> dict:
    import pandas as pd
    from data_loader import load_patient_dict
    model, prep, info = load_artifacts()
    df    = load_patient_dict(patient_dict)
    X     = prep.transform(df)
    label = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])
    conf  = "high" if (proba > 0.75 or proba < 0.25) else "medium"
    return {"stroke_label": label, "stroke_risk": round(proba, 4),
            "confidence": conf, "model_version": info.get("model_name")}
''')

# ── Résumé ────────────────────────────────────────────────────
print("✅ Modules src/ créés :")
for f in sorted(SRC.glob("*.py")):
    print(f"   ├── {f}")
