# =============================================================
# ÉTAPE 3 : PRÉPROCESSING
# Compatible : VS Code Local | CPU only
# =============================================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.impute           import KNNImputer, SimpleImputer
from sklearn.preprocessing    import OneHotEncoder, RobustScaler
from imblearn.over_sampling   import SMOTE

DATA_PATH        = "data/raw/stroke_data.csv"
PROCESSED_DIR    = Path("data/processed")
PREPROCESSOR_DIR = Path("models/preprocessors")
REPORTS_DIR      = Path("reports")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL       = "stroke"
DROP_COLS        = ["id"]
NUMERIC_COLS     = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL_COLS = ["gender", "work_type", "Residence_type",
                    "smoking_status", "ever_married"]
BINARY_COLS      = ["hypertension", "heart_disease"]

TEST_SIZE   = 0.20
SEED        = 42
SMOTE_RATIO = 1.0


# ── Nettoyage ─────────────────────────────────────────────────
def clean(df):
    df = df.drop(columns=DROP_COLS, errors="ignore").copy()
    df = df.drop_duplicates()
    df.loc[df["bmi"] == 0, "bmi"]  = np.nan
    df.loc[df["age"] < 0, "age"]   = np.nan
    df["gender"]         = df["gender"].replace("Other", "Unknown")
    df["smoking_status"] = df["smoking_status"].replace("Unknown", np.nan)
    return df


# ── Pipeline Sklearn ──────────────────────────────────────────
def build_preprocessor():
    num_pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler",  RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe,      NUMERIC_COLS),
        ("cat", cat_pipe,      CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ], remainder="drop")


# ── Pipeline complet ──────────────────────────────────────────
def plot_class_balance(before, after=None):
    data = []
    if before is not None:
        for label, count in before.items():
            data.append({"stage": "Avant SMOTE", "stroke": int(label), "count": int(count)})
    if after is not None:
        for label, count in after.items():
            data.append({"stage": "Après SMOTE", "stroke": int(label), "count": int(count)})

    df_plot = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df_plot, x="stroke", y="count", hue="stage", ax=ax)
    ax.set_title("Équilibre des classes avant/après SMOTE")
    ax.set_xlabel("stroke")
    ax.set_ylabel("Nombre de cas")
    ax.set_xticks([0, 1])
    ax.set_ylim(0, df_plot["count"].max() * 1.1)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_balance_smote.png")
    plt.close(fig)
    print(f"  Visualisation équilibrage sauvegardée -> {REPORTS_DIR / 'class_balance_smote.png'}")


def run_preprocessing(path=DATA_PATH):
    print("\nDEMARRAGE PREPROCESSING\n")

    df = clean(pd.read_csv(path))
    print(f"  Shape après nettoyage : {df.shape}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"  Train : {X_train.shape}  |  Test : {X_test.shape}")

    prep = build_preprocessor()
    X_train_p = prep.fit_transform(X_train)
    X_test_p  = prep.transform(X_test)

    balance_before = y_train.value_counts().sort_index()
    print(f"\n  Avant SMOTE : {balance_before.to_dict()}")
    plot_class_balance(balance_before, None)

    print("  Application de SMOTE... ")
    if len(set(y_train)) > 1 and y_train.value_counts().min() < y_train.value_counts().max():
       smote = SMOTE(random_state=42)
       X_train_p, y_train = smote.fit_resample(X_train_p, y_train)
       balance_after = pd.Series(y_train).value_counts().sort_index()
       print("Après SMOTE :", balance_after.to_dict())
       plot_class_balance(balance_before, balance_after)
    else:
       balance_after = balance_before.copy()
       print("⚠️ SMOTE skipped (data already balanced)")
       plot_class_balance(balance_before, balance_after)

    print(f"  Après SMOTE : {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Noms des features
    ohe_names = prep.named_transformers_["cat"]["ohe"] \
                    .get_feature_names_out(CATEGORICAL_COLS)
    feature_names = NUMERIC_COLS + list(ohe_names) + BINARY_COLS

    # Sauvegardes
    joblib.dump(prep,          PREPROCESSOR_DIR / "stroke_preprocessor.joblib")
    joblib.dump(feature_names, PREPROCESSOR_DIR / "feature_names.joblib")

    train_df = pd.DataFrame(X_train_p, columns=feature_names)
    train_df[TARGET_COL] = y_train.values
    test_df  = pd.DataFrame(X_test_p,  columns=feature_names)
    test_df[TARGET_COL]  = y_test.values

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv( PROCESSED_DIR / "test.csv",  index=False)

    print("\nPreprocessing termine")
    print("   Preprocessor -> models/preprocessors/stroke_preprocessor.joblib")
    print(f"   Train CSV    -> data/processed/train.csv  ({len(train_df)} lignes)")
    print(f"   Test CSV     -> data/processed/test.csv   ({len(test_df)} lignes)")

    return X_train_p, X_test_p, y_train, y_test, feature_names


if __name__ == "__main__":
    run_preprocessing()