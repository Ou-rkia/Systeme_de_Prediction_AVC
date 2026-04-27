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
