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
