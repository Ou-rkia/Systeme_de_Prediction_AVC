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
