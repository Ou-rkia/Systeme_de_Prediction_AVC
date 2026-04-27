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
