# =============================================================
# ÉTAPE 7 : API FASTAPI
# Compatible : VS Code Local | CPU only
# Lancer : uvicorn etape7_api_fastapi:app --reload --port 8000
# Docs   : http://localhost:8000/docs
# =============================================================

import sys, json, uuid, joblib
import numpy as np
import pandas as pd
from pathlib       import Path
from datetime      import datetime
from typing        import Optional

from fastapi           import FastAPI, HTTPException
from pydantic          import BaseModel, Field, field_validator

# ── Chemins ──────────────────────────────────────────────────
PREPROCESSOR_PATH = Path("models/preprocessors/stroke_preprocessor.joblib")
MODEL_DIR         = Path("models/trained")
BEST_MODEL_INFO   = Path("models/best_model_info.json")
MODEL_VERSION     = "1.0.0"

# ── Schémas Pydantic ─────────────────────────────────────────
class PatientFeatures(BaseModel):
    age:               float         = Field(..., ge=0,  le=120)
    gender:            str
    hypertension:      int           = Field(..., ge=0,  le=1)
    heart_disease:     int           = Field(..., ge=0,  le=1)
    ever_married:      str
    work_type:         str
    Residence_type:    str
    avg_glucose_level: float         = Field(..., ge=50, le=400)
    bmi:               Optional[float] = Field(None, ge=10, le=80)
    smoking_status:    str

    @field_validator("gender")
    @classmethod
    def val_gender(cls, v):
        if v not in {"Male", "Female", "Unknown"}:
            raise ValueError("gender doit être Male | Female | Unknown")
        return v

    @field_validator("ever_married")
    @classmethod
    def val_married(cls, v):
        if v not in {"Yes", "No"}:
            raise ValueError("ever_married doit être Yes | No")
        return v

    @field_validator("work_type")
    @classmethod
    def val_work(cls, v):
        allowed = {"Private", "Self-employed", "Govt_job", "children", "Never_worked"}
        if v not in allowed:
            raise ValueError(f"work_type doit être parmi {allowed}")
        return v

    @field_validator("Residence_type")
    @classmethod
    def val_residence(cls, v):
        if v not in {"Urban", "Rural"}:
            raise ValueError("Residence_type doit être Urban | Rural")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 67, "gender": "Male",
                "hypertension": 1, "heart_disease": 1,
                "ever_married": "Yes", "work_type": "Private",
                "Residence_type": "Urban", "avg_glucose_level": 228.69,
                "bmi": 36.6, "smoking_status": "formerly smoked"
            }
        }
    }


class PredictionResponse(BaseModel):
    request_id:    str
    prediction:    dict
    model_version: str
    timestamp:     str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_name:   str
    api_version:  str
    timestamp:    str


# ── App FastAPI ───────────────────────────────────────────────
app = FastAPI(
    title="StrokePredict API",
    description="API de prédiction du risque d'AVC",
    version=MODEL_VERSION,
)

_model        = None
_preprocessor = None
_model_info   = {}


def load_artifacts():
    global _model, _preprocessor, _model_info
    try:
        with open(BEST_MODEL_INFO) as f:
            _model_info = json.load(f)
        name       = _model_info["model_name"].lower()
        model_path = MODEL_DIR / f"best_model_{name}.joblib"
        _model        = joblib.load(model_path)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"✅ Modèle chargé : {_model_info['model_name']}")
        return True
    except FileNotFoundError as e:
        print(f"⚠️  Artefacts manquants ({e}). Exécutez les étapes 1-5 d'abord.")
        return False


@app.on_event("startup")
async def startup():
    load_artifacts()


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {"message": "StrokePredict API 🧠", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    return HealthResponse(
        status       = "healthy" if _model else "degraded",
        model_loaded = _model is not None,
        model_name   = _model_info.get("model_name", "N/A"),
        api_version  = MODEL_VERSION,
        timestamp    = datetime.utcnow().isoformat() + "Z",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientFeatures):
    """
    Prédit le risque d'AVC pour un patient.
    - **stroke_label** : 0 ou 1
    - **stroke_risk**  : probabilité [0-1]
    - **confidence**   : high | medium | low
    """
    if _model is None:
        raise HTTPException(503, detail="Modèle non disponible.")
    try:
        df    = pd.DataFrame([patient.model_dump()])
        X     = _preprocessor.transform(df)
        label = int(_model.predict(X)[0])
        proba = float(_model.predict_proba(X)[0][1])
        conf  = "high" if (proba > 0.80 or proba < 0.20) else \
                "medium" if (proba > 0.65 or proba < 0.35) else "low"

        return PredictionResponse(
            request_id    = str(uuid.uuid4()),
            prediction    = {
                "stroke_label":    label,
                "stroke_risk":     round(proba, 4),
                "confidence":      conf,
                "interpretation":  (
                    "⚠️  Risque élevé d'AVC — consultation médicale urgente recommandée."
                    if label == 1 else
                    "✅ Risque faible d'AVC — maintenir le suivi préventif."
                ),
            },
            model_version = _model_info.get("model_name", MODEL_VERSION),
            timestamp     = datetime.utcnow().isoformat() + "Z",
        )
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/model-info", tags=["Monitoring"])
def model_info():
    if not _model_info:
        raise HTTPException(503, detail="Modèle non chargé.")
    return {
        "model_name":     _model_info.get("model_name"),
        "test_f1":        _model_info.get("test_f1"),
        "test_recall":    _model_info.get("test_recall"),
        "test_auc_roc":   _model_info.get("test_auc_roc"),
        "registered_at":  _model_info.get("registered_at"),
        "primary_metric": "Recall — minimiser les faux négatifs",
    }


# ── Point d'entrée direct ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    load_artifacts()
    uvicorn.run("etape7_api_fastapi:app",
                host="0.0.0.0", port=8000, reload=True)
