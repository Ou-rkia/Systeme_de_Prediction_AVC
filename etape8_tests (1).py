# =============================================================
# ÉTAPE 8 : TESTS AUTOMATISÉS
# Compatible : VS Code Local | CPU only
# Lancer : pytest etape8_tests.py -v
# =============================================================

import sys
import json
import numpy as np
import pandas as pd
import pytest
from pathlib       import Path
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════
# FIXTURES COMMUNES
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def valid_patient():
    return {
        "age": 67.0, "gender": "Male", "hypertension": 1,
        "heart_disease": 1, "ever_married": "Yes",
        "work_type": "Private", "Residence_type": "Urban",
        "avg_glucose_level": 228.69, "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age":               [67.0, 55.0, 30.0, 80.0, 45.0],
        "gender":            ["Male", "Female", "Male", "Female", "Male"],
        "hypertension":      [1, 0, 0, 1, 0],
        "heart_disease":     [1, 0, 0, 0, 1],
        "ever_married":      ["Yes", "Yes", "No", "Yes", "No"],
        "work_type":         ["Private","Self-employed","Private","Govt_job","Private"],
        "Residence_type":    ["Urban", "Rural", "Urban", "Rural", "Urban"],
        "avg_glucose_level": [228.69, 95.0, 80.0, 150.0, np.nan],
        "bmi":               [36.6, np.nan, 22.0, 28.0, 25.0],
        "smoking_status":    ["formerly smoked","never smoked","smokes",np.nan,"never smoked"],
        "stroke":            [1, 0, 0, 1, 0],
    })

@pytest.fixture
def mock_model():
    m = MagicMock()
    m.predict.return_value       = np.array([1])
    m.predict_proba.return_value = np.array([[0.12, 0.88]])
    return m

@pytest.fixture
def mock_preprocessor():
    p = MagicMock()
    p.transform.return_value = np.random.rand(1, 20)
    return p


# ══════════════════════════════════════════════════════════════
# TESTS 1 — GÉNÉRATION DES DONNÉES
# ══════════════════════════════════════════════════════════════

class TestGenerationDonnees:

    def test_csv_created(self, tmp_path, monkeypatch):
        """Le fichier CSV doit être créé."""
        import importlib, types
        # Injecter le chemin tmp
        out = str(tmp_path / "stroke_data.csv")
        # Exécuter la fonction sans modifier le module global
        import etape1_generation_donnees as gen
        monkeypatch.setattr(gen, "OUTPUT_PATH", out)
        df = gen.generate_dataset()
        assert Path(out).exists(), "Fichier CSV non créé"

    def test_correct_shape(self, tmp_path, monkeypatch):
        import etape1_generation_donnees as gen
        out = str(tmp_path / "s.csv")
        monkeypatch.setattr(gen, "OUTPUT_PATH", out)
        df = gen.generate_dataset()
        assert len(df) == gen.N_STROKE + gen.N_NO_STROKE

    def test_balanced_classes(self, tmp_path, monkeypatch):
        import etape1_generation_donnees as gen
        out = str(tmp_path / "s.csv")
        monkeypatch.setattr(gen, "OUTPUT_PATH", out)
        df = gen.generate_dataset()
        assert df["stroke"].sum() == gen.N_STROKE
        assert (df["stroke"] == 0).sum() == gen.N_NO_STROKE

    def test_missing_values_present(self, tmp_path, monkeypatch):
        import etape1_generation_donnees as gen
        out = str(tmp_path / "s.csv")
        monkeypatch.setattr(gen, "OUTPUT_PATH", out)
        df = gen.generate_dataset()
        assert df["bmi"].isnull().sum() > 0, "Il devrait y avoir des NaN dans bmi"

    def test_expected_columns(self, tmp_path, monkeypatch):
        import etape1_generation_donnees as gen
        out = str(tmp_path / "s.csv")
        monkeypatch.setattr(gen, "OUTPUT_PATH", out)
        df = gen.generate_dataset()
        expected = {"id","age","gender","hypertension","heart_disease",
                    "ever_married","work_type","Residence_type",
                    "avg_glucose_level","bmi","smoking_status","stroke"}
        assert expected.issubset(set(df.columns))


# ══════════════════════════════════════════════════════════════
# TESTS 2 — PREPROCESSING
# ══════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_clean_removes_duplicates(self, sample_df):
        import etape3_preprocessing as prep
        df_dup = pd.concat([sample_df, sample_df], ignore_index=True)
        result = prep.clean(df_dup)
        assert len(result) == len(sample_df)

    def test_clean_bmi_zero_becomes_nan(self):
        import etape3_preprocessing as prep
        df = pd.DataFrame({
            "age":[30.0],"gender":["Male"],"hypertension":[0],
            "heart_disease":[0],"ever_married":["Yes"],
            "work_type":["Private"],"Residence_type":["Urban"],
            "avg_glucose_level":[80.0],"bmi":[0.0],
            "smoking_status":["never smoked"],"stroke":[0]
        })
        result = prep.clean(df)
        assert pd.isna(result["bmi"].iloc[0])

    def test_preprocessor_no_nan_output(self, sample_df):
        import etape3_preprocessing as prep
        X = sample_df.drop(columns=["stroke"])
        p = prep.build_preprocessor()
        X_out = p.fit_transform(X)
        assert not np.isnan(X_out).any(), "Des NaN persistent après preprocessing"

    def test_preprocessor_shape_preserved(self, sample_df):
        import etape3_preprocessing as prep
        X = sample_df.drop(columns=["stroke"])
        p = prep.build_preprocessor()
        X_out = p.fit_transform(X)
        assert X_out.shape[0] == len(sample_df)
        assert X_out.shape[1] > 0

    def test_values_in_reasonable_range(self, sample_df):
        import etape3_preprocessing as prep
        X = sample_df.drop(columns=["stroke"])
        p = prep.build_preprocessor()
        X_out = p.fit_transform(X)
        # Colonnes numériques centrées-réduites : pas de valeur > 100
        assert np.abs(X_out[:, :3]).max() < 100


# ══════════════════════════════════════════════════════════════
# TESTS 3 — PRÉDICTION
# ══════════════════════════════════════════════════════════════

class TestPrediction:

    def _make_info(self):
        return {"model_name": "XGBoost", "test_f1": 0.9,
                "test_recall": 0.95, "test_auc_roc": 0.97}

    def test_output_keys_present(self, valid_patient, mock_model, mock_preprocessor):
        import etape7_api_fastapi as api
        api._model        = mock_model
        api._preprocessor = mock_preprocessor
        api._model_info   = self._make_info()

        from fastapi.testclient import TestClient
        client = TestClient(api.app)
        r = client.post("/predict", json=valid_patient)
        assert r.status_code == 200
        data = r.json()
        assert "request_id"    in data
        assert "prediction"    in data
        assert "model_version" in data

    def test_stroke_label_binary(self, valid_patient, mock_model, mock_preprocessor):
        import etape7_api_fastapi as api
        api._model        = mock_model
        api._preprocessor = mock_preprocessor
        api._model_info   = self._make_info()

        from fastapi.testclient import TestClient
        client = TestClient(api.app)
        r = client.post("/predict", json=valid_patient)
        assert r.json()["prediction"]["stroke_label"] in [0, 1]

    def test_risk_in_range(self, valid_patient, mock_model, mock_preprocessor):
        import etape7_api_fastapi as api
        api._model        = mock_model
        api._preprocessor = mock_preprocessor
        api._model_info   = self._make_info()

        from fastapi.testclient import TestClient
        client = TestClient(api.app)
        r = client.post("/predict", json=valid_patient)
        risk = r.json()["prediction"]["stroke_risk"]
        assert 0.0 <= risk <= 1.0


# ══════════════════════════════════════════════════════════════
# TESTS 4 — API
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def client(mock_model, mock_preprocessor):
    import etape7_api_fastapi as api
    api._model        = mock_model
    api._preprocessor = mock_preprocessor
    api._model_info   = {
        "model_name": "XGBoost", "test_f1": 0.91,
        "test_recall": 0.94,     "test_auc_roc": 0.97
    }
    from fastapi.testclient import TestClient
    return TestClient(api.app)


class TestAPI:

    def test_health_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_model_loaded(self, client):
        assert client.get("/health").json()["model_loaded"] is True

    def test_predict_200(self, client, valid_patient):
        assert client.post("/predict", json=valid_patient).status_code == 200

    def test_predict_invalid_gender_422(self, client, valid_patient):
        valid_patient["gender"] = "Robot"
        assert client.post("/predict", json=valid_patient).status_code == 422

    def test_predict_missing_age_422(self, client, valid_patient):
        del valid_patient["age"]
        assert client.post("/predict", json=valid_patient).status_code == 422

    def test_predict_negative_age_422(self, client, valid_patient):
        valid_patient["age"] = -5.0
        assert client.post("/predict", json=valid_patient).status_code == 422

    def test_predict_null_bmi_200(self, client, valid_patient):
        valid_patient["bmi"] = None
        assert client.post("/predict", json=valid_patient).status_code == 200

    def test_model_info_200(self, client):
        assert client.get("/model-info").status_code == 200

    def test_model_info_has_recall(self, client):
        assert "test_recall" in client.get("/model-info").json()


# ══════════════════════════════════════════════════════════════
# TEST 5 — END-TO-END (mini pipeline)
# ══════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_full_pipeline_10_rows(self, tmp_path):
        """Mini-pipeline sur 10 lignes — vérifier cohérence finale."""
        # Générer mini-dataset
        import etape1_generation_donnees as gen
        out = str(tmp_path / "mini.csv")
        import unittest.mock as mock
        with mock.patch.object(gen, "N_STROKE",    5), \
             mock.patch.object(gen, "N_NO_STROKE", 5), \
             mock.patch.object(gen, "OUTPUT_PATH", out):
            df = gen.generate_dataset()

        assert len(df) == 10
        assert "stroke" in df.columns
        assert df["stroke"].isin([0, 1]).all()

    def test_api_health_then_predict(self, client, valid_patient):
        """Appel health puis predict — vérifier cohérence."""
        h = client.get("/health").json()
        assert h["status"] == "healthy"

        p = client.post("/predict", json=valid_patient).json()
        assert p["prediction"]["stroke_label"] in [0, 1]
        assert 0.0 <= p["prediction"]["stroke_risk"] <= 1.0


# ── Point d'entrée ────────────────────────────────────────────
if __name__ == "__main__":
    # Lancer avec : python etape8_tests.py
    # ou          : pytest etape8_tests.py -v
    pytest.main([__file__, "-v", "--tb=short"])
