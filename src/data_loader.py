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
            raise TypeError(f"Colonne '{col}' doit être numérique.")
    return df

def load_patient_dict(patient: dict) -> pd.DataFrame:
    required = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS
    missing  = [f for f in required if f not in patient]
    if missing:
        raise ValueError(f"Champs manquants : {missing}")
    return pd.DataFrame([patient])
