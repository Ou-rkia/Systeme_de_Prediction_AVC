# =============================================================
# ÉTAPE 1 : GÉNÉRATION DES DONNÉES SIMULÉES
# Compatible : VS Code Local | CPU only | Pas de GPU requis
# =============================================================

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

N_STROKE    = 500
N_NO_STROKE = 500
OUTPUT_PATH = "data/raw/stroke_data.csv"


def generate_stroke_cases(n):
    return pd.DataFrame({
        "age":               np.random.normal(68, 10, n).clip(30, 95),
        "gender":            np.random.choice(["Male", "Female"], n, p=[0.48, 0.52]),
        "hypertension":      np.random.choice([0, 1], n, p=[0.35, 0.65]),
        "heart_disease":     np.random.choice([0, 1], n, p=[0.40, 0.60]),
        "ever_married":      np.random.choice(["Yes", "No"], n, p=[0.85, 0.15]),
        "work_type":         np.random.choice(
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            n, p=[0.55, 0.25, 0.15, 0.03, 0.02]),
        "Residence_type":    np.random.choice(["Urban", "Rural"], n, p=[0.55, 0.45]),
        "avg_glucose_level": np.random.normal(130, 40, n).clip(55, 300),
        "bmi":               np.random.normal(32, 7, n).clip(15, 55),
        "smoking_status":    np.random.choice(
            ["formerly smoked", "never smoked", "smokes", "Unknown"],
            n, p=[0.35, 0.30, 0.25, 0.10]),
        "stroke": np.ones(n, dtype=int),
    })


def generate_no_stroke_cases(n):
    return pd.DataFrame({
        "age":               np.random.normal(43, 15, n).clip(10, 90),
        "gender":            np.random.choice(["Male", "Female"], n, p=[0.49, 0.51]),
        "hypertension":      np.random.choice([0, 1], n, p=[0.80, 0.20]),
        "heart_disease":     np.random.choice([0, 1], n, p=[0.90, 0.10]),
        "ever_married":      np.random.choice(["Yes", "No"], n, p=[0.60, 0.40]),
        "work_type":         np.random.choice(
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            n, p=[0.58, 0.16, 0.14, 0.09, 0.03]),
        "Residence_type":    np.random.choice(["Urban", "Rural"], n, p=[0.52, 0.48]),
        "avg_glucose_level": np.random.normal(90, 25, n).clip(55, 250),
        "bmi":               np.random.normal(27, 6, n).clip(12, 50),
        "smoking_status":    np.random.choice(
            ["formerly smoked", "never smoked", "smokes", "Unknown"],
            n, p=[0.18, 0.48, 0.18, 0.16]),
        "stroke": np.zeros(n, dtype=int),
    })


def generate_dataset():
    df = pd.concat(
        [generate_stroke_cases(N_STROKE), generate_no_stroke_cases(N_NO_STROKE)],
        ignore_index=True
    )
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    # Introduire ~13% de NaN dans bmi
    idx = df.sample(frac=0.13, random_state=SEED).index
    df.loc[idx, "bmi"] = np.nan

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Dataset sauvegardé → {OUTPUT_PATH}")
    print(f"   Lignes       : {len(df)}")
    print(f"   AVC (1)      : {df['stroke'].sum()}")
    print(f"   Sain (0)     : {(df['stroke']==0).sum()}")
    print(f"   NaN total    : {df.isnull().sum().sum()}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
