# =============================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES RÉELLES (Kaggle)
# Dataset : Healthcare Dataset Stroke Data
# Compatible : VS Code Local | CPU only | Pas de GPU requis
# =============================================================

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/stroke_data.csv")

EXPECTED_COLUMNS = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke"
]

def validate_dataset(df: pd.DataFrame) -> None:
    """Vérifie que le fichier chargé est bien le bon dataset Kaggle."""

    # Vérification des colonnes
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans le fichier : {missing_cols}\n"
            f"Colonnes trouvées : {list(df.columns)}\n"
            f"Vérifie que c'est bien le fichier Kaggle 'healthcare-dataset-stroke-data.csv'."
        )

    # Vérification de la colonne cible
    if not set(df["stroke"].dropna().unique()).issubset({0, 1}):
        raise ValueError("La colonne 'stroke' doit contenir uniquement 0 et 1.")

    # Vérification taille minimale
    if len(df) < 100:
        raise ValueError(f"Dataset trop petit ({len(df)} lignes). Vérifie le fichier.")

    print("   Validation OK.")


def load_dataset() -> pd.DataFrame:
    """
    Charge le dataset réel Kaggle depuis data/raw/stroke_data.csv.
    Affiche un résumé clair des données.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\n❌ Fichier introuvable : {DATA_PATH}\n\n"
            f"Pour obtenir le dataset :\n"
            f"   1. Va sur : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset\n"
            f"   2. Télécharge 'healthcare-dataset-stroke-data.csv'\n"
            f"   3. Renomme-le en 'stroke_data.csv'\n"
            f"   4. Place-le dans le dossier : data/raw/\n"
        )

    print(f"Chargement des données réelles -> {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    validate_dataset(df)

    # Résumé
    n_total   = len(df)
    n_stroke  = int(df["stroke"].sum())
    n_healthy = int((df["stroke"] == 0).sum())
    n_nan     = int(df.isnull().sum().sum())
    pct_pos   = n_stroke / n_total * 100

    print(f"   Lignes totales  : {n_total}")
    print(f"   AVC (stroke=1)  : {n_stroke}  ({pct_pos:.1f}%)")
    print(f"   Sains (stroke=0): {n_healthy}")
    print(f"   Valeurs NaN     : {n_nan}  (colonne bmi principalement)")
    print(f"   Colonnes        : {list(df.columns)}")

    # Avertissement déséquilibre
    if pct_pos < 10:
        print(
            f"\n   ⚠  Dataset déséquilibré ({pct_pos:.1f}% positifs).\n"
            f"      SMOTE sera appliqué à l'étape 3 pour corriger ce déséquilibre."
        )

    return df


# Alias pour compatibilité avec le reste du pipeline
def load_or_generate_dataset(force_generate=False) -> pd.DataFrame:
    """
    Remplace l'ancienne fonction qui générait des données synthétiques.
    Maintenant charge uniquement les données réelles Kaggle.
    Le paramètre force_generate est conservé pour compatibilité mais ignoré.
    """
    if force_generate:
        print("⚠  Paramètre force_generate ignoré : les données sont réelles, pas générées.")

    return load_dataset()


if __name__ == "__main__":
    df = load_or_generate_dataset()
    print("\nAperçu des données :")
    print(df.head())
    print("\nTypes des colonnes :")
    print(df.dtypes)
    print("\nValeurs manquantes par colonne :")
    print(df.isnull().sum()[df.isnull().sum() > 0])
