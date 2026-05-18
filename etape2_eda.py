# =============================================================
# ÉTAPE 2 : EDA — ANALYSE EXPLORATOIRE
# Compatible : VS Code Local | CPU only
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Pas d'affichage GUI — sauvegarde en PNG
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})

DATA_PATH   = "data/raw/stroke_data.csv"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLS = ["age", "avg_glucose_level", "bmi"]
CAT_COLS     = ["gender", "work_type", "Residence_type",
                "smoking_status", "hypertension", "heart_disease", "ever_married"]


# ── 1. Vue d'ensemble ────────────────────────────────────────
def overview(df):
    print("=" * 55)
    print("  VUE D'ENSEMBLE")
    print("=" * 55)
    print(f"  Shape      : {df.shape}")
    print(f"  Doublons   : {df.duplicated().sum()}")
    print(f"  NaN total  : {df.isnull().sum().sum()}")
    print("\n── Types ──")
    print(df.dtypes.to_string())
    print("\n── Statistiques ──")
    print(df.describe(include="all").T.to_string())


# ── 2. Distribution cible ────────────────────────────────────
def plot_target(df):
    counts = df["stroke"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(["Pas d'AVC (0)", "AVC (1)"], counts.values,
                color=["#4C9BE8", "#E8574C"], edgecolor="white")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 3, str(v), ha="center", fontweight="bold")
    axes[0].set_title("Distribution de la variable cible", fontweight="bold")

    axes[1].pie(counts.values, labels=["Pas d'AVC", "AVC"],
                autopct="%1.1f%%", colors=["#4C9BE8", "#E8574C"],
                wedgeprops=dict(edgecolor="white"))
    axes[1].set_title("Proportion des classes", fontweight="bold")

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "01_target.png")
    plt.close(fig)
    print(f"\n✅ Graphique sauvegardé → reports/01_target.png")
    print(f"   Ratio déséquilibre : {counts[1]/counts[0]:.3f}")


# ── 3. Variables numériques ──────────────────────────────────
def plot_numeric(df):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    for i, col in enumerate(NUMERIC_COLS):
        # Histogramme
        axes[0, i].hist(df[col].dropna(), bins=30,
                        color="#4C9BE8", edgecolor="white", alpha=0.85)
        axes[0, i].axvline(df[col].median(), color="#E8574C",
                           linestyle="--", label=f"Médiane={df[col].median():.1f}")
        axes[0, i].set_title(f"Distribution — {col}", fontweight="bold")
        axes[0, i].legend(fontsize=8)

        # Boxplot
        axes[1, i].boxplot(df[col].dropna(), vert=False, patch_artist=True,
                           boxprops=dict(facecolor="#4C9BE8", alpha=0.5))
        axes[1, i].set_title(f"Boxplot — {col}", fontweight="bold")

    plt.suptitle("Variables numériques", fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "02_numeric.png")
    plt.close(fig)

    print("\n── Skewness / Kurtosis ──")
    print(df[NUMERIC_COLS].agg(["mean", "median", "std", "skew", "kurt"]).T.round(3).to_string())
    print("✅ Graphique sauvegardé → reports/02_numeric.png")


# ── 4. Variables catégorielles ───────────────────────────────
def plot_categorical(df):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(CAT_COLS):
        counts = df[col].value_counts()
        axes[i].bar(counts.index.astype(str), counts.values,
                    color=sns.color_palette("muted", len(counts)), edgecolor="white")
        axes[i].set_title(col, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=20)

    axes[-1].set_visible(False)
    plt.suptitle("Variables catégorielles", fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "03_categorical.png")
    plt.close(fig)
    print("✅ Graphique sauvegardé → reports/03_categorical.png")


# ── 5. Valeurs manquantes ────────────────────────────────────
def plot_missing(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"N": missing, "%": missing_pct})
    miss_df = miss_df[miss_df["N"] > 0].sort_values("N", ascending=False)

    print("\n── Valeurs manquantes ──")
    print(miss_df.to_string() if not miss_df.empty else "  Aucune.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Heatmap
    sample = df.sample(min(150, len(df)), random_state=42)
    sns.heatmap(sample.isnull(), cbar=False, yticklabels=False,
                cmap=["#E8EAF6", "#E8574C"], ax=axes[0])
    axes[0].set_title("Carte NaN (150 lignes)", fontweight="bold")

    # Barplot %
    if not miss_df.empty:
        axes[1].barh(miss_df.index, miss_df["%"], color="#E8574C", edgecolor="white")
        axes[1].set_xlabel("% Manquant")
        axes[1].set_title("Taux de NaN par variable", fontweight="bold")
        for j, v in enumerate(miss_df["%"]):
            axes[1].text(v + 0.1, j, f"{v}%", va="center", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "Aucune valeur manquante",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "04_missing.png")
    plt.close(fig)
    print("✅ Graphique sauvegardé → reports/04_missing.png")


# ── 6. Corrélations ──────────────────────────────────────────
def plot_correlation(df):
    num_df = df[NUMERIC_COLS + ["hypertension", "heart_disease", "stroke"]].copy()
    corr   = num_df.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, annot_kws={"size": 10})
    ax.set_title("Matrice de corrélation", fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "05_correlation.png")
    plt.close(fig)
    print("✅ Graphique sauvegardé → reports/05_correlation.png")


# ── 7. Analyse bivariée ──────────────────────────────────────
def plot_bivariate(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for i, col in enumerate(NUMERIC_COLS):
        for val, color, label in zip([0, 1], ["#4C9BE8", "#E8574C"],
                                     ["Pas d'AVC", "AVC"]):
            data = df[df["stroke"] == val][col].dropna()
            axes[i].hist(data, bins=25, alpha=0.6, color=color, label=label, edgecolor="white")
        axes[i].set_title(f"{col} par groupe", fontweight="bold")
        axes[i].legend()

    plt.suptitle("Distribution par classe (AVC vs Sain)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "06_bivariate.png")
    plt.close(fig)
    print("✅ Graphique sauvegardé → reports/06_bivariate.png")


# ── Pipeline EDA complet ─────────────────────────────────────
def run_full_eda(path=DATA_PATH):
    print("\n🔍 DÉMARRAGE EDA\n")
    df = pd.read_csv(path)
    overview(df)
    plot_target(df)
    plot_numeric(df)
    plot_categorical(df)
    plot_missing(df)
    plot_correlation(df)
    plot_bivariate(df)
    print(f"\n✅ EDA terminée — graphiques dans '{REPORTS_DIR}/'")
    return df


if __name__ == "__main__":
    run_full_eda()