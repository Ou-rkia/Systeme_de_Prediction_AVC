# 🧠 StrokePredict — Pipeline MLOps de Prédiction d'AVC

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.2-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.12-blue?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/Licence-MIT-yellow)

**Projet MLOps complet — de l'exploration des données au déploiement d'une API médicale**

*Filière 2IA — Semestre 4 — ENSIAS Université Mohammed V de Rabat*

</div>

---

## 📋 Table des Matières

- [Contexte](#-contexte)
- [Problématique médicale](#-problématique-médicale)
- [Architecture du pipeline](#-architecture-du-pipeline)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Exécution](#-exécution)
- [API REST](#-api-rest)
- [MLflow UI](#-mlflow-ui)
- [Tests](#-tests)
- [Docker](#-docker)
- [Résultats](#-résultats)
- [Technologies](#-technologies)
- [Auteurs](#-auteurs)

---

## 🏥 Contexte

Ce projet implémente un **pipeline MLOps complet** pour la prédiction du risque d'**Accident Vasculaire Cérébral (AVC)**. Il couvre l'intégralité du cycle de vie d'un projet de Machine Learning en milieu médical : exploration des données, preprocessing, modélisation, tracking des expériences, exposition via une API REST et conteneurisation Docker.

> ⚠️ **Avertissement médical** : Ce système est un outil d'aide à la décision à usage pédagogique. Il ne remplace en aucun cas le diagnostic d'un professionnel de santé.

---

## 🩺 Problématique Médicale

Le problème est formulé comme une **classification binaire** :

> Étant donné un ensemble de mesures cliniques d'un patient, prédire si ce patient présente un risque d'AVC (`stroke = 1`) ou non (`stroke = 0`).

### Pourquoi le Recall est la métrique principale ?

| Type d'erreur | Description | Conséquence |
|---|---|---|
| **Faux Négatif** | Prédit sain → patient à risque | ❌ AVC non détecté — conséquences graves |
| **Faux Positif** | Prédit AVC → patient sain | ⚠️ Examens inutiles — stress, coût |

En contexte médical, **manquer un AVC est bien plus grave** qu'une fausse alarme. Le pipeline est donc optimisé pour **maximiser le Recall**.

---

## 🔄 Architecture du Pipeline

```
Données brutes (CSV)
       │
       ▼
┌─────────────────┐
│  Étape 1 : EDA  │  ← Exploration, distributions, corrélations
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Étape 2 : Preproc   │  ← KNN Imputer + RobustScaler + SMOTE
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Étape 3 : Modèles   │  ← MLP / SVM / XGBoost + RandomizedSearchCV
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Étape 4 : MLflow    │  ← Tracking, comparaison, Model Registry
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Étape 5 : API REST  │  ← FastAPI → /predict /health /model-info
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Étape 6 : Docker    │  ← Image portable, docker-compose
└──────────────────────┘
```

---

## 📁 Structure du Projet

```
stroke_mlops/
│
├── 📂 data/
│   ├── raw/                        # Données brutes (gérées par DVC)
│   │   └── stroke_data.csv
│   └── processed/                  # Données après preprocessing
│       ├── train.csv
│       └── test.csv
│
├── 📂 src/                         # Modules Python production
│   ├── config.py                   # Paramètres centralisés
│   ├── data_loader.py              # Chargement + validation
│   ├── preprocessing.py            # Pipeline de transformation
│   ├── train.py                    # Entraînement + logging MLflow
│   ├── evaluate.py                 # Métriques + visualisations
│   └── predict.py                  # Inférence
│
├── 📂 models/
│   ├── trained/                    # Modèles sérialisés (.joblib)
│   │   ├── best_model_xgboost.joblib
│   │   ├── best_model_mlp.joblib
│   │   └── best_model_svm.joblib
│   ├── preprocessors/
│   │   ├── stroke_preprocessor.joblib
│   │   └── feature_names.joblib
│   └── best_model_info.json        # Infos du meilleur modèle
│
├── 📂 reports/                     # Graphiques générés automatiquement
│   ├── 01_target.png
│   ├── 02_numeric.png
│   ├── 03_categorical.png
│   ├── 04_missing.png
│   ├── 05_correlation.png
│   ├── 06_bivariate.png
│   ├── 07_confusion_matrices.png
│   ├── 08_roc_curves.png
│   └── 09_metrics_comparison.png
│
├── 📂 docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── 📂 mlruns/                      # Expériences MLflow (local)
│
├── etape1_generation_donnees.py    # Génération des données simulées
├── etape2_eda.py                   # Analyse exploratoire
├── etape3_preprocessing.py         # Preprocessing complet
├── etape4_modelisation.py          # Modélisation + comparaison
├── etape5_mlflow.py                # Tracking MLflow
├── etape6_structure_projet.py      # Génération des modules src/
├── etape7_api_fastapi.py           # API REST FastAPI
├── etape8_tests.py                 # Tests automatisés
├── etape9_docker.py                # Génération fichiers Docker
├── run_pipeline.py                 # Script maître d'exécution
│
├── requirements.txt                # Dépendances Python épinglées
├── .gitignore
├── .dockerignore
└── README.md
```

---

## ⚙️ Installation

### Prérequis

- Python **3.10+**
- Git
- Docker Desktop *(optionnel — pour l'étape 9)*

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-username>/stroke_mlops.git
cd stroke_mlops
```

### 2. Créer l'environnement virtuel

```bash
# Créer
python -m venv venv

# Activer — Windows
venv\Scripts\activate

# Activer — Linux / Mac
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> ⏱️ Installation : **3 à 5 minutes** selon la connexion.

---

## ▶️ Exécution

### Option A — Pipeline complet automatique

```bash
python run_pipeline.py
```

### Option B — Étape par étape

```bash
# Étape 1 — Générer les données simulées
python etape1_generation_donnees.py

# Étape 2 — Analyse exploratoire (graphiques → reports/)
python etape2_eda.py

# Étape 3 — Preprocessing (KNN + SMOTE + RobustScaler)
python etape3_preprocessing.py

# Étape 4 — Modélisation MLP / SVM / XGBoost  (~2-8 min CPU)
python etape4_modelisation.py

# Étape 5 — Tracking MLflow
python etape5_mlflow.py

# Étape 6 — Générer les modules src/
python etape6_structure_projet.py

# Étape 7 — Tester l'API (mode inline)
python etape7_api_fastapi.py

# Étape 8 — Tests automatisés
pytest etape8_tests.py -v

# Étape 9 — Générer les fichiers Docker
python etape9_docker.py
```

### Option C — Étape unique

```bash
python run_pipeline.py --step 4   # Exécuter uniquement l'étape 4
```

---

## 🌐 API REST

### Lancer le serveur

```bash
uvicorn etape7_api_fastapi:app --reload --port 8000
```

### Documentation interactive

```
http://localhost:8000/docs
```

### Endpoints disponibles

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Statut de l'API et du modèle |
| `POST` | `/predict` | Prédiction AVC pour un patient |
| `GET` | `/model-info` | Métriques et métadonnées du modèle |

### Exemple de requête

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67,
    "gender": "Male",
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

### Exemple de réponse

```json
{
  "request_id": "a3f2c1d4-...",
  "prediction": {
    "stroke_label": 1,
    "stroke_risk": 0.8912,
    "confidence": "high",
    "interpretation": "⚠️ Risque élevé d'AVC — consultation médicale urgente recommandée."
  },
  "model_version": "XGBoost",
  "timestamp": "2026-04-18T18:37:00Z"
}
```

---

## 📊 MLflow UI

```bash
mlflow ui
```

Ouvrir dans le navigateur : **http://localhost:5000**

Vous y trouverez :
- La comparaison des 3 modèles (MLP, SVM, XGBoost)
- Les hyperparamètres de chaque run
- Les courbes ROC et matrices de confusion
- Le modèle champion enregistré en stage **Production**

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest etape8_tests.py -v

# Avec rapport de couverture
pytest etape8_tests.py -v --tb=short
```

### Couverture des tests

| Classe de tests | Ce qui est testé |
|---|---|
| `TestGenerationDonnees` | CSV créé, shape, classes, NaN, colonnes |
| `TestPreprocessing` | Nettoyage, absence de NaN, forme de sortie |
| `TestPrediction` | Clés de sortie, label binaire, probabilité dans [0,1] |
| `TestAPI` | Codes HTTP, validation Pydantic, edge cases |
| `TestEndToEnd` | Mini-pipeline 10 lignes, health → predict |

---

## 🐳 Docker

### Construire et lancer

```bash
# Construire l'image
docker build -f docker/Dockerfile -t strokepredict .

# Lancer le conteneur
docker run -d \
  --name strokepredict \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  strokepredict

# Vérifier
curl http://localhost:8000/health
```

### Avec Docker Compose

```bash
cd docker/

# API seule
docker-compose up -d strokepredict-api

# API + MLflow UI
docker-compose --profile mlflow up -d

# Arrêter
docker-compose down
```

---

## 📈 Résultats

### Tableau comparatif des modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **XGBoost** ⭐ | ~0.91 | ~0.89 | ~0.94 | ~0.91 | ~0.97 |
| MLP | ~0.88 | ~0.85 | ~0.91 | ~0.88 | ~0.95 |
| SVM | ~0.86 | ~0.83 | ~0.89 | ~0.86 | ~0.93 |

> 🏆 **XGBoost** sélectionné comme modèle de production grâce au meilleur compromis Recall / F1-Score.

### Choix de preprocessing

| Décision | Justification |
|---|---|
| **KNN Imputer** pour BMI | Préserve les relations entre variables (vs médiane simple) |
| **RobustScaler** | Résistant aux hyperglycémies extrêmes (outliers) |
| **SMOTE ratio 1:1** | Compense le fort déséquilibre (4.9% positifs) |
| **Recall** comme métrique | Faux négatif = AVC manqué → conséquence médicale grave |

---

## 🛠️ Technologies

| Catégorie | Outil | Version |
|---|---|---|
| Langage | Python | 3.11 |
| ML | Scikit-learn | 1.4.2 |
| Boosting | XGBoost | 2.0.3 |
| Déséquilibre | Imbalanced-learn | 0.12.2 |
| Tracking | MLflow | 2.12.1 |
| API | FastAPI + Uvicorn | 0.111 |
| Validation | Pydantic | 2.7.0 |
| Tests | Pytest + HTTPX | 8.2.0 |
| Conteneur | Docker | — |
| Visualisation | Matplotlib + Seaborn | 3.8 / 0.13 |

---

## 👨‍💻 Auteurs

| Nom | Rôle |
|---|---|
| **[Votre Nom]** | Data Science, MLOps, API |
| **[Nom Binôme]** | EDA, Preprocessing, Tests |

**Encadrant :** *[Nom du professeur]*

**Établissement :** ENSIAS — Université Mohammed V de Rabat
**Filière :** 2IA — Semestre 4 — Module MLOps
**Année :** 2025-2026

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**.
Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

<div align="center">

*Projet réalisé dans le cadre du module MLOps — ENSIAS Rabat 2026*

</div>