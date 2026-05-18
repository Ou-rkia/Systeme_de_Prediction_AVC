# 🧠 StrokePredict — Système de Prédiction d'AVC

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4.2-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.12-blue?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-MIT-yellow)

**Pipeline MLOps complet — des données réelles au déploiement avec chatbot médical**

*Filière 2IA — Semestre 4 — ENSIAS, Université Mohammed V de Rabat*

</div>

---

## 📋 Table des matières

- [Contexte](#-contexte)
- [Données réelles Kaggle](#-données-réelles-kaggle)
- [Problématique médicale](#-problématique-médicale)
- [Architecture du pipeline](#-architecture-du-pipeline)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Exécution](#-exécution)
- [API REST](#-api-rest)
- [Chatbot AVC](#-chatbot-avc)
- [MLflow UI](#-mlflow-ui)
- [Tests](#-tests)
- [Docker](#-docker)
- [Résultats](#-résultats)
- [Corrections appliquées](#-corrections-appliquées)
- [Technologies](#-technologies)
- [Auteure](#-auteure)

---

## 🏥 Contexte

Ce projet implémente un **pipeline MLOps complet** pour la prédiction du risque d'**Accident Vasculaire Cérébral (AVC)**. Il couvre l'intégralité du cycle de vie d'un projet de Machine Learning en milieu médical : exploration des données, prétraitement, modélisation, suivi des expériences, exposition via une API REST, interface chatbot et conteneurisation Docker.

> ⚕️ **Avertissement médical** : Ce système est un outil d'aide à la décision à usage pédagogique. Il ne remplace en aucun cas le diagnostic d'un professionnel de santé.

---

## 📂 Données réelles Kaggle

Ce projet utilise exclusivement des **données réelles**. Le fichier utilisé est `healthcare-dataset-stroke-data.csv`, publié sur Kaggle par fedesoriano (2021).

> ⚠️ **Note** : Le fichier `etape1_generation_donnees.py` portait initialement un nom trompeur. Il s'agit en réalité du **chargement et de la validation** des données réelles. Aucune donnée synthétique n'est utilisée dans ce projet.

### Télécharger le dataset

1. Aller sur : [https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
2. Télécharger `healthcare-dataset-stroke-data.csv`
3. Le renommer en `stroke_data.csv`
4. Le placer dans le dossier `data/raw/`

### Description des variables

| Variable | Type | Description |
|---|---|---|
| `age` | Numérique | Âge du patient (années) |
| `gender` | Catégorielle | Male / Female |
| `hypertension` | Binaire | 0 = non, 1 = oui |
| `heart_disease` | Binaire | 0 = non, 1 = oui |
| `ever_married` | Catégorielle | Yes / No |
| `work_type` | Catégorielle | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Catégorielle | Urban / Rural |
| `avg_glucose_level` | Numérique | Taux de glucose moyen (mg/dL) |
| `bmi` | Numérique | Indice de masse corporelle (~13 % de valeurs manquantes) |
| `smoking_status` | Catégorielle | formerly smoked, never smoked, smokes, Unknown |
| `stroke` | **Cible binaire** | 0 = pas d'AVC, 1 = AVC |

- **5 110 patients** au total
- **~4,9 % de cas positifs** — dataset fortement déséquilibré, représentatif de la réalité clinique

---

## 🩺 Problématique médicale

Le problème est formulé comme une **classification binaire** :

> Étant donné un ensemble de mesures cliniques d'un patient, prédire si ce patient présente un risque d'AVC (`stroke = 1`) ou non (`stroke = 0`).

### Pourquoi le Recall est la métrique principale ?

| Type d'erreur | Description | Conséquence |
|---|---|---|
| **Faux Négatif (FN)** | Prédit « sain » alors que le patient est à risque | ❌ AVC non détecté — conséquences médicales graves |
| **Faux Positif (FP)** | Prédit « AVC » alors que le patient est sain | ⚠️ Examens inutiles — stress et coût supplémentaires |

En contexte médical, **manquer un AVC est bien plus grave** qu'une fausse alarme. Le pipeline est donc optimisé pour **maximiser le Recall** (sensibilité).

---

## 🔄 Architecture du pipeline

```
Données réelles Kaggle (stroke_data.csv)
         │
         ▼
┌──────────────────────────┐
│  Étape 1 : Chargement    │  ← Validation des colonnes, types, taille minimale
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Étape 2 : EDA           │  ← Distributions, corrélations, valeurs manquantes
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Étape 3 : Prétraitement │  ← Split → KNN Imputer → RobustScaler → SMOTE (train seulement)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Étape 4 : Modélisation  │  ← MLP / SVM / XGBoost + RandomizedSearchCV (critère : Recall)
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Étape 5 : MLflow        │  ← Suivi des expériences + Model Registry
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Étape 6 : Modules src/  │  ← config, data_loader, train, evaluate, predict
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Étape 7 : API FastAPI   │  ← /predict  /health  /model-info
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Étape 8 : Tests pytest  │  ← Données, prétraitement, prédiction, API, end-to-end
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Étape 9 : Docker        │  ← Image portable + docker-compose
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Étape 10 : Chatbot AVC  │  ← Interface conversationnelle (Mode Claude + Mode FastAPI)
└──────────────────────────┘
```

---

## 📁 Structure du projet

```
stroke_mlops/
│
├── 📂 data/
│   ├── raw/
│   │   └── stroke_data.csv              ← Données réelles Kaggle (à télécharger)
│   └── processed/
│       ├── train.csv                    ← Après prétraitement + SMOTE
│       └── test.csv
│
├── 📂 src/                              ← Modules Python production
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── 📂 models/
│   ├── trained/
│   │   ├── best_model_xgboost.joblib
│   │   ├── best_model_mlp.joblib
│   │   ├── best_model_svm.joblib
│   │   └── best_model_production.joblib ← Meilleur modèle (objet sérialisé)
│   ├── preprocessors/
│   │   ├── stroke_preprocessor.joblib
│   │   └── feature_names.joblib
│   └── best_model_info.json             ← Métadonnées utilisées par l'API
│
├── 📂 reports/                          ← Graphiques générés automatiquement
│   ├── 01_target.png
│   ├── 02_numeric.png
│   ├── 03_categorical.png
│   ├── 04_missing.png
│   ├── 05_correlation.png
│   ├── 06_bivariate.png
│   ├── 07_confusion_matrices.png
│   ├── 08_roc_curves.png
│   ├── 09_metrics_comparison.png
│   └── 10_metrics_comparison_threshold.png
│
├── 📂 docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── 📂 mlruns/                           ← Expériences MLflow (local)
│
├── etape1_generation_donnees.py         ← Chargement et validation des données réelles
├── etape2_eda.py                        ← Analyse exploratoire
├── etape3_preprocessing.py              ← Prétraitement complet
├── etape4_modelisation.py               ← Modélisation et comparaison
├── etape5_mlflow.py                     ← Suivi MLflow
├── etape6_structure_projet.py           ← Génération des modules src/
├── etape7_api_fastapi.py                ← API REST FastAPI
├── etape8_tests.py                      ← Tests automatisés
├── etape9_docker.py                     ← Génération fichiers Docker
├── chatbot_avc.html                     ← Chatbot interface web ✨
├── run_pipeline.py                      ← Script maître d'exécution
│
├── requirements.txt
├── .gitignore
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
git clone https://github.com/Ou-rkia/Systeme_de_Prediction_AVC.git
cd Systeme_de_Prediction_AVC
```

### 2. Placer les données réelles

```
data/
└── raw/
    └── stroke_data.csv   ← fichier Kaggle renommé (obligatoire)
```

### 3. Créer l'environnement virtuel

```bash
# Créer l'environnement
python -m venv venv

# Activer — Windows
venv\Scripts\activate

# Activer — Linux / Mac
source venv/bin/activate
```

### 4. Installer les dépendances

```bash
pip install -r requirements.txt
```

> ⏱️ Durée estimée : 3 à 5 minutes selon la connexion.

---

## ▶️ Exécution

### Option A — Pipeline complet automatique

```bash
python run_pipeline.py
```

### Option B — Étape par étape

```bash
# Étape 1 — Charger et valider les données réelles
python etape1_generation_donnees.py

# Étape 2 — Analyse exploratoire (graphiques → reports/)
python etape2_eda.py

# Étape 3 — Prétraitement (KNN Imputer + SMOTE + RobustScaler)
python etape3_preprocessing.py

# Étape 4 — Modélisation MLP / SVM / XGBoost  (~2 à 8 min sur CPU)
python etape4_modelisation.py

# Étape 5 — Suivi MLflow
python etape5_mlflow.py

# Étape 6 — Générer les modules src/
python etape6_structure_projet.py

# Étape 7 — Lancer l'API FastAPI
uvicorn etape7_api_fastapi:app --reload --port 8000

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
| `GET` | `/health` | Statut de l'API et du modèle chargé |
| `POST` | `/predict` | Prédiction AVC pour un patient |
| `GET` | `/model-info` | Métriques et métadonnées du meilleur modèle |

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

## 🤖 Chatbot AVC

Le projet inclut un **chatbot conversationnel** (`chatbot_avc.html`) qui permet d'évaluer le risque d'AVC d'un patient par un dialogue naturel, puis de fournir des conseils de prévention personnalisés.

### Fonctionnalités

- Collecte progressive des **10 facteurs de risque** par questions successives
- Estimation du niveau de risque avec badge coloré : **Faible** / **Modéré** / **Élevé**
- Conseils de prévention adaptés au profil clinique du patient
- Rappel des signes d'urgence AVC — méthode **F.A.S.T.**
- Boutons de raccourcis pour les questions fréquentes
- Disclaimer médical intégré dans l'interface

### Deux modes de fonctionnement

| Mode | Description | Quand l'utiliser |
|---|---|---|
| **Mode Claude** (défaut) | L'IA Claude analyse les réponses et fournit les conseils | Démonstration, sans serveur local |
| **Mode FastAPI** | Connecté au modèle XGBoost entraîné localement | Évaluation avec le modèle réel |

### Utilisation — Mode Claude (sans installation supplémentaire)

```
1. Ouvrir chatbot_avc.html dans un navigateur
2. Le mode "Mode Claude (IA générale)" est actif par défaut
3. Commencer à chatter directement
```

### Utilisation — Mode FastAPI (connecté au modèle XGBoost)

```bash
# 1. Lancer le pipeline complet pour entraîner le modèle
python run_pipeline.py

# 2. Lancer le serveur FastAPI
uvicorn etape7_api_fastapi:app --reload --port 8000

# 3. Ouvrir chatbot_avc.html dans le navigateur
# 4. Cliquer sur "Mode FastAPI"
# 5. Vérifier l'URL : http://localhost:8000/predict
# 6. Commencer l'évaluation du patient
```

### Flux de collecte des données (Mode FastAPI)

| Étape | Question posée | Variable collectée |
|---|---|---|
| 1 | Quel est votre âge ? | `age` |
| 2 | Quel est votre sexe ? (Male / Female) | `gender` |
| 3 | Avez-vous de l'hypertension artérielle ? (0/1) | `hypertension` |
| 4 | Avez-vous une maladie cardiaque ? (0/1) | `heart_disease` |
| 5 | Êtes-vous marié(e) ? (Yes / No) | `ever_married` |
| 6 | Quel est votre type de travail ? | `work_type` |
| 7 | Zone de résidence ? (Urban / Rural) | `Residence_type` |
| 8 | Taux de glucose moyen (mg/dL) ? | `avg_glucose_level` |
| 9 | Indice de masse corporelle (IMC) ? | `bmi` |
| 10 | Statut tabagique ? | `smoking_status` |

### Exemple d'interaction

```
🧠 Chatbot  : Bonjour ! Je suis StrokePredict Assistant.
              Pour évaluer votre risque d'AVC, je vais vous poser
              quelques questions. Quel est votre âge ?

👤 Patient  : 67 ans

🧠 Chatbot  : Avez-vous de l'hypertension artérielle ? (0 = non, 1 = oui)

👤 Patient  : 1

              ... (collecte des 10 variables) ...

🧠 Chatbot  : ⚠️ Résultat de votre modèle XGBoost :
              Probabilité d'AVC = 89,1 %
              ✗ RISQUE ÉLEVÉ — Consultation médicale urgente recommandée.
```

---

## 📊 MLflow UI

```bash
mlflow ui
```

Ouvrir dans le navigateur : **http://localhost:5000**

L'interface permet de visualiser :
- La comparaison des 3 modèles (MLP, SVM, XGBoost)
- Les hyperparamètres de chaque expérience
- Les courbes ROC et matrices de confusion
- Le modèle champion enregistré en stage **Production**

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest etape8_tests.py -v

# Avec rapport détaillé
pytest etape8_tests.py -v --tb=short
```

| Classe de tests | Ce qui est testé |
|---|---|
| `TestChargementDonnees` | Fichier présent, colonnes, types, valeurs 0/1, taille minimale |
| `TestPreprocessing` | Nettoyage, absence de NaN, forme de sortie, SMOTE après split |
| `TestPrediction` | Clés de sortie, label binaire, probabilité dans [0, 1] |
| `TestAPI` | Codes HTTP, validation Pydantic, cas limites |
| `TestEndToEnd` | Mini-pipeline sur 10 lignes, health → predict |

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

# Vérifier que l'API répond
curl http://localhost:8000/health
```

### Avec Docker Compose

```bash
cd docker/

# API seule
docker-compose up -d strokepredict-api

# API + MLflow UI
docker-compose --profile mlflow up -d

# Arrêter tous les services
docker-compose down
```

---

## 📈 Résultats

### Tableau comparatif des modèles

| Modèle | Accuracy | Precision | Recall ⭐ | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **XGBoost** ⭐ | ~0,91 | ~0,89 | ~0,94 | ~0,91 | ~0,97 |
| MLP | ~0,88 | ~0,85 | ~0,91 | ~0,88 | ~0,95 |
| SVM | ~0,86 | ~0,83 | ~0,89 | ~0,86 | ~0,93 |

> 🏆 **XGBoost** sélectionné comme modèle de production. Critère de sélection : **Recall** (métrique principale médicale), puis F1-Score en cas d'égalité.

### Décisions de prétraitement

| Décision | Justification |
|---|---|
| **KNN Imputer** pour BMI | Préserve les corrélations entre variables (versus médiane simple) |
| **RobustScaler** | Résistant aux valeurs extrêmes de glycémie et d'IMC |
| **SMOTE après split** | Évite le data leakage — les données de test restent intactes |
| **SMOTE ratio 1:1** | Compense le déséquilibre réel (~4,9 % de cas positifs) |
| **Seuil ajusté ≥ 0,80** | Maximise la sensibilité tout en limitant les faux positifs |

---

## 🔧 Corrections appliquées

| Fichier | Problème initial | Correction apportée |
|---|---|---|
| `etape1` | Génération de données synthétiques avec ratio 50/50 artificiel | Chargement uniquement des données réelles Kaggle avec validation stricte des colonnes |
| `etape3` | SMOTE potentiellement appliqué avant le split (data leakage) | SMOTE appliqué uniquement sur `X_train`, après `train_test_split` |
| `etape4` | Sélection du meilleur modèle par F1 (incohérent avec l'objectif médical) | Sélection par **Recall** en priorité, puis F1-Score en cas d'égalité |
| `etape4` | `best_model_name.joblib` sauvegardait le nom (string) | Sauvegarde de l'objet modèle + `best_model_info.json` pour l'API |
| `etape5` | Tag `dataset: synthetic` dans MLflow | Tag corrigé : `dataset: Kaggle — healthcare-stroke-data (réel)` |
| `etape7` | Pas de validation des plages médicales | Validateurs Pydantic : âge [0–120], glucose [50–400], IMC [10–80] |

---

## 🛠️ Technologies

| Catégorie | Outil | Version |
|---|---|---|
| Langage | Python | 3.11 |
| Machine Learning | Scikit-learn | 1.4.2 |
| Boosting | XGBoost | 2.0.3 |
| Déséquilibre des classes | Imbalanced-learn | 0.12.2 |
| Suivi des expériences | MLflow | 2.12.1 |
| API REST | FastAPI + Uvicorn | 0.111 |
| Validation des données | Pydantic | 2.7.0 |
| Tests automatisés | Pytest + HTTPX | 8.2.0 |
| Chatbot | HTML + JavaScript + Claude API | claude-sonnet-4-20250514 |
| Conteneurisation | Docker | — |
| Visualisation | Matplotlib + Seaborn | 3.8 / 0.13 |

---

## 👩‍💻 Auteure

**RKIA OUHSAIN**

**Établissement :** ENSIAS — Université Mohammed V de Rabat
**Filière :** 2IA — Semestre 4 — Module MLOps
**Année :** 2025-2026
**Dépôt GitHub :** [github.com/Ou-rkia/Systeme_de_Prediction_AVC](https://github.com/Ou-rkia/Systeme_de_Prediction_AVC)

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**.
Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

<div align="center">

*Projet réalisé dans le cadre du module MLOps — ENSIAS Rabat 2026*

</div>