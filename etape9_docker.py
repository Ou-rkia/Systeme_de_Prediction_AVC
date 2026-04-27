# =============================================================
# ÉTAPE 9 : DOCKER + REQUIREMENTS
# Compatible : VS Code Local | CPU only
# Exécuter : python etape9_docker.py  pour générer les fichiers
# =============================================================

from pathlib import Path

BASE = Path(".")

# ── requirements.txt ─────────────────────────────────────────
(BASE / "requirements.txt").write_text("""...""", encoding="utf-8")
print("✅ requirements.txt créé")


# ── Dockerfile ───────────────────────────────────────────────
Path("docker").mkdir(exist_ok=True)

(Path("docker") / "Dockerfile").write_text("""\
# ═══════════════════════════════════════════════════════
# Dockerfile — StrokePredict API
# Image : python:3.11-slim  |  CPU only, pas de GPU requis
# ═══════════════════════════════════════════════════════

FROM python:3.11-slim

LABEL maintainer="StrokePredict"
LABEL description="API prédiction AVC — CPU only"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PYTHONPATH=/app \\
    PORT=8000

WORKDIR /app

# Dépendances système (minimales)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential curl \\
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python (layer cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \\
    && pip install --no-cache-dir -r requirements.txt

# Code source
COPY etape7_api_fastapi.py ./
COPY models/               ./models/

# Utilisateur non-root (sécurité)
RUN useradd --create-home appuser
USER appuser

# Port
EXPOSE 8000

# Healthcheck intégré
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Démarrage
CMD ["uvicorn", "etape7_api_fastapi:app", \\
     "--host", "0.0.0.0", "--port", "8000"]
""", encoding="utf-8")
print(" docker/Dockerfile créé")


# ── docker-compose.yml ───────────────────────────────────────
(Path("docker") / "docker-compose.yml").write_text("""\
version: "3.9"

services:

  # ── API principale ────────────────────────────────────────
  strokepredict-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: strokepredict-api
    ports:
      - "8000:8000"
    volumes:
      - ../models:/app/models    # Modèles montés (pas de rebuild)
      - ../reports:/app/reports
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ── MLflow UI ────────────────────────────────────────────
  mlflow-ui:
    image: python:3.11-slim
    container_name: mlflow-ui
    ports:
      - "5000:5000"
    volumes:
      - ../mlruns:/mlruns
    command: >
      sh -c "pip install mlflow --quiet &&
             mlflow ui --host 0.0.0.0 --port 5000
             --backend-store-uri /mlruns"
    profiles:
      - mlflow
""", encoding="utf-8")
print(" docker/docker-compose.yml créé")


# ── .dockerignore ────────────────────────────────────────────
(BASE / ".dockerignore").write_text("""\
__pycache__/
*.pyc
*.pyo
.venv/
venv/
data/raw/
data/processed/
notebooks/
*.ipynb
.ipynb_checkpoints/
tests/
.git/
.github/
mlruns/
.dvc/
*.dvc
.DS_Store
*.log
""", encoding="utf-8")
print(".dockerignore créé")


# ── .gitignore ───────────────────────────────────────────────
(BASE / ".gitignore").write_text("""\
# Python
__pycache__/
*.pyc
*.pyo
.Python
*.egg-info/
dist/
build/

# Environnements
.venv/
venv/
env/

# Données (trackées par DVC)
data/raw/
data/processed/
*.csv

# Modèles volumineux
models/trained/*.joblib
models/preprocessors/*.joblib

# MLflow
mlruns/

# DVC
.dvc/cache/

# Système
.DS_Store
*.log
*.tmp

# VS Code
.vscode/
""", encoding="utf-8")
print(".gitignore créé")


# ── Instructions Docker ───────────────────────────────────────
INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════╗
║              INSTRUCTIONS DOCKER LOCALES                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. Construire l'image                                   ║
║  ─────────────────────────────────────────────────────── ║
║  cd stroke_mlops/                                        ║
║  docker build -f docker/Dockerfile -t strokepredict .   ║
║                                                          ║
║  2. Démarrer le conteneur                                ║
║  ─────────────────────────────────────────────────────── ║
║  docker run -d \\                                        ║
║    --name strokepredict \\                               ║
║    -p 8000:8000 \\                                       ║
║    -v $(pwd)/models:/app/models \\                       ║
║    strokepredict                                         ║
║                                                          ║
║  3. Tester                                               ║
║  ─────────────────────────────────────────────────────── ║
║  curl http://localhost:8000/health                       ║
║  → http://localhost:8000/docs                            ║
║                                                          ║
║  4. Avec docker-compose                                  ║
║  ─────────────────────────────────────────────────────── ║
║  cd docker/                                              ║
║  docker-compose up -d strokepredict-api                  ║
║  docker-compose --profile mlflow up -d  (+ MLflow UI)   ║
║                                                          ║
║  5. Commandes utiles                                     ║
║  ─────────────────────────────────────────────────────── ║
║  docker logs strokepredict -f                            ║
║  docker exec -it strokepredict bash                      ║
║  docker stop strokepredict                               ║
║  docker-compose down                                     ║
╚══════════════════════════════════════════════════════════╝
"""
print(INSTRUCTIONS)

if __name__ == "__main__":
    print("Tous les fichiers Docker ont été générés.")
