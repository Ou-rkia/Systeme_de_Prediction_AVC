# ═══════════════════════════════════════════════════════
# Dockerfile — StrokePredict API
# Image : python:3.11-slim  |  CPU only, pas de GPU requis
# ═══════════════════════════════════════════════════════

FROM python:3.11-slim

LABEL maintainer="StrokePredict"
LABEL description="API prédiction AVC — CPU only"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Dépendances système (minimales)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python (layer cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
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
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Démarrage
CMD ["uvicorn", "etape7_api_fastapi:app", \
     "--host", "0.0.0.0", "--port", "8000"]
