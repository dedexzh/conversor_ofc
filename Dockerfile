# Conversor GP7 — monitora uma pasta (watchdog) e joga os arquivos no
# LiveFolder (Postgres). O .env NÃO entra na imagem — é injetado em runtime
# (env_file no docker-compose).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=America/Belem

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY conversor_db.py .
COPY livefolder/ ./livefolder/

# PASTA_MONITORADA aponta pra dentro do container — o compose bind-monta a
# pasta real (planilhas/CSVs) em /dados.
ENV PASTA_MONITORADA=/dados

CMD ["python", "conversor_db.py"]
