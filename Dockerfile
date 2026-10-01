FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py core.py train_csv.py docker-entrypoint.sh ./

RUN mkdir -p /app/data /data \
    && chmod +x /app/docker-entrypoint.sh \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app /data

ENV ANOMALY_MODEL_PATH=/app/data/anomaly_rf.pkl \
    ANOMALY_CSV_PATH=/data/metrics.csv \
    PORT=5001

USER appuser
VOLUME ["/data", "/app/data"]

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5001/health').status==200 else 1)"

ENTRYPOINT ["/app/docker-entrypoint.sh"]