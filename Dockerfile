# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=5001 \
    ANOMALY_MODEL_PATH=/app/anomaly_rf.pkl

WORKDIR /app

# Dependencies first so the layer stays cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + training data
COPY app.py core.py train_csv.py ./
COPY data/metrics.csv ./data/metrics.csv

# Train at build time. train_csv.py exits non-zero on failure, so a bad
# dataset breaks the build instead of shipping an untrained image.
RUN python train_csv.py data/metrics.csv --out "$ANOMALY_MODEL_PATH"

# Run unprivileged. /app stays writable so POST /train and /train_csv
# can persist a freshly trained model.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,sys,urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','5001')+'/health',timeout=4).status==200 else 1)"

CMD ["python", "app.py"]
