FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements-mlops.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py cnn_model.py preprocessing.py tracking.py app.py streamlit_app.py api_client.py ./
COPY model/protein_model.pth ./model/protein_model.pth

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/mlflow_data \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000 8501 5000

FROM base AS api

ENV PROTEIN_DEVICE=cpu \
    PROTEIN_MODEL_PATH=/app/model/protein_model.pth \
    PROTEIN_API_RELOAD=0

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS streamlit

ENV PROTEIN_DEVICE=cpu \
    PROTEIN_MODEL_PATH=/app/model/protein_model.pth \
    PROTEIN_API_URL=http://fastapi:8000

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--browser.gatherUsageStats", "false"]

FROM base AS mlflow

USER root
RUN pip install --no-cache-dir -r requirements-mlops.txt \
    && mkdir -p /app/mlflow_data \
    && chown -R appuser:appuser /app
USER appuser

ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow_data/mlflow.db \
    MLFLOW_EXPERIMENT_NAME=human-protein-atlas-classifier

EXPOSE 5000
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////app/mlflow_data/mlflow.db", "--artifacts-destination", "/app/mlflow_data/mlartifacts"]
