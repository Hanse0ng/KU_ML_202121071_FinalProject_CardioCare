FROM python:3.14-slim

WORKDIR /ML_pipeline

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY mlruns/ ./mlruns/
COPY mlflow.db ./mlflow.db

ENTRYPOINT ["python", "src/inference.py"]