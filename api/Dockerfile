# Usa la MISMA versión de Python que Colab (la imprime notebooks/02_modelo_ml.py)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_DIR=/app/modelo

WORKDIR /app

# 1) Dependencias (capa cacheada mientras requirements.txt no cambie)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Código + artefactos del modelo
COPY main.py predictor.py schemas.py ./
COPY modelo/ ./modelo/

# Usuario sin privilegios (UID 1000 también es lo que espera Hugging Face Spaces)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health', timeout=4)"

# Render / HF Spaces inyectan $PORT; localmente se usa 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
