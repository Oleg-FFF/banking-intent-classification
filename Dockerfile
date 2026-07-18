FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements-app.txt

COPY app ./app
COPY src ./src

RUN mkdir -p /app/.streamlit \
    && mkdir -p /app/.cache/huggingface

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD streamlit run app/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=${PORT:-8501} \
    --server.headless=true \
    --browser.gatherUsageStats=false