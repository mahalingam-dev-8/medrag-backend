FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Pre-download embedding model into a fixed path so it survives layer caching
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from fastembed.common.model_description import ModelSource as S; from fastembed.rerank.cross_encoder import TextCrossEncoder as T; m = 'Xenova/ms-marco-TinyBERT-L-2-v2'; T.add_custom_model(model=m, sources=S(hf=m)); T(m)"

COPY app/ app/
COPY scripts/ scripts/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
