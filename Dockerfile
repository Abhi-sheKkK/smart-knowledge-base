FROM python:3.12-slim

# Tesseract for scanned (image-only) PDF pages
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# download the embedding model at build time so the first /index does not wait for it
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

COPY kb/ kb/

# collections and caches live here; mount a volume to keep them across restarts
ENV DATA_DIR=/data/collections
VOLUME ["/data"]

EXPOSE 8000
# secrets (LANGFUSE_*, optional OPENCODE_API_KEY) are passed at run time: docker run --env-file .env ...
# PORT is set by platforms such as Railway; 8000 otherwise
CMD ["sh", "-c", "exec uvicorn kb.api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 75"]
