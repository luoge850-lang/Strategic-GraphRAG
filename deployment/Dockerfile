# syntax=docker/dockerfile:1

FROM node:22-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    GRAPH_VECTOR_DB_PATH=/app/data/chroma_db
WORKDIR /app

# The 2026-09-19 lock was captured on Windows; Linux compatibility is unverified.
# Use the project runtime requirements until a Linux lock is produced and tested.
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Keep every input used by source_fingerprint(ROOT=/app) in the image.
COPY strategic_graphrag/ ./strategic_graphrag/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY frontend/src/ ./frontend/src/
COPY .env.example ./
COPY deployment/preflight.py ./deployment/preflight.py
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

RUN mkdir -p /app/data/chroma_db /app/data/pdfs /app/data/pdfs_other /runtime && \
    chown -R 10001:10001 /app/data /runtime
USER 10001:10001
EXPOSE 8000

# The vector snapshot is immutable in /candidate. Chroma may mutate SQLite/HNSW
# files, so each container start copies it into a writable runtime tmpfs.
CMD ["/bin/sh", "-ec", "test -f /candidate/vector_index/chroma.sqlite3; cp -R /candidate/vector_index/. /app/data/chroma_db/; python deployment/preflight.py --production --output /runtime/preflight.json; exec uvicorn strategic_graphrag.api.server:app --host 0.0.0.0 --port 8000"]

FROM caddy:2.10-alpine AS caddy
RUN mkdir -p /data /config && chown -R 10001:10001 /data /config
COPY deployment/Caddyfile /etc/caddy/Caddyfile
USER 10001:10001
