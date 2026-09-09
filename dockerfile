# ── Stage 1: Build React Frontend ────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /build

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python Runtime & FastAPI Backend ────────────────
FROM python:3.11-slim
WORKDIR /app

# Install curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create runtime directories
RUN mkdir -p /app/faiss_db /app/data/docs /app/mlruns

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY . .

# Ensure production frontend assets are copied from Stage 1
COPY --from=frontend-builder /build/dist ./frontend/dist

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/status || exit 1

CMD ["python", "app/main.py"]