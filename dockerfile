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

# Create user with UID 1000 required by Hugging Face Spaces
RUN useradd -m -u 1000 user && \
    mkdir -p /app/faiss_db /app/data/docs /app/mlruns /home/user/.cache && \
    chown -R user:user /app /home/user

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

# Ensure production frontend assets are copied from Stage 1
COPY --chown=user:user --from=frontend-builder /build/dist ./frontend/dist

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/status || exit 1

CMD ["python", "app/main.py"]