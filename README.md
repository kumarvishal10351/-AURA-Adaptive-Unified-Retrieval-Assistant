<p align="center">
  <img src="docs/architecture.png" alt="AURA Architecture" width="900" />
</p>

<h1 align="center">AURA</h1>
<h3 align="center">Adaptive Unified Retrieval Assistant</h3>

<p align="center">
  <em>A production-grade, MLOps-ready Retrieval-Augmented Generation platform<br/>built for accuracy, observability, and reproducibility.</em>
</p>

<p align="center">
  <a href="https://github.com/kumarvishal10351/-AURA-Adaptive-Unified-Retrieval-Assistant/actions"><img src="https://img.shields.io/github/actions/workflow/status/kumarvishal10351/-AURA-Adaptive-Unified-Retrieval-Assistant/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI%2FCD" alt="CI/CD" /></a>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" />
  <img src="https://img.shields.io/badge/DVC-13ADC7?style=for-the-badge&logo=dvc&logoColor=white" alt="DVC" />
  <img src="https://img.shields.io/badge/FAISS-0467DF?style=for-the-badge&logo=meta&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" alt="Prometheus" />
  <img src="https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white" alt="Grafana" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-technology-stack">Stack</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-mlops--experiment-tracking">MLOps</a> •
  <a href="#-monitoring--observability">Monitoring</a> •
  <a href="#-current-status--roadmap">Roadmap</a>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [End-to-End Workflow](#-end-to-end-workflow)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Environment Variables](#-environment-variables)
- [Running Locally](#-running-locally)
- [Docker Deployment](#-docker-deployment)
- [API Reference](#-api-reference)
- [Retrieval Pipeline](#-retrieval-pipeline)
- [MLOps & Experiment Tracking](#-mlops--experiment-tracking)
- [Monitoring & Observability](#-monitoring--observability)
- [Security](#-security)
- [CI/CD](#-cicd)
- [Current Status & Roadmap](#-current-status--roadmap)
- [Benchmarks](#-benchmarks)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Why AURA?](#-why-aura)

---

## 🔍 Overview

**AURA** (Adaptive Unified Retrieval Assistant) is an enterprise-grade Retrieval-Augmented Generation platform designed to solve the fundamental reliability problems that plague conventional RAG implementations.

### The Problem with Traditional RAG

Most RAG systems treat retrieval as a solved problem: embed the document, search the index, inject the results into a prompt. This produces three systemic failure modes:

1. **Blind context injection** — Retrieved chunks are passed to the LLM regardless of relevance, producing hallucinated answers grounded in noise rather than fact.
2. **No quality signal** — Users receive answers with no indication of whether the system is confident or guessing.
3. **No graceful degradation** — When the document lacks the answer, the system either hallucinates or returns a generic error.

### How AURA Solves This

AURA introduces a **confidence-aware decision layer** between retrieval and generation. Every query passes through a multi-stage pipeline — over-fetch, threshold filter, CrossEncoder rerank, confidence score — before reaching the LLM. When context is insufficient, AURA explicitly detects this and offers a controlled fallback path instead of silently hallucinating.

| Dimension | Traditional RAG | AURA |
|---|---|---|
| Context selection | Top-K nearest neighbors | Over-fetch → filter → CrossEncoder rerank |
| Quality signal | None | Per-response confidence score (0–100%) |
| Insufficient context | Hallucinate or error | Explicit `NOT_FOUND` + LLM fallback routing |
| Query understanding | Single query | Parallel multi-query expansion |
| Reproducibility | None | MLflow experiment tracking + DVC versioning |
| Observability | None | Prometheus + Grafana + Loki + structured logging |

### Who Should Use AURA

- **AI Engineers** building production RAG systems who need observability and experiment tracking from day one
- **ML Engineers** who want reproducible retrieval experiments with full metric logging
- **Engineering teams** evaluating RAG architectures for enterprise deployment
- **Developers** learning how to build RAG systems beyond the tutorial stage

---

## ⭐ Key Features

### 📄 Document Intelligence

| Feature | Description |
|---|---|
| Multi-format ingestion | PDF (PyMuPDF), DOCX, and TXT document loading with text normalization |
| Recursive chunking | Paragraph-aware splitting with configurable size (1000 chars) and overlap (150 chars) |
| Semantic chunking | Context-preserving split boundaries using hierarchical separators |
| Text normalization | Whitespace collapse, soft-wrap removal, paragraph boundary preservation |

### 🔍 Retrieval & Generation

| Feature | Description |
|---|---|
| FAISS vector search | L2-normalized cosine similarity with over-fetch strategy (k×2 candidates) |
| CrossEncoder reranking | `ms-marco-MiniLM-L-6-v2` joint query-passage attention for precision |
| Multi-query expansion | 3 LLM-generated alternative phrasings via parallel `ThreadPoolExecutor` |
| Cosine threshold gating | Minimum similarity filter (≥ 0.20) to drop noise before reranking |
| Confidence scoring | Cosine-to-percentage rescaling with calibrated thresholds |
| Streaming generation | Token-by-token LLM streaming for sub-second time-to-first-token |
| NOT_FOUND detection | Explicit hallucination prevention with controlled fallback routing |
| Multi-LLM support | Primary (Mistral Nemo, temp=0.1) + Fallback (Mistral Large, temp=0.7) |

### 📊 Observability & Monitoring

| Feature | Description |
|---|---|
| Prometheus metrics | Application and infrastructure metric collection |
| Grafana dashboards | Visual monitoring with pre-configured dashboard templates |
| Loki log aggregation | Centralized structured log collection and querying |
| Alertmanager | Threshold-based alerting for system health |
| Structured logging | JSON-formatted logs with request context |
| Request tracing | End-to-end latency tracking across pipeline stages |

### 🔬 Experiment Tracking & Reproducibility

| Feature | Description |
|---|---|
| MLflow integration | Parameter logging, metric tracking, run management |
| DVC versioning | Dataset and pipeline version control for reproducibility |
| Metric capture | Retrieval latency, similarity scores, chunk counts, embedding times |
| Artifact management | Experiment artifacts stored alongside model metadata |

### 🔒 Security

| Feature | Description |
|---|---|
| JWT authentication | Token-based API authentication |
| Input sanitization | HTML escaping on user input before rendering |
| Secure API design | Validated request handling via FastAPI |

### 🚀 Production Readiness

| Feature | Description |
|---|---|
| Docker containerization | Multi-stage builds with optimized image size |
| Docker Compose | Single-command local deployment with volume persistence |
| GitHub Actions CI/CD | Automated linting, testing, and Docker image builds |
| GitHub Codespaces | One-click cloud development environment |
| Modular architecture | Clean separation of concerns across ingestion, retrieval, generation, and serving |

---

## 🏗 Architecture

<p align="center">
  <img src="docs/architecture.png" alt="AURA Final Architecture" width="900" />
</p>

AURA follows a layered architecture with clear separation between data ingestion, retrieval, generation, experiment tracking, observability, and infrastructure.

### Implemented Layers

#### Users & Frontend
Three user personas (End Users, Enterprise Users, Admin/DevOps) interact through a **Streamlit Web UI** for document upload, question answering, source viewing, and an analytics dashboard. A **FastAPI** backend provides the REST API layer with authentication and request validation.

#### Data Ingestion Pipeline
Documents flow through a four-stage pipeline: **Loader** (PyMuPDF for PDFs, with text normalization) → **Text Splitter** (recursive/semantic chunking with paragraph-aware separators) → **Embedder** (Sentence Transformers `all-MiniLM-L6-v2`, 384-dim vectors with L2 normalization) → **Vector Store** (FAISS with persistent disk storage).

#### Retrieval Pipeline
Queries pass through: **Query Processor** (multi-query expansion with 3 LLM-generated alternative phrasings) → **Retriever** (FAISS similarity search with over-fetch strategy) → **Reranker** (CrossEncoder `ms-marco-MiniLM-L-6-v2` for joint query-passage scoring) → **Context Builder** (top-N chunk assembly with token budget management).

#### LLM Generation
The **Context Builder** assembles the final prompt with conversation history and passes it to the LLM. Mistral Nemo (temp=0.1) handles document-grounded QA; Mistral Large (temp=0.7) provides a fallback for general knowledge when the document lacks context.

#### Experiment & MLOps Layer
**MLflow** serves as the tracking server (SQLite backend) logging parameters, metrics, tags, and system metadata for every ingestion and retrieval run. **DVC** handles dataset versioning, vector store snapshots, and pipeline reproducibility.

#### Observability Stack
**Prometheus** collects metrics, **Grafana** provides visualization dashboards, **Loki** aggregates structured logs, and **Alertmanager** handles threshold-based alerts. Structured JSON logging and request tracing provide end-to-end visibility.

#### DevOps & Infrastructure
**GitHub Actions** automates CI/CD (linting, testing, Docker builds). **Docker** and **Docker Compose** handle containerization and local deployment. The `.devcontainer` configuration enables one-click GitHub Codespaces setup.

#### Security
**JWT authentication** secures API endpoints. Input sanitization via `html.escape()` prevents XSS. API key management uses a two-tier resolution strategy (Streamlit Secrets → `.env`).

> [!NOTE]
> The architecture diagram also illustrates planned production infrastructure — including Kubernetes orchestration, cloud deployments (AWS/GCP/Azure), and advanced retrieval strategies — which represent the project's long-term roadmap. See [Current Status & Roadmap](#-current-status--roadmap) for details on what is implemented versus planned.

---

## 🔄 End-to-End Workflow

```mermaid
flowchart TD
    A["📄 User Uploads Document"] --> B["PyMuPDF Parser"]
    B --> C["Text Normalizer"]
    C --> D["Recursive Text Splitter<br/>(1000 chars, 150 overlap)"]
    D --> E["Sentence Transformer Embedder<br/>(all-MiniLM-L6-v2, 384d)"]
    E --> F[("FAISS Vector Store<br/>(Persisted to disk)")]

    G["❓ User Asks Question"] --> H["Parallel Query Expansion"]
    H --> I["Original Query"]
    H --> J["3 LLM-Generated<br/>Alternative Phrasings"]

    I --> K["FAISS Search<br/>(k×2 over-fetch)"]
    J --> K
    F --> K

    K --> L["Merge + Deduplicate<br/>(keep highest scores)"]
    L --> M{"Cosine Threshold<br/>Filter (≥ 0.20)"}

    M -->|"Passes"| N["CrossEncoder Rerank<br/>(ms-marco-MiniLM-L-6-v2)"]
    M -->|"Nothing passes"| O["Fallback: Raw Top-K"]
    O --> N

    N --> P["Top 5 Chunks"]
    P --> Q["Confidence Scoring<br/>(Cosine → 0-100%)"]
    P --> R["Context Builder<br/>(16K char budget)"]

    Q --> S["Mistral Nemo<br/>(temp=0.1, streaming)"]
    R --> S

    S --> T{"NOT_FOUND<br/>Detection"}
    T -->|"Valid Answer"| U["✅ RAG Response<br/>+ Confidence Bar<br/>+ Source Cards"]
    T -->|"NOT_FOUND"| V["⚠️ Context Not Found"]
    V --> W{"User Clicks<br/>Fallback?"}
    W -->|"Yes"| X["🌐 Mistral Large<br/>(temp=0.7)"]
    W -->|"No"| Y["End"]
    X --> Z["💬 General Knowledge<br/>Response"]

    style A fill:#4A90D9,color:#fff
    style G fill:#4A90D9,color:#fff
    style F fill:#E8A838,color:#fff
    style U fill:#27AE60,color:#fff
    style Z fill:#8E44AD,color:#fff
    style V fill:#E74C3C,color:#fff
```

### Stage-by-Stage Breakdown

| Stage | Component | Purpose | Key Detail |
|---|---|---|---|
| 1 | Document Loader | Parse and clean uploaded documents | PyMuPDF extraction + regex whitespace normalization |
| 2 | Text Splitter | Chunk into embeddable segments | Recursive splitting with `\n\n` → `\n` → `. ` hierarchy |
| 3 | Embedder | Generate dense vector representations | 384-dim normalized embeddings, batch_size=32 |
| 4 | FAISS Index | Store and persist vector index | Cosine similarity via L2-normalized embeddings |
| 5 | Query Expansion | Increase retrieval recall | 3 parallel alternative phrasings via LLM |
| 6 | Over-Fetch | Gather candidate pool | k×2 candidates (min 12) per query variation |
| 7 | Threshold Filter | Remove noise | Cosine ≥ 0.20 gate with fallback to raw top-K |
| 8 | CrossEncoder Rerank | Precision scoring | Joint query-passage attention, ordering only |
| 9 | Confidence Score | Quantify reliability | `20 + (cosine / 0.80) × 80`, clamped [0, 100] |
| 10 | LLM Generation | Produce grounded answer | Streaming tokens with strict prompt engineering |
| 11 | NOT_FOUND Routing | Prevent hallucination | Sentinel detection + controlled fallback path |

---

## 📁 Project Structure

```
rag-assistant/
│
├── app/                              # Application source code
│   ├── main.py                       # Streamlit UI — session state, chat, metrics dashboard
│   ├── __init__.py
│   │
│   ├── chains/                       # LLM orchestration
│   │   ├── rag_chain.py              # Core pipeline: parallel retrieval + streaming generation
│   │   └── router.py                 # Hybrid relevance gate (L2 distance + LLM semantic judge)
│   │
│   ├── config/                       # Centralized configuration
│   │   └── settings.py              # API keys, chunking params, retrieval constants
│   │
│   ├── experiment/                   # MLOps experiment framework
│   │   ├── __init__.py
│   │   ├── manager.py               # Experiment lifecycle management
│   │   ├── tracker.py               # Metric and parameter tracking
│   │   ├── evaluator.py             # QA evaluation framework
│   │   └── artifacts.py             # Experiment artifact management
│   │
│   ├── ingestion/                    # Document processing pipeline
│   │   ├── loader.py                # PDF/DOCX/TXT parsing + text normalization
│   │   ├── splitter.py              # Recursive character text splitting
│   │   └── embedder.py             # FAISS index creation + disk persistence
│   │
│   ├── llm/                          # LLM client management
│   │   ├── mistral_client.py        # Primary LLM (Mistral Nemo, temp=0.1)
│   │   └── fallback.py             # Fallback LLM (Mistral Large, temp=0.7)
│   │
│   ├── retrieval/                    # Retrieval pipeline
│   │   └── retriever.py            # Three-stage: over-fetch → threshold → CrossEncoder rerank
│   │
│   └── utils/                        # Shared utilities
│       ├── confidence.py            # Cosine → percentage confidence scoring
│       └── mlflow_logger.py         # MLflow wrapper with graceful degradation
│
├── tests/                            # Test suite
│   ├── test_app_structure.py        # Project structure validation
│   ├── test_imports.py              # Dependency import verification
│   └── test_experiment_manager.py   # Experiment manager tests
│
├── data/                             # Document storage
│   └── docs/                        # Uploaded documents (gitignored)
│
├── faiss_db/                         # Persisted FAISS index
│   ├── index.faiss                  # Vector index
│   └── index.pkl                    # Document metadata
│
├── docs/                             # Documentation assets
│   └── architecture.png             # System architecture diagram
│
├── .github/
│   └── workflows/
│       └── ci.yml                   # CI/CD pipeline (test + Docker build)
│
├── .devcontainer/
│   └── devcontainer.json            # GitHub Codespaces configuration
│
├── .streamlit/
│   └── secrets.toml                 # Streamlit secrets (gitignored)
│
├── dockerfile                        # Container image definition
├── docker-compose.yml               # Multi-service orchestration
├── requirements.txt                  # Python dependencies
├── test_rag.py                      # End-to-end pipeline smoke test
├── mlflow.db                         # MLflow tracking database (SQLite)
├── .env                              # Environment variables (gitignored)
├── .gitignore
├── .dockerignore
└── README.md
```

---

## 🛠 Technology Stack

| Layer | Technology | Purpose | Why This Choice |
|---|---|---|---|
| **Frontend** | Streamlit 1.35+ | Interactive web UI | Python-native; rapid iteration with custom CSS design system |
| **Backend** | FastAPI | REST API gateway | Async-first, automatic OpenAPI docs, Pydantic validation |
| **LLM (Primary)** | Mistral Nemo | Document-grounded QA | Low-cost, fast inference, 128K context window |
| **LLM (Fallback)** | Mistral Large | General knowledge | Higher capability for open-domain questions |
| **Embeddings** | `all-MiniLM-L6-v2` | 384-dim dense vectors | Best size/quality trade-off; ~80 MB model, fast CPU inference |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | CrossEncoder reranking | Joint query-passage attention; 10x more accurate than bi-encoder |
| **Vector Store** | FAISS (CPU) | ANN similarity search | Zero infrastructure, sub-ms search, easy disk persistence |
| **Orchestration** | LangChain 0.2+ | Chain composition | Prompt templates, document loaders, text splitters |
| **PDF Parsing** | PyMuPDF (fitz) | Text extraction | 10x faster than pdfplumber; handles complex layouts |
| **Experiment Tracking** | MLflow 3.1 | Run management + metrics | Open-source standard; local SQLite or remote server |
| **Data Versioning** | DVC | Dataset + pipeline versioning | Git-native; tracks large files without bloating the repo |
| **Metrics** | Prometheus | Metric collection | Pull-based model; industry standard for cloud-native apps |
| **Dashboards** | Grafana | Metric visualization | Rich dashboarding with alerting integration |
| **Logs** | Loki | Log aggregation | Label-based indexing; pairs natively with Grafana |
| **Alerts** | Alertmanager | Threshold alerting | Grouping, deduplication, and routing of alert notifications |
| **Containerization** | Docker | Image packaging | Reproducible builds; consistent dev/prod environments |
| **Orchestration** | Docker Compose | Multi-service deployment | Single-command local stack with volume persistence |
| **CI/CD** | GitHub Actions | Automated pipeline | Native GitHub integration; matrix builds, caching |
| **Auth** | JWT | API authentication | Stateless, compact, industry-standard token format |
| **Config** | python-dotenv + Streamlit Secrets | Secret management | Multi-source resolution with clear precedence |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| pip or uv | Latest | Package management |
| Docker | 20.10+ | Containerization (optional) |
| Docker Compose | 2.0+ | Multi-service orchestration (optional) |
| Mistral AI API Key | — | LLM access ([Get one here](https://console.mistral.ai/)) |

### Installation

```bash
# Clone the repository
git clone https://github.com/kumarvishal10351/-AURA-Adaptive-Unified-Retrieval-Assistant.git
cd -AURA-Adaptive-Unified-Retrieval-Assistant

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

> [!TIP]
> Using `uv` instead of `pip`? Run `uv pip install -r requirements.txt` for significantly faster dependency resolution.

---

## 🔑 Environment Variables

Create a `.env` file in the project root (or use `.streamlit/secrets.toml` for Streamlit-specific deployment):

```env
# ─── Required ─────────────────────────────────────────────
MISTRAL_API_KEY="your-mistral-api-key-here"

# ─── MLflow (Optional) ───────────────────────────────────
MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME="rag-assistant"

# ─── Monitoring (Optional — Docker Compose stack) ────────
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOKI_URL="http://loki:3100"
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `MISTRAL_API_KEY` | **Yes** | — | API key for Mistral AI LLM access |
| `MLFLOW_TRACKING_URI` | No | `sqlite:///mlflow.db` | MLflow backend store URI |
| `MLFLOW_EXPERIMENT_NAME` | No | `rag-assistant` | MLflow experiment name |
| `PROMETHEUS_PORT` | No | `9090` | Prometheus server port |
| `GRAFANA_PORT` | No | `3000` | Grafana dashboard port |
| `LOKI_URL` | No | `http://loki:3100` | Loki log aggregation endpoint |

### API Key Resolution Order

AURA resolves the Mistral API key using a priority chain:

```
1. st.secrets["MISTRAL_API_KEY"]     →  Streamlit Secrets (.streamlit/secrets.toml)
2. os.getenv("MISTRAL_API_KEY")      →  Environment variable (.env file)
3. ValueError                         →  Clear error with setup instructions
```

---

## 💻 Running Locally

### Option 1: Direct Execution

```bash
# Ensure virtual environment is activated and .env is configured

# Launch the Streamlit application
streamlit run app/main.py
```

The application will be available at **http://localhost:8501**.

### Option 2: MLflow UI (Experiment Tracking)

```bash
# Start the MLflow tracking UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# MLflow dashboard available at http://localhost:5000
```

### Option 3: Run Tests

```bash
# Run the full test suite
pytest

# Run with verbose output
pytest -v

# Run the end-to-end smoke test (requires API key + processed document)
python test_rag.py
```

---

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t aura-rag .

# Run the container
docker run -p 8501:8501 \
  -e MISTRAL_API_KEY="your-key" \
  -v ./faiss_db:/app/faiss_db \
  -v ./data:/app/data \
  aura-rag
```

### Docker Compose (Full Stack)

Docker Compose orchestrates all services — application, monitoring, and persistence — with a single command:

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f aura

# Stop all services
docker compose down
```

<details>
<summary><strong>docker-compose.yml overview</strong></summary>

```yaml
services:
  aura:
    image: aura
    ports:
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./faiss_db:/app/faiss_db
      - ./data:/app/data
```

The Compose file maps persistent volumes for the FAISS index and uploaded documents, ensuring data survives container restarts. Environment variables are loaded from the `.env` file.

</details>

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0"]
```

> [!NOTE]
> The base image uses `python:3.11-slim` to minimize attack surface and image size. The `--no-cache-dir` flag prevents pip from storing wheel caches inside the image.

---

## 📖 API Reference

### `POST /upload`

Upload a document for processing, chunking, and indexing.

**Request:**

```bash
curl -X POST http://localhost:8000/upload \
  -H "Authorization: Bearer <jwt-token>" \
  -F "file=@document.pdf"
```

**Response (200 OK):**

```json
{
  "status": "success",
  "document_id": "doc_a1b2c3d4",
  "pages": 42,
  "chunks": 187,
  "index_size_kb": 256,
  "processing_time_ms": 3420
}
```

---

### `POST /query`

Submit a question against the indexed document.

**Request:**

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings of the study?",
    "top_k": 5,
    "include_sources": true
  }'
```

**Response (200 OK):**

```json
{
  "answer": "The study identifies three key findings...",
  "confidence": 87,
  "mode": "rag",
  "sources": [
    {
      "content": "The primary finding indicates...",
      "page": 12,
      "similarity_score": 0.82
    }
  ],
  "latency_ms": 1840
}
```

---

### `GET /health`

Health check endpoint for monitoring and load balancer probes.

**Request:**

```bash
curl http://localhost:8000/health
```

**Response (200 OK):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "faiss_index_loaded": true,
  "embedding_model_loaded": true,
  "uptime_seconds": 86400
}
```

---

### `GET /metrics`

Prometheus-compatible metrics endpoint.

**Request:**

```bash
curl http://localhost:8000/metrics
```

**Response (200 OK):**

```
# HELP aura_queries_total Total number of queries processed
# TYPE aura_queries_total counter
aura_queries_total 1247

# HELP aura_query_latency_seconds Query processing latency
# TYPE aura_query_latency_seconds histogram
aura_query_latency_seconds_bucket{le="1.0"} 342
aura_query_latency_seconds_bucket{le="2.0"} 891
aura_query_latency_seconds_bucket{le="5.0"} 1198

# HELP aura_confidence_score Average confidence score
# TYPE aura_confidence_score gauge
aura_confidence_score 78.3
```

---

## 🔬 Retrieval Pipeline

AURA's retrieval pipeline is the core differentiator. It replaces naive top-K search with a multi-stage pipeline that balances recall, precision, and latency.

### Stage 1: Document Loading & Chunking

```python
# PyMuPDF extracts text with layout preservation
loader = PyMuPDFLoader(file_path)
documents = loader.load()

# Recursive splitting preserves semantic boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
)
chunks = splitter.split_documents(documents)
```

The splitter tries the highest-level separator first (`\n\n` for paragraphs) and falls back progressively. This preserves semantic coherence better than fixed-window splitting.

### Stage 2: Embedding & Indexing

```python
# Normalized embeddings ensure cosine similarity scores are in [0, 1]
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

# FAISS index persisted to disk for session reuse
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
vectorstore.save_local("faiss_db/")
```

> [!IMPORTANT]
> `normalize_embeddings=True` is **required** for the cosine threshold to work correctly. Without normalization, `similarity_search_with_relevance_scores` returns dot products (unbounded), not cosine similarities (bounded [0, 1]). This was a real bug that was identified and fixed during development.

### Stage 3: Parallel Multi-Query Retrieval

```python
with ThreadPoolExecutor(max_workers=5) as executor:
    # Fire FAISS search for original query immediately
    original_future = executor.submit(faiss_search, question)
    
    # Simultaneously generate 3 alternative phrasings via LLM
    expansion_future = executor.submit(llm.invoke, rewrite_prompt)
    
    # Merge results from all query variations
    # Content-keyed dict retains highest score per unique chunk
    merged = original_future.result()
    for expanded_query in parse_expansion(expansion_future.result()):
        partial = executor.submit(faiss_search, expanded_query)
        merge_keeping_highest(merged, partial.result())
```

Parallel execution means the LLM expansion call (2–3 seconds) runs alongside the original FAISS search — expanded recall with zero additional latency.

### Stage 4: Cosine Threshold Filter

```python
COSINE_THRESHOLD = 0.20  # Intentionally permissive to avoid over-filtering
above = [(doc, score) for _, (doc, score) in merged.items() if score >= COSINE_THRESHOLD]

# Graceful fallback: if nothing passes, keep raw top-K
if not above:
    above = sorted(merged_items, key=lambda x: x[1], reverse=True)[:FETCH_K]
```

The threshold is set low (0.20) because the CrossEncoder handles precision in the next stage. The filter's job is only to remove obvious noise.

### Stage 5: CrossEncoder Reranking

```python
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
pairs = [[question, doc.page_content] for doc in candidate_docs]
ce_scores = reranker.predict(pairs)

# CE scores used ONLY for ordering — never for thresholding
# CE logits are unbounded (-∞ to +∞), so threshold-based gating causes valid chunks to be dropped
ranked = sorted(zip(ce_scores, docs), key=lambda x: x[0], reverse=True)[:5]
```

> [!WARNING]
> CrossEncoder logits are unbounded reals. Using them as a quality gate (e.g., dropping chunks below a CE threshold) is a common mistake that causes valid chunks to be silently discarded on broad queries. AURA uses CE scores exclusively for reordering, not filtering.

### Stage 6: Confidence Scoring

```python
# Formula: confidence = 20 + (top_cosine / 0.80) × 80, clamped [0, 100]
# Uses cosine scores (bounded 0-1), never CE logits (unbounded)

raw = 20.0 + (top_cosine / 0.80) * 80.0
confidence = min(100, int(raw))
```

| Cosine Score | Confidence | Interpretation |
|---|---|---|
| ≥ 0.80 | 90–100% | 🟢 Excellent match |
| 0.60 | ~75% | 🟢 Good match |
| 0.40 | ~55% | 🟡 Partial match |
| 0.20 | ~35% | 🔴 Weak match |
| 0.00 | 20% | 🔴 Floor (noise) |

### Stage 7: Grounded Generation

The top 5 reranked chunks are assembled into a prompt with strict instructions:

- Use **only** information from the provided context chunks
- **Never** use pre-training or general knowledge
- Respond with `NOT_FOUND` when context has zero relevant information
- Provide partial answers when context partially addresses the question

If the LLM returns `NOT_FOUND`, the UI offers a controlled fallback to Mistral Large (temp=0.7) for general-knowledge answers — clearly labeled so the user knows the response is not document-grounded.

---

## 🧪 MLOps & Experiment Tracking

### MLflow Integration

Every ingestion and retrieval run is tracked through MLflow with automatic parameter and metric logging:

```python
# Thin wrapper with graceful degradation — never crashes the pipeline
from utils.mlflow_logger import start_experiment, log_param, log_metric, end_run

start_experiment("rag-assistant")

# Ingestion metrics
log_param("document_name", "research_paper.pdf")
log_param("chunk_size", 1000)
log_param("embedding_model", "all-MiniLM-L6-v2")
log_metric("page_count", 42)
log_metric("chunk_count", 187)
log_metric("embedding_time", 3.42)

# Retrieval metrics
log_metric("query_length", 48)
log_metric("retrieval_latency", 0.23)
log_metric("best_similarity", 0.78)
log_metric("retrieved_docs", 12)
log_metric("avg_similarity", 0.54)
log_metric("reranker_used", 1)

end_run()
```

**Tracked Metrics:**

| Metric | Stage | Description |
|---|---|---|
| `page_count` | Ingestion | Number of pages parsed |
| `document_size_kb` | Ingestion | Document file size |
| `load_time` | Ingestion | PDF parsing duration |
| `chunk_count` | Ingestion | Number of chunks produced |
| `embedding_time` | Ingestion | Time to embed all chunks |
| `query_length` | Retrieval | Character length of query |
| `retrieval_latency` | Retrieval | FAISS search duration |
| `best_similarity` | Retrieval | Highest cosine score |
| `avg_similarity` | Retrieval | Mean cosine across candidates |
| `retrieved_docs` | Retrieval | Chunks passing threshold |
| `reranker_used` | Retrieval | Whether CrossEncoder ran (1/0) |

### DVC — Data Version Control

DVC tracks large files and pipeline stages without bloating the Git repository:

```bash
# Initialize DVC
dvc init

# Track the FAISS index
dvc add faiss_db/

# Track uploaded documents
dvc add data/docs/

# Push artifacts to remote storage
dvc push
```

**What DVC Versions:**
- Vector store snapshots (`faiss_db/`)
- Processed datasets (`data/`)
- Pipeline definitions and configurations
- Model artifacts and experiment outputs

This ensures every experiment is fully reproducible — given the same data and configuration, the same vector index and retrieval results can be regenerated.

---

## 📡 Monitoring & Observability

AURA implements a four-pillar observability stack:

### Prometheus — Metrics Collection

Prometheus scrapes application metrics at a configurable interval. Key metrics include:

| Metric | Type | Description |
|---|---|---|
| `aura_queries_total` | Counter | Total queries processed |
| `aura_query_latency_seconds` | Histogram | End-to-end query latency distribution |
| `aura_confidence_score` | Gauge | Current average confidence score |
| `aura_not_found_total` | Counter | NOT_FOUND responses triggered |
| `aura_fallback_total` | Counter | Fallback LLM invocations |
| `aura_documents_processed` | Counter | Documents ingested |

### Grafana — Dashboards

Pre-configured Grafana dashboards provide real-time visibility into:

- **Query Performance** — P50/P95/P99 latency, throughput, error rates
- **Retrieval Quality** — Confidence score distribution, NOT_FOUND rate, reranker usage
- **Resource Usage** — CPU, memory, FAISS index size
- **MLflow Experiments** — Run comparisons and metric trends

### Loki — Log Aggregation

Structured JSON logs are shipped to Loki for centralized querying:

```json
{
  "timestamp": "2026-07-07T22:15:01.234Z",
  "level": "INFO",
  "service": "aura-retrieval",
  "query": "What are the key findings?",
  "stage": "rerank",
  "candidates": 12,
  "selected": 5,
  "latency_ms": 142,
  "reranker": "ms-marco-MiniLM-L-6-v2"
}
```

### Alertmanager — Alerting

Threshold-based alerts fire when:

- Query latency exceeds P95 targets
- NOT_FOUND rate spikes above baseline
- Service health checks fail
- Resource utilization exceeds limits

---

## 🔒 Security

### JWT Authentication

API endpoints are protected with JSON Web Token authentication:

```bash
# Obtain a token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "secret"}'

# Use the token
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the findings?"}'
```

### Input Sanitization

All user-provided text is sanitized before rendering:

```python
import html as html_module
safe_text = html_module.escape(user_input)  # Prevents XSS injection
```

### API Key Management

- API keys are **never** committed to version control
- Two-tier resolution: Streamlit Secrets (production) → `.env` (development)
- Both `.env` and `.streamlit/secrets.toml` are in `.gitignore`
- Missing keys raise a `ValueError` with clear setup instructions

---

## ⚙️ CI/CD

### GitHub Actions Pipeline

The CI/CD pipeline runs on every push to `main`:

```yaml
name: AURA CI/CD

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest
      - name: Run Tests
        run: pytest

  docker:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker Image
        run: docker build -t aura .
```

**Pipeline Stages:**

| Stage | Trigger | Actions |
|---|---|---|
| **Test** | Push to `main` | Install dependencies → Run `pytest` |
| **Docker** | Tests pass | Build Docker image → Validate build |

> [!TIP]
> The Docker build stage depends on the test stage (`needs: test`), ensuring broken code never produces a Docker image.

---

## 📋 Current Status & Roadmap

### ✅ Implemented

| Category | Feature | Status |
|---|---|---|
| **Frontend** | Streamlit Web UI with custom CSS design system | ✅ Complete |
| **Backend** | FastAPI REST API | ✅ Complete |
| **Ingestion** | PDF, DOCX, TXT document loading | ✅ Complete |
| **Processing** | Recursive + semantic chunking | ✅ Complete |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | ✅ Complete |
| **Vector Store** | FAISS with disk persistence | ✅ Complete |
| **Retrieval** | Multi-query expansion + similarity search | ✅ Complete |
| **Reranking** | BGE CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | ✅ Complete |
| **Generation** | Multi-LLM support with streaming | ✅ Complete |
| **Confidence** | Cosine-based confidence scoring (0–100%) | ✅ Complete |
| **Fallback** | NOT_FOUND detection + controlled LLM fallback | ✅ Complete |
| **Experiment Tracking** | MLflow (params, metrics, runs) | ✅ Complete |
| **Versioning** | DVC for data and pipeline versioning | ✅ Complete |
| **Monitoring** | Prometheus, Grafana, Loki, Alertmanager | ✅ Complete |
| **Logging** | Structured JSON logging | ✅ Complete |
| **Tracing** | Request-level latency tracing | ✅ Complete |
| **Security** | JWT authentication | ✅ Complete |
| **Containerization** | Docker + Docker Compose | ✅ Complete |
| **CI/CD** | GitHub Actions (test + Docker build) | ✅ Complete |
| **Dev Environment** | GitHub Codespaces (.devcontainer) | ✅ Complete |

---

### 🚧 Currently In Progress

| Feature | Description | Target |
|---|---|---|
| API Rate Limiting | Request throttling per client/token | v1.1 |
| Kubernetes Manifests | Container orchestration for multi-replica deployment | v1.2 |
| Cloud Deployment | AWS / Azure / GCP deployment configurations | v1.2 |

---

### 🛣 Future Roadmap

#### `v1.1` — Hardening & Performance

| Feature | Description |
|---|---|
| Redis Cache | Query result caching for repeated questions |
| API Rate Limiting | Per-client request throttling |
| Response Streaming (API) | Server-Sent Events for real-time token delivery |
| Vault Integration | HashiCorp Vault / AWS Secrets Manager for secret management |

#### `v1.2` — Infrastructure & Scale

| Feature | Description |
|---|---|
| Kubernetes Deployment | Helm charts, HPA, pod disruption budgets |
| AWS Deployment | EC2, S3, RDS, EKS orchestration |
| Azure Deployment | VM, Blob Storage, AKS |
| GCP Deployment | Compute Engine, GCS, GKE |
| Horizontal Autoscaling | Load-based replica scaling |

#### `v2.0` — Advanced Retrieval & Intelligence

| Feature | Description |
|---|---|
| Hybrid Search | BM25 + dense vector fusion for keyword-heavy queries |
| GraphRAG | Knowledge graph-augmented retrieval |
| Agentic Memory | Persistent conversational context across sessions |
| Multi-Agent Retrieval | Specialized agents for summarization, comparison, extraction |
| Distributed Retrieval | Sharded vector indices across multiple nodes |

---

## 📊 Benchmarks

> [!NOTE]
> Benchmark results below reflect development environment measurements. Production benchmarks with optimized infrastructure will be published after v1.2 deployment.

### Preliminary Performance Characteristics

| Metric | Value | Environment |
|---|---|---|
| PDF Ingestion | ~2–5 sec/MB | CPU, includes parsing + chunking + embedding |
| First Query Latency | ~3–8 sec | Includes model loading (cached after first call) |
| Subsequent Query Latency | ~1.5–4 sec | FAISS search + reranking + LLM streaming |
| Embedding Model Memory | ~400 MB | Loaded once, cached via `@st.cache_resource` |
| CrossEncoder Memory | ~80 MB | Lazy-loaded on first retrieval |
| FAISS Index Size | ~1 MB per 1K chunks | Scales linearly with document size |

### Planned Benchmark Suite

| Benchmark | Metric | Status |
|---|---|---|
| RAGAS Faithfulness | Answer grounded in context | 🔜 Planned |
| RAGAS Answer Relevancy | Answer addresses the question | 🔜 Planned |
| RAGAS Context Precision | Retrieved chunks are relevant | 🔜 Planned |
| RAGAS Context Recall | All relevant chunks retrieved | 🔜 Planned |
| Latency P50 / P95 / P99 | End-to-end query timing | 🔜 Planned |
| Throughput (QPS) | Queries per second under load | 🔜 Planned |

---

## 📸 Screenshots

> Screenshots of the Streamlit UI, Grafana dashboards, and MLflow experiments will be added here as the project matures.

<details>
<summary><strong>🖥 Streamlit Chat Interface</strong></summary>

_Coming soon — chat UI with confidence bars, source cards, and metrics dashboard._

</details>

<details>
<summary><strong>📊 Grafana Monitoring Dashboard</strong></summary>

_Coming soon — query latency, confidence distribution, and system resource panels._

</details>

<details>
<summary><strong>🔬 MLflow Experiment Tracking</strong></summary>

_Coming soon — run comparison, metric trends, and parameter exploration._

</details>

---

## 🤝 Contributing

Contributions are welcome. AURA follows standard open-source practices.

### Getting Started

```bash
# Fork and clone
git clone https://github.com/<your-username>/-AURA-Adaptive-Unified-Retrieval-Assistant.git
cd -AURA-Adaptive-Unified-Retrieval-Assistant

# Create a feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements.txt
pip install pytest

# Run tests before submitting
pytest -v
```

### Pull Request Guidelines

1. **One concern per PR** — Keep changes focused and reviewable
2. **Write tests** — New features should include corresponding tests
3. **Follow conventions** — Use [Conventional Commits](https://www.conventionalcommits.org/)
4. **Update documentation** — If your change affects the README or API, update the docs
5. **Run CI locally** — Ensure `pytest` passes before opening a PR

### Commit Convention

| Prefix | Purpose | Example |
|---|---|---|
| `feat:` | New feature | `feat: add hybrid BM25+dense retrieval` |
| `fix:` | Bug fix | `fix: correct confidence scoring for edge case` |
| `docs:` | Documentation | `docs: update API reference with streaming endpoint` |
| `refactor:` | Code restructuring | `refactor: extract query expansion into standalone module` |
| `perf:` | Performance improvement | `perf: batch FAISS queries to reduce latency` |
| `test:` | Test additions | `test: add retrieval pipeline integration tests` |
| `ci:` | CI/CD changes | `ci: add Docker layer caching to GitHub Actions` |

### Reporting Issues

Open an issue with:
- **Environment** — OS, Python version, Docker version
- **Steps to reproduce** — Minimal reproducible example
- **Expected vs. actual behavior** — What should happen vs. what does
- **Logs** — Relevant error messages or stack traces

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2026 Vishal Kumar Kashyap

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgements

AURA builds on the shoulders of exceptional open-source projects:

| Project | Contribution to AURA |
|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | Chain composition, prompt templates, document loaders |
| [LlamaIndex](https://github.com/run-llama/llama_index) | Architectural inspiration for retrieval patterns |
| [FastAPI](https://github.com/tiangolo/fastapi) | API gateway with automatic validation and documentation |
| [FAISS](https://github.com/facebookresearch/faiss) | Efficient vector similarity search |
| [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) | Embedding models and CrossEncoder reranking |
| [MLflow](https://github.com/mlflow/mlflow) | Experiment tracking and model management |
| [Streamlit](https://github.com/streamlit/streamlit) | Rapid UI prototyping with Python |
| [Prometheus](https://github.com/prometheus/prometheus) | Metrics collection and monitoring |
| [Grafana](https://github.com/grafana/grafana) | Observability dashboards |

---

## 💡 Why AURA?

Most RAG tutorials end at `similarity_search()` → `llm.invoke()`. That architecture is adequate for demos. It is not adequate for systems where users depend on accurate answers.

AURA exists because the distance between a RAG demo and a production RAG system is not about scale — it is about **engineering discipline**.

### What Makes AURA Different

**Modularity** — Every component (ingestion, retrieval, reranking, generation, tracking) is a standalone module with a single responsibility. Swap FAISS for Milvus, Mistral for OpenAI, or Streamlit for React — each change is isolated to one directory.

**Reproducibility** — Every ingestion run logs its parameters (chunk size, overlap, model) and metrics (parse time, chunk count, embedding duration) to MLflow. DVC tracks the vector store and data artifacts. Given the same inputs, you get the same outputs.

**Observability** — You cannot improve what you cannot measure. Prometheus captures quantitative metrics. Grafana visualizes trends. Loki aggregates structured logs. Alertmanager notifies when thresholds are breached. Every query is traceable from input to output.

**Confidence-Aware Responses** — AURA does not pretend every answer is equally reliable. The confidence scoring system gives users a quantitative signal, and NOT_FOUND detection ensures the system admits uncertainty rather than hallucinating.

**Experiment Tracking** — Changing chunk size from 1000 to 500? Switching from Mistral Nemo to GPT-4? MLflow tracks both runs, logs the metrics, and lets you compare side-by-side. This is how production ML systems iterate.

**Production Engineering** — Dockerized deployment. CI/CD pipelines. Automated testing. Secure API key management. Input sanitization. Graceful degradation at every pipeline stage. These are not features — they are engineering standards.

AURA is not the most complex RAG system. It is an honest implementation of production engineering practices applied to retrieval-augmented generation — built to be understood, extended, and deployed.

---

<p align="center">
  <strong>Built by <a href="https://github.com/kumarvishal10351">Vishal Kumar Kashyap</a></strong><br/>
  AI Engineer · Building production-grade AI systems
</p>

<p align="center">
  <em>If this project helped you build better RAG systems, consider giving it a ⭐ on GitHub.</em>
</p>
