<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-Meta-0467DF?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/Mistral_AI-Small-FF7000?style=for-the-badge&logo=mistral&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🧠 AURA — Adaptive Unified Retrieval Assistant</h1>

<p align="center">
  <strong>A production-grade, reliability-first Retrieval-Augmented Generation system with confidence-aware routing, CrossEncoder reranking, and intelligent LLM fallback.</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#%EF%B8%8F-configuration">Configuration</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Solution — AURA](#-solution--aura)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Retrieval Pipeline Deep Dive](#-retrieval-pipeline-deep-dive)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Configuration](#%EF%B8%8F-configuration)
- [API Reference](#-api-reference)
- [Performance Characteristics](#-performance-characteristics)
- [Design Decisions](#-design-decisions)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

**AURA** is a confidence-aware RAG system that treats document retrieval as a *probabilistic process* rather than a deterministic one. Unlike conventional RAG implementations that blindly trust retrieved context, AURA introduces a multi-stage decision pipeline with cosine-threshold gating, CrossEncoder reranking, and confidence-scored responses — enabling graceful degradation when document context is insufficient.

### Why This Matters

| Traditional RAG | AURA |
|---|---|
| Always trusts retrieved context | Evaluates confidence before answering |
| Single-stage retrieval | Three-stage pipeline (fetch → filter → rerank) |
| Silent hallucinations on irrelevant context | Explicit `NOT_FOUND` detection + fallback |
| Fixed retrieval queries | Parallel multi-query expansion |
| No quality signal to the user | Per-response confidence scoring (0–100%) |

---

## 🎯 Problem Statement

Traditional RAG systems suffer from a critical architectural flaw:

1. **Blind Trust in Retrieval** — They always inject retrieved chunks into the prompt, even when the chunks are semantically irrelevant to the query.
2. **Silent Hallucination** — The LLM generates plausible-sounding answers from noise, giving users no signal that the response is unreliable.
3. **No Graceful Degradation** — When the document doesn't contain the answer, the system either hallucinates or returns an unhelpful error.

These issues make standard RAG pipelines unsuitable for production deployments where **answer reliability** is non-negotiable.

---

## ✅ Solution — AURA

AURA introduces a **confidence-aware decision layer** between retrieval and generation:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                               │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Parallel Retrieval + Query Expansion              │
│  ┌──────────────────┐   ┌────────────────────────────────┐  │
│  │ FAISS Similarity │   │ LLM-based Query Rewriting      │  │
│  │ Search (k×2)     │   │ (3 alternative phrasings)      │  │
│  └────────┬─────────┘   └──────────────┬─────────────────┘  │
│           │                            │                    │
│           └──────────┬─────────────────┘                    │
│                      ▼                                      │
│              Merged + Deduplicated Candidates                │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Cosine Threshold Filter (≥ 0.20)                  │
│  Drop noise chunks · Keep relevant candidates               │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: CrossEncoder Reranking (ms-marco-MiniLM-L-6-v2)  │
│  Joint query-passage attention · Top-5 selection            │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Confidence Scoring (0–100%)                       │
│  cosine ≥ 0.80 → 100%  ·  cosine 0.60 → ~75%              │
│  cosine 0.40 → ~55%    ·  cosine 0.20 → ~35%               │
└─────────────┬───────────────────────────────────────────────┘
              │
              ├── High Confidence ──▶ RAG Response (LLM + Context)
              │
              ├── NOT_FOUND ────────▶ Neutral message + Fallback option
              │
              └── User clicks 🌐 ──▶ Fallback LLM (Mistral Large)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.11+
- **Mistral AI API Key** — [Get one here](https://console.mistral.ai/)
- ~2 GB disk space (for model weights)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kumarvishal10351/rag-assistant.git
cd rag-assistant

# 2. Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key (choose one method)
# Method A: Environment file
echo 'MISTRAL_API_KEY="your-key-here"' > .env

# Method B: Streamlit secrets
mkdir -p .streamlit
echo 'MISTRAL_API_KEY = "your-key-here"' > .streamlit/secrets.toml

# 5. Launch the application
streamlit run app/main.py
```

The app will be available at **http://localhost:8501**.

### Docker (Alternative)

```bash
docker build -t aura-rag .
docker run -p 8501:8501 -e MISTRAL_API_KEY="your-key" aura-rag
```

### GitHub Codespaces

This project includes a `.devcontainer/devcontainer.json` for one-click Codespaces setup. Open in Codespaces and the app will auto-start.

---

## 🏗 Architecture

### High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Streamlit Frontend                          │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Sidebar    │  │  Chat UI     │  │  Metrics     │  │ Confidence│  │
│  │  Upload     │  │  Streaming   │  │  Dashboard   │  │ Bars      │  │
│  │  Controls   │  │  Messages    │  │  KPIs        │  │ Sources   │  │
│  └────────────┘  └──────┬───────┘  └──────────────┘  └───────────┘  │
└─────────────────────────┼────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────────┐
│  Ingestion   │ │  RAG Chain   │ │  Fallback                        │
│  Pipeline    │ │  Pipeline    │ │  Pipeline                        │
│              │ │              │ │                                   │
│  loader.py   │ │  rag_chain   │ │  fallback.py                     │
│  splitter.py │ │  router.py   │ │  (Mistral Large, temp=0.7)       │
│  embedder.py │ │  retriever   │ │                                   │
│              │ │  confidence  │ └──────────────────────────────────┘
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FAISS Vector Store                            │
│                   (Persisted to disk: faiss_db/)                     │
│                   L2-normalized cosine similarity                    │
└─────────────────────────────────────────────────────────────────────┘
       ▲                                          ▲
       │                                          │
┌──────┴──────────┐                    ┌──────────┴──────────┐
│  HuggingFace    │                    │  CrossEncoder       │
│  Embeddings     │                    │  Reranker           │
│  all-MiniLM-    │                    │  ms-marco-MiniLM-   │
│  L6-v2 (384d)   │                    │  L-6-v2             │
└─────────────────┘                    └─────────────────────┘
```

### Module Responsibility Matrix

| Module | Responsibility | Key Design Decision |
|---|---|---|
| `ingestion/loader.py` | PDF parsing + text cleaning | PyMuPDF for speed; regex-based whitespace normalization |
| `ingestion/splitter.py` | Recursive text chunking | 1000-char chunks, 150-char overlap, paragraph-aware separators |
| `ingestion/embedder.py` | FAISS index creation | Shared embedding model (avoids duplicate 400 MB load) |
| `retrieval/retriever.py` | Three-stage retrieval pipeline | Over-fetch → threshold → rerank; graceful CE fallback |
| `chains/rag_chain.py` | Orchestrates RAG pipeline | Parallel query expansion; streaming token generation |
| `chains/router.py` | Hybrid relevance gate | L2 distance pre-filter + LLM semantic judge |
| `llm/mistral_client.py` | Primary LLM client | Mistral Nemo, temp=0.1, cached per session |
| `llm/fallback.py` | Fallback LLM client | Mistral Large, temp=0.7, for general knowledge |
| `utils/confidence.py` | Confidence scoring | Cosine-to-percentage rescaling (floor=20%, ceiling=100%) |
| `config/settings.py` | Centralized configuration | API key resolution, chunking params, retrieval constants |
| `app/main.py` | Streamlit UI + orchestration | Custom CSS design system, session state management |

---

## ⭐ Key Features

### 🎯 Confidence-Aware Responses
Every RAG response includes a calculated confidence score (0–100%) derived from cosine similarity, giving users a clear signal of answer reliability.

### 🔄 Intelligent Fallback Routing
When the document doesn't contain the answer, AURA detects `NOT_FOUND` responses and offers a one-click fallback to Mistral Large for general knowledge.

### 🔍 Multi-Query Expansion
The system generates 3 alternative phrasings of each query in parallel, increasing recall for ambiguous or broad questions.

### 🏆 CrossEncoder Reranking
A second-stage `ms-marco-MiniLM-L-6-v2` CrossEncoder performs joint query-passage attention scoring, dramatically improving precision over bi-encoder-only retrieval.

### 💬 Streaming Responses
Token-by-token streaming from the Mistral API for sub-second time-to-first-token, with real-time rendering in the chat UI.

### 📚 Source Attribution
Expandable source cards show the exact document chunks used to generate each answer, with page numbers for traceability.

### 🗂️ Session State Management
Persistent chat history, query counters, and confidence score tracking across the session using Streamlit's `session_state`.

### 🎨 Production UI
Fully custom CSS design system with glassmorphism, gradient accents, micro-animations, and a dark-mode-first aesthetic.

---

## 🔬 Retrieval Pipeline Deep Dive

### Stage 1: FAISS Approximate Nearest-Neighbor Search

```python
# Over-fetch candidates: k × 2 (min 12) to give the reranker enough material
FETCH_K = max(TOP_K * 2, 12)
results = vs.similarity_search_with_relevance_scores(query, k=FETCH_K)
```

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, ~80 MB)
- **Normalization**: `normalize_embeddings=True` ensures scores are true cosine similarities in [0, 1]
- **Parallel Execution**: Original query + 3 expanded queries run concurrently via `ThreadPoolExecutor`
- **Deduplication**: Content-keyed dictionary retains highest-scoring instance per chunk

### Stage 2: Cosine Threshold Filter

```python
COSINE_THRESHOLD = 0.20  # Intentionally low to avoid over-filtering broad queries
above = [(doc, score) for _, (doc, score) in merged.items() if score >= COSINE_THRESHOLD]
```

- Filters out noise while preserving partial matches
- Graceful fallback: if nothing passes, keep raw top-K candidates

### Stage 3: CrossEncoder Reranking

```python
# CrossEncoder reads (query, passage) jointly — much more accurate than bi-encoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
ce_scores = reranker.predict([[query, doc.page_content] for doc in docs])
```

- CE scores are used **only for ordering** — never for thresholding (logits are unbounded)
- Graceful degradation: if CE fails, results fall back to cosine-sorted order

### Confidence Scoring Formula

```
confidence = 20 + (top_cosine / 0.80) × 80, clamped to [0, 100]
```

| Cosine Score | Confidence | UI Label |
|---|---|---|
| ≥ 0.80 | 90–100% | 🟢 High |
| 0.60 | ~75% | 🟢 High |
| 0.40 | ~55% | 🟡 Medium |
| 0.20 | ~35% | 🔴 Low |
| 0.00 | 20% | 🔴 Low |

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.35+ | Interactive web UI with custom CSS |
| **LLM (Primary)** | Mistral Nemo (`open-mistral-nemo`) | Document-grounded QA, temp=0.1 |
| **LLM (Fallback)** | Mistral Large (`mistral-large-latest`) | General knowledge, temp=0.7 |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) | 384-dim dense vectors |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` (CrossEncoder) | Joint query-passage scoring |
| **Vector Store** | FAISS (CPU) | ANN search with L2 distance |
| **Orchestration** | LangChain 0.2+ | Prompt templates, chain composition |
| **PDF Parsing** | PyMuPDF (fitz) | Fast, accurate text extraction |
| **Text Splitting** | RecursiveCharacterTextSplitter | Paragraph-aware chunking |
| **Config** | python-dotenv + Streamlit Secrets | Multi-source API key resolution |

---

## 📁 Project Structure

```
rag-assistant/
├── app/
│   ├── main.py                  # Streamlit UI (1600+ lines, custom CSS design system)
│   ├── __init__.py
│   │
│   ├── chains/
│   │   ├── rag_chain.py         # Core RAG pipeline (parallel retrieval + streaming)
│   │   └── router.py            # Hybrid relevance gate (L2 + LLM judge)
│   │
│   ├── config/
│   │   └── settings.py          # Centralized config (API keys, chunking, retrieval)
│   │
│   ├── ingestion/
│   │   ├── loader.py            # PDF loading + text normalization
│   │   ├── splitter.py          # Recursive character text splitting
│   │   └── embedder.py          # FAISS index creation + persistence
│   │
│   ├── llm/
│   │   ├── mistral_client.py    # Primary LLM (Mistral Nemo, temp=0.1)
│   │   └── fallback.py          # Fallback LLM (Mistral Large, temp=0.7)
│   │
│   ├── retrieval/
│   │   └── retriever.py         # Three-stage retrieval (fetch → filter → rerank)
│   │
│   └── utils/
│       └── confidence.py        # Cosine → percentage confidence scoring
│
├── data/
│   └── docs/                    # Uploaded PDFs (gitignored)
│
├── faiss_db/                    # Persisted FAISS index
│   ├── index.faiss              # Vector index (~800 KB)
│   └── index.pkl                # Document metadata (~500 KB)
│
├── .devcontainer/
│   └── devcontainer.json        # GitHub Codespaces configuration
│
├── .streamlit/
│   └── secrets.toml             # Streamlit secrets (gitignored)
│
├── test_rag.py                  # End-to-end pipeline smoke test
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (gitignored)
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

All configuration is centralized in `app/config/settings.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `12` | FAISS candidates per query |
| `MAX_CONTEXT_LENGTH` | `16,000` | Max characters sent to LLM (~4K tokens) |
| `COSINE_THRESHOLD` | `0.20` | Minimum cosine similarity to keep a chunk |
| `FINAL_TOP_N` | `5` | Chunks sent to LLM after reranking |
| `EXPAND_TIMEOUT` | `18s` | Timeout for query expansion LLM call |
| `HISTORY_TURNS` | `3` | Recent conversation turns in prompt |

### API Key Resolution Order

1. `st.secrets["MISTRAL_API_KEY"]` (Streamlit Secrets)
2. `os.getenv("MISTRAL_API_KEY")` (`.env` file)
3. Raises `ValueError` with a clear error message

---

## 📖 API Reference

### Core Functions

#### `create_rag_chain(llm, vectorstore) → Callable`
Factory function returning the RAG pipeline callable.

```python
rag_pipeline = create_rag_chain(llm, vectorstore)
generator, docs, results = rag_pipeline(question="What is deep learning?", history=[])
answer = "".join(list(generator))  # Stream to string
```

#### `calculate_confidence(results) → int`
Converts retrieval results to a 0–100 confidence score.

```python
confidence = calculate_confidence(results)  # e.g., 87
```

#### `retrieve(query, *, k=8, rerank_top_n=4) → list[Document]`
Standalone three-stage retrieval pipeline.

```python
docs = retrieve("What are neural networks?", k=8, rerank_top_n=4)
```

---

## 📊 Performance Characteristics

| Metric | Value | Notes |
|---|---|---|
| **PDF Ingestion** | ~2–5 sec/MB | Includes parsing + chunking + embedding |
| **First Query Latency** | ~3–8 sec | Includes model loading (cached after first call) |
| **Subsequent Query Latency** | ~1.5–4 sec | FAISS search + reranking + LLM streaming |
| **Embedding Model Load** | ~400 MB RAM | Loaded once, cached via `@st.cache_resource` |
| **CrossEncoder Load** | ~80 MB RAM | Lazy-loaded on first retrieval |
| **FAISS Index Size** | ~1 MB per 1K chunks | Scales linearly with document size |

---

## 🧠 Design Decisions

### Why Cosine Threshold Instead of CrossEncoder Gating?
CrossEncoder logits are unbounded reals (−∞ to +∞). Using them as a quality gate causes valid chunks to be dropped on broad queries. Cosine scores from normalized embeddings are bounded [0, 1] and provide a consistent, interpretable threshold.

### Why Parallel Query Expansion?
Single-query retrieval suffers from vocabulary mismatch. By generating 3 alternative phrasings concurrently, we increase recall without adding sequential latency.

### Why NOT_FOUND Detection?
LLMs tend to hallucinate when given irrelevant context. Explicit `NOT_FOUND` detection allows the UI to offer a clean fallback path instead of serving a hallucinated answer.

### Why Separate LLMs for RAG vs. Fallback?
- **RAG (Mistral Nemo, temp=0.1)**: Low temperature for factual, document-grounded responses
- **Fallback (Mistral Large, temp=0.7)**: Higher temperature for creative, general-knowledge responses

### Why Streamlit Over React/Next.js?
Rapid prototyping with Python-native tooling. The custom CSS design system demonstrates that Streamlit can produce production-quality UIs when properly styled.

---

## 🔮 Future Roadmap

- [ ] **Hybrid Retrieval** — BM25 + dense vector fusion for improved recall on keyword-heavy queries
- [ ] **Multi-Document Support** — Upload and query across multiple PDFs simultaneously
- [ ] **Adaptive Thresholding** — Dynamic cosine threshold based on document characteristics
- [ ] **Evaluation Framework** — Automated RAGAS metrics (faithfulness, answer relevancy, context precision)
- [ ] **Authentication & Multi-tenancy** — User-scoped document stores
- [ ] **Reinforcement Learning for Routing** — Learn optimal RAG vs. fallback decisions from user feedback
- [ ] **Multi-Agent Reasoning** — Specialized agents for summarization, comparison, and extraction tasks
- [ ] **GPU Acceleration** — FAISS-GPU and CUDA-enabled embedding for large-scale deployments

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'feat: add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Commit Convention

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New features
- `fix:` — Bug fixes
- `docs:` — Documentation changes
- `refactor:` — Code refactoring
- `perf:` — Performance improvements
- `test:` — Test additions/modifications

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Vishal Kumar Kashyap**
AI Engineer · Building production-grade AI systems

<p align="center">
  <i>If you found this project useful, consider giving it a ⭐ on GitHub!</i>
</p>
