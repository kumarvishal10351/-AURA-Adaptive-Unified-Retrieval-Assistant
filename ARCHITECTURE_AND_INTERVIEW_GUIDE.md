# 🧠 AURA — Architecture Flow & Interview Preparation Guide

> **Purpose**: This document serves as a comprehensive technical reference for understanding the AURA RAG system's architecture and preparing for AI/ML engineering interviews.

---

## 📋 Table of Contents

1. [End-to-End Architecture Flow](#1-end-to-end-architecture-flow)
2. [Data Ingestion Pipeline](#2-data-ingestion-pipeline)
3. [Retrieval Pipeline](#3-retrieval-pipeline)
4. [Generation Pipeline](#4-generation-pipeline)
5. [Confidence Scoring System](#5-confidence-scoring-system)
6. [Fallback Mechanism](#6-fallback-mechanism)
7. [Interview Questions & Answers](#7-interview-questions--answers)
8. [System Design Discussion Points](#8-system-design-discussion-points)
9. [Common Follow-Up Questions](#9-common-follow-up-questions)
10. [Key Terminology Glossary](#10-key-terminology-glossary)

---

## 1. End-to-End Architecture Flow

### Complete Request Lifecycle

```
USER UPLOADS PDF
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              INGESTION PIPELINE                      │
│                                                      │
│  PDF ──► PyMuPDF Parser ──► Text Cleaner             │
│                                  │                   │
│                                  ▼                   │
│                    RecursiveCharacterTextSplitter     │
│                    (1000 chars, 150 overlap)          │
│                                  │                   │
│                                  ▼                   │
│                    HuggingFace Embeddings             │
│                    (all-MiniLM-L6-v2, 384d)          │
│                                  │                   │
│                                  ▼                   │
│                    FAISS Index (saved to disk)        │
└─────────────────────────────────────────────────────┘

USER ASKS A QUESTION
       │
       ▼
┌─────────────────────────────────────────────────────┐
│           PARALLEL RETRIEVAL (Stage 1)               │
│                                                      │
│  ┌──────────────┐     ┌─────────────────────────┐   │
│  │ Original     │     │ LLM Query Expansion     │   │
│  │ FAISS Search │     │ (3 alternative queries)  │   │
│  │ (k=24)       │     │ via ThreadPoolExecutor   │   │
│  └──────┬───────┘     └────────────┬────────────┘   │
│         │                          │                 │
│         └──────────┬───────────────┘                 │
│                    ▼                                  │
│         Merge + Deduplicate (keep highest scores)    │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         COSINE THRESHOLD FILTER (Stage 2)            │
│                                                      │
│  Keep chunks with cosine similarity ≥ 0.20           │
│  If nothing passes → keep raw top-K as fallback      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│         CROSSENCODER RERANKING (Stage 3)             │
│                                                      │
│  Model: ms-marco-MiniLM-L-6-v2                      │
│  Joint (query, passage) attention scoring            │
│  Select Top-5 by CE score                            │
│  CE scores used ONLY for ordering, NOT thresholding  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           CONFIDENCE SCORING (Stage 4)               │
│                                                      │
│  Formula: conf = 20 + (top_cosine / 0.80) × 80      │
│  Clamped to [0, 100]                                 │
│  Uses COSINE scores (not CE logits)                  │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           LLM GENERATION (Stage 5)                   │
│                                                      │
│  Build context from top-5 chunks                     │
│  Include last 3 conversation turns                   │
│  Stream tokens via Mistral Nemo (temp=0.1)           │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │ IF answer == "NOT_FOUND"                     │    │
│  │   → Show neutral message                     │    │
│  │   → Offer "Use web model" fallback button    │    │
│  │                                               │    │
│  │ IF user clicks fallback                       │    │
│  │   → Route to Mistral Large (temp=0.7)        │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                     │
                     ▼
            RENDER IN STREAMLIT UI
            (chat bubbles, confidence bar,
             expandable source cards)
```

---

## 2. Data Ingestion Pipeline

### Step-by-Step Flow

| Step | Module | What Happens | Why |
|------|--------|-------------|-----|
| 1 | `loader.py` | PyMuPDF extracts text from PDF | Faster than pdfplumber; handles complex layouts |
| 2 | `loader.py` | Normalize whitespace (regex) | Remove soft-wraps, preserve paragraph breaks |
| 3 | `splitter.py` | Split into 1000-char chunks with 150-char overlap | Overlap prevents context loss at boundaries |
| 4 | `embedder.py` | Embed chunks with `all-MiniLM-L6-v2` | 384-dim vectors; normalized for cosine similarity |
| 5 | `embedder.py` | Save FAISS index to `faiss_db/` | Persistent storage; reload without re-embedding |

### Key Design Choices

**Why RecursiveCharacterTextSplitter?**
- Uses hierarchical separators: `\n\n` → `\n` → `. ` → ` ` → `""`
- Preserves semantic boundaries (paragraphs > sentences > words)
- Better than naive fixed-size splitting

**Why normalize_embeddings=True?**
- Without normalization, `similarity_search_with_relevance_scores` returns dot products, not cosine
- The 0.20 threshold becomes meaningless without normalization
- This was a real bug that was fixed in this project

---

## 3. Retrieval Pipeline

### Three-Stage Architecture

```
Stage 1: FAISS Over-Fetch
├── Fetch k×2 candidates (min 12)
├── Run 3 expanded queries in parallel
├── Merge results, deduplicate by content
└── Keep highest score per unique chunk

Stage 2: Cosine Threshold Filter
├── Drop chunks below 0.20 cosine similarity
├── If nothing passes → keep raw top-K
└── This prevents over-filtering on broad queries

Stage 3: CrossEncoder Rerank
├── Score each (query, chunk) pair jointly
├── Sort by CE score descending
├── Take top 5
└── If CE fails → fall back to cosine-sorted order
```

### Why Over-Fetch + Rerank?

**Problem**: Bi-encoder (FAISS) is fast but imprecise. It encodes query and document independently.

**Solution**: Over-fetch candidates, then use a CrossEncoder that reads query+document together for precise scoring.

```
Bi-Encoder (FAISS):   query ──► embed ──► compare with doc embeddings
CrossEncoder:         (query, doc) ──► joint attention ──► relevance score
```

The CrossEncoder is ~10x more accurate but ~100x slower, so we use it only on pre-filtered candidates.

---

## 4. Generation Pipeline

### Prompt Engineering

```
System: "You are a strict document analyst. You answer ONLY from the 
         provided context chunks..."

Rules enforced:
1. Use ONLY information from context chunks
2. NEVER use pre-training knowledge
3. Respond "NOT_FOUND" when context has zero relevant info
4. No chunk references like [Chunk 1] in output
5. Structure with headings/bullets when appropriate
```

### Why temp=0.1 for RAG?
- Low temperature = deterministic, factual responses
- Higher temperature would introduce creative hallucination
- We want the LLM to be a faithful summarizer, not a creative writer

### Streaming Architecture
```python
def token_generator():
    for chunk in llm.stream(formatted_prompt):
        if chunk.content:
            yield chunk.content
```
- Yields tokens as they arrive from the API
- Enables real-time rendering in the UI
- Fallback: yields "NOT_FOUND" if no tokens produced

---

## 5. Confidence Scoring System

### The Bug That Was Fixed

**Original bug**: CrossEncoder logits (unbounded: -10 to +5) were passed directly to confidence calculation. A perfectly good match scoring CE logit of -1.0 would produce `sigmoid(-1.0) × 100 = 27%` — completely misleading.

**Fix**: Always use cosine similarity scores (bounded 0–1) from FAISS for confidence. CE scores used only for reranking order.

### Rescaling Formula

```
confidence = 20 + (top_cosine / 0.80) × 80
clamped to [0, 100]
```

**Why this formula?**
- Raw cosine 0.55 as "55%" feels too low for a good match
- 0.80 cosine = 100% confidence (practical ceiling for non-identical text)
- Floor of 20% ensures any non-zero retrieval shows something

---

## 6. Fallback Mechanism

### Decision Flow

```
RAG Pipeline Returns Answer
         │
         ├── Answer starts with "NOT_FOUND"
         │        │
         │        ▼
         │   Show: "Context not found in document"
         │   Show: "🌐 Use web model" button
         │        │
         │        ├── User clicks button
         │        │        │
         │        │        ▼
         │        │   Mistral Large (temp=0.7)
         │        │   General knowledge answer
         │        │
         │        └── User ignores → no action
         │
         └── Answer is valid
                  │
                  ▼
             Display RAG answer with:
             - Confidence bar
             - Source cards
             - "From Document" badge
```

### Why Two Different Models?

| Aspect | RAG (Mistral Nemo) | Fallback (Mistral Large) |
|--------|-------------------|------------------------|
| Temperature | 0.1 | 0.7 |
| Purpose | Factual extraction | Creative general knowledge |
| Context | Document chunks | No document context |
| Cost | Lower | Higher |
| Speed | Faster | Slower |

---

## 7. Interview Questions & Answers

### Category A: RAG Fundamentals

---

**Q1: What is RAG and why is it needed?**

**A:** RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses by first retrieving relevant documents from a knowledge base, then providing them as context to the LLM. It's needed because:
- LLMs have knowledge cutoff dates
- LLMs hallucinate when they don't know something
- RAG grounds responses in actual documents, reducing hallucination
- It enables domain-specific Q&A without fine-tuning the model

**In AURA specifically**: We go beyond basic RAG by adding confidence scoring and fallback routing — the system knows when it doesn't know.

---

**Q2: Explain the difference between bi-encoder and cross-encoder retrieval.**

**A:**

| Aspect | Bi-Encoder (FAISS) | Cross-Encoder |
|--------|-------------------|---------------|
| How it works | Encodes query and document separately, compares via cosine | Encodes (query, document) pair jointly |
| Speed | Very fast (pre-computed doc embeddings) | Slow (must process each pair) |
| Accuracy | Good for recall | Excellent for precision |
| Use case | First-stage retrieval | Second-stage reranking |

**In AURA**: We use bi-encoder for initial fetch (24 candidates) and cross-encoder to rerank down to 5. This gives us both speed and accuracy.

---

**Q3: What is embedding normalization and why does it matter?**

**A:** Embedding normalization scales vectors to unit length (L2 norm = 1). This matters because:
- Without normalization, `similarity_search_with_relevance_scores` returns dot products, not cosine similarity
- Dot products are unbounded; cosine similarity is bounded [-1, 1]
- Our threshold of 0.20 only makes sense with cosine similarity
- **Real bug we fixed**: Without `normalize_embeddings=True`, our thresholding was essentially random

---

**Q4: Why use chunking? Why not embed entire documents?**

**A:** 
- Embedding models have token limits (512 tokens for MiniLM)
- Longer text → diluted semantic meaning in the embedding
- Chunking allows precise retrieval of relevant sections
- Overlap (150 chars) prevents information loss at chunk boundaries

**Our chunking strategy**: 1000 chars with paragraph-aware separators (`\n\n` → `\n` → `. `) to preserve semantic boundaries.

---

**Q5: How do you handle the "hallucination on irrelevant context" problem?**

**A:** Three-layer defense:
1. **Cosine threshold** (0.20): Drops clearly irrelevant chunks before they reach the LLM
2. **Strict prompt engineering**: LLM instructed to output "NOT_FOUND" when context is insufficient
3. **Fallback routing**: If NOT_FOUND detected, offer user a separate general-knowledge LLM instead of forcing a hallucinated answer

---

### Category B: System Design & Architecture

---

**Q6: How would you scale this system to handle 1000 concurrent users?**

**A:** Several changes needed:
1. **FAISS → Milvus/Pinecone**: Managed vector DB with horizontal scaling
2. **API Gateway**: Rate limiting, load balancing across multiple Streamlit instances
3. **Caching Layer**: Redis cache for repeated queries (same query + same document = same answer)
4. **Async Processing**: Document ingestion via task queue (Celery/Bull)
5. **Model Serving**: Dedicated embedding model server (Triton/TGI) instead of loading per-process
6. **Multi-tenancy**: User-scoped vector stores, not a shared index

---

**Q7: How would you evaluate RAG quality in production?**

**A:** Use the RAGAS framework with these metrics:
- **Faithfulness**: Is the answer supported by the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Precision**: Are the retrieved chunks actually relevant?
- **Context Recall**: Did we retrieve all relevant chunks?

Additionally:
- **A/B testing**: Compare RAG responses vs. LLM-only responses
- **Human evaluation**: Sample-based review of answer quality
- **Confidence calibration**: Does 80% confidence actually mean 80% accuracy?

---

**Q8: Why did you choose FAISS over ChromaDB, Pinecone, or Weaviate?**

**A:**
- **FAISS**: Free, no server needed, fast for single-machine deployment, good for prototyping
- **ChromaDB**: Also local-first, but adds SQLite overhead; FAISS is leaner
- **Pinecone**: Managed cloud service — better for production but adds cost and network latency
- **Weaviate**: Full-featured but heavyweight for a single-user assistant

For this project, FAISS was the right choice: zero infrastructure, sub-millisecond search, and easy persistence to disk.

---

**Q9: Explain the parallel query expansion architecture.**

**A:**
```python
with ThreadPoolExecutor(max_workers=5) as ex:
    # Fire original FAISS search immediately
    orig_future = ex.submit(_fetch_candidates, vs, question)
    
    # Simultaneously ask LLM to generate 3 alternative queries
    expand_future = ex.submit(llm.invoke, rewrite_prompt)
    
    # Merge results from all queries
    merged = orig_future.result()
    for expanded_query in parse_expansion(expand_future.result()):
        extra = ex.submit(_fetch_candidates, vs, expanded_query)
        # Merge, keeping highest score per chunk
```

**Why parallel?** The LLM expansion call takes ~2-3 seconds. By running it alongside the original FAISS search, we get expanded recall with zero additional latency.

**Why 3 alternative queries?** Vocabulary mismatch is the #1 retrieval failure mode. "What are the conclusions?" might miss chunks containing "In summary..." — but an expanded query "summarize the final findings" would catch it.

---

**Q10: How does your session state management work?**

**A:** Streamlit reruns the entire script on every interaction. We use `st.session_state` to persist:
- `chat_history`: List of conversation turns (question, answer, mode, confidence, docs)
- `db_ready`: Whether a document has been processed
- `conf_scores`: Running list for average confidence calculation
- `input_key`: Incremented to force text input clearing after submission
- `prefill_query`: For suggested query buttons to populate the input

Resource caching via `@st.cache_resource` prevents reloading the 400MB embedding model on every rerun.

---

### Category C: ML/NLP Concepts

---

**Q11: What is cosine similarity and why is it preferred for text embeddings?**

**A:** Cosine similarity measures the angle between two vectors, ignoring magnitude:
```
cos(A, B) = (A · B) / (||A|| × ||B||)
```
Range: [-1, 1] (or [0, 1] for normalized positive embeddings)

**Why preferred for text:**
- Text embeddings should be compared by direction (meaning), not magnitude (length)
- A longer document shouldn't be "more similar" just because its vector is larger
- Cosine is invariant to vector magnitude

---

**Q12: Explain the difference between temperature 0.1 and 0.7 in LLMs.**

**A:** Temperature controls the randomness of token selection:
- **temp=0.1**: Nearly deterministic; always picks the highest-probability token
- **temp=0.7**: More diverse; allows lower-probability tokens to be selected

**In AURA:**
- RAG uses 0.1 because we want faithful extraction from documents (no creativity)
- Fallback uses 0.7 because general knowledge answers benefit from natural, varied phrasing

---

**Q13: What is the RecursiveCharacterTextSplitter and how does it work?**

**A:** It splits text using a hierarchy of separators, trying each in order:
1. `\n\n` (paragraph breaks) — best semantic boundary
2. `\n` (line breaks)
3. `. ` (sentence endings)
4. ` ` (word boundaries)
5. `""` (character-level fallback)

It tries to split on the highest-level separator that keeps chunks within the size limit. This preserves semantic coherence better than naive fixed-size splitting.

---

**Q14: What is FAISS and how does approximate nearest-neighbor search work?**

**A:** FAISS (Facebook AI Similarity Search) is a library for efficient similarity search on dense vectors.

For exact search: Compare query against all N vectors → O(N) time.

For approximate search (what FAISS uses):
- **IVF (Inverted File Index)**: Clusters vectors, only searches nearest clusters
- **HNSW (Hierarchical Navigable Small World)**: Graph-based; walks through a hierarchy of proximity graphs
- **PQ (Product Quantization)**: Compresses vectors for memory efficiency

Trade-off: Slight accuracy loss for 10-100x speed improvement.

---

**Q15: How does the CrossEncoder model work internally?**

**A:** Unlike bi-encoders that embed query and document separately:

```
Bi-Encoder:    [CLS] query [SEP] → embedding₁
               [CLS] document [SEP] → embedding₂
               score = cosine(embedding₁, embedding₂)

Cross-Encoder: [CLS] query [SEP] document [SEP] → single score
               Full cross-attention between query and document tokens
```

The cross-encoder can capture fine-grained interactions (e.g., negation, entity matching) that bi-encoders miss. But it's ~100x slower because it can't pre-compute document embeddings.

---

### Category D: Production & Engineering

---

**Q16: How do you handle API failures and timeouts?**

**A:** Multiple layers of resilience:
1. **Mistral client**: `timeout=30s`, `max_retries=2` configured at client level
2. **Query expansion**: 18-second timeout; if expansion fails, original query proceeds alone
3. **CrossEncoder**: try/except fallback to cosine-sorted results if CE unavailable
4. **FAISS search**: Individual try/except per query variation; partial results still usable
5. **UI**: Specific error messages for TimeoutError vs. generic exceptions

---

**Q17: What caching strategies are used and why?**

**A:**
| What | Mechanism | Why |
|------|-----------|-----|
| Embedding model | `@st.cache_resource` | 400MB model; load once per session |
| CrossEncoder | `@st.cache_resource` | 80MB model; lazy-loaded |
| FAISS index | `@st.cache_resource` | Avoid re-reading from disk |
| Mistral LLM client | `@st.cache_resource` | Reuse connection/auth |

**Key insight**: `@st.cache_resource` caches across reruns but NOT across sessions. This is correct because different sessions may have different documents.

---

**Q18: How do you prevent the tokenizer deadlock on Streamlit hot-reloads?**

**A:** Setting `os.environ["TOKENIZERS_PARALLELISM"] = "false"` at the top of `main.py`. 

HuggingFace tokenizers use Rust parallelism by default. When Streamlit hot-reloads the script, the forked process inherits locked mutexes from the parent → deadlock. Disabling parallelism prevents this.

---

**Q19: What are the security considerations in this system?**

**A:**
1. **API Key Management**: Two-tier resolution (Streamlit secrets > .env), both gitignored
2. **FAISS Deserialization**: `allow_dangerous_deserialization=True` is required for pickle-based FAISS — mitigated by only loading indices we created ourselves
3. **File Upload**: Only PDF files accepted; processed server-side with PyMuPDF
4. **XSS Prevention**: `html_module.escape()` on all user input before rendering
5. **No Authentication**: Current limitation; production would need auth + multi-tenancy

---

**Q20: Walk me through what happens when a user clicks "Send" on a query.**

**A:** Complete execution flow:

1. **Input Validation**: Check `db_ready` flag and non-empty query
2. **UI Update**: Show "thinking" animation with bouncing dots
3. **Load Resources**: Get cached vectorstore + Mistral LLM
4. **Create Pipeline**: `create_rag_chain(llm, vectorstore)` returns a callable
5. **Execute Pipeline**:
   - Parallel: FAISS search + LLM query expansion
   - Merge candidates from all queries
   - Filter by cosine threshold (≥ 0.20)
   - Rerank with CrossEncoder → top 5 chunks
6. **Calculate Confidence**: Top chunk's cosine score → 0-100% via rescaling formula
7. **Stream Answer**: LLM generates tokens from context + prompt
8. **NOT_FOUND Check**: If response starts with "NOT_FOUND" → set mode to "not_found"
9. **Update Session State**: Append to chat_history, increment counters
10. **Re-render**: Streamlit reruns, renders all messages including new one
11. **Show Sources**: Expandable cards with page numbers and chunk previews

---

## 8. System Design Discussion Points

### How to Explain This Project in 2 Minutes

> "I built AURA, a production-grade RAG system that solves the hallucination problem in document Q&A. Most RAG systems blindly trust retrieved context — AURA doesn't. It uses a three-stage retrieval pipeline: FAISS over-fetch for recall, cosine threshold filtering to remove noise, and CrossEncoder reranking for precision. Every response includes a confidence score so users know how reliable the answer is. When the document doesn't contain the answer, instead of hallucinating, the system explicitly detects this and offers a fallback to a general-knowledge LLM. The architecture uses parallel query expansion, streaming responses, and session-aware conversation history."

### Key Talking Points for Interviews

1. **"I treat RAG as a probabilistic system"** — Not all retrievals are equal; confidence scoring acknowledges this
2. **"I fixed a real production bug"** — CrossEncoder logits were being used as confidence scores; unbounded values made confidence meaningless
3. **"I optimized resource usage"** — Single shared embedding model instance (avoided duplicate 400MB loads)
4. **"I designed for graceful degradation"** — Every stage has a fallback path
5. **"I understand the bi-encoder vs. cross-encoder trade-off"** — Speed vs. accuracy, and how to combine them

### Architecture Diagram for Whiteboard

```
[User Query]
     │
     ├──► [FAISS Search] ──────────────────┐
     │                                      │
     └──► [LLM Expand] ──► [FAISS × 3] ───┤
                                            │
                                            ▼
                                    [Merge + Dedup]
                                            │
                                            ▼
                                    [Cosine Filter ≥ 0.20]
                                            │
                                            ▼
                                    [CrossEncoder Rerank]
                                            │
                                            ▼
                                    [Top 5 Chunks]
                                            │
                              ┌─────────────┼──────────────┐
                              │             │              │
                              ▼             ▼              ▼
                         [Confidence]  [Build Prompt]  [Sources]
                              │             │              │
                              └─────────────┼──────────────┘
                                            │
                                            ▼
                                    [LLM Stream Answer]
                                            │
                                    ┌───────┴────────┐
                                    │                │
                               [Valid Answer]   [NOT_FOUND]
                                    │                │
                                    ▼                ▼
                               [Show with      [Offer Fallback
                                confidence]      to Mistral Large]
```

---

## 9. Common Follow-Up Questions

### "What would you do differently if starting over?"

- Use **LangGraph** instead of manual chain composition for better observability
- Add **RAGAS evaluation** from day one
- Implement **hybrid retrieval** (BM25 + dense) for better keyword matching
- Use **async** throughout (aiohttp for API calls, async FAISS)
- Add **structured logging** (not just print statements)

### "How would you add multi-document support?"

- Namespace FAISS indices by document ID
- Metadata filtering: tag each chunk with its source document
- Query routing: let user select which documents to search
- Index merging: combine multiple FAISS indices for cross-document queries

### "How would you deploy this to production?"

- **Backend**: FastAPI + Celery for async processing
- **Vector DB**: Migrate from FAISS to Milvus/Qdrant (managed, scalable)
- **Frontend**: React/Next.js for richer interactivity
- **Infrastructure**: Docker + Kubernetes, or serverless (AWS Lambda + S3)
- **Monitoring**: LangSmith/Langfuse for LLM observability
- **CI/CD**: GitHub Actions with automated RAGAS evaluation on PRs

### "What metrics would you track in production?"

| Metric | Why |
|--------|-----|
| P95 query latency | User experience |
| Confidence score distribution | Retrieval quality |
| NOT_FOUND rate | Document coverage gaps |
| Fallback trigger rate | How often RAG fails |
| Token usage per query | Cost optimization |
| User feedback (thumbs up/down) | Ground truth quality signal |

---

## 10. Key Terminology Glossary

| Term | Definition |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation — grounding LLM responses in retrieved documents |
| **Bi-Encoder** | Encodes query and document independently; fast but less accurate |
| **Cross-Encoder** | Encodes (query, document) jointly; slow but highly accurate |
| **FAISS** | Facebook AI Similarity Search — efficient vector similarity library |
| **Cosine Similarity** | Measures angle between vectors; range [-1, 1] |
| **Embedding** | Dense vector representation of text (384 dimensions in our case) |
| **Chunking** | Splitting documents into smaller pieces for embedding |
| **Reranking** | Second-stage scoring to improve retrieval precision |
| **Hallucination** | LLM generating plausible but factually incorrect information |
| **Temperature** | Controls randomness in LLM token selection |
| **Top-K** | Number of nearest neighbors to retrieve |
| **NOT_FOUND** | Sentinel token indicating document lacks relevant context |
| **Session State** | Streamlit's mechanism for persisting data across reruns |
| **Query Expansion** | Generating alternative phrasings to improve retrieval recall |
| **Graceful Degradation** | System continues functioning (with reduced quality) when components fail |

---

## 💡 Pro Tips for the Interview

1. **Start with the problem, not the solution**: "Traditional RAG hallucinates because it blindly trusts retrieved context..."
2. **Mention real bugs you fixed**: The CrossEncoder confidence bug shows debugging depth
3. **Show trade-off awareness**: "We chose FAISS for simplicity, but I'd use Milvus at scale"
4. **Demonstrate system thinking**: "Every stage has a fallback path — that's production engineering"
5. **Quantify when possible**: "384-dim embeddings, 1000-char chunks, 0.20 cosine threshold"
6. **Connect to business impact**: "Confidence scoring lets users trust the system — that's the difference between a demo and a product"

---

> **Remember**: The goal is not to recite this document — it's to deeply understand every design decision so you can discuss them naturally and handle any follow-up question with confidence.

---

*Built by Vishal Kumar Kashyap — AI Engineer*
