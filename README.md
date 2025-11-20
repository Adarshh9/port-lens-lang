# Port Lens Lang: Production RAG with Hybrid Routing

> **Status:** Production Ready (Phase 3+)
>
> **Core:** Dual-Orchestration (LangGraph & Smart Router)
>
> **Models:** Llama 3.1 (Groq), Phi-3 (Ollama Local), GPT-4o (OpenAI)

***

## 🕹️ Problem Statement

Building RAG systems with great demo metrics means nothing when your cloud LLM vendor throws a 500 error, cost spikes out of nowhere, or the pipeline faces a query it's not fit for. Most "RAG starters" are locked to a single LLM, waste money on trivial queries, or crumble at scale.

Port Lens Lang solves real production pain—balancing resilience, cost, and depth using hybrid local/cloud models, dynamic routing, smart memory, observability, and bulletproof fallback. **No single point of failure, every query gets the best fit.**

***

## 🚀 TL;DR Solution & Approach

- **Multi-Pipeline Architecture:** Graph-based for complex, stateful flows; stateless smart router for blazing fast cost/performance wins.
- **Local-First Routing:** Cheap, fast local inferencing on Ollama for basic/factual queries. Cloud providers (Groq/OpenAI) are fallback or for high-complexity.
- **Quality-Aware Fallback:** Answers are always judged; failover is instant and automatic on any error or subpar quality.
- **Cost and Latency-Optimized:** Routing respects your `cost`, `speed`, `quality`, or `balanced` preference at every request. Real, working trade-off.
- **Observability & Monitoring:** Integrated with LangSmith and custom logs for instant debugging and analytics.

***

## 🧬 Technologies & Techniques Used

- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **RAG Orchestration:** LangGraph, LangChain
- **Vector DB:** ChromaDB (disk-backed, persistent)
- **Inference:**
  - **Local:** Ollama (phi3:mini)
  - **Cloud-Fast:** Groq API (llama-3.1-8b-instant)
  - **Cloud-Premium:** OpenAI API (gpt-4o-mini)
- **Judging:** LLM-as-a-judge for automatic answer scoring
- **Caching:** Redis or filesystem, with two-layer fallback
- **Memory:** RAM buffer & SQL persistent (long-term)
- **Monitoring:** LangSmith tracing, structured JSON logs, ready for Grafana/Prometheus
- **Containerization:** Docker and Compose setup
- **Testing:** pytest, plus `debug_router.py` for hands-on pipeline probes

***

## 🔬 Techniques/Patterns Demonstrated

- **Query Complexity Classification:** Fast heuristics (keyword, length) to avoid the embarrassment of LLM-for-everything routing
- **Dynamic Multi-Model Routing:** Cost and quality-driven model selection
- **Resilient Error Handling:** All exceptions, API errors, timeouts transparently trigger next-best model
- **Self-correcting graph:** Bad answers don’t pollute memory—automatic retry/fallback
- **Pluggable Model Configuration:** YAML-based, env-variable controls, instant new model onboarding

***

## 🛠️ Architecture & Workflow

### 1. Dual-Pipeline
```
User → FastAPI
    ├─ /api/v1/query (LangGraph: for stateful, complex)
    |
    └─ /api/v1/query/smart (Smart Router: optimized for cost/speed/quality)
```

### 2. Graph Pipeline (LangGraph)
```
Cache Check → Retrieve Context → Generate (LLM) → Judge Score
     └─(Low Quality)
            ↓
        Fallback Node
     └─(Quality Pass)
           ↓
     Memory Update → Cache Write → Return
```

### 3. Smart Routing Workflow
```
User Query
  ↓
Query Classifier [complexity ∈ (0,1)]
  ├─ <0.3    → Local/Ollama (phi3:mini)
  ├─ <0.6    → Groq Cloud (llama3_8b)
  └─ otherwise → OpenAI (gpt-4o) or fallback

Each answer judged; fallback chain is always ready.
```

- **All layers share resources:** Vector db, memory, cache.
- **Monitoring at every step:** Logs, traces, metrics.

flowchart TD
    Start[User Request or API Call]
    Start --> ChoosePipeline{Query Type?}

    ChoosePipeline -->|Stateful or Conversational| GraphPipeline[Graph Pipeline - LangGraph]
    ChoosePipeline -->|Simple or Stateless| RouterPipeline[Smart Router Pipeline]

    %% Graph Pipeline
    GraphPipeline --> GCacheCheck[Cache Check]
    GCacheCheck -- No Hit --> GRetrieve[Retrieve from Vector DB]
    GCacheCheck -- Hit --> GReturnCached[Return Cached Answer]
    GRetrieve --> GLLMGen[LLM Generation - Groq]
    GLLMGen --> GJudge[Judge Quality]
    GJudge -- Pass --> GStore[Update Memory, Write Cache]
    GJudge -- Fail --> GFallback[Graph Fallback Node]
    GStore --> GResponse[Return Response]
    GReturnCached --> GResponse
    GFallback --> GResponse

    %% Smart Routing Pipeline
    RouterPipeline --> Classifier[Query Classifier - Complexity Score 0-1]
    Classifier -->|Simple <0.3| LocalLLM[Ollama - Phi3 Mini]
    Classifier -->|Medium <0.6| CloudLLM[Groq - Llama3]
    Classifier -->|Complex >=0.6| PremiumLLM[OpenAI - GPT-4o]

    LocalLLM --> LocalJudge[Judge Quality]
    CloudLLM --> CloudJudge[Judge Quality]
    PremiumLLM --> PremiumJudge[Judge Quality]

    LocalJudge -- Quality Pass --> SmartMem[Write to Short/Long Memory]
    CloudJudge -- Quality Pass --> SmartMem
    PremiumJudge -- Quality Pass --> SmartMem

    LocalJudge -- Fail or Error --> CloudLLM
    CloudJudge -- Fail or Error --> PremiumLLM
    LocalLLM -- Error or Exception --> CloudLLM
    CloudLLM -- Error or Exception --> PremiumLLM

    SmartMem --> SmartResp[Return Routed Response]

    %% Docs
    subgraph DataLayer
        VectorDB[ChromaDB Vector Store]
        RedisCache[Redis or FS Cache]
        SQLiteMem[SQLite Memory]
    end

    GRetrieve --> VectorDB
    GCacheCheck --> RedisCache
    GStore --> SQLiteMem
    SmartMem --> SQLiteMem
    RouterPipeline --> VectorDB

    %% Monitoring
    GJudge -.-> MonitorSynapse[Monitoring and Logging - LangSmith, Grafana]
    Classifier -.-> MonitorSynapse
    SmartResp -.-> MonitorSynapse
    GResponse -.-> MonitorSynapse

***

## 🔥 Challenges & What Worked

### 1. **Cloud Outages (Groq 500 errors)**
  - **What didn’t work:** Assuming the primary cloud model will always be up.
  - **What worked:** Local-first routing. As soon as Groq failed, Ollama handled the query (instant, zero downtime).

### 2. **Swallowing Local Errors**
  - **What didn’t work:** Returning failed Ollama runs as "bad answers" flagged as low quality.
  - **What worked:** Refactored to raise exceptions so router instantly tries fallback. Now, no accidental silent failures.

### 3. **Efficient Cost Control**
  - **What didn’t work:** Fixed-model, high-quality-everywhere design (too expensive).
  - **What worked:** Query classifier actually routed ~50% of prod queries to local, saving >21% at volume.

### 4. **Debugging Blackhole Bugs**
  - **What didn’t work:** FastAPI’s lack of error transparency in prod.
  - **What worked:** Added explicit debug scripts (`debug_router.py`) and made API endpoints print exception tracebacks to console.

### 5. **Model Management**
  - **What didn’t work:** Reliance on `ollama pull` if registry/connection flaky.
  - **What worked:** Manual local model creation script, guaranteed that the correct model is always available.

***

## ⚡ Quick Setup Guide

### 1. Prerequisites

- Python 3.10+
- Ollama (must be running for local inference)
- (Optional) Redis, Docker
- Git

### 2. Install & Configure

```bash
git clone https://github.com/yourusername/port-lens-lang.git
cd port-lens-lang
pip install -r requirements.txt

# Start Ollama (new terminal)
ollama serve

# Pull local model (critical)
ollama pull phi3:mini

# Copy .env and edit keys
cp .env.example .env
# Fill in: GROQ_API_KEY=...  [OPENAI_API_KEY=...]
```

**Model config**: Check `config/models.yaml` is present and matches your actual model names (especially Ollama).

### 3. Run the System

```bash
python -m uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
```

***

## 🏃 Run, Inference, and Test Guide

### Run Interactive Debug/Smoke Tests

```bash
python debug_router.py   # Probes Ollama, Groq, routes a fake query
```

### Query the Smart Router

```bash
curl -X POST "http://localhost:8000/api/v1/query/smart" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?", "optimize_for": "balanced"}'
```

### Query the LangGraph Pipeline (Stateful)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain clustering.", "session_id": "s123"}'
```

### Ingest Documents

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "./docs/your.pdf"}'
```

***

## 👩‍💻 Contributing Guide

1. Fork the repo & create a branch (`feature/xyz`)
2. Write clean, tested code (pytest or debug_router.py for edge cases)
3. Document new endpoints/config options
4. Submit pull request and describe the use case, new config, and any migration steps

**Tips:**
- If you’re adding a new provider/model, edit `config/models.yaml`, test using `debug_router.py`, then document in the README.
- All exceptions should be catchable by users – never let a user "lose" a query without a trace.

***

## 📝 Additional Notes

- **Monitoring/Tracing:** Enable LangSmith or custom logs for real-world analytics.
- **Cache:** For best production speed, enable Redis. For solo/self-hosted, simple FS caching works.
- **Security:** Protect your .env. Add FastAPI security/rate-limiting if deploying public.

***

## 🚧 Roadmap & Next Steps

- Phase 4: Metrics Dashboard (Prometheus/Grafana or use LangSmith)
- Phase 5: Adaptive query learning/true hybrid-orchestration
- Visual query history in the upcoming web UI

***