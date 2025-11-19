# Port Lens Lang: Hybrid Production RAG System

> **Status:** Production Ready (Phase 3)
> **Architecture:** Hybrid (LangGraph Orchestration + Cost-Aware Routing)
> **Models:** Llama 3.1 (Groq), Phi-3 (Ollama Local), GPT-4o (OpenAI)

A high-performance, modular Retrieval Augmented Generation (RAG) system engineered for cost-efficiency, resilience, and complexity handling. It features a dual-pipeline architecture that can orchestrate complex stateful workflows or route simple queries to local models to minimize costs.



## 🚀 Key Features

### 1. Smart Multi-Model Routing (New)
An intelligent routing layer that classifies query complexity and selects the optimal model to balance cost and performance.
-   **Local First Strategy:** Routes simple factual queries to **Phi-3 Mini (via Ollama)** for $0 inference cost.
-   **Cloud Fallback:** Automatically switches to **Groq (Llama 3.1)** or OpenAI if the local model fails or the query requires deep reasoning.
-   **Resilience:** If a cloud provider goes down (e.g., 500 Internal Error), the system seamlessly falls back to local or alternative models.
-   **Optimization Modes:** User-selectable strategies: `cost`, `speed`, `quality`, or `balanced`.

### 2. Advanced Graph Orchestration
For complex workflows requiring memory and self-correction, utilizing **LangGraph**:
-   **Stateful Execution:** Manages conversation history and retrieval state across turns.
-   **LLM-as-a-Judge:** Automatically evaluates answer quality and triggers re-generation if the initial response is poor.
-   **Hierarchical Caching:** L1 (Redis) for high-speed hits and L2 (SQLite) for persistent storage.
-   **Persistent Memory:** Short-term (RAM) and Long-term (SQLite) conversational memory.

---

## 🛠️ Architecture

The system operates via two primary pipelines managed by a central Router:

```mermaid
graph LR
    User -->|Query| API[FastAPI Gateway]
    API -->|/query (Legacy)| Graph[LangGraph Pipeline]
    API -->|/query/smart| Router[Smart Router]
    
    subgraph "Smart Routing Logic"
        Router -->|Simple Query| Local[Ollama / Phi-3]
        Router -->|Complex Query| Cloud[Groq / Llama-3]
        Router -->|Expert Query| Premium[OpenAI / GPT-4o]
        Local -.->|Fallback| Cloud
    end
    
    Graph --> Nodes[Cache -> Retrieve -> Generate -> Judge]
````

### Tech Stack

  * **Backend:** Python 3.11+, FastAPI, Uvicorn
  * **Orchestration:** LangGraph, LangChain
  * **Vector Store:** ChromaDB
  * **Inference:**
      * *Local:* Ollama (`phi3:mini`)
      * *Fast Cloud:* Groq API (`llama-3.1-8b-instant`)
      * *Premium Cloud:* OpenAI API (`gpt-4o-mini`)
  * **Monitoring:** LangSmith, Custom JSON Logging

-----

## ⚡ Setup & Installation

### 1\. Prerequisites

  * Python 3.10+
  * [Ollama](https://ollama.com/) (Required for local routing)
  * Git

### 2\. Clone & Install

```bash
git clone [https://github.com/yourusername/port-lens-lang.git](https://github.com/yourusername/port-lens-lang.git)
cd port-lens-lang
pip install -r requirements.txt
```

### 3\. Local Model Setup (Critical)

You must initialize the local model for the router to work.

```bash
# 1. Start Ollama in a separate terminal
ollama serve

# 2. Pull the specific model used in config
ollama pull phi3:mini
```

### 4\. Configuration

Create a `.env` file in the root directory:

```ini
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk_...  # Optional, system will skip if missing
LANGSMITH_API_KEY=lsv2_... # Optional for tracing
```

Ensure `config/models.yaml` exists. This file controls cost thresholds, routing logic, and model definitions.

### 5\. Running the System

```bash
python -m uvicorn app.main:app --reload
```

  * **API Docs:** `http://localhost:8000/docs`
  * **Frontend:** `http://localhost:8501` (if running Streamlit)

-----

## 🔌 API Usage

### 🧠 Smart Query (Routing Enabled)

**Endpoint:** `POST /api/v1/query/smart`

Automatically selects the best model based on your strategy.

```json
{
  "query": "What is the capital of France?",
  "optimize_for": "cost",
  "user_id": "test_user"
}
```

| Parameter | Options | Description |
| :--- | :--- | :--- |
| `optimize_for` | `cost` | Forces cheapest model (usually Local). |
| | `speed` | Uses fastest model available (Local or Groq). |
| | `quality` | Uses best capability model (GPT-4o/Llama 3). |
| | `balanced` | (Default) Uses Complexity Classifier to decide. |

### 🔄 Graph Query (Stateful)

**Endpoint:** `POST /api/v1/query`

Uses the full LangGraph pipeline with memory and caching.

```json
{
  "query": "Explain the previous concept in more detail.",
  "session_id": "session_123",
  "use_cache": true
}
```

### 📄 Ingestion

**Endpoint:** `POST /api/v1/ingest`

```json
{ "file_path": "./documents/whitepaper.pdf" }
```

-----

## 📂 Project Structure

```
app/
├── api/                 # Routes for Graph and Smart Routing
├── graph/               # LangGraph Nodes, State, and Builder
├── ingestion/           # PDF/Text Loaders & Chunking
├── llm/                 # Multi-Provider Wrappers (Groq, Ollama, OpenAI)
├── models/              # Configuration Logic
├── routing/             # Query Classifier & Router Logic
├── vector/              # ChromaDB & Retriever
└── main.py              # App Entry Point
config/
└── models.yaml          # Routing strategies and model specs
```

## 🧪 Testing

  * **Unit Tests:** `pytest tests/`
  * **Router Debug:** `python debug_router.py` (Tests connection to all providers and simulates routing logic directly).

<!-- end list -->

````

---