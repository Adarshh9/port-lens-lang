# RAG + LLM System with LangGraph

A production-ready Retrieval Augmented Generation (RAG) system with LLM integration using LangChain, LangGraph, Groq API, and advanced orchestration.

## Features

- **Advanced RAG Pipeline**: Multi-layer orchestration with LangGraph
- **Groq LLM Integration**: Fast, reliable LLM inference
- **Vector Database**: Chroma for persistent embeddings storage
- **Multi-layer Memory**: Short-term (buffer) and long-term (persistent) memory
- **LLM as Judge**: Automatic quality evaluation of generated responses
- **Caching Layer**: Redis or filesystem-based caching
- **Monitoring**: LangSmith integration for tracing
- **REST API**: FastAPI-based production-ready API
- **Docker Support**: Full containerization with Docker Compose

## Architecture

```
User Query
    ↓
Cache Check (Fast retrieval if available)
    ↓
Document Retrieval (From vector store)
    ↓
LLM Generation (Using Groq)
    ↓
Quality Judge (Evaluates response)
    ↓
Fallback Check (If quality threshold not met)
    ↓
Memory Update (Store in short/long term)
    ↓
Response
```

## System Requirements

- Python 3.11+
- Redis (optional, for caching)
- Docker & Docker Compose (optional)

## Installation

### 1. Clone Repository

```
git clone <repository-url>
cd rag-llm-system
```

### 2. Create Virtual Environment

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```
cp .env.example .env
```

Required environment variables:
- `GROQ_API_KEY`: Your Groq API key
- `LANGSMITH_API_KEY`: Your LangSmith API key (optional)

### 5. Initialize Database

```
python -c "from app.memory.long_term import LongTermMemory; LongTermMemory()"
```

## Usage

### Start Development Server

```
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start with Docker

```
docker-compose up -d
```

### Ingest Documents

```
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "path/to/document.pdf"}'
```

### Query the System

```
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "session_id": "session_123",
    "user_id": "user_123"
  }'
```

### Health Check

```
curl http://localhost:8000/api/v1/health
```

## Configuration

All configuration is managed through `.env` file. Key settings:

- **LLM**: `GROQ_MODEL` (default: mixtral-8x7b-32768)
- **Vector DB**: `CHROMA_DB_PATH`, `CHROMA_COLLECTION_NAME`
- **Embeddings**: `EMBEDDING_MODEL` (default: sentence-transformers/all-MiniLM-L6-v2)
- **Cache**: `CACHE_TYPE` (filesystem or redis)
- **Judge**: `JUDGE_QUALITY_THRESHOLD` (default: 7.0/10)
- **Memory**: `SHORT_TERM_MEMORY_MAX_MESSAGES` (default: 20)

## API Endpoints

### POST /api/v1/query

Process a query through the RAG system.

Request:
```
{
  "query": "Your question here",
  "session_id": "unique_session_id",
  "user_id": "optional_user_id",
  "use_cache": true
}
```

Response:
```
{
  "query": "Your question here",
  "answer": "Generated answer",
  "retrieved_docs": [...],
  "judge_evaluation": {...},
  "cache_hit": false,
  "processing_time": 2.34,
  "quality_passed": true
}
```

### POST /api/v1/ingest

Ingest a document into the system.

Request:
```
{
  "file_path": "path/to/document.pdf"
}
```

### GET /api/v1/health

Health check endpoint.

### GET /api/v1/cache/clear

Clear the cache.

## Testing

Run all tests:

```
pytest
```

Run specific test:

```
pytest tests/test_rag_flow.py -v
```

Run with coverage:

```
pytest --cov=app tests/
```

## Monitoring

### LangSmith Integration

Set `LANGSMITH_API_KEY` to enable tracing. Access traces at:
https://smith.langchain.com/

### Logging

Logs are stored in `./logs/app.log` with JSON formatting for structured logging.

Access logs:
```
tail -f logs/app.log
```

## Performance Optimization

1. **Cache**: Enable Redis for better performance
   ```
   CACHE_TYPE=redis
   REDIS_URL=redis://localhost:6379
   ```

2. **Embedding Model**: Use smaller models for faster inference
   ```
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

3. **Chunk Size**: Adjust for better retrieval
   ```
   CHUNK_SIZE=1024
   CHUNK_OVERLAP=256
   ```

## Troubleshooting

### Redis Connection Error

Ensure Redis is running:
```
docker-compose up redis
```

### Out of Memory

Reduce `SHORT_TERM_MEMORY_MAX_MESSAGES` or `CHUNK_SIZE`

### Low Quality Responses

Increase `JUDGE_QUALITY_THRESHOLD` or improve document ingestion

## Development

### Project Structure

- `app/`: Main application code
- `app/ingestion/`: Document loading and processing
- `app/vector/`: Vector database operations
- `app/llm/`: LLM integration
- `app/memory/`: Memory management
- `app/graph/`: LangGraph orchestration
- `app/cache/`: Caching layer
- `app/api/`: FastAPI routes
- `tests/`: Unit and integration tests

### Adding New Nodes

1. Create file in `app/graph/nodes/`
2. Implement async function with RAGState
3. Add to graph in `graph_builder.py`
4. Add edges for routing

Example:
```
async def custom_node(state: RAGState) -> RAGState:
    # Process state
    return state
```

## Production Deployment

### Using Docker

```
docker-compose -f docker-compose.yml up -d
```

### Using Kubernetes

Generate deployment manifests and deploy:
```
kubectl apply -f k8s/deployment.yaml
```

### Environment Variables for Production

```
ENVIRONMENT=production
DEBUG=false
LANGSMITH_TRACING_V2=true
JUDGE_ENABLE_FALLBACK=true
CACHE_TYPE=redis
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License

## Support

For issues and questions, open an issue on GitHub or contact the team.

## Roadmap

- [ ] Multi-model support
- [ ] Fine-tuning pipeline
- [ ] Advanced retrieval strategies
- [ ] Web UI
- [ ] Batch processing
- [ ] Distributed processing
```