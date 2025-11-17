import logging
import time
from typing import Optional
from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    JudgeEvaluation,
    IngestRequest,
    IngestResponse,
    HealthResponse,
)
from app.graph.graph_builder import RAGGraphBuilder
from app.monitoring.rag_evaluators import RAGEvaluator
from app.llm.groq_wrapper import GroqLLM

logger = logging.getLogger("rag_llm_system")

router = APIRouter(prefix="/api/v1", tags=["RAG"])

# Global dependencies
_graph_builder: Optional[RAGGraphBuilder] = None
_indexer = None
_rag_evaluator: Optional[RAGEvaluator] = None


def init_routes(graph_builder: RAGGraphBuilder, indexer) -> None:
    """Initialize routes with dependencies."""
    global _graph_builder, _indexer, _rag_evaluator
    
    _graph_builder = graph_builder
    _indexer = indexer
    
    # Initialize evaluator
    judge_llm = GroqLLM()
    _rag_evaluator = RAGEvaluator()
    
    logger.info("✅ Routes initialized with graph builder, indexer, and evaluator")


# ============================================================================
# QUERY ENDPOINTS
# ============================================================================

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a query through the RAG system.
    
    Returns answer, retrieved docs, judge evaluation, and metadata.
    """
    logger.info(f"Processing query: {request.query[:100]}")
    start_time = time.time()
    
    try:
        if not _graph_builder:
            raise HTTPException(
                status_code=500,
                detail="Graph builder not initialized"
            )
        
        # Create initial state as dict
        state = {
            "query": request.query,
            "session_id": request.session_id,
            "user_id": request.user_id or "",
        }
        
        # Execute graph - returns dict
        result = _graph_builder.invoke(state)
        
        # Parse retrieved docs
        retrieved_docs = []
        for doc in result.get("retrieved_docs", []):
            if isinstance(doc, dict):
                retrieved_docs.append(
                    RetrievedDocument(
                        content=doc.get("content", ""),
                        metadata=doc.get("metadata", {}),
                        distance=doc.get("distance", 0.0),
                    )
                )
        
        # Parse judge evaluation
        judge_eval = None
        je = result.get("judge_evaluation", {})
        if je and isinstance(je, dict):
            try:
                judge_eval = JudgeEvaluation(
                    score=float(je.get("score", 0.0)),
                    reasons=str(je.get("reasons", "")),
                    criteria=je.get("criteria", {}),
                )
            except Exception as e:
                logger.warning(f"Failed to parse judge evaluation: {str(e)}")
        
        # Get final answer
        final_answer = (
            result.get("final_answer")
            or result.get("generated_answer")
            or "I apologize, but I couldn't find enough information to answer your question."
        )
        
        processing_time = time.time() - start_time
        
        response = QueryResponse(
            query=request.query,
            answer=final_answer,
            retrieved_docs=retrieved_docs,
            judge_evaluation=judge_eval,
            cache_hit=result.get("cache_hit", False),
            processing_time=processing_time,
            quality_passed=result.get("quality_passed", False),
        )
        
        logger.info(
            f"✅ Query processed: {processing_time:.2f}s, "
            f"cache_hit={response.cache_hit}, "
            f"quality_passed={response.quality_passed}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


# ============================================================================
# EVALUATION ENDPOINTS (NEW)
# ============================================================================

@router.get("/evaluation/summary")
async def get_evaluation_summary(last_n: int = 100):
    """
    Get RAG evaluation summary statistics.
    
    Args:
        last_n: Number of recent evaluations to analyze
        
    Returns:
        Summary statistics including:
        - Overall RAG score
        - Retrieval quality
        - Generation quality
        - Average latency
        - Total cost
    """
    logger.info(f"Getting evaluation summary for last {last_n} evaluations")
    
    try:
        if not _rag_evaluator:
            raise HTTPException(
                status_code=500,
                detail="Evaluator not initialized"
            )
        
        summary = _rag_evaluator.get_evaluation_summary(last_n=last_n)
        
        if not summary:
            return {
                "message": "No evaluations found",
                "total_evaluations": 0
            }
        
        logger.info(f"✅ Evaluation summary generated: {summary.get('total_evaluations', 0)} evaluations")
        
        return summary
        
    except Exception as e:
        logger.error(f"Evaluation summary failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation summary failed: {str(e)}"
        )


@router.post("/evaluation/evaluate")
async def evaluate_response(
    query: str,
    answer: str,
    retrieved_docs: list,
    ground_truth_answer: Optional[str] = None,
):
    """
    Manually evaluate a RAG response.
    
    Useful for:
    - Testing different answers
    - Comparing model outputs
    - Debugging quality issues
    
    Args:
        query: Original query
        answer: Generated answer
        retrieved_docs: Retrieved documents
        ground_truth_answer: (Optional) Reference answer for correctness
        
    Returns:
        Complete evaluation breakdown:
        - Retrieval metrics
        - Generation metrics
        - Overall score
    """
    logger.info(f"Manual evaluation requested for query: {query[:50]}")
    
    try:
        if not _rag_evaluator:
            raise HTTPException(
                status_code=500,
                detail="Evaluator not initialized"
            )
        
        evaluation = _rag_evaluator.evaluate_rag_response(
            query=query,
            retrieved_docs=retrieved_docs,
            answer=answer,
            latency_ms=0,  # Manual eval, no latency
            cost_usd=0,    # Manual eval, no cost
            ground_truth_answer=ground_truth_answer,
            session_id="manual_eval",
            user_id="api_user",
        )
        
        logger.info(f"✅ Evaluation complete: overall_score={evaluation.get('overall_score', 0):.2f}")
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


# ============================================================================
# INGESTION ENDPOINTS
# ============================================================================

@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Ingest a document into the vector store.
    
    Args:
        request: Ingest request with file path
        
    Returns:
        Status and number of chunks indexed
    """
    logger.info(f"Ingesting document: {request.file_path}")
    
    try:
        if not _indexer:
            raise HTTPException(
                status_code=500,
                detail="Indexer not initialized"
            )
        
        chunks_indexed = _indexer.ingest_file(request.file_path)
        
        logger.info(f"✅ Document ingested: {chunks_indexed} chunks")
        
        return IngestResponse(
            status="success",
            chunks_indexed=chunks_indexed,
            file_path=request.file_path,
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns system status and version.
    """
    logger.info("Health check requested")
    
    try:
        status = {
            "status": "healthy",
            "version": "1.0.0",
            "database": "ok",
            "graph_builder": "ok" if _graph_builder else "not_initialized",
            "evaluator": "ok" if _rag_evaluator else "not_initialized",
        }
        
        logger.info("✅ Health check successful")
        
        return HealthResponse(**status)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


@router.get("/cache/clear")
async def clear_cache():
    """
    Clear the cache.
    
    Useful for:
    - Testing
    - Forcing fresh responses
    - Clearing stale data
    """
    logger.warning("Cache clear requested")
    
    try:
        if not _graph_builder:
            raise HTTPException(
                status_code=500,
                detail="Graph builder not initialized"
            )
        
        _graph_builder.cache.clear()
        logger.info("✅ Cache cleared")
        
        return {"status": "cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Cache clear failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cache clear failed: {str(e)}"
        )


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns:
        - Cache size
        - Hit rate
        - Miss rate
    """
    logger.info("Cache stats requested")
    
    try:
        if not _graph_builder:
            raise HTTPException(
                status_code=500,
                detail="Graph builder not initialized"
            )
        
        # Try to get stats from cache
        try:
            stats = _graph_builder.cache.get_stats()
        except AttributeError:
            # Cache doesn't have get_stats method
            stats = {"message": "Cache stats not available"}
        
        logger.info("✅ Cache stats retrieved")
        
        return stats
        
    except Exception as e:
        logger.error(f"Cache stats failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cache stats failed: {str(e)}"
        )