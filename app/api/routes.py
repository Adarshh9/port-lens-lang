"""
FastAPI routes for RAG system.
FIXED: Handle dict state returned from graph.
"""

import logging
import time
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
from app.logger import logger

router = APIRouter(prefix="/api/v1", tags=["RAG"])

_graph_builder: RAGGraphBuilder = None
_indexer = None


def init_routes(graph_builder: RAGGraphBuilder, indexer) -> None:
    """Initialize routes with dependencies."""
    global _graph_builder, _indexer
    _graph_builder = graph_builder
    _indexer = indexer
    logger.info("Routes initialized with graph builder and indexer")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Process a query through the RAG system."""
    logger.info(f"Processing query: {request.query[:100]}")
    start_time = time.time()

    try:
        if not _graph_builder:
            raise HTTPException(
                status_code=500,
                detail="Graph builder not initialized",
            )

        # Create initial state as dict
        state = {
            "query": request.query,
            "session_id": request.session_id,
            "user_id": request.user_id or "",
        }

        # Execute graph - returns dict
        result = _graph_builder.invoke(state)

        # result is now a dict, not an object
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

        final_answer = result.get("final_answer") or result.get("generated_answer") or ""
        if not final_answer:
            final_answer = "I apologize, but I couldn't find enough information to answer your question."

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

        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}",
        )


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """Ingest a document into the system."""
    logger.info(f"Ingesting document: {request.file_path}")

    try:
        if not _indexer:
            raise HTTPException(
                status_code=500,
                detail="Indexer not initialized",
            )

        chunks_indexed = _indexer.ingest_file(request.file_path)
        
        logger.info(f"Document ingested: {chunks_indexed} chunks")
        return IngestResponse(
            status="success",
            chunks_indexed=chunks_indexed,
            file_path=request.file_path,
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    logger.info("Health check requested")

    try:
        logger.info("Health check successful")
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            database="ok",
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Health check failed",
        )


@router.get("/cache/clear")
async def clear_cache():
    """Clear the cache."""
    logger.warning("Cache clear requested")
    try:
        if not _graph_builder:
            raise HTTPException(
                status_code=500,
                detail="Graph builder not initialized",
            )

        _graph_builder.cache.clear()
        return {"status": "cache cleared"}

    except Exception as e:
        logger.error(f"Cache clear failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cache clear failed: {str(e)}",
        )
