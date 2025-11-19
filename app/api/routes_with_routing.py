from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import traceback # Added for detailed debug prints

from app.routing.model_router import CostAwareRouter, RoutingResult
from app.models.model_config import MultiModelConfig
from app.vector.retriever import Retriever

logger = logging.getLogger("rag_llm_system")

router = APIRouter(prefix="/api/v1", tags=["RAG with Routing"])

# Global dependencies
_router: Optional[CostAwareRouter] = None
_retriever: Optional[Retriever] = None

def init_routing(model_config: MultiModelConfig, retriever: Retriever):
    global _router, _retriever
    _router = CostAwareRouter(model_config)
    _retriever = retriever
    logger.info("✅ Multi-model routing initialized (Global)")

class SmartQueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    user_id: str = "default"
    optimize_for: str = "balanced"

# REMOVED response_model=SmartQueryResponse temporarily for debugging
@router.post("/query/smart")
async def smart_query(request: SmartQueryRequest):
    print(f"DEBUG: Received smart query request: {request.query}") # Direct console print
    
    if not _router or not _retriever:
        print("DEBUG: Router or Retriever is None!")
        raise HTTPException(status_code=500, detail="Routing system not initialized")
    
    try:
        # Step 1: Retrieve context
        print("DEBUG: Starting Retrieval...")
        docs = _retriever.retrieve(request.query, k=2)
        print(f"DEBUG: Retrieved {len(docs)} docs")
        
        context = "\n\n".join([doc.get("content", "")[:500] for doc in docs])
        
        # Step 2: Route and generate
        print(f"DEBUG: calling router with optimize_for={request.optimize_for}")
        result: RoutingResult = await _router.route_and_generate(
            query=request.query,
            context=context,
            optimize_for=request.optimize_for,
            user_id=request.user_id,
        )
        print(f"DEBUG: Generation complete. Model: {result.model_used}")

        # Step 3: Build Response (Manually, as a dict)
        response_data = {
            "query": request.query,
            "answer": result.answer,
            "model_used": result.model_used,
            "judge_score": result.judge_score,
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
            "retrieval_docs_count": len(docs),
            
            # Routing metadata
            "query_complexity": result.classification.complexity_score,
            "query_difficulty": result.classification.difficulty,
            "attempts": result.attempts,
            "fallback_used": result.fallback_used,
            "routing_reasoning": result.classification.routing_reasoning,
            
            # Performance - SAFETY CHECK these keys
            "input_tokens": result.routing_metadata.get("input_tokens", 0),
            "output_tokens": result.routing_metadata.get("output_tokens", 0),
            "model_latency_ms": result.routing_metadata.get("model_latency_ms", 0.0),
        }
        
        print("DEBUG: Response object built successfully")
        return response_data
        
    except Exception as e:
        # Print full traceback to console so you definitely see it
        traceback.print_exc()
        logger.error(f"Smart query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Smart query failed: {str(e)}"
        )