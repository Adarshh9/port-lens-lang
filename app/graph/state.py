"""
State definition for RAG orchestration graph.
Using TypedDict for LangGraph compatibility.
CRITICAL: TypedDict is required for StateGraph to properly merge dict updates.
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class RAGState(TypedDict, total=False):
    """
    State for RAG orchestration graph.
    Using TypedDict ensures LangGraph can properly merge dict updates.
    
    total=False means all fields are optional for updates.
    """
    # Input fields
    query: str
    session_id: str
    user_id: str
    
    # Retrieval fields
    retrieved_docs: List[Dict[str, Any]]
    retrieval_metadata: Dict[str, Any]
    
    # Generation fields
    generated_answer: str
    generation_metadata: Dict[str, Any]
    
    # Quality evaluation fields
    judge_score: float
    judge_evaluation: Dict[str, Any]
    quality_passed: bool
    
    # Caching fields
    cache_hit: bool
    cached_answer: Optional[str]
    
    # Memory fields
    conversation_history: List[Dict[str, str]]
    facts: Dict[str, Any]
    
    # Flow control fields
    used_fallback: bool
    final_answer: str
    
    # Metadata fields
    processing_time: float
    timestamp: str
    errors: List[str]
