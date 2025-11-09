"""
API request/response schemas.
FIXED: Proper schema validation for judge evaluation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Query request schema."""

    query: str = Field(..., description="User query")
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    use_cache: bool = Field(True, description="Whether to use cache")


class RetrievedDocument(BaseModel):
    """Retrieved document schema."""

    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    distance: float = Field(..., description="Distance/similarity score")


class JudgeEvaluation(BaseModel):
    """Judge evaluation schema - FIXED."""

    score: float = Field(..., description="Overall score (0-1)")
    reasons: str = Field(..., description="Evaluation reasons")
    criteria: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Criteria scores (can be dict or floats)"
    )


class QueryResponse(BaseModel):
    """Query response schema."""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    retrieved_docs: List[RetrievedDocument] = Field(
        ..., description="Retrieved documents"
    )
    judge_evaluation: Optional[JudgeEvaluation] = Field(
        None, description="Judge evaluation"
    )
    cache_hit: bool = Field(..., description="Whether cache was hit")
    processing_time: float = Field(..., description="Processing time in seconds")
    quality_passed: bool = Field(..., description="Whether quality threshold passed")


class IngestRequest(BaseModel):
    """Document ingestion request schema."""

    file_path: str = Field(..., description="Path to file to ingest")


class IngestResponse(BaseModel):
    """Document ingestion response schema."""

    status: str = Field(..., description="Ingestion status")
    chunks_indexed: int = Field(..., description="Number of chunks indexed")
    file_path: str = Field(..., description="File path")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="System status")
    version: str = Field(..., description="API version")
    database: str = Field(..., description="Database status")
