"""
Structured logging for all RAG operations.
Captures: latency, judge scores, cache stats, model routing.
"""

import logging
import json
import time
from typing import Dict, Any
from datetime import datetime
from pathlib import Path


class MetricsLogger:
    """Structured metrics logging to JSON."""

    def __init__(self, log_file: str = "./logs/metrics.jsonl"):
        """
        Initialize metrics logger.
        
        Args:
            log_file: JSONL file for metrics
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_query_execution(
        self,
        query: str,
        cache_status: str,  # "HIT", "MISS"
        cache_level: str,  # "L1", "L2", "NONE"
        retrieval_time_ms: float,
        generation_time_ms: float,
        judge_score: float,
        model_used: str,
        docs_retrieved: int,
        session_id: str,
        user_id: str,
        judge_details: Dict[str, Any] = None,
    ) -> None:
        """
        Log complete query execution metrics.
        
        Args:
            query: User query
            cache_status: HIT/MISS
            cache_level: L1/L2/NONE
            retrieval_time_ms: Retrieval latency
            generation_time_ms: Generation latency
            judge_score: Quality score
            model_used: LLM model name
            docs_retrieved: Number of docs
            session_id: Session ID
            user_id: User ID
            judge_details: Judge evaluation details
        """
        total_time_ms = retrieval_time_ms + generation_time_ms
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "query_execution",
            "query_hash": self._hash_query(query),
            "query_length": len(query),
            "session_id": session_id,
            "user_id": user_id,
            "cache": {
                "status": cache_status,
                "level": cache_level,
                "hit": cache_status == "HIT"
            },
            "performance": {
                "retrieval_ms": retrieval_time_ms,
                "generation_ms": generation_time_ms,
                "total_ms": total_time_ms
            },
            "quality": {
                "judge_score": judge_score,
                "threshold_passed": judge_score >= 0.7,
                "details": judge_details or {}
            },
            "model": model_used,
            "docs_retrieved": docs_retrieved,
            "metrics": {
                "retrieval_latency_percentile": self._estimate_percentile(retrieval_time_ms),
                "total_latency_percentile": self._estimate_percentile(total_time_ms)
            }
        }
        
        self._write_metric(metrics)

    def log_cache_operation(
        self,
        operation: str,  # "GET", "SET", "CLEAR"
        cache_level: str,  # "L1", "L2"
        hit: bool,
        latency_ms: float,
        key_hash: str,
    ) -> None:
        """Log cache operation."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "cache_operation",
            "operation": operation,
            "cache_level": cache_level,
            "hit": hit,
            "latency_ms": latency_ms,
            "key_hash": key_hash
        }
        
        self._write_metric(metrics)

    def log_model_routing(
        self,
        query: str,
        preferred_model: str,
        selected_model: str,
        reason: str,
        expected_cost_usd: float,
        expected_latency_ms: float,
    ) -> None:
        """Log model routing decision."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "model_routing",
            "query_hash": self._hash_query(query),
            "preferred_model": preferred_model,
            "selected_model": selected_model,
            "routing_reason": reason,
            "cost_estimate_usd": expected_cost_usd,
            "latency_estimate_ms": expected_latency_ms
        }
        
        self._write_metric(metrics)

    def _write_metric(self, metric: Dict[str, Any]) -> None:
        """Write metric to JSONL file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(metric) + "\n")
        except Exception as e:
            logging.error(f"Failed to write metric: {str(e)}")

    @staticmethod
    def _hash_query(query: str) -> str:
        """Hash query for anonymized tracking."""
        import hashlib
        return hashlib.sha256(query.encode()).hexdigest()[:16]

    @staticmethod
    def _estimate_percentile(latency_ms: float) -> str:
        """Estimate latency percentile."""
        if latency_ms < 50:
            return "p25"
        elif latency_ms < 500:
            return "p50"
        elif latency_ms < 1500:
            return "p75"
        else:
            return "p99"