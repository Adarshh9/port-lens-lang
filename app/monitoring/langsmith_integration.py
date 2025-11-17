"""
LangSmith integration for full query tracing and observability.
Tracks every step: retrieval, generation, judging, caching.
"""

import logging
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime
from functools import wraps
import os

try:
    from langsmith import Client as LangSmithClient
    from langsmith.wrappers import wrap_openai
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logger = logging.getLogger("rag_llm_system")
    logger.warning("LangSmith not installed. Install with: pip install langsmith")

logger = logging.getLogger("rag_llm_system")


class LangSmithMonitor:
    """Unified monitoring and tracing with LangSmith."""

    def __init__(self, project_name: str = "rag-llm-system"):
        """
        Initialize LangSmith monitor.
        
        Args:
            project_name: LangSmith project name
        """
        self.project_name = project_name
        
        if LANGSMITH_AVAILABLE:
            try:
                self.client = LangSmithClient(project_name=project_name)
                logger.info(f"✅ LangSmith connected - Project: {project_name}")
            except Exception as e:
                logger.warning(f"LangSmith initialization failed: {str(e)}")
                self.client = None
        else:
            self.client = None

    def trace_query_pipeline(self, query: str, session_id: str, user_id: str):
        """
        Decorator to trace full query pipeline.
        
        Usage:
        @monitor.trace_query_pipeline(query, session_id, user_id)
        def my_rag_pipeline(query):
            ...
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create trace entry
                trace_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "query": query,
                    "session_id": session_id,
                    "user_id": user_id,
                    "function": func.__name__,
                }
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    trace_data["status"] = "SUCCESS"
                    trace_data["result"] = result
                except Exception as e:
                    trace_data["status"] = "FAILED"
                    trace_data["error"] = str(e)
                    raise
                finally:
                    trace_data["elapsed_ms"] = (time.time() - start_time) * 1000
                    
                    # Log to LangSmith
                    if self.client:
                        try:
                            self.client.create_run(
                                name=func.__name__,
                                inputs={"query": query},
                                outputs=result if trace_data["status"] == "SUCCESS" else None,
                                run_type="chain"
                            )
                        except Exception as e:
                            logger.warning(f"LangSmith trace failed: {str(e)}")
                    
                    # Log structured metrics
                    logger.info(json.dumps(trace_data))
                
                return result
            
            return wrapper
        return decorator

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log structured metrics at any point.
        
        Usage:
        monitor.log_metrics({
            "step": "retrieval",
            "num_docs": 5,
            "latency_ms": 42,
            "query_hash": "abc123"
        })
        """
        metrics["timestamp"] = datetime.utcnow().isoformat()
        
        # Log to LangSmith
        if self.client:
            try:
                # LangSmith has custom logging capabilities
                pass
            except Exception as e:
                logger.warning(f"LangSmith metrics log failed: {str(e)}")
        
        # Always log to local JSON for analysis
        logger.info(json.dumps(metrics))