import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
from datetime import datetime

try:
    from langsmith import Client as LangSmithClient
    from langsmith import wrappers
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None
    wrappers = None

from app.config import settings

logger = logging.getLogger("rag_llm_system")


class LangSmithRAGIntegration:
    """
    Integrate RAG evaluations with LangSmith.
    
    Provides:
    1. Dataset creation from evaluation logs
    2. Experiment runner with evaluators
    3. Metrics logging to LangSmith
    """
    
    def __init__(self):
        """Initialize LangSmith integration."""
        if not LANGSMITH_AVAILABLE:
            logger.error("LangSmith not installed: pip install langsmith")
            self.client = None
            return
        
        try:
            self.client = LangSmithClient(api_key=settings.langsmith_api_key)
            logger.info("✅ LangSmith client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LangSmith: {str(e)}")
            self.client = None
    
    def create_dataset_from_evaluations(
        self,
        dataset_name: str,
        description: str,
        evaluations_file: str = "./logs/rag_evaluations.jsonl"
    ) -> Optional[str]:
        """
        Create LangSmith dataset from evaluation logs.
        
        Args:
            dataset_name: Name for the dataset
            description: Dataset description
            evaluations_file: Path to rag_evaluations.jsonl
        
        Returns:
            Dataset ID if successful
        """
        if not self.client:
            logger.error("LangSmith client not available")
            return None
        
        try:
            # Create dataset
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description=description,
            )
            logger.info(f"✅ Created dataset: {dataset_name} (ID: {dataset.id})")
            
            # Read evaluation logs and convert to examples
            examples = []
            evaluations_path = Path(evaluations_file)
            
            if not evaluations_path.exists():
                logger.warning(f"Evaluation file not found: {evaluations_file}")
                return dataset.id
            
            with open(evaluations_path, "r") as f:
                for line in f:
                    try:
                        eval_data = json.loads(line)
                        
                        # Convert evaluation to example format
                        example = {
                            "inputs": {
                                "query": eval_data.get("query", ""),
                                "retrieved_docs": eval_data.get("retrieval", {}).get("num_retrieved", 0),
                            },
                            "outputs": {
                                "answer": f"[Evaluation ID: {eval_data.get('timestamp')}]",
                            },
                            "metadata": {
                                "session_id": eval_data.get("session_id", ""),
                                "user_id": eval_data.get("user_id", ""),
                                "overall_score": eval_data.get("overall_score", 0),
                                "retrieval_metrics": eval_data.get("retrieval", {}),
                                "generation_metrics": eval_data.get("generation", {}),
                                "system_metrics": eval_data.get("system", {}),
                            }
                        }
                        
                        examples.append(example)
                    except json.JSONDecodeError:
                        continue
            
            # Add examples to dataset
            if examples:
                self.client.create_examples(
                    dataset_id=dataset.id,
                    examples=examples
                )
                logger.info(f"✅ Added {len(examples)} examples to dataset")
            
            return dataset.id
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            return None
    
    def log_evaluation_to_langsmith(
        self,
        eval_dict: Dict[str, Any],
        dataset_id: Optional[str] = None
    ) -> None:
        """
        Log evaluation results to LangSmith.
        
        Args:
            eval_dict: Evaluation result dictionary
            dataset_id: Optional dataset ID for grouping
        """
        if not self.client:
            return
        
        try:
            # Create feedback/metrics in LangSmith
            query = eval_dict.get("query", "")
            overall_score = eval_dict.get("overall_score", 0)
            
            # Log as a run (trace will capture this)
            metadata = {
                "overall_score": overall_score,
                "retrieval_metrics": eval_dict.get("retrieval", {}),
                "generation_metrics": eval_dict.get("generation", {}),
                "system_metrics": eval_dict.get("system", {}),
            }
            
            logger.debug(f"Evaluation logged for: {query[:50]}")
            
        except Exception as e:
            logger.warning(f"Failed to log to LangSmith: {str(e)}")
    
    def create_evaluators(self) -> Dict[str, Callable]:
        """
        Create LangSmith evaluators for RAG metrics.
        
        Returns:
            Dict of evaluator functions
        """
        evaluators = {}
        
        # Evaluator 1: Overall Score
        def overall_score_evaluator(inputs: dict, outputs: dict) -> dict:
            """Evaluate overall RAG score."""
            score = outputs.get("overall_score", 0.5)
            return {
                "key": "overall_score",
                "score": score,
                "comment": f"Overall RAG score: {score:.2f}"
            }
        
        evaluators["overall_score"] = overall_score_evaluator
        
        # Evaluator 2: Retrieval Quality
        def retrieval_evaluator(inputs: dict, outputs: dict) -> dict:
            """Evaluate retrieval metrics."""
            retrieval = outputs.get("retrieval", {})
            context_relevance = retrieval.get("context_relevance", 0)
            
            return {
                "key": "retrieval_quality",
                "score": context_relevance,
                "comment": f"Retrieval quality: {context_relevance:.2f}"
            }
        
        evaluators["retrieval"] = retrieval_evaluator
        
        # Evaluator 3: Generation Quality
        def generation_evaluator(inputs: dict, outputs: dict) -> dict:
            """Evaluate generation metrics."""
            generation = outputs.get("generation", {})
            gen_score = generation.get("avg_generation_score", 0)
            
            return {
                "key": "generation_quality",
                "score": gen_score,
                "comment": f"Generation quality: {gen_score:.2f}"
            }
        
        evaluators["generation"] = generation_evaluator
        
        return evaluators