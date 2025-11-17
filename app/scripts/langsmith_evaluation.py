import sys
from pathlib import Path

try:
    from langsmith import Client as LangSmithClient
except ImportError:
    print("Error: langsmith not installed. Run: pip install langsmith")
    sys.exit(1)

from app.monitoring.langsmith_integration import LangSmithRAGIntegration
from app.monitoring.langsmith_evaluators import (
    retrieval_precision_evaluator,
    retrieval_recall_evaluator,
    generation_relevance_evaluator,
    generation_groundedness_evaluator,
    overall_rag_evaluator,
)
from app.config import settings


def create_rag_dataset():
    """Create LangSmith dataset from RAG evaluations."""
    integration = LangSmithRAGIntegration()
    
    dataset_id = integration.create_dataset_from_evaluations(
        dataset_name="rag-evaluation-metrics",
        description="RAG system evaluation metrics with retrieval and generation scores",
        evaluations_file="./logs/rag_evaluations.jsonl"
    )
    
    if dataset_id:
        print(f"✅ Dataset created: {dataset_id}")
        print(f"View at: https://smith.langchain.com/datasets/{dataset_id}")
    else:
        print("❌ Failed to create dataset")
    
    return dataset_id


def run_rag_experiment(dataset_id: str, target_func, experiment_prefix: str = "rag-eval"):
    """
    Run LangSmith experiment on RAG evaluations.
    
    Args:
        dataset_id: LangSmith dataset ID
        target_func: Function to evaluate (e.g., your query handler)
        experiment_prefix: Prefix for experiment name
    """
    try:
        client = LangSmithClient(api_key=settings.langsmith_api_key)
        
        evaluators = [
            retrieval_precision_evaluator,
            retrieval_recall_evaluator,
            generation_relevance_evaluator,
            generation_groundedness_evaluator,
            overall_rag_evaluator,
        ]
        
        print(f"Running experiment with {len(evaluators)} evaluators...")
        
        experiment_results = client.evaluate(
            target_func,
            data=dataset_id,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            max_concurrency=2,
        )
        
        print(f"✅ Experiment complete!")
        print(f"View results: https://smith.langchain.com")
        
        return experiment_results
        
    except Exception as e:
        print(f"❌ Experiment failed: {str(e)}")
        return None


if __name__ == "__main__":
    print("🚀 LangSmith RAG Evaluation Pipeline")
    print("=" * 50)
    
    # Step 1: Create dataset
    print("\n1. Creating dataset from evaluations...")
    dataset_id = create_rag_dataset()
    
    if not dataset_id:
        print("Cannot proceed without dataset")
        sys.exit(1)
    
    print("\n2. Dataset ready!")
    print(f"   Dataset ID: {dataset_id}")
    print(f"   View at: https://smith.langchain.com/datasets/{dataset_id}")
    print("\n   To run experiments, provide a target function")
    print("   See documentation for example")