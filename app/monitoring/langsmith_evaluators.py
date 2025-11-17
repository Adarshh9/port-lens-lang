try:
    from langsmith import Client
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None


def retrieval_precision_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """
    Evaluate retrieval precision.
    
    Args:
        inputs: {"query": str, "ground_truth_docs": List[str]}
        outputs: {"retrieved_docs": List[Dict], ...}
        reference_outputs: Reference/ground truth
    
    Returns:
        {"key": "retrieval_precision", "score": float, ...}
    """
    retrieval_metrics = outputs.get("retrieval", {})
    precision = retrieval_metrics.get("precision")
    
    if precision is None:
        precision = retrieval_metrics.get("context_relevance", 0.5)
    
    return {
        "key": "retrieval_precision",
        "score": float(precision),
        "comment": f"Retrieval precision: {precision:.2f}"
    }


def retrieval_recall_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """Evaluate retrieval recall."""
    retrieval_metrics = outputs.get("retrieval", {})
    recall = retrieval_metrics.get("recall")
    
    if recall is None:
        recall = retrieval_metrics.get("hit_rate", 0.5)
    
    return {
        "key": "retrieval_recall",
        "score": float(recall),
        "comment": f"Retrieval recall: {recall:.2f}"
    }


def generation_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """Evaluate generation relevance."""
    generation_metrics = outputs.get("generation", {})
    relevance = generation_metrics.get("relevance_score", 0.5)
    
    return {
        "key": "generation_relevance",
        "score": float(relevance),
        "comment": f"Generation relevance: {relevance:.2f}"
    }


def generation_groundedness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """Evaluate generation groundedness (no hallucinations)."""
    generation_metrics = outputs.get("generation", {})
    groundedness = generation_metrics.get("groundedness_score", 0.5)
    
    return {
        "key": "generation_groundedness",
        "score": float(groundedness),
        "comment": f"Generation groundedness: {groundedness:.2f}"
    }


def overall_rag_evaluator(inputs: dict, outputs: dict, reference_outputs: dict = None) -> dict:
    """Evaluate overall RAG system performance."""
    overall_score = outputs.get("overall_score", 0.5)
    
    return {
        "key": "overall_rag_score",
        "score": float(overall_score),
        "comment": f"Overall RAG score: {overall_score:.2f}"
    }