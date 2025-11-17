import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger("rag_llm_system")


class RetrievalEvaluator:
    """
    Comprehensive retrieval evaluation with:
    - Context Relevance (from distance scores)
    - Precision (relevant docs / total retrieved)
    - Recall (relevant docs / total relevant)
    - Mean Reciprocal Rank (MRR)
    - NDCG (Normalized Discounted Cumulative Gain)
    - Hit Rate (at least 1 relevant doc retrieved)
    """
    
    def __init__(self):
        """Initialize retrieval evaluator."""
        pass
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth_docs: Optional[List[str]] = None,
        relevance_labels: Optional[List[float]] = None,  # 0-1 scores per doc
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval comprehensively.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents with distance scores
            ground_truth_docs: (Optional) Known relevant documents
            relevance_labels: (Optional) Manual relevance scores 0-1 per doc
        
        Returns:
            Dict with all retrieval metrics
        """
        metrics = {
            "num_retrieved": len(retrieved_docs),
            "timestamp": datetime.now().isoformat(),
        }
        
        if not retrieved_docs:
            # No docs retrieved
            return {
                **metrics,
                "avg_distance": 1.0,
                "context_relevance": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mrr": 0.0,
                "ndcg": 0.0,
                "hit_rate": 0.0,
            }
        
        # ================================================================
        # Metric 1: Context Relevance (from distance)
        # ================================================================
        distances = [doc.get("distance", 1.0) for doc in retrieved_docs]
        avg_distance = np.mean(distances)
        context_relevance = float(1.0 - avg_distance)
        
        metrics["avg_distance"] = round(float(avg_distance), 4)
        metrics["context_relevance"] = round(context_relevance, 4)
        
        # ================================================================
        # Metric 2: Hit Rate (at least 1 relevant doc?)
        # ================================================================
        # Use distance as proxy: distance < 0.5 = relevant
        hit_rate = 1.0 if min(distances) < 0.5 else 0.0
        metrics["hit_rate"] = float(hit_rate)
        
        # ================================================================
        # Metrics 3-6: Precision, Recall, MRR, NDCG
        # (If ground truth or manual labels provided)
        # ================================================================
        
        if relevance_labels:
            # Use provided relevance labels
            metrics.update(self._compute_metrics_from_labels(
                relevance_labels,
                ground_truth_docs
            ))
        elif ground_truth_docs:
            # Infer relevance from ground truth docs
            metrics.update(self._compute_metrics_from_ground_truth(
                retrieved_docs,
                ground_truth_docs
            ))
        else:
            # No ground truth - use heuristics
            metrics.update({
                "precision": None,  # Can't compute without ground truth
                "recall": None,
                "mrr": self._compute_mrr_from_distance(distances),
                "ndcg": self._compute_ndcg_from_distance(distances),
            })
        
        return metrics
    
    def _compute_metrics_from_labels(
        self,
        relevance_labels: List[float],
        ground_truth_docs: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Compute metrics from manual relevance labels (0-1)."""
        # Consider relevant if label > 0.5
        relevant_mask = [label > 0.5 for label in relevance_labels]
        num_relevant = sum(relevant_mask)
        
        # Precision: relevant / total retrieved
        precision = num_relevant / len(relevance_labels) if relevance_labels else 0.0
        
        # Recall: only if ground truth known
        total_relevant_docs = len(ground_truth_docs) if ground_truth_docs else num_relevant
        recall = num_relevant / total_relevant_docs if total_relevant_docs > 0 else 0.0
        
        # MRR: 1 / rank of first relevant
        mrr = 0.0
        for rank, is_relevant in enumerate(relevant_mask, 1):
            if is_relevant:
                mrr = 1.0 / rank
                break
        
        # NDCG: normalized DCG
        ndcg = self._compute_ndcg_from_relevance(relevance_labels)
        
        return {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "mrr": round(float(mrr), 4),
            "ndcg": round(float(ndcg), 4),
        }
    
    def _compute_metrics_from_ground_truth(
        self,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth_docs: List[str]
    ) -> Dict[str, Any]:
        """Compute metrics by comparing retrieved docs to ground truth."""
        # Simple string matching
        retrieved_content = [
            doc.get("content", "").lower()
            for doc in retrieved_docs
        ]
        
        ground_truth_lower = [
            doc.lower() for doc in ground_truth_docs
        ]
        
        # Find matches
        matches = []
        for retrieved in retrieved_content:
            for gt in ground_truth_lower:
                if gt in retrieved or retrieved in gt:
                    matches.append(True)
                    break
            else:
                matches.append(False)
        
        num_relevant = sum(matches)
        
        # Precision: relevant / total retrieved
        precision = num_relevant / len(retrieved_docs) if retrieved_docs else 0.0
        
        # Recall: relevant / total relevant
        recall = num_relevant / len(ground_truth_docs) if ground_truth_docs else 0.0
        
        # MRR
        mrr = 0.0
        for rank, is_relevant in enumerate(matches, 1):
            if is_relevant:
                mrr = 1.0 / rank
                break
        
        # NDCG: use relevance as binary (0 or 1)
        ndcg = self._compute_ndcg_from_relevance(
            [float(m) for m in matches]
        )
        
        return {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "mrr": round(float(mrr), 4),
            "ndcg": round(float(ndcg), 4),
        }
    
    def _compute_mrr_from_distance(self, distances: List[float]) -> float:
        """Compute MRR from distance scores."""
        for rank, distance in enumerate(distances, 1):
            if distance < 0.5:  # Relevant
                return 1.0 / rank
        return 0.0
    
    def _compute_ndcg_from_distance(
        self,
        distances: List[float],
        k: int = 10
    ) -> float:
        """Compute NDCG from distance scores."""
        # Convert distance to relevance (0-1)
        relevances = [max(0, 1.0 - d) for d in distances[:k]]
        
        return self._compute_ndcg_from_relevance(relevances)
    
    def _compute_ndcg_from_relevance(
        self,
        relevances: List[float],
        k: int = 10
    ) -> float:
        """
        Compute NDCG@k.
        
        NDCG = DCG / IDCG
        DCG = sum(relevance_i / log2(rank_i + 1))
        IDCG = ideal DCG (if all docs ranked perfectly)
        """
        # Actual DCG
        dcg = 0.0
        for rank, rel in enumerate(relevances[:k], 1):
            dcg += rel / np.log2(rank + 1)
        
        # Ideal DCG (perfect ranking)
        ideal_relevances = sorted(relevances, reverse=True)[:k]
        idcg = 0.0
        for rank, rel in enumerate(ideal_relevances, 1):
            idcg += rel / np.log2(rank + 1)
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return float(ndcg)


class GenerationEvaluator:
    """
    Generation evaluation (reuses judge_evaluation from GroqLLM).
    Already includes all necessary metrics from LLM-as-judge.
    """
    
    def evaluate_generation_from_judge(
        self,
        judge_evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract all metrics from judge evaluation."""
        if not judge_evaluation:
            judge_evaluation = {"score": 0.5, "reasons": "", "criteria": {}}
        
        score = float(judge_evaluation.get("score", 0.5))
        criteria = judge_evaluation.get("criteria", {})
        
        # Extract individual criteria
        metrics = {
            "judge_score": score,
            "relevance_score": float(criteria.get("relevance", 5)) / 10.0,
            "groundedness_score": float(criteria.get("correctness", 5)) / 10.0,
            "completeness_score": float(criteria.get("completeness", 5)) / 10.0,
            "clarity_score": float(criteria.get("clarity", 5)) / 10.0,
            "citations_score": float(criteria.get("citations", 5)) / 10.0,
            "avg_generation_score": score,
            "explanation": judge_evaluation.get("reasons", ""),
        }
        
        return metrics


class RAGEvaluator:
    """
    COMPLETE RAG evaluation system.
    
    Logs all metrics:
    Retrieval: precision, recall, mrr, ndcg, hit_rate, context_relevance
    Generation: relevance, groundedness, completeness, clarity, citations
    System: latency, cost
    """
    
    def __init__(self, metrics_file: str = "./logs/rag_evaluations.jsonl"):
        """Initialize evaluator."""
        self.retrieval_eval = RetrievalEvaluator()
        self.generation_eval = GenerationEvaluator()
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Complete RAG evaluator initialized: {self.metrics_file}")
    
    def evaluate_rag_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        answer: str,
        judge_evaluation: Dict[str, Any],
        latency_ms: float,
        cost_usd: float,
        session_id: str = "",
        user_id: str = "",
        ground_truth_docs: Optional[List[str]] = None,
        relevance_labels: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate RAG response comprehensively.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents
            answer: Generated answer
            judge_evaluation: LLM judge output
            latency_ms: End-to-end latency
            cost_usd: API cost
            session_id: Session ID
            user_id: User ID
            ground_truth_docs: (Optional) Ground truth documents
            relevance_labels: (Optional) Manual relevance labels per doc (0-1)
        
        Returns:
            Complete evaluation dict with all metrics
        """
        evaluation = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:200],
            "session_id": session_id,
            "user_id": user_id,
            
            # ================================================================
            # RETRIEVAL METRICS
            # ================================================================
            "retrieval": self.retrieval_eval.evaluate_retrieval(
                query=query,
                retrieved_docs=retrieved_docs,
                ground_truth_docs=ground_truth_docs,
                relevance_labels=relevance_labels,
            ),
            
            # ================================================================
            # GENERATION METRICS (from LLM-as-judge)
            # ================================================================
            "generation": self.generation_eval.evaluate_generation_from_judge(
                judge_evaluation
            ),
            
            # ================================================================
            # SYSTEM METRICS
            # ================================================================
            "system": {
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
            },
            
            # ================================================================
            # METADATA
            # ================================================================
            "metadata": {
                "has_ground_truth": ground_truth_docs is not None,
                "has_relevance_labels": relevance_labels is not None,
                "num_docs_retrieved": len(retrieved_docs),
                "answer_length": len(answer),
            }
        }
        
        # Compute overall score
        retrieval_score = evaluation["retrieval"].get("context_relevance", 0.5)
        generation_score = evaluation["generation"].get("avg_generation_score", 0.5)
        evaluation["overall_score"] = (retrieval_score * 0.3) + (generation_score * 0.7)
        
        # Log to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(evaluation) + "\n")
        
        logger.info(
            f"✅ Evaluation logged: "
            f"retrieval={retrieval_score:.2f}, "
            f"generation={generation_score:.2f}, "
            f"overall={evaluation['overall_score']:.2f}"
        )
        
        return evaluation
    
    def get_evaluation_summary(self, last_n: int = 100) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics_file.exists():
            return {}
        
        evaluations = []
        with open(self.metrics_file, "r") as f:
            for line in f:
                try:
                    evaluations.append(json.loads(line))
                except:
                    pass
        
        if not evaluations:
            return {}
        
        recent = evaluations[-last_n:]
        
        # Extract metrics that have values
        retrieval_metrics = {}
        generation_metrics = {}
        
        for ret_key in ["precision", "recall", "mrr", "ndcg", "hit_rate", "context_relevance"]:
            values = [
                e.get("retrieval", {}).get(ret_key)
                for e in recent
                if e.get("retrieval", {}).get(ret_key) is not None
            ]
            if values:
                retrieval_metrics[ret_key] = round(np.mean(values), 4)
        
        for gen_key in ["relevance_score", "groundedness_score", "completeness_score", "clarity_score", "citations_score"]:
            values = [
                e.get("generation", {}).get(gen_key)
                for e in recent
            ]
            if values:
                generation_metrics[gen_key] = round(np.mean(values), 4)
        
        return {
            "total_evaluations": len(recent),
            "avg_overall_score": round(np.mean([e.get("overall_score", 0.5) for e in recent]), 4),
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "avg_latency_ms": round(np.mean([e.get("system", {}).get("latency_ms", 0) for e in recent]), 2),
            "total_cost_usd": round(sum([e.get("system", {}).get("cost_usd", 0) for e in recent]), 6),
        }