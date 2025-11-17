# tests/test_rag_evaluators.py

import pytest
from app.monitoring.rag_evaluators import RAGEvaluator, RetrievalEvaluator, GenerationEvaluator
from app.llm.groq_wrapper import GroqLLM


class TestRAGEvaluation:
    """Test RAG evaluation system."""
    
    @pytest.fixture
    def judge_llm(self):
        """Create judge LLM."""
        return GroqLLM()
    
    @pytest.fixture
    def evaluator(self, judge_llm):
        """Create evaluator."""
        return RAGEvaluator(judge_llm)
    
    def test_retrieval_evaluation(self, judge_llm):
        """Test retrieval evaluation."""
        retriever = RetrievalEvaluator(judge_llm)
        
        result = retriever.evaluate_retrieval(
            query="What is AI?",
            retrieved_docs=[
                {"content": "AI is artificial intelligence"},
                {"content": "ML is machine learning"},
            ]
        )
        
        assert "context_relevance" in result
        assert "num_retrieved" in result
        assert result["num_retrieved"] == 2
    
    def test_generation_evaluation(self, judge_llm):
        """Test generation evaluation."""
        generator = GenerationEvaluator(judge_llm)
        
        result = generator.evaluate_generation(
            query="What is AI?",
            answer="AI is artificial intelligence, the simulation of human intelligence.",
            retrieved_docs=[
                {"content": "AI is artificial intelligence"}
            ]
        )
        
        assert "relevance" in result
        assert "groundedness" in result
        assert "completeness" in result
        assert 0 <= result["generation"]["avg_generation_score"] <= 1
    
    def test_rag_evaluation_complete(self, evaluator):
        """Test complete RAG evaluation."""
        result = evaluator.evaluate_rag_response(
            query="What is AI?",
            retrieved_docs=[
                {"content": "AI is artificial intelligence"}
            ],
            answer="AI is artificial intelligence.",
            latency_ms=2000,
            cost_usd=0.015,
            session_id="test",
            user_id="test_user"
        )
        
        assert "retrieval" in result
        assert "generation" in result
        assert "system" in result
        assert "overall_score" in result
        assert 0 <= result["overall_score"] <= 1
