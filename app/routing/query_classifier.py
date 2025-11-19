import re
from dataclasses import dataclass
import logging
from app.models.model_config import MultiModelConfig

logger = logging.getLogger("rag_llm_system")

@dataclass
class QueryClassification:
    complexity_score: float
    difficulty: str
    requires_reasoning: bool
    estimated_tokens_out: int
    domain: str
    preferred_model: str
    routing_reasoning: str

class QueryClassifier:
    def __init__(self, model_config: MultiModelConfig):
        self.model_config = model_config
        self.complexity_thresholds = model_config.get_complexity_thresholds()

    def classify(self, query: str) -> QueryClassification:
        query_lower = query.lower()
        
        # 1. Length Score
        word_count = len(query.split())
        length_score = min(word_count / 50, 0.3)

        # 2. Reasoning Score
        reasoning_keywords = ["why", "how", "explain", "reason", "analyze", "compare"]
        reasoning_score = 0.15 * sum(1 for kw in reasoning_keywords if kw in query_lower)
        reasoning_score = min(reasoning_score, 0.3)

        # 3. Technical Score
        technical_keywords = ["code", "debug", "function", "api", "error"]
        technical_score = 0.1 if any(kw in query_lower for kw in technical_keywords) else 0

        # 4. Domain & Complexity
        complexity_score = min(length_score + reasoning_score + technical_score, 1.0)

        # Determine Difficulty
        if complexity_score < self.complexity_thresholds.get("simple", 0.3):
            difficulty = "simple"
            preferred_model = "phi3_mini"
        elif complexity_score < self.complexity_thresholds.get("medium", 0.6):
            difficulty = "medium"
            preferred_model = "llama3_8b"
        else:
            difficulty = "complex"
            preferred_model = "gpt4o_mini" # or llama3_8b if gpt4 not avail

        return QueryClassification(
            complexity_score=complexity_score,
            difficulty=difficulty,
            requires_reasoning=reasoning_score > 0,
            estimated_tokens_out=max(100, word_count * 5),
            domain="technical" if technical_score > 0 else "general",
            preferred_model=preferred_model,
            routing_reasoning=f"Complexity: {complexity_score:.2f} ({difficulty})"
        )