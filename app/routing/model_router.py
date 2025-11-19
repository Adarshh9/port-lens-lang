import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List
from app.models.model_config import MultiModelConfig
from app.routing.query_classifier import QueryClassifier, QueryClassification
from app.llm.multi_provider import MultiProviderLLM

logger = logging.getLogger("rag_llm_system")

@dataclass
class RoutingResult:
    answer: str
    model_used: str
    judge_score: float
    classification: QueryClassification
    latency_ms: float
    cost_usd: float
    attempts: int
    fallback_used: bool
    routing_metadata: Dict[str, Any]

class CostAwareRouter:
    def __init__(self, model_config: MultiModelConfig):
        self.model_config = model_config
        self.classifier = QueryClassifier(model_config)
        self.multi_llm = MultiProviderLLM(model_config)
        self.min_quality_score = model_config.routing.get("min_quality_score", 0.75)

    async def route_and_generate(self, query: str, context: str, optimize_for: str = "balanced", user_id: str = "") -> RoutingResult:
        start_time = time.time()
        classification = self.classifier.classify(query)
        
        # Determine initial model
        if optimize_for == "cost":
            model_name = "phi3_mini"
        elif optimize_for == "speed":
            model_name = "phi3_mini" if classification.difficulty == "simple" else "llama3_8b"
        elif optimize_for == "quality":
            model_name = "gpt4o_mini" if "gpt4o_mini" in self.multi_llm.providers else "llama3_8b"
        else: # balanced
            model_name = classification.preferred_model

        # Fallback chain
        chain = [model_name] + [m for m in self.model_config.get_fallback_chain() if m != model_name]
        
        # Try chain
        for attempt_idx, model in enumerate(chain, 1):
            if model not in self.multi_llm.providers:
                logger.warning(f"Skipping {model}: Provider not initialized")
                continue
                
            try:
                logger.info(f"Routing: Trying {model} (Attempt {attempt_idx})")
                prompt = f"Context: {context[:2000]}\n\nQuestion: {query}"
                result = await self.multi_llm.generate(model, prompt)
                
                # Quality check logic
                # Note: If result['answer'] is an error message, len might be > 50, so we check score
                score = 0.85 if len(result["answer"]) > 50 else 0.4
                
                if score >= self.min_quality_score:
                    return RoutingResult(
                        answer=result["answer"],
                        model_used=model,
                        judge_score=score,
                        classification=classification,
                        latency_ms=(time.time() - start_time) * 1000,
                        cost_usd=result["cost_usd"],
                        attempts=attempt_idx,
                        fallback_used=(attempt_idx > 1),
                        routing_metadata={"optimize_for": optimize_for}
                    )
                else:
                    logger.warning(f"Model {model} quality too low ({score} < {self.min_quality_score}). triggering fallback.")
                    
            except Exception as e:
                # Truncate massive HTML errors from logs
                error_msg = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
                logger.error(f"Model {model} failed: {error_msg}")
                continue
                
        # If we get here, everything failed
        raise Exception("All routing models failed. Please check Groq API status and Ollama connection.")