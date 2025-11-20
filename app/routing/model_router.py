import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from langsmith import traceable

from app.models.model_config import MultiModelConfig
from app.routing.query_classifier import QueryClassifier, QueryClassification
from app.llm.multi_provider import MultiProviderLLM
from app.llm.groq_wrapper import GroqLLM

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
        self.judge_llm = GroqLLM()
        
        # Routing settings
        self.routing_config = model_config.routing
        self.min_quality_score = self.routing_config.get("min_quality_score", 0.75)
        self.fallback_chain = model_config.get_fallback_chain()

    @traceable(run_type="chain", name="smart_router_orchestrator")
    async def route_and_generate(
        self,
        query: str,
        context: str,
        optimize_for: str = "balanced",
        user_id: str = "",
    ) -> RoutingResult:
        """Classify query → Select model → Generate → Judge → Fallback."""
        start_time = time.time()
        
        classification = self.classifier.classify(query)
        model_to_try = self._select_initial_model(classification, optimize_for)
        attempt_chain = self._build_attempt_chain(model_to_try, optimize_for)

        REFUSAL_PHRASE = "I cannot find this information in Adarsh's portfolio documents"
        for attempt_idx, model_name in enumerate(attempt_chain, 1):
            try:
                # Generate answer
                result = await self._generate_with_model(
                    model_name, query, context
                )
                answer_text = result["answer"]
                # --- NEW: Check for Valid Refusal ---
                if REFUSAL_PHRASE in answer_text:
                    logger.info(f"Model {model_name} correctly refused to answer due to missing context.")
                    total_latency = (time.time() - start_time) * 1000
                    
                    return RoutingResult(
                        answer=answer_text,
                        model_used=model_name,
                        judge_score=1.0,  # Give it a perfect score for following rules
                        classification=classification,
                        latency_ms=total_latency,
                        cost_usd=result["cost_usd"],
                        attempts=attempt_idx,
                        fallback_used=attempt_idx > 1,
                        routing_metadata={
                            "optimize_for": optimize_for,
                            "status": "valid_refusal"
                        }
                    )
                
                # Judge quality
                judge_score = await self._judge_quality(
                    query, result["answer"], context
                )

                if judge_score >= self.min_quality_score:
                    total_latency = (time.time() - start_time) * 1000
                    return RoutingResult(
                        answer=result["answer"],
                        model_used=model_name,
                        judge_score=judge_score,
                        classification=classification,
                        latency_ms=total_latency,
                        cost_usd=result["cost_usd"],
                        attempts=attempt_idx,
                        fallback_used=attempt_idx > 1,
                        routing_metadata={
                            "optimize_for": optimize_for,
                            "model_latency_ms": result["latency_ms"],
                            "input_tokens": result["input_tokens"],
                            "output_tokens": result["output_tokens"],
                        }
                    )
                
                logger.warning(f"Quality too low ({judge_score:.2f}), trying fallback")

            except Exception as e:
                logger.error(f"Model {model_name} failed: {str(e)}")
                continue

        raise Exception(f"All models failed or scored below {self.min_quality_score}")

    @traceable(run_type="tool", name="model_selection")
    def _select_initial_model(self, classification: QueryClassification, optimize_for: str) -> str:
        if optimize_for == "cost": return "phi3_mini"
        elif optimize_for == "speed": return "phi3_mini" if classification.complexity_score < 0.3 else "llama3_8b"
        elif optimize_for == "quality": return "gpt4o_mini"
        return classification.preferred_model

    def _build_attempt_chain(self, initial_model: str, optimize_for: str) -> List[str]:
        if optimize_for == "quality": return [initial_model]
        chain = [initial_model]
        for model in self.fallback_chain:
            if model not in chain: chain.append(model)
        return chain

    @traceable(run_type="chain", name="execute_model_attempt")
    async def _generate_with_model(
        self,
        model_name: str,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """Generate answer with strict portfolio-only constraints."""
        
        # STRICT SYSTEM PROMPT
        system_instruction = """You are a specialized assistant for Adarsh Kesharwani's portfolio.
STRICT RULES:
1. Use ONLY the provided Context below to answer the question.
2. If the answer is NOT in the Context, you MUST say: "I cannot find this information in Adarsh's portfolio documents."
3. Do NOT use outside knowledge. Do NOT answer general questions (like "what is the capital of France").
4. Keep the answer professional and relevant to the provided documents.
"""

        # Combine into the prompt string (since MultiProviderLLM takes a raw string)
        final_prompt = f"""{system_instruction}

Context:
{context[:3000]}

Question: {query}

Answer:"""

        return await self.multi_llm.generate(
            model_name=model_name,
            prompt=final_prompt,
            max_tokens=1024,
            temperature=0.3 
        )

    @traceable(run_type="llm", name="quality_judge")
    async def _judge_quality(
        self,
        query: str,
        answer: str,
        context: str
    ) -> float:
        """Judge answer quality."""
        # Use existing judge from GroqLLM
        evaluation = await asyncio.to_thread(
            self.judge_llm.judge_answer,
            query=query,
            answer=answer,
            context=[{"content": context}]
        )
        return float(evaluation.get("score", 0.5))