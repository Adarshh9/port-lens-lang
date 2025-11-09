"""
Groq LLM wrapper for unified API access.
Handles API calls to Groq models.
"""

import logging
import json
from typing import Optional, List, Dict, Any
from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from app.config import settings
from app.llm.prompts import (
    SYSTEM_PROMPT_RAG,
    SYSTEM_PROMPT_JUDGE,
    RAG_PROMPT_TEMPLATE,
    JUDGE_PROMPT_TEMPLATE,
)

logger = logging.getLogger("rag_llm_system")


class GroqLLM:
    """Wrapper for Groq LLM API."""

    def __init__(self, model: str = settings.groq_model):
        """
        Initialize Groq LLM.
        
        Args:
            model: Model name to use
        """
        logger.info(f"Initializing Groq LLM with model: {model}")
        try:
            self.client = Groq(api_key=settings.groq_api_key)
            self.model = model
            logger.info("Groq LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def generate(
        self,
        query: str,
        context: str = "",
        conversation_history: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate answer to query using RAG.
        
        Args:
            query: User query
            context: Retrieved context documents
            conversation_history: Previous conversation
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer
        """
        logger.info(f"Generating answer for query (length: {len(query)})")
        try:
            # Build the prompt
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context or "No context available",
                question=query
            )
            
            # Add conversation history if available
            if conversation_history:
                prompt = f"Previous conversation:\n{conversation_history}\n\n{prompt}"
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer (length: {len(answer)})")
            return answer

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def judge_answer(
        self,
        query: str,
        answer: str,
        context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate answer quality.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Retrieved context documents
            
        Returns:
            Evaluation dict with score and criteria
        """
        try:
            # Format context for judge
            context_text = "\n\n".join([
                doc.get("content", "")[:500]  # Limit context length
                for doc in context[:3]  # Use top 3 docs
            ]) if context else "No context available"
            
            # Build judge prompt
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=query,
                context=context_text,
                answer=answer
            )
            
            logger.info("Evaluating answer quality")
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for consistent evaluation
                max_tokens=500,
            )

            eval_text = response.choices[0].message.content.strip()
            
            # Clean up response - remove markdown code blocks if present
            if eval_text.startswith("```json"):
                eval_text = eval_text.split("```json")[1].split("```")[0].strip()
            elif eval_text.startswith("```"):
                eval_text = eval_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            try:
                evaluation = json.loads(eval_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse judge response: {eval_text[:200]}")
                # Return default evaluation
                evaluation = {
                    "score": 5.0,
                    "reasons": "Failed to parse evaluation response",
                    "criteria": {
                        "correctness": 5.0,
                        "relevance": 5.0,
                        "completeness": 5.0,
                        "clarity": 5.0,
                        "citations": 5.0
                    }
                }
            
            # Normalize score to 0-1 range (from 0-10)
            if "score" in evaluation:
                evaluation["score"] = evaluation["score"] / 10.0
            
            logger.info(f"Evaluation score: {evaluation.get('score', 0):.2f}")
            return evaluation

        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            # Return default evaluation on error
            return {
                "score": 0.5,
                "reasons": f"Evaluation error: {str(e)}",
                "criteria": {
                    "correctness": 5.0,
                    "relevance": 5.0,
                    "completeness": 5.0,
                    "clarity": 5.0,
                    "citations": 5.0
                }
            }

    def generate_with_context(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text with context documents (legacy method).
        
        Args:
            prompt: Input prompt
            context: List of context documents
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        logger.info("Generating text with context (legacy method)")
        try:
            context_str = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(context)])
            
            # Use the new generate method
            return self.generate(
                query=prompt,
                context=context_str,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        except Exception as e:
            logger.error(f"Context-based generation failed: {str(e)}")
            raise