import logging
import json
from typing import Optional, List, Dict, Any
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
from app.llm.prompts import (
    SYSTEM_PROMPT_RAG,
    SYSTEM_PROMPT_JUDGE,
    RAG_PROMPT_TEMPLATE,
    JUDGE_PROMPT_TEMPLATE,
)

logger = logging.getLogger("rag_llm_system")


class GroqLLM:
    """Groq LLM wrapper with robust error handling."""

    def __init__(self, model: str = settings.groq_model):
        """Initialize Groq client."""
        logger.info(f"Initializing Groq LLM: {model}")
        
        try:
            self.client = Groq(api_key=settings.groq_api_key)
            self.model = model
            logger.info("✅ Groq LLM initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(
        self,
        query: str,
        context: str = "",
        conversation_history: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate answer with retry logic."""
        logger.info(f"Generating answer for query (length: {len(query)})")
        
        try:
            # Build prompt
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context or "No context available",
                question=query
            )
            
            if conversation_history:
                prompt = f"Previous conversation:\n{conversation_history}\n\n{prompt}"
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user", "content": prompt}
            ]
            
            # Call Groq
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
            logger.error(f"Generation failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def judge_answer(
        self,
        query: str,
        answer: str,
        context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Judge answer quality with robust JSON parsing.
        Handles truncated, malformed, or incomplete responses.
        """
        try:
            # Format context
            context_text = "\n\n".join([
                doc.get("content", "")[:200]
                for doc in context[:3]
            ]) if context else "No context"
            
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
            
            # Call judge with longer timeout
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
            
            eval_text = response.choices[0].message.content.strip()
            
            # Step 1: Extract JSON from markdown code blocks
            if "```json" in eval_text:
                eval_text = eval_text.split("```json")[1].split("```")[0].strip()
            elif "```" in eval_text:
                eval_text = eval_text.split("```")[1].split("```")[0].strip()
            
            # Step 2: Try parsing as-is
            try:
                evaluation = json.loads(eval_text)
                logger.info(f"Successfully parsed judge response")
            except json.JSONDecodeError as e:
                # Step 3: Try fixing common JSON issues
                logger.warning(f"Initial JSON parse failed: {str(e)[:100]}")
                evaluation = self._repair_json(eval_text)
                
                if evaluation is None:
                    logger.error(f"Failed to repair JSON, using defaults")
                    evaluation = self._get_default_evaluation()
            
            # Step 4: Validate and normalize score
            if "score" in evaluation:
                score = evaluation["score"]
                
                # Handle string scores
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        score = 5.0
                
                # Normalize to 0-1
                if score > 10:
                    evaluation["score"] = score / 100.0
                elif score > 1:
                    evaluation["score"] = score / 10.0
                else:
                    evaluation["score"] = score
            else:
                evaluation["score"] = 0.5
            
            final_score = float(evaluation.get("score", 0.5))
            logger.info(f"Evaluation score: {final_score:.2f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Judge failed: {str(e)}")
            return self._get_default_evaluation()

    def _repair_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair malformed JSON.
        
        Strategies:
        1. Remove trailing incomplete keys
        2. Add missing closing braces
        3. Fix unquoted strings
        4. Fix single quotes
        """
        text = text.strip()
        
        # Strategy 1: Remove truncated final key
        if text.endswith('",'):
            text = text[:-1]
        
        # Strategy 2: Add missing closing braces
        open_braces = text.count('{') - text.count('}')
        if open_braces > 0:
            text += '}' * open_braces
        
        # Strategy 3: Replace single quotes with double quotes
        # (be careful not to break strings containing apostrophes)
        text = text.replace("'", '"')
        
        # Strategy 4: Try parsing again
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug(f"Could not repair JSON: {text[:100]}")
            return None

    def _get_default_evaluation(self) -> Dict[str, Any]:
        """Return safe default evaluation."""
        return {
            "score": 0.5,
            "reasons": "Evaluation failed - using default score",
            "criteria": {
                "correctness": 5,
                "relevance": 5,
                "completeness": 5,
                "clarity": 5,
                "citations": 5,
            }
        }