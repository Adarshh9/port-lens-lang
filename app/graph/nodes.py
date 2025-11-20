import logging
import time
from typing import Dict, Any
from app.vector.retriever import Retriever
from app.llm.groq_wrapper import GroqLLM
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.monitoring.rag_evaluators import RAGEvaluator

logger = logging.getLogger("rag_llm_system")

evaluator = RAGEvaluator()


def cache_node(state: Dict[str, Any], cache) -> Dict[str, Any]:
    """Check cache for existing answer. RETURNS DICT."""
    query = state.get("query", "")
    session_id = state.get("session_id", "")
    user_id = state.get("user_id", "")
    
    logger.info(f"Checking cache for query: {query[:50]}")
    
    try:
        cached_answer = cache.get(
            query=query,
            session_id=session_id,
            user_id=user_id
        )
        
        if cached_answer:
            logger.info("✅ Cache HIT - returning cached answer")
            return {
                "cache_hit": True,
                "cached_answer": cached_answer,
                "final_answer": cached_answer
            }
        
        logger.info("❌ Cache MISS - continuing to retrieval")
        return {"cache_hit": False}
        
    except Exception as e:
        logger.error(f"Cache check failed: {str(e)}")
        return {
            "cache_hit": False,
            "errors": state.get("errors", []) + [f"Cache error: {str(e)}"]
        }


def retrieval_node(state: Dict[str, Any], retriever: Retriever) -> Dict[str, Any]:
    """Retrieve documents. RETURNS DICT."""
    if state.get("cache_hit"):
        logger.info("Skipping retrieval due to cache hit")
        return {}
    
    query = state.get("query", "")
    logger.info(f"Retrieving documents for query: {query[:50]}")
    start_time = time.time()
    
    try:
        # Retrieve top 2 docs (optimized for performance)
        docs = retriever.retrieve(query, k=2)
        
        retrieved_docs = []
        for doc in docs:
            retrieved_docs.append({
                "content": doc.get("content", "") if isinstance(doc, dict) else str(doc),
                "metadata": doc.get("metadata", {}) if isinstance(doc, dict) else {},
                "distance": doc.get("distance", 0.0) if isinstance(doc, dict) else 0.0
            })
        
        retrieval_time = time.time() - start_time
        logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")
        
        return {
            "retrieved_docs": retrieved_docs,
            "retrieval_metadata": {
                "num_docs": len(retrieved_docs),
                "retrieval_time_ms": retrieval_time * 1000,
            }
        }
        
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return {
            "retrieved_docs": [],
            "retrieval_metadata": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Retrieval error: {str(e)}"]
        }


def llm_node(state: Dict[str, Any], llm: GroqLLM) -> Dict[str, Any]:
    """Generate answer with optimized context handling. RETURNS DICT."""
    if state.get("cache_hit"):
        logger.info("Skipping LLM generation due to cache hit")
        return {}
    
    logger.info("Generating answer with LLM")
    start_time = time.time()
    
    try:
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_docs", [])
        conversation_history = state.get("conversation_history", [])
        
        # Build context with doc numbers (improved formatting)
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")[:300] if isinstance(doc, dict) else str(doc)[:300]
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            
            if content:
                doc_header = f"Document {i}:"
                if metadata.get("source"):
                    doc_header += f" (Source: {metadata['source']})"
                
                context_parts.append(f"{doc_header}\n{content}")
        
        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
        
        # Build history (last 4 messages)
        history_text = ""
        if conversation_history:
            history_items = []
            for msg in conversation_history[-4:]:
                role = msg.get("role", "") if isinstance(msg, dict) else ""
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if role and content:
                    history_items.append(f"{role.upper()}: {content[:100]}")
            history_text = "\n".join(history_items)
        
        # Generate answer (optimized: max_tokens=1024)
        answer = llm.generate(
            query=query,
            context=context,
            conversation_history=history_text,
            max_tokens=1024
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Generated answer ({len(answer)} chars) in {generation_time:.2f}s")
        
        return {
            "generated_answer": answer,
            "generation_metadata": {
                "context_length": len(context),
                "generation_time_ms": generation_time * 1000,
                "num_context_docs": len(retrieved_docs),
                "answer_length": len(answer),
                "has_history": bool(conversation_history)
            }
        }
        
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}")
        return {
            "generated_answer": "",
            "generation_metadata": {"error": str(e)},
            "errors": state.get("errors", []) + [f"Generation error: {str(e)}"]
        }


def judge_node(state: Dict[str, Any], llm: GroqLLM) -> Dict[str, Any]:
    """Evaluate answer quality. RETURNS DICT."""
    logger.info("Evaluating answer quality")
    
    if state.get("cache_hit"):
        logger.info("Cache hit - skipping quality evaluation")
        return {
            "quality_passed": True,
            "judge_score": 1.0,
            "judge_evaluation": {
                "score": 1.0,
                "reasons": "Cached answer",
                "criteria": {"cached": True}
            },
            "final_answer": state.get("cached_answer", "")
        }
    
    generated_answer = state.get("generated_answer", "")
    
    if not generated_answer:
        logger.warning("No answer generated - quality check failed")
        return {
            "quality_passed": False,
            "judge_score": 0.0,
            "judge_evaluation": {
                "score": 0.0,
                "reasons": "No answer generated",
                "criteria": {}
            },
            "final_answer": ""
        }
    
    try:
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_docs", [])
        
        evaluation = llm.judge_answer(
            query=query,
            answer=generated_answer,
            context=retrieved_docs
        )
        
        score = float(evaluation.get("score", 0.0))
        from app.config import settings
        threshold = settings.judge_quality_threshold
        passed = score >= threshold
        
        logger.info(f"Answer quality score: {score:.2f} (threshold: {threshold}) - {'PASSED' if passed else 'FAILED'}")
        
        return {
            "judge_score": score,
            "judge_evaluation": evaluation,
            "quality_passed": passed,
            "final_answer": generated_answer if passed else ""
        }
        
    except Exception as e:
        logger.error(f"Quality evaluation failed: {str(e)}")
        # Fallback: accept answer with warning
        return {
            "judge_score": 0.5,
            "judge_evaluation": {
                "score": 0.5,
                "reasons": f"Evaluation error: {str(e)}",
                "criteria": {}
            },
            "quality_passed": True,
            "final_answer": generated_answer,
            "errors": state.get("errors", []) + [f"Judge error: {str(e)}"]
        }


def memory_node(
    state: Dict[str, Any],
    short_term_memory: ShortTermMemory,
    long_term_memory: LongTermMemory,
    cache
) -> Dict[str, Any]:
    """
    Update SHORT-TERM and LONG-TERM memory, cache results, and log evaluation.
    RETURNS DICT.
    """
    logger.info("Updating memory, cache, and logging evaluation")
    
    try:
        query = state.get("query", "")
        final_answer = state.get("final_answer", "")
        session_id = state.get("session_id", "")
        user_id = state.get("user_id", "")
        quality_passed = state.get("quality_passed", False)
        used_fallback = state.get("used_fallback", False)
        judge_score = state.get("judge_score", 0.0)
        
        if not final_answer:
            logger.warning("No final answer to save in memory")
            return {}
        
        # ================================================================
        # SHORT-TERM MEMORY (session-based, temporary)
        # ================================================================
        short_term_memory.add_message(
            role="user",
            content=query,
            metadata={"session_id": session_id, "timestamp": time.time()}
        )
        
        short_term_memory.add_message(
            role="assistant",
            content=final_answer,
            metadata={
                "session_id": session_id,
                "timestamp": time.time(),
                "judge_score": judge_score,
                "quality_passed": quality_passed
            }
        )
        
        # Session-specific history
        if session_id:
            short_term_memory.add_message_for_session(
                session_id=session_id,
                role="user",
                content=query
            )
            short_term_memory.add_message_for_session(
                session_id=session_id,
                role="assistant",
                content=final_answer
            )
        
        # ================================================================
        # LONG-TERM MEMORY (persistent, across sessions)
        # ================================================================
        if long_term_memory and quality_passed and not used_fallback:
            try:
                # Store high-quality Q&A pairs for future retrieval/training
                long_term_memory.add_qa_pair(
                    query=query,
                    answer=final_answer,
                    metadata={
                        "user_id": user_id,
                        "session_id": session_id,
                        "judge_score": judge_score,
                        "timestamp": time.time(),
                        "num_docs_retrieved": len(state.get("retrieved_docs", []))
                    }
                )
                logger.info("✅ Q&A pair stored in long-term memory")
            except Exception as ltm_err:
                logger.warning(f"Failed to store in long-term memory: {ltm_err}")
        
        # ================================================================
        # CACHE (for fast retrieval of repeated queries)
        # ================================================================
        if cache and quality_passed and not used_fallback:
            try:
                cache.set(
                    query=query,
                    answer=final_answer,
                    session_id=session_id,
                    user_id=user_id,
                    metadata={"judge_score": judge_score, "timestamp": time.time()}
                )
                logger.info(f"✅ Cached answer for query: {query[:50]}")
            except Exception as cache_err:
                logger.warning(f"Failed to cache answer: {cache_err}")
        
        # ================================================================
        # COMPREHENSIVE EVALUATION LOGGING (LLM-based metrics)
        # ================================================================
        try:        
            # Compute processing time
            generation_time = state.get("generation_metadata", {}).get("generation_time_ms", 0)
            retrieval_time = state.get("retrieval_metadata", {}).get("retrieval_time_ms", 0)
            total_time_ms = generation_time + retrieval_time
            
            # Estimate cost (rough calculation)
            answer_length = len(final_answer)
            query_length = len(query)
            estimated_tokens = (query_length + answer_length) / 4  # ~4 chars per token
            cost_per_1k = 0.02  # Groq Llama pricing
            estimated_cost = (estimated_tokens / 1000) * cost_per_1k
            
            eval_result = evaluator.evaluate_rag_response(
                query=query,
                retrieved_docs=state.get("retrieved_docs", []),
                answer=final_answer,
                judge_evaluation=state.get("judge_evaluation", {}),  # REUSE!
                latency_ms=total_time_ms,
                cost_usd=estimated_cost,
                session_id=session_id,
                user_id=user_id,
            )
            
            logger.info(
                f"✅ Evaluation complete: "
                f"overall={eval_result.get('overall_score', 0):.2f}, "
                f"retrieval={eval_result.get('retrieval', {}).get('context_relevance', 0):.2f}, "
                f"generation={eval_result.get('generation', {}).get('avg_generation_score', 0):.2f}"
            )
        except Exception as eval_err:
            logger.warning(f"Evaluation logging failed: {str(eval_err)}")
        
        # ================================================================
        # RETURN UPDATED CONVERSATION HISTORY
        # ================================================================
        updated_history = (
            short_term_memory.get_history(session_id)
            if session_id
            else short_term_memory.get_messages()
        )
        
        logger.info(f"Memory updated - history length: {len(updated_history)}")
        
        return {"conversation_history": updated_history}
        
    except Exception as e:
        logger.error(f"Memory update failed: {str(e)}")
        return {"errors": state.get("errors", []) + [f"Memory error: {str(e)}"]}


def fallback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback response when quality is too low. RETURNS DICT."""
    judge_score = state.get("judge_score", 0.0)
    logger.warning(f"Fallback triggered (score: {judge_score:.2f})")
    
    from app.llm.prompts import FALLBACK_MESSAGE
    
    return {
        "used_fallback": True,
        "final_answer": FALLBACK_MESSAGE
    }