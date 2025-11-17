"""
LangGraph graph construction - BULLETPROOF FINAL VERSION.
Ensures ALL nodes return dicts, never state objects.
"""

import logging
from typing import Union, Dict, Any, Callable
from langgraph.graph import StateGraph, END
from langsmith import traceable
from app.graph.state import RAGState
from app.vector.retriever import Retriever
from app.llm.groq_wrapper import GroqLLM
from app.cache.fs_cache import FilesystemCache
from app.cache.redis_cache import RedisCache
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.llm.prompts import FALLBACK_MESSAGE
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class RAGGraphBuilder:
    """Build and manage RAG orchestration graph."""

    def __init__(
        self,
        retriever: Retriever,
        llm: GroqLLM,
        cache: Union[FilesystemCache, RedisCache],
        short_term_memory: ShortTermMemory,
        long_term_memory: LongTermMemory,
    ):
        """
        Initialize graph builder.
        
        Args:
            retriever: Retriever instance
            llm: LLM instance
            cache: Cache instance
            short_term_memory: Short-term memory instance
            long_term_memory: Long-term memory instance
        """
        self.retriever = retriever
        self.llm = llm
        self.cache = cache
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.graph = None

    def _create_node_wrapper(
        self, node_func: Callable, node_name: str
    ) -> Callable:
        """
        Create a wrapper for a node that ensures it returns a dict.
        
        Args:
            node_func: The actual node function
            node_name: Name for logging
            
        Returns:
            Wrapped function that always returns dict
        """
        def wrapper(state: RAGState) -> Dict[str, Any]:
            try:
                logger.debug(f"Executing node: {node_name}")
                result = node_func(state)
                
                # CRITICAL: Ensure result is a dict
                if not isinstance(result, dict):
                    logger.error(f"❌ {node_name} returned {type(result)}, converting to dict")
                    return {}
                
                logger.debug(f"✅ {node_name} returned dict with keys: {list(result.keys())}")
                return result
                
            except Exception as e:
                logger.error(f"❌ {node_name} raised exception: {str(e)}")
                return {}
        
        return wrapper

    def _cache_node(self, state: RAGState) -> Dict[str, Any]:
        """Cache check node."""
        from app.graph.nodes import cache_node
        return cache_node(state, self.cache)

    def _retrieval_node(self, state: RAGState) -> Dict[str, Any]:
        """Document retrieval node."""
        from app.graph.nodes import retrieval_node
        return retrieval_node(state, self.retriever)

    def _llm_node(self, state: RAGState) -> Dict[str, Any]:
        """LLM generation node."""
        from app.graph.nodes import llm_node
        return llm_node(state, self.llm)

    def _judge_node(self, state: RAGState) -> Dict[str, Any]:
        """Quality judgment node."""
        from app.graph.nodes import judge_node
        return judge_node(state, self.llm)

    def _memory_node(self, state: RAGState) -> Dict[str, Any]:
        """Memory update node - NOW WITH CACHE."""
        from app.graph.nodes import memory_node
        # PASS CACHE to memory_node so it can cache the answer
        return memory_node(state, self.short_term_memory, self.long_term_memory, self.cache)

    def _fallback_node(self, state: RAGState) -> Dict[str, Any]:
        """Fallback response node."""
        from app.graph.nodes import fallback_node
        return fallback_node(state)

    def _route_after_judge(self, state: RAGState) -> str:
        """Route based on judge quality."""
        if state.get("quality_passed", False):
            return "memory_update"
        elif settings.judge_enable_fallback:
            return "fallback"
        else:
            return "memory_update"

    def build(self) -> StateGraph:
        """
        Build the RAG graph.
        Every node is wrapped to guarantee dict returns.
        """
        logger.info("Building RAG orchestration graph")

        try:
            # Create StateGraph with TypedDict state
            graph = StateGraph(RAGState)

            # Add nodes with wrappers to ensure dict returns
            graph.add_node(
                "cache_check",
                self._create_node_wrapper(self._cache_node, "cache_node")
            )
            graph.add_node(
                "retrieval",
                self._create_node_wrapper(self._retrieval_node, "retrieval_node")
            )
            graph.add_node(
                "llm_generation",
                self._create_node_wrapper(self._llm_node, "llm_node")
            )
            graph.add_node(
                "judge",
                self._create_node_wrapper(self._judge_node, "judge_node")
            )
            graph.add_node(
                "memory_update",
                self._create_node_wrapper(self._memory_node, "memory_node")
            )
            graph.add_node(
                "fallback",
                self._create_node_wrapper(self._fallback_node, "fallback_node")
            )

            # Add edges
            graph.add_edge("cache_check", "retrieval")
            graph.add_edge("retrieval", "llm_generation")
            graph.add_edge("llm_generation", "judge")

            # Conditional routing after judge
            graph.add_conditional_edges(
                "judge",
                self._route_after_judge,
                {
                    "memory_update": "memory_update",
                    "fallback": "fallback",
                },
            )

            # Fallback to memory
            graph.add_edge("fallback", "memory_update")

            # Memory to end
            graph.add_edge("memory_update", END)

            # Set entry point
            graph.set_entry_point("cache_check")

            # Compile
            self.graph = graph.compile()
            logger.info("✅ Graph compiled successfully")

            return self.graph

        except Exception as e:
            logger.error(f"Graph build failed: {str(e)}", exc_info=True)
            raise
        
    @traceable(run_type="chain", name="rag_pipeline")
    def invoke(self, state: RAGState) -> Dict[str, Any]:
        """
        Execute the graph.
        
        Args:
            state: Initial RAG state (TypedDict)
            
        Returns:
            Final state dict from graph execution
        """
        logger.info("Executing RAG graph")
        
        if not self.graph:
            logger.warning("Graph not built, building now...")
            self.build()

        try:
            # Invoke graph - StateGraph handles state merging with dicts
            result = self.graph.invoke(state)
            
            logger.info(f"✅ Graph execution succeeded")
            logger.debug(f"Result keys: {list(result.keys())}")
            
            return result

        except Exception as e:
            logger.error(f"❌ Graph execution failed: {str(e)}", exc_info=True)
            
            # Return safe fallback state
            fallback_state = dict(state)  # Copy state as dict
            fallback_state["final_answer"] = FALLBACK_MESSAGE
            fallback_state["used_fallback"] = True
            if "errors" not in fallback_state:
                fallback_state["errors"] = []
            fallback_state["errors"].append(f"Graph error: {str(e)}")
            
            logger.info("Graph execution completed with fallback")
            return fallback_state
