import logging
from app.vector.store import VectorStore
from app.vector.retriever import Retriever
from app.ingestion.embedder import EmbeddingGenerator
from app.ingestion.indexer import DocumentIndexer
from app.ingestion.splitter import TextSplitter
from app.llm.groq_wrapper import GroqLLM
from app.cache.fs_cache import FilesystemCache
from app.cache.redis_cache import RedisCache
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.graph.graph_builder import RAGGraphBuilder
from app.config import settings

# NEW IMPORTS for Routing
from app.models.model_config import MultiModelConfig
from app.routing.model_router import CostAwareRouter

logger = logging.getLogger("rag_llm_system")


class AppRouter:
    """Central router for application components."""

    def __init__(self):
        """Initialize all application components."""
        logger.info("Initializing application components")

        # Initialize embeddings
        self.embedding_generator = EmbeddingGenerator()

        # Initialize vector store
        self.vector_store = VectorStore()

        # Initialize retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
        )

        # Initialize text splitter
        self.text_splitter = TextSplitter()

        # Initialize indexer
        self.indexer = DocumentIndexer(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
            text_splitter=self.text_splitter,
        )

        # Initialize LLM
        self.llm = GroqLLM()

        # NEW: Initialize Model Configuration & Router
        try:
            self.model_config = MultiModelConfig("config/models.yaml")
            self.cost_router = CostAwareRouter(self.model_config)
            logger.info("✅ Multi-model routing initialized in AppRouter")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Routing: {e}. Smart query will likely fail.")
            self.model_config = None
            self.cost_router = None

        # Initialize cache
        if settings.cache_type == "redis":
            self.cache = RedisCache()
        else:
            self.cache = FilesystemCache()

        # Initialize memory
        self.short_term_memory = ShortTermMemory()
        self.long_term_memory = LongTermMemory()

        # Initialize graph
        self.graph_builder = RAGGraphBuilder(
            retriever=self.retriever,
            llm=self.llm,
            cache=self.cache,
            short_term_memory=self.short_term_memory,
            long_term_memory=self.long_term_memory,
        )
        self.graph_builder.build()

        logger.info("All components initialized successfully")