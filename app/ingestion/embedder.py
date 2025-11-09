"""
Embedding generation utilities using Sentence Transformers.
"""

from typing import List, Optional
import logging
# Updated import to use community embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class EmbeddingGenerator:
    """Generate embeddings for documents and queries."""

    def __init__(self, model_name: str = settings.embedding_model):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the embedding model
        """
        logger.info(f"Initializing embeddings with model: {model_name}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                cache_folder="./data/embeddings_cache",
            )
            self.model_name = model_name
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Embedding {len(documents)} documents")
            embeddings = self.embeddings.embed_documents(documents)
            logger.info("Documents embedded successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise