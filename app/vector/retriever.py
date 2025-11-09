"""
Retrieval utilities for fetching relevant documents.
"""

import logging
from typing import List, Dict, Any
from app.vector.store import VectorStore
from app.ingestion.embedder import EmbeddingGenerator

logger = logging.getLogger("rag_llm_system")


class Retriever:
    """Retrieve relevant documents from vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

    def retrieve(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Retrieving {k} documents for query: {query[:100]}")
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.embed_query(query)

            # Search vector store
            results = self.vector_store.search(query_embedding, k=k)

            logger.info(f"Retrieved {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise

    def retrieve_with_threshold(
        self, query: str, k: int = 5, distance_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with distance threshold filtering.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            distance_threshold: Maximum distance to include
            
        Returns:
            List of relevant documents above threshold
        """
        logger.info(
            f"Retrieving documents with threshold {distance_threshold}"
        )
        try:
            results = self.retrieve(query, k=k)

            # Filter by threshold
            filtered_results = [
                r for r in results
                if r.get("distance", 1.0) <= distance_threshold
            ]

            logger.info(
                f"Filtered to {len(filtered_results)} documents above threshold"
            )
            return filtered_results
        except Exception as e:
            logger.error(f"Threshold retrieval failed: {str(e)}")
            raise
