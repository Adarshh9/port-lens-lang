"""
Vector database management using Chroma.
Persistent storage for embeddings and document metadata.
"""

import logging
from typing import List, Optional, Dict, Any
import chromadb
from langchain.schema import Document
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class VectorStore:
    """Manage vector database operations with Chroma."""

    def __init__(
        self,
        db_path: str = settings.chroma_db_path,
        collection_name: str = settings.chroma_collection_name,
    ):
        """
        Initialize Chroma vector store.
        
        Args:
            db_path: Path to Chroma database
            collection_name: Name of collection
        """
        logger.info(f"Initializing Chroma vector store at {db_path}")
        try:
            chromadb.utils.embedding_functions.DEFAULT_TELEMETRY = False
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.collection_name = collection_name
            logger.info(f"Vector store initialized with collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with embeddings
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        try:
            ids = []
            embeddings = []
            documents_text = []
            metadatas = []

            for i, doc in enumerate(documents):
                from app.ingestion.embedder import EmbeddingGenerator
                embedder = EmbeddingGenerator()
                embedding = embedder.embed_query(doc.page_content)

                doc_id = f"doc_{hash(doc.page_content) % 10**8}"
                ids.append(doc_id)
                embeddings.append(embedding)
                documents_text.append(doc.page_content)
                metadatas.append(doc.metadata or {})

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadatas,
            )

            logger.info(f"Successfully added {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def search(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
            )

            formatted_results = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append(
                        {
                            "content": doc,
                            "metadata": results["metadatas"][0][i],
                            "distance": results["distances"][0][i],
                        }
                    )

            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def delete_collection(self) -> None:
        """Delete the current collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info("Collection deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            raise

    def clear(self) -> None:
        """Clear all documents from the collection."""
        logger.warning("Clearing all documents from collection")
        try:
            # Get all IDs and delete them
            results = self.collection.get()
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
            logger.info("Collection cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            raise
