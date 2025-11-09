"""
Document indexing and ingestion pipeline.
Orchestrates loading, splitting, embedding, and storing documents.
"""

import logging
from typing import List, Optional
from app.ingestion.loader import DocumentLoader
from app.ingestion.splitter import TextSplitter
from app.ingestion.embedder import EmbeddingGenerator
from app.vector.store import VectorStore

logger = logging.getLogger("rag_llm_system")


class DocumentIndexer:
    """Orchestrate complete document ingestion pipeline."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        text_splitter: TextSplitter,
    ):
        """
        Initialize document indexer.
        
        Args:
            vector_store: Vector database instance
            embedding_generator: Embedding generator instance
            text_splitter: Text splitter instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.text_splitter = text_splitter
        self.loader = DocumentLoader()

    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single file into the vector database.
        
        Args:
            file_path: Path to file
            
        Returns:
            Number of chunks indexed
        """
        logger.info(f"Starting ingestion for file: {file_path}")
        try:
            # Load
            documents = self.loader.load_file(file_path)
            documents = self.loader.normalize_metadata(documents)

            # Split
            chunks = self.text_splitter.split_documents(documents)

            # Index
            self.vector_store.add_documents(chunks)

            logger.info(f"Successfully indexed {len(chunks)} chunks from {file_path}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {str(e)}")
            raise

    def ingest_directory(self, directory_path: str) -> int:
        """
        Ingest all documents from a directory.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            Total number of chunks indexed
        """
        logger.info(f"Starting directory ingestion: {directory_path}")
        try:
            # Load all documents
            documents = self.loader.load_directory(directory_path)
            documents = self.loader.normalize_metadata(documents)

            # Split
            chunks = self.text_splitter.split_documents(documents)

            # Index
            self.vector_store.add_documents(chunks)

            logger.info(f"Successfully indexed {len(chunks)} chunks from directory")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to ingest directory {directory_path}: {str(e)}")
            raise

    def clear_index(self) -> None:
        """Clear all documents from the index."""
        logger.warning("Clearing entire vector store index")
        self.vector_store.clear()
