"""
Text splitting utilities for document chunking.
Uses recursive character splitting with configurable chunk size and overlap.
"""

from typing import List
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class TextSplitter:
    """Handle text splitting and chunking."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ):
        """
        Initialize text splitter.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents
        """
        logger.info(f"Splitting {len(documents)} documents")
        split_docs = self.splitter.split_documents(documents)
        logger.info(f"Generated {len(split_docs)} chunks")

        # Add chunk metadata
        for i, doc in enumerate(split_docs):
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["chunk_id"] = i
            doc.metadata["chunk_size"] = len(doc.page_content)

        return split_docs

    def split_text(self, text: str) -> List[str]:
        """
        Split raw text into chunks.
        
        Args:
            text: Raw text to split
            
        Returns:
            List of text chunks
        """
        logger.info("Splitting raw text")
        chunks = self.splitter.split_text(text)
        logger.info(f"Generated {len(chunks)} chunks from raw text")
        return chunks
