"""
Document loading utilities for various file formats.
Supports PDF, TXT, and Markdown files.
"""

from typing import List, Dict, Any
from pathlib import Path
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain.schema import Document

logger = logging.getLogger("rag_llm_system")


class DocumentLoader:
    """Load documents from various file formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """
        Load PDF document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading PDF: {file_path}")
        try:
            # Resolve absolute path
            abs_path = Path(file_path).resolve()
            logger.info(f"Resolved PDF path: {abs_path}")
            
            if not abs_path.exists():
                raise FileNotFoundError(f"PDF file not found: {abs_path}")
                
            loader = PyPDFLoader(str(abs_path))
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """
        Load text or markdown document.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading text file: {file_path}")
        try:
            # Resolve absolute path
            abs_path = Path(file_path).resolve()
            logger.info(f"Resolved text path: {abs_path}")
            
            if not abs_path.exists():
                raise FileNotFoundError(f"Text file not found: {abs_path}")
                
            loader = TextLoader(str(abs_path))
            documents = loader.load()
            logger.info(f"Successfully loaded text file")
            return documents
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            raise

    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """
        Load a file based on its extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file type not supported
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {cls.SUPPORTED_EXTENSIONS}"
            )

        if extension == ".pdf":
            return cls.load_pdf(file_path)
        else:
            return cls.load_text(file_path)

    @classmethod
    def load_directory(cls, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            Combined list of Document objects
        """
        logger.info(f"Loading documents from directory: {directory_path}")
        documents = []
        dir_path = Path(directory_path).resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        for file_path in dir_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                try:
                    docs = cls.load_file(str(file_path))
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {str(e)}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    @staticmethod
    def normalize_metadata(documents: List[Document]) -> List[Document]:
        """
        Normalize metadata across all documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Normalized documents
        """
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}

            # Ensure standard metadata fields
            if "source" not in doc.metadata:
                doc.metadata["source"] = "unknown"
            if "created_at" not in doc.metadata:
                from datetime import datetime
                doc.metadata["created_at"] = datetime.utcnow().isoformat()

        return documents