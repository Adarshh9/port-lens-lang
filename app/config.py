from pydantic_settings import BaseSettings
from typing import Literal
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Groq API - FIXED: correct model name
    groq_api_key: str
    groq_model: str = "llama-3.1-8b-instant"  # Changed from openai/gpt-oss-20b

    # LangSmith
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-llm-system"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Vector DB (Chroma)
    chroma_db_path: str = "./data/chroma_db"
    chroma_collection_name: str = "rag_documents"

    # Cache
    cache_type: Literal["filesystem", "redis"] = "filesystem"
    redis_url: str = "redis://localhost:6379"
    filesystem_cache_dir: str = "./data/cache"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Memory
    short_term_memory_max_messages: int = 20
    long_term_memory_db_path: str = "./data/long_term_memory.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # Judge - FIXED: threshold should be 0-1 range (not 0-10)
    judge_quality_threshold: float = 0.7  # Changed from 7.0
    judge_enable_fallback: bool = True

    # Document Processing
    chunk_size: int = 1024
    chunk_overlap: int = 256
    pdf_extraction_method: str = "pypdf"

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **data):
        super().__init__(**data)
        self._create_directories()

    def _create_directories(self) -> None:
        """Create required directories if they don't exist."""
        dirs = [
            Path(self.chroma_db_path),
            Path(self.filesystem_cache_dir),
            Path(self.long_term_memory_db_path).parent,
            Path(self.log_file).parent,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)


# Create global settings instance
settings = Settings()

# Configure LangSmith at import time
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    print("✅ LangSmith tracing enabled")