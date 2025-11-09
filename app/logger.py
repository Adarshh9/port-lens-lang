"""
Logging configuration with LangSmith integration.
Provides structured logging for the RAG + LLM system.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import json
from typing import Optional
from app.config import settings

# Import LangSmith callbacks
try:
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False


class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs JSON logs for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def setup_logging() -> logging.Logger:
    """
    Configure logging with both file and console handlers.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("rag_llm_system")
    logger.setLevel(getattr(logging, settings.log_level))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler with rotation
    log_file = Path(settings.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(getattr(logging, settings.log_level))
    file_formatter = JsonFormatter()
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, settings.log_level))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logging()


def log_with_context(message: str, level: str = "INFO", **context) -> None:
    """
    Log a message with additional context.
    
    Args:
        message: The log message
        level: Log level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
        **context: Additional context data to include
    """
    log_func = getattr(logger, level.lower(), logger.info)
    record = logging.LogRecord(
        name="rag_llm_system",
        level=getattr(logging, level),
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.extra_data = context
    log_func(record)


def init_langsmith() -> Optional[LangSmithClient]:
    """
    Initialize LangSmith client if available.
    
    Returns:
        LangSmithClient or None if not configured
    """
    if not LANGSMITH_AVAILABLE or not settings.langsmith_api_key:
        logger.warning("LangSmith not configured or unavailable")
        return None

    try:
        # Updated initialization without project_name parameter
        client = LangSmithClient(api_key=settings.langsmith_api_key)
        logger.info("LangSmith initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize LangSmith: {str(e)}")
        return None


# Initialize LangSmith
langsmith_client = init_langsmith()