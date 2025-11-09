"""
Filesystem-based cache implementation.
Provides caching using local filesystem.
"""

import logging
import hashlib
import json
from typing import Optional, Any, Dict
from pathlib import Path
from datetime import datetime, timedelta
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class FilesystemCache:
    """Filesystem-based cache for storing query results."""

    def __init__(self, cache_dir: str = settings.filesystem_cache_dir):
        """
        Initialize filesystem cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized filesystem cache at {cache_dir}")

    @staticmethod
    def _generate_key(query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{key}.json"

    def get(self, query: str, ttl_hours: int = 24) -> Optional[Dict[str, Any]]:
        """
        Get cached result.
        
        Args:
            query: Query string
            ttl_hours: Time-to-live in hours
            
        Returns:
            Cached data or None if expired/missing
        """
        try:
            key = self._generate_key(query)
            cache_file = self._get_cache_file(key)

            if not cache_file.exists():
                return None

            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            # Check if cache is expired
            created_at = datetime.fromisoformat(cache_data["created_at"])
            if datetime.utcnow() - created_at > timedelta(hours=ttl_hours):
                logger.info(f"Cache expired for key: {key}")
                cache_file.unlink()
                return None

            logger.debug(f"Cache hit for key: {key}")
            return cache_data["data"]

        except Exception as e:
            logger.error(f"Cache get failed: {str(e)}")
            return None

    def set(self, query: str, data: Any) -> None:
        """
        Set cache value.
        
        Args:
            query: Query string
            data: Data to cache
        """
        try:
            key = self._generate_key(query)
            cache_file = self._get_cache_file(key)

            cache_data = {
                "created_at": datetime.utcnow().isoformat(),
                "key": key,
                "data": data,
            }

            with open(cache_file, "w") as f:
                json.dump(cache_data, f)

            logger.debug(f"Cache set for key: {key}")

        except Exception as e:
            logger.error(f"Cache set failed: {str(e)}")

    def clear(self) -> None:
        """Clear all cache."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            return {
                "total_entries": len(cache_files),
                "total_size_bytes": total_size,
                "cache_dir": str(self.cache_dir),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
