"""
Filesystem cache implementation with better key generation.
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class FilesystemCache:
    """Cache implementation using filesystem with MD5 hashing."""

    def __init__(self, cache_dir: str = settings.filesystem_cache_dir):
        """
        Initialize filesystem cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized filesystem cache at {cache_dir}")

    def _generate_key(self, query: str, session_id: str, user_id: str = "") -> str:
        """
        Generate cache key from query, session, and user.
        Uses MD5 hash for short, collision-resistant keys.
        
        Args:
            query: User query
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Cache key (hash)
        """
        # Create composite key
        composite = f"{session_id}:{user_id}:{query}".lower().strip()
        
        # Generate MD5 hash
        key_hash = hashlib.md5(composite.encode()).hexdigest()
        
        logger.debug(f"Generated cache key: {key_hash} for query: {query[:50]}")
        return key_hash

    def get(self, query: str, session_id: str = "", user_id: str = "") -> Optional[str]:
        """
        Retrieve cached answer.
        
        Args:
            query: Query string
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Cached answer or None
        """
        try:
            key = self._generate_key(query, session_id, user_id)
            cache_file = self.cache_dir / f"{key}.json"
            
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    data = json.load(f)
                
                logger.info(f"✅ Cache HIT for query: {query[:50]}")
                return data.get("answer")
            
            logger.debug(f"Cache MISS for query: {query[:50]}")
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {str(e)}")
            return None

    def set(
        self,
        query: str,
        answer: str,
        session_id: str = "",
        user_id: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store answer in cache.
        
        Args:
            query: Query string
            answer: Generated answer
            session_id: Session identifier
            user_id: User identifier
            metadata: Optional metadata (judge_score, etc.)
            
        Returns:
            True if successful
        """
        try:
            key = self._generate_key(query, session_id, user_id)
            cache_file = self.cache_dir / f"{key}.json"
            
            cache_data = {
                "query": query,
                "answer": answer,
                "session_id": session_id,
                "user_id": user_id,
                "metadata": metadata or {},
                "timestamp": str(Path(cache_file).stem)
            }
            
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            
            logger.info(f"✅ Cached answer for query: {query[:50]}")
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed: {str(e)}")
            return False

    def clear(self) -> bool:
        """Clear all cache."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        return {
            "total_cached_queries": len(cache_files),
            "cache_size_mb": sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        }
