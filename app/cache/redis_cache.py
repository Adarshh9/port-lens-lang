"""
Redis-based cache implementation.
Provides high-performance caching using Redis.
"""

import logging
import json
from typing import Optional, Any, Dict
import redis
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class RedisCache:
    """Redis-based cache for storing query results."""

    def __init__(self, redis_url: str = settings.redis_url):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
        """
        logger.info(f"Initializing Redis cache at {redis_url}")
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {str(e)}")
            raise

    def get(self, query: str, ttl_hours: int = 24) -> Optional[Dict[str, Any]]:
        """
        Get cached result.
        
        Args:
            query: Query string
            ttl_hours: Time-to-live in hours (ignored for Redis TTL)
            
        Returns:
            Cached data or None if missing
        """
        try:
            cached = self.redis_client.get(f"query:{query}")
            if cached:
                logger.debug(f"Cache hit for query")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Cache get failed: {str(e)}")
            return None

    def set(self, query: str, data: Any, ttl_hours: int = 24) -> None:
        """
        Set cache value.
        
        Args:
            query: Query string
            data: Data to cache
            ttl_hours: Time-to-live in hours
        """
        try:
            self.redis_client.setex(
                f"query:{query}",
                ttl_hours * 3600,
                json.dumps(data),
            )
            logger.debug(f"Cache set for query")
        except Exception as e:
            logger.error(f"Cache set failed: {str(e)}")

    def clear(self) -> None:
        """Clear all cache."""
        try:
            self.redis_client.flushdb()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis_client.info()
            return {
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands": info.get("total_commands_processed"),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
