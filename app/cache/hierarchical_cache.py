"""
Hierarchical caching system:
L1: Redis (50ms, 1hr TTL) - hot queries
L2: SQLite (persistent, permanent) - cold queries
"""

import logging
import json
import hashlib
import time
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import redis
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class HierarchicalCache:
    """L1 Redis + L2 SQLite cache with intelligent routing."""

    def __init__(
        self,
        redis_url: str = settings.redis_url,
        sqlite_path: str = settings.filesystem_cache_dir,
    ):
        """
        Initialize hierarchical cache.
        
        Args:
            redis_url: Redis connection string
            sqlite_path: SQLite database path
        """
        self.sqlite_path = Path(sqlite_path) / "cache.db"
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try Redis, fallback to memory if unavailable
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.redis_available = True
            logger.info("✅ Redis L1 cache available")
        except Exception as e:
            logger.warning(f"Redis unavailable: {str(e)}, using SQLite only")
            self.redis_client = None
            self.redis_available = False
        
        # Initialize SQLite L2
        self._init_sqlite()
        logger.info(f"✅ SQLite L2 cache initialized at {self.sqlite_path}")

    def _init_sqlite(self) -> None:
        """Initialize SQLite cache schema."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT UNIQUE NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                judge_score REAL,
                user_id TEXT,
                session_id TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_key ON cache(cache_key)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)
        """)
        
        conn.commit()
        conn.close()

    def _generate_cache_key(
        self,
        query: str,
        user_id: str = "",
        session_id: str = "",
        doc_set_version: str = "v1"
    ) -> str:
        """
        Generate normalized cache key using SHA256.
        Ensures consistency across sessions and users.
        
        Args:
            query: User query
            user_id: User identifier
            session_id: Session identifier
            doc_set_version: Document set version (for invalidation)
            
        Returns:
            SHA256 hash of composite key
        """
        # Normalize query: lowercase, strip whitespace, remove duplicates
        normalized_query = " ".join(query.lower().strip().split())
        
        # Composite key includes user, session, doc version
        composite = f"{normalized_query}|{user_id}|{session_id}|{doc_set_version}"
        
        # Generate SHA256
        cache_key = hashlib.sha256(composite.encode()).hexdigest()
        
        logger.debug(f"Generated cache key: {cache_key} for query: {query[:50]}")
        return cache_key

    def get(
        self,
        query: str,
        user_id: str = "",
        session_id: str = "",
        doc_set_version: str = "v1"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve from cache (L1 → L2).
        
        Args:
            query: Query string
            user_id: User identifier
            session_id: Session identifier
            doc_set_version: Document version
            
        Returns:
            Cached result dict or None
        """
        cache_key = self._generate_cache_key(query, user_id, session_id, doc_set_version)
        
        # Try L1: Redis (hot, recent)
        if self.redis_available:
            try:
                result = self.redis_client.get(cache_key)
                if result:
                    logger.info(f"✅ CACHE_HIT_L1 (Redis) - query: {query[:50]}")
                    # Parse and return
                    cached = json.loads(result)
                    cached["cache_level"] = "L1"
                    cached["cache_hit"] = True
                    return cached
            except Exception as e:
                logger.warning(f"Redis L1 get failed: {str(e)}")
        
        # Try L2: SQLite (persistent)
        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM cache WHERE cache_key = ? LIMIT 1
            """, (cache_key,))
            
            row = cursor.fetchone()
            
            if row:
                # Update access metadata
                cursor.execute("""
                    UPDATE cache 
                    SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
                    WHERE cache_key = ?
                """, (cache_key,))
                conn.commit()
                
                logger.info(f"✅ CACHE_HIT_L2 (SQLite) - query: {query[:50]}")
                
                # Convert to dict
                cached = dict(row)
                if cached.get("metadata"):
                    cached["metadata"] = json.loads(cached["metadata"])
                cached["cache_level"] = "L2"
                cached["cache_hit"] = True
                
                conn.close()
                return cached
            
            conn.close()
            
        except Exception as e:
            logger.error(f"SQLite L2 get failed: {str(e)}")
        
        logger.info(f"❌ CACHE_MISS - query: {query[:50]}")
        return None

    def set(
        self,
        query: str,
        answer: str,
        judge_score: float,
        user_id: str = "",
        session_id: str = "",
        doc_set_version: str = "v1",
        metadata: Optional[Dict[str, Any]] = None,
        l1_ttl_seconds: int = 3600
    ) -> bool:
        """
        Store in cache (L1 + L2).
        
        Args:
            query: Query string
            answer: Generated answer
            judge_score: Quality score (0-1)
            user_id: User identifier
            session_id: Session identifier
            doc_set_version: Document version
            metadata: Optional metadata
            l1_ttl_seconds: Redis TTL (default 1hr)
            
        Returns:
            True if successful
        """
        cache_key = self._generate_cache_key(query, user_id, session_id, doc_set_version)
        
        cache_data = {
            "query": query,
            "answer": answer,
            "judge_score": judge_score,
            "user_id": user_id,
            "session_id": session_id,
            "metadata": metadata or {},
            "cached_at": datetime.utcnow().isoformat()
        }
        
        # Store in L1: Redis (hot cache, short TTL)
        if self.redis_available:
            try:
                self.redis_client.setex(
                    cache_key,
                    l1_ttl_seconds,
                    json.dumps(cache_data)
                )
                logger.info(f"✅ Cached in L1 (Redis, TTL={l1_ttl_seconds}s) - query: {query[:50]}")
            except Exception as e:
                logger.warning(f"Redis L1 set failed: {str(e)}")
        
        # Store in L2: SQLite (persistent)
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache 
                (cache_key, query, answer, judge_score, user_id, session_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                query,
                answer,
                judge_score,
                user_id,
                session_id,
                json.dumps(metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Cached in L2 (SQLite, permanent) - query: {query[:50]}")
            return True
            
        except Exception as e:
            logger.error(f"SQLite L2 set failed: {str(e)}")
            return False

    def clear(self) -> bool:
        """Clear all caches."""
        try:
            # Clear Redis
            if self.redis_available:
                self.redis_client.flushdb()
                logger.info("Cleared Redis L1 cache")
            
            # Clear SQLite
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache")
            conn.commit()
            conn.close()
            
            logger.info("Cleared SQLite L2 cache")
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Total cached entries
            cursor.execute("SELECT COUNT(*) as count FROM cache")
            total_entries = cursor.fetchone()[0]
            
            # Cache hit rate (last 24 hours)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_accesses,
                    SUM(CASE WHEN accessed_at > datetime('now', '-1 day') THEN 1 ELSE 0 END) as recent_accesses
                FROM cache
            """)
            row = cursor.fetchone()
            
            # Average judge score in cache
            cursor.execute("SELECT AVG(judge_score) FROM cache WHERE judge_score > 0")
            avg_score = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "total_cached_queries": total_entries,
                "avg_judge_score": avg_score,
                "redis_available": self.redis_available,
                "sqlite_location": str(self.sqlite_path)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {}