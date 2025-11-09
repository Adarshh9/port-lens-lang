"""
Long-term persistent memory storage.
Stores conversations and important information in a database.
"""

import logging
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class LongTermMemory:
    """Long-term persistent memory using SQLite."""

    def __init__(self, db_path: str = settings.long_term_memory_db_path):
        """
        Initialize long-term memory.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        logger.info(f"Initializing long-term memory at {db_path}")
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES conversations(session_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES conversations(session_id)
                )
            """)

            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def create_session(
        self, session_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a new conversation session.
        
        Args:
            session_id: Unique session identifier
            metadata: Optional session metadata
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR IGNORE INTO conversations 
                (session_id, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                session_id,
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat(),
                json.dumps(metadata or {}),
            ))

            conn.commit()
            conn.close()
            logger.info(f"Created session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add message to session.
        
        Args:
            session_id: Session identifier
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        try:
            # Create session if it doesn't exist
            self.create_session(session_id)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO messages (session_id, role, content, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session_id,
                role,
                content,
                datetime.utcnow().isoformat(),
                json.dumps(metadata or {}),
            ))

            # Update session updated_at
            cursor.execute("""
                UPDATE conversations SET updated_at = ? WHERE session_id = ?
            """, (datetime.utcnow().isoformat(), session_id))

            conn.commit()
            conn.close()
            logger.debug(f"Added message to session {session_id}")
        except Exception as e:
            logger.error(f"Failed to add message: {str(e)}")

    def store_interaction(
        self,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a query-answer interaction.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            query: User query
            answer: Assistant answer
            metadata: Optional metadata
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO interactions 
                (user_id, session_id, query, answer, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                session_id,
                query,
                answer,
                datetime.utcnow().isoformat(),
                json.dumps(metadata or {}),
            ))

            conn.commit()
            conn.close()
            logger.debug(f"Stored interaction for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to store interaction: {str(e)}")

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages from a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT role, content, timestamp, metadata FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))

            rows = cursor.fetchall()
            conn.close()

            messages = []
            for role, content, timestamp, metadata_str in rows:
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,
                    "metadata": json.loads(metadata_str or "{}"),
                })

            return messages
        except Exception as e:
            logger.error(f"Failed to get session messages: {str(e)}")
            return []

    def store_fact(
        self, session_id: str, key: str, value: str
    ) -> None:
        """
        Store a fact about the conversation.
        
        Args:
            session_id: Session identifier
            key: Fact key
            value: Fact value
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO facts (session_id, key, value, created_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, key, value, datetime.utcnow().isoformat()))

            conn.commit()
            conn.close()
            logger.debug(f"Stored fact: {key}={value}")
        except Exception as e:
            logger.error(f"Failed to store fact: {str(e)}")

    def get_facts(self, session_id: str) -> Dict[str, str]:
        """
        Get all facts from a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of facts
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT key, value FROM facts
                WHERE session_id = ?
            """, (session_id,))

            rows = cursor.fetchall()
            conn.close()

            return {key: value for key, value in rows}
        except Exception as e:
            logger.error(f"Failed to get facts: {str(e)}")
            return {}

    def get_user_interactions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get interactions for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of interactions to return
            
        Returns:
            List of interactions
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT user_id, session_id, query, answer, created_at, metadata
                FROM interactions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))

            rows = cursor.fetchall()
            conn.close()

            interactions = []
            for user_id, session_id, query, answer, created_at, metadata_str in rows:
                interactions.append({
                    "user_id": user_id,
                    "session_id": session_id,
                    "query": query,
                    "answer": answer,
                    "created_at": created_at,
                    "metadata": json.loads(metadata_str or "{}"),
                })

            return interactions
        except Exception as e:
            logger.error(f"Failed to get user interactions: {str(e)}")
            return []
