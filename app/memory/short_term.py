"""
Short-term conversation memory buffer.
Stores recent messages in memory for context.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime
from app.config import settings

logger = logging.getLogger("rag_llm_system")


class ShortTermMemory:
    """Short-term conversation buffer for recent messages."""

    def __init__(self, max_messages: int = settings.short_term_memory_max_messages):
        """
        Initialize short-term memory.
        
        Args:
            max_messages: Maximum number of messages to keep
        """
        self.max_messages = max_messages
        self.messages: deque = deque(maxlen=max_messages)
        # Store by session_id for multi-session support
        self.sessions: Dict[str, deque] = {}
        logger.info(f"Initialized short-term memory with max {max_messages} messages")

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to short-term memory.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata (for compatibility)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.messages.append(message)
        logger.debug(f"Added {role} message to short-term memory")

    def add_message_for_session(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a message for a specific session.
        
        Args:
            session_id: Session identifier
            role: Message role
            content: Message content
            metadata: Optional metadata
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_messages)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.sessions[session_id].append(message)
        logger.debug(f"Added {role} message to session {session_id}")

    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get messages from short-term memory.
        
        Args:
            limit: Optional limit on number of messages
            
        Returns:
            List of messages
        """
        messages = list(self.messages)
        if limit:
            messages = messages[-limit:]
        return messages

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get message history for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages for session
        """
        if session_id not in self.sessions:
            return []
        return list(self.sessions[session_id])

    def get_context_string(self) -> str:
        """
        Get formatted context string from messages.
        
        Returns:
            Formatted conversation context
        """
        context_parts = []
        for msg in self.messages:
            context_parts.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.sessions.clear()
        logger.info("Short-term memory cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_messages": len(self.messages),
            "total_sessions": len(self.sessions),
            "max_capacity": self.max_messages,
            "utilization": len(self.messages) / self.max_messages,
        }
