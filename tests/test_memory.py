"""
Tests for memory modules.
"""

import pytest
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory


def test_short_term_memory():
    """Test short-term memory."""
    memory = ShortTermMemory(max_messages=10)

    memory.add_message(role="user", content="Hello")
    memory.add_message(role="assistant", content="Hi there!")

    messages = memory.get_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    stats = memory.get_stats()
    assert stats["total_messages"] == 2
    assert stats["max_capacity"] == 10


@pytest.mark.asyncio
async def test_long_term_memory():
    """Test long-term memory."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "memory.db"
        memory = LongTermMemory(str(db_path))

        session_id = "test_session"
        memory.create_session(session_id)

        memory.add_message(
            session_id=session_id,
            role="user",
            content="What is AI?",
        )

        memory.store_fact(session_id, "user_interest", "machine_learning")

        messages = memory.get_session_messages(session_id)
        assert len(messages) == 1

        facts = memory.get_facts(session_id)
        assert "user_interest" in facts
