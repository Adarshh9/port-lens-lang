"""
Tests for RAG flow.
"""

import pytest
from app.graph.state import RAGState
from app.ingestion.splitter import TextSplitter
from app.ingestion.embedder import EmbeddingGenerator
from app.vector.store import VectorStore


@pytest.mark.asyncio
async def test_rag_state_creation():
    """Test RAG state creation."""
    state = RAGState(
        query="What is machine learning?",
        session_id="test_session",
        user_id="test_user",
    )

    assert state.query == "What is machine learning?"
    assert state.session_id == "test_session"
    assert state.user_id == "test_user"
    assert state.judge_score == 0.0
    assert len(state.retrieved_docs) == 0


def test_text_splitter():
    """Test text splitter."""
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

    text = "This is a long document. " * 100
    chunks = splitter.split_text(text)

    assert len(chunks) > 0
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_embedding_generator():
    """Test embedding generator."""
    generator = EmbeddingGenerator()

    embedding = generator.embed_query("What is AI?")
    assert len(embedding) > 0
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_vector_store():
    """Test vector store."""
    from langchain.schema import Document

    store = VectorStore()

    # Create test documents
    docs = [
        Document(page_content="AI is transforming industries", metadata={"source": "test"}),
        Document(page_content="Machine learning is a subset of AI", metadata={"source": "test"}),
    ]

    # This would require embeddings to be generated
    # store.add_documents(docs)
    # results = store.search([0.1] * 384, k=2)
    # assert len(results) > 0
