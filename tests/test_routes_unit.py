import asyncio
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import init_routes


class DummyGraphBuilder:
    def __init__(self, result):
        self._result = result

    async def execute(self, state):
        # simulate small processing delay
        await asyncio.sleep(0)
        return self._result


def test_query_route_with_dict_result():
    # Prepare a dict-shaped graph result
    graph_result = {
        "query": "What is AI?",
        "final_answer": "AI is ...",
        "retrieved_docs": [],
        "judge_evaluation": None,
        "cache_hit": False,
        "quality_passed": False,
    }

    init_routes(DummyGraphBuilder(graph_result), indexer=None)

    client = TestClient(app)

    payload = {"query": "What is AI?", "session_id": "s1", "user_id": "tester"}
    resp = client.post("/api/v1/query", json=payload)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "AI is ..."
    assert data["query"] == "What is AI?"


def test_query_route_with_ragstate_like_object():
    # Create a simple object that has a to_dict method
    class FakeState:
        def to_dict(self):
            return {
                "query": "What is AI?",
                "final_answer": "AI is ...",
                "retrieved_docs": [],
                "judge_evaluation": None,
                "cache_hit": False,
                "quality_passed": False,
            }

    init_routes(DummyGraphBuilder(FakeState()), indexer=None)

    client = TestClient(app)

    payload = {"query": "What is AI?", "session_id": "s2", "user_id": "tester"}
    resp = client.post("/api/v1/query", json=payload)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "AI is ..."
    assert data["query"] == "What is AI?"
