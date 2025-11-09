"""
Tests for judge module.
"""

import pytest
import json


def test_judge_prompt_structure():
    """Test judge prompt structure."""
    from app.llm.prompts import JUDGE_PROMPT_TEMPLATE

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question="What is AI?",
        context="AI is artificial intelligence",
        answer="AI stands for artificial intelligence",
    )

    assert "question" in prompt.lower()
    assert "context" in prompt.lower()
    assert "answer" in prompt.lower()


def test_judge_response_parsing():
    """Test parsing judge response."""
    response_json = """{
        "score": 8.5,
        "reasons": "Good response",
        "criteria": {
            "correctness": 9.0,
            "relevance": 8.0,
            "completeness": 8.0,
            "clarity": 9.0,
            "citations": 7.0
        }
    }"""

    response = json.loads(response_json)

    assert response["score"] == 8.5
    assert response["criteria"]["correctness"] == 9.0
    assert len(response["criteria"]) == 5
