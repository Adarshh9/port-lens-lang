"""
Fixed test script for RAG + LLM System
"""

import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print("Health Check:", response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def create_test_document():
    """Create a test document with absolute path."""
    test_content = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals or humans. 
    
    Machine learning is a subset of AI that focuses on building systems that learn from data.
    
    Neural networks are computing systems inspired by biological neural networks.
    
    Deep learning uses multiple layers in neural networks to analyze various factors in data.
    
    Large Language Models (LLMs) are AI models that can understand and generate human-like text.
    
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language.
    
    Computer vision is a field of AI that enables computers to interpret and understand the visual world.
    
    Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.
    """
    
    # Create test_documents directory if it doesn't exist
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    file_path = test_dir / "test_document.txt"
    # with open(file_path, "w", encoding="utf-8") as f:
    #     f.write(test_content)
    
    return str(file_path.absolute())

def test_ingestion():
    """Test document ingestion."""
    try:
        file_path = create_test_document()
        print(f"Created test document at: {file_path}")
        
        payload = {
            "file_path": file_path
        }
        
        response = requests.post(f"{BASE_URL}/ingest", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Ingested {result['chunks_indexed']} chunks from {result['file_path']}")
            return True
        else:
            print(f"❌ Ingestion failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ingestion request failed: {e}")
        return False

def test_query(question, session_id="test_session"):
    """Test query endpoint."""
    payload = {
        "query": question,
        "session_id": session_id,
        "user_id": "test_user"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Query: {question}")
            
            if result['answer']:
                answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
                print(f"✅ Answer: {answer_preview}")
            else:
                print("✅ Answer: [No answer generated]")
                
            print(f"✅ Cache Hit: {result['cache_hit']}")
            print(f"✅ Processing Time: {result['processing_time']:.2f}s")
            print(f"✅ Quality Passed: {result['quality_passed']}")
            
            if result.get('judge_evaluation'):
                print(f"✅ Judge Score: {result['judge_evaluation']['score']}/10")
            
            print(f"✅ Retrieved Documents: {len(result['retrieved_docs'])}")
            for i, doc in enumerate(result['retrieved_docs'][:2]):  # Show first 2
                content_preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                print(f"   📄 Doc {i+1}: {content_preview}")
            
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_empty_query():
    """Test the system with no documents ingested."""
    print("\n🧪 Testing with empty database (no documents ingested)...")
    
    questions = [
        "What is artificial intelligence?",
        "Explain machine learning basics"
    ]
    
    for i, question in enumerate(questions):
        print(f"\n{'='*60}")
        print(f"Empty DB Test {i+1}/{len(questions)}")
        success = test_query(question, f"empty_db_session_{i+1}")
        if not success:
            print("❌ Empty DB query test failed!")
            break
    
    return True

def main():
    print("🧪 Testing RAG + LLM System...")
    
    # Test health
    print("1. Testing health endpoint...")
    if not test_health():
        print("❌ Health check failed!")
        return
    
    # Test with empty database first
    test_empty_query()
    
    # Test ingestion
    print("\n2. Testing document ingestion...")
    if test_ingestion():
        print("✅ Ingestion successful!")
        # Wait a moment for processing
        time.sleep(3)
        
        # Test queries after ingestion
        print("\n3. Testing query endpoints after ingestion...")
        test_questions = [
            "What is artificial intelligence?",
            "Explain machine learning",
        ]
        
        for i, question in enumerate(test_questions):
            print(f"\n{'='*60}")
            print(f"Test {i+1}/{len(test_questions)}")
            success = test_query(question, f"session_{i+1}")
            if not success:
                print("❌ Query test failed!")
                break
    else:
        print("⚠️ Ingestion failed, but continuing with empty DB tests...")
    
    print(f"\n{'='*60}")
    print("🎉 Testing completed!")

if __name__ == "__main__":
    main()