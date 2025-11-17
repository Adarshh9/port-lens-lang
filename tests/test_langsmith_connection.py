# tests/test_langsmith_connection.py

from langsmith import Client
from app.config import settings

def test_langsmith_connection():
    """Verify LangSmith is configured correctly."""
    
    if not settings.langsmith_api_key:
        print("❌ LANGSMITH_API_KEY not set")
        return False
    
    try:
        client = Client()
        projects = client.list_projects()
        print(f"✅ Connected to LangSmith")
        print(f"   Projects: {[p.name for p in projects]}")
        return True
    except Exception as e:
        print(f"❌ LangSmith connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_langsmith_connection()
