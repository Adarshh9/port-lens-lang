import asyncio
import os
import sys
from dotenv import load_dotenv
from app.models.model_config import MultiModelConfig
from app.routing.model_router import CostAwareRouter
from app.llm.multi_provider import MultiProviderLLM

# Setup basic logging to console
import logging

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_llm_system")

async def test_routing_logic():
    print("\n" + "="*50)
    print("🛠️  DEBUGGING ROUTER LOGIC DIRECTLY")
    print("="*50 + "\n")

    # 1. Check Config
    config_path = "config/models.yaml"
    if not os.path.exists(config_path):
        print(f"❌ CRITICAL: Config file not found at {config_path}")
        return

    print(f"✅ Found config: {config_path}")
    
    try:
        config = MultiModelConfig(config_path)
        print("✅ Config loaded successfully")
        print(f"   Models configured: {list(config.models.keys())}")
    except Exception as e:
        print(f"❌ Config Load Failed: {e}")
        return

    # 2. Check Router Initialization
    try:
        router = CostAwareRouter(config)
        print("✅ Router initialized")
    except Exception as e:
        print(f"❌ Router Init Failed: {e}")
        return

    # 3. Test Connections (Ping Providers)
    print("\nTesting Model Providers...")
    llm = MultiProviderLLM(config)
    
    # Test Local Ollama (Phi3)
    print("\n--- Testing Local Ollama (phi3_mini) ---")
    try:
        # Simple ping prompt
        response = await llm.generate("phi3_mini", "Hi", max_tokens=5)
        print(f"✅ Ollama Success: {response['answer']}")
    except Exception as e:
        print(f"❌ Ollama Failed. Is 'ollama serve' running? Error: {str(e)[:100]}...")

    # Test Cloud (Llama3)
    print("\n--- Testing Cloud Groq (llama3_8b) ---")
    try:
        response = await llm.generate("llama3_8b", "Hi", max_tokens=5)
        print(f"✅ Groq Success: {response['answer']}")
    except Exception as e:
        print(f"❌ Groq Failed. Check API Key/Status. Error: {str(e)[:100]}...")

    # 4. Test Full Routing
    print("\n" + "="*50)
    print("🚦 TESTING FULL ROUTING FLOW")
    print("="*50)
    
    query = "What is machine learning?"
    context = "Machine learning is a field of AI."
    
    try:
        print(f"Attempting route with query: '{query}'")
        result = await router.route_and_generate(query, context, optimize_for="balanced")
        
        print("\n🎉 SUCCESS!")
        print(f"   Model Used: {result.model_used}")
        print(f"   Judge Score: {result.judge_score}")
        print(f"   Answer: {result.answer[:100]}...")
    except Exception as e:
        print(f"\n🔥 ROUTING CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_routing_logic())