import asyncio
import time
from typing import Dict, Any
from abc import ABC, abstractmethod
import logging
from langsmith import traceable

logger = logging.getLogger("rag_llm_system")

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_async(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str: pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float: pass

class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str, cost_per_1k: float):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.cost_per_1k = cost_per_1k

    async def generate_async(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return ((input_tokens + output_tokens) / 1000) * self.cost_per_1k

class OllamaProvider(BaseLLMProvider):
    def __init__(self, endpoint: str, model_name: str, cost_per_1k: float):
        import requests
        self.endpoint = endpoint
        self.model_name = model_name
        self.cost_per_1k = cost_per_1k

    async def generate_async(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        import requests
        try:
            # INCREASE TIMEOUT to 60s for slower local hardware
            response = await asyncio.to_thread(
                requests.post,
                f"{self.endpoint}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False},
                timeout=60 
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            
            # FIX: Raise exception instead of returning error string
            raise Exception(f"Ollama API Error {response.status_code}: {response.text}")
            
        except Exception as e:
            # Log the error here if needed, but let it propagate
            raise Exception(f"Ollama Connection Failed: {str(e)}. Is 'ollama serve' running?")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

class MultiProviderLLM:
    def __init__(self, model_config):
        self.model_config = model_config
        self.providers = {}
        self._init_providers()
    
    def _init_providers(self):
        for name, cfg in self.model_config.models.items():
            if cfg.provider == "groq":
                self.providers[name] = GroqProvider(cfg.api_key, cfg.model_name, cfg.cost_per_1k_tokens)
            elif cfg.provider == "ollama":
                self.providers[name] = OllamaProvider(cfg.endpoint, cfg.model_name, cfg.cost_per_1k_tokens)
            # Add OpenAI here if needed

    @traceable(run_type="llm", name="llm_generation")
    async def generate(self, model_name: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> Dict[str, Any]:
        if model_name not in self.providers:
            raise ValueError(f"Model {model_name} not initialized")
        
        provider = self.providers[model_name]
        start = time.time()
        answer = await provider.generate_async(prompt, max_tokens, temperature)
        latency = (time.time() - start) * 1000
        
        input_tokens = len(prompt) // 4
        output_tokens = len(answer) // 4
        cost = provider.estimate_cost(input_tokens, output_tokens)
        
        return {
            "answer": answer,
            "latency_ms": latency,
            "cost_usd": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }