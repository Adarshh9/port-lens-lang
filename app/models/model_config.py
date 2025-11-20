import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from string import Template
from dotenv import load_dotenv

@dataclass
class ModelConfig:
    name: str
    provider: str
    model_name: str
    cost_per_1k_tokens: float
    latency_ms_estimate: int
    context_window: int
    quality_tier: str
    ideal_for: List[str]
    max_complexity: float
    endpoint: str = ""
    api_key: str = ""

class MultiModelConfig:
    def __init__(self, config_path: str = "config/models.yaml"):
        load_dotenv()
        config_file = Path(config_path)
        if not config_file.exists():
            # Create default if missing to prevent crash
            Path("config").mkdir(exist_ok=True)
            raise FileNotFoundError(f"Models config not found: {config_path}")
            
        with open(config_file, "r") as f:
            config_str = f.read()
            
        # Substitute environment variables
        # Use safe_substitute to avoid crashing if OPENAI_API_KEY is missing but not used
        config_str = Template(config_str).safe_substitute(os.environ)
        
        self.config = yaml.safe_load(config_str)
        self.models = self._load_models()
        self.routing = self.config.get("routing", {})
        
    def _load_models(self) -> Dict[str, ModelConfig]:
        models = {}
        for name, cfg in self.config.get("models", {}).items():
            models[name] = ModelConfig(
                name=name,
                provider=cfg["provider"],
                model_name=cfg["model_name"],
                cost_per_1k_tokens=cfg["cost_per_1k_tokens"],
                latency_ms_estimate=cfg["latency_ms_estimate"],
                context_window=cfg["context_window"],
                quality_tier=cfg["quality_tier"],
                ideal_for=cfg.get("ideal_for", []),
                max_complexity=cfg.get("max_complexity", 1.0),
                endpoint=cfg.get("endpoint", ""),
                api_key=cfg.get("api_key", ""),
            )
        return models

    def get_model(self, name: str) -> ModelConfig:
        if name not in self.models:
            raise ValueError(f"Model not found: {name}")
        return self.models[name]

    def get_fallback_chain(self) -> List[str]:
        return self.routing.get("fallback_chain", [])
        
    def get_complexity_thresholds(self) -> Dict[str, float]:
        return self.routing.get("complexity_thresholds", {})