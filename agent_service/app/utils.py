import os
from pathlib import Path
import yaml
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_llm_configs(config_path: str = None):
    """
    Loads YAML config and merges with environment variables (if set).
    If no config_path is provided, it computes the path relative to this file.
    """
    if config_path is None:
        current_dir = Path(__file__).resolve().parent  # e.g., agent_service/app
        repo_root = current_dir.parent.parent         # repository root
        config_path = str(repo_root / "config.yaml")
    
    try:
        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file) or {}
            llm_configs_raw = yaml_config.get("llm_configs", {})

            merged_configs = {}
            for name, config in llm_configs_raw.items():
                merged_configs[name] = {
                    "api_key": os.getenv("OPENAI_API_KEY", config.get("api_key")),
                    "base_url": config.get("base_url"),
                    "model_id": config.get("model_id"), 
                    "max_new_tokens": config.get("max_new_tokens", 512), 
                    "temperature": float(os.getenv("TEMPERATURE", config.get("temperature", 1.0))),
                    "top_p": float(os.getenv("TOP_P", config.get("top_p", 1.0))),
                    "repetition_penalty": float(os.getenv("REPETITION_PENALTY", config.get("repetition_penalty", 1.0))),
                }
            print(merged_configs)
            return merged_configs

    except Exception as e:
        logger.error(f"Failed to load config file: {e}", exc_info=True)
        raise RuntimeError(f"Error loading configuration: {e}")
