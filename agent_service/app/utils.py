# agent_service/app/utils.py
import os
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_llm_configs(config_path: str = None):
    """
    Loads YAML config (with top-level "llm_configs"), returning a dict like:
      {
        "openai": {
          "default_llm": {...},
          "alternate_llm": {...}
        },
        "vllm": {
          "default_llm": {...},
          "alternate_llm": {...}
        }
      }
    """
    if config_path is None:
        current_dir = Path(__file__).resolve().parent 
        repo_root = current_dir.parent.parent         
        config_path = str(repo_root / "config.yaml")

    try:
        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file) or {}
            top = yaml_config.get("llm_configs", {})  
            out = {}

            for provider_name, provider_data in top.items():
                sub_dict = {}
                for llm_key, llm_conf in provider_data.items():
                    if not llm_conf:
                        llm_conf = {}

                    if llm_conf.get("model_id") is None and "model" in llm_conf:
                        llm_conf["model_id"] = llm_conf["model"]

                    merged = {
                        "api_key": os.getenv("OPENAI_API_KEY", llm_conf.get("api_key")),
                        "base_url": llm_conf.get("base_url"),
                        "model_id": llm_conf.get("model_id"), 
                        "max_new_tokens": llm_conf.get("max_new_tokens", llm_conf.get("max_tokens", 512)),
                        "temperature": float(os.getenv("TEMPERATURE", llm_conf.get("temperature", 1.0))),
                        "top_p": float(os.getenv("TOP_P", llm_conf.get("top_p", 1.0))),
                        "repetition_penalty": float(os.getenv("REPETITION_PENALTY", llm_conf.get("repetition_penalty", 1.0))),
                    }
                    sub_dict[llm_key] = merged

                out[provider_name] = sub_dict
            return out

    except Exception as e:
        logger.error(f"Failed to load config file: {e}", exc_info=True)
        raise RuntimeError(f"Error loading configuration: {e}")
