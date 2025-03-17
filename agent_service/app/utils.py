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
        current_dir = Path(__file__).resolve().parent  # e.g., agent_service/app
        repo_root = current_dir.parent.parent         # repository root
        config_path = str(repo_root / "config.yaml")

    try:
        with open(config_path, "r") as file:
            yaml_config = yaml.safe_load(file) or {}
            top = yaml_config.get("llm_configs", {})  # e.g. {'openai': {...}, 'vllm': {...}}

            # We'll build a dict like { 'openai': {...}, 'vllm': {...} }
            # Each sub-key should have 'default_llm' and 'alternate_llm'
            out = {}

            # top is: {"openai": {default_llm: { ... }, alternate_llm: { ... }}, "vllm": {...}}
            for provider_name, provider_data in top.items():
                # e.g., provider_name = "openai" or "vllm"
                # provider_data = {"default_llm": {...}, "alternate_llm": {...}}
                sub_dict = {}
                for llm_key, llm_conf in provider_data.items():
                    # e.g., llm_key = "default_llm" or "alternate_llm"
                    # llm_conf = { "api_key": null, "model": "gpt-3.5-turbo", ... }
                    if not llm_conf:
                        llm_conf = {}

                    # Merge config with environment variables
                    # If environment variable not set, fallback to the YAML
                    if llm_conf.get("model_id") is None and "model" in llm_conf:
                        # Some people store "model" for openai. You might unify or rename
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

            # out now looks like:
            # {
            #   "openai": {
            #       "default_llm": {"api_key": ..., "base_url": ..., "model_id": ..., ...},
            #       "alternate_llm": {...}
            #   },
            #   "vllm": {
            #       "default_llm": {...},
            #       "alternate_llm": {...}
            #   }
            # }

            print("Merged LLM configs:\n", out)
            return out

    except Exception as e:
        logger.error(f"Failed to load config file: {e}", exc_info=True)
        raise RuntimeError(f"Error loading configuration: {e}")
