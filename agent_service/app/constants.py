import os
from dotenv import load_dotenv

load_dotenv()

# Check environment configs
DOWNSTREAM_SERVER = os.getenv("VLLM_DOWNSTREAM_HOST")
if not DOWNSTREAM_SERVER:
    raise ValueError("VLLM_DOWNSTREAM_HOST environment variable is not set.")

API_BASE_URL = f"{DOWNSTREAM_SERVER}/v1"
API_KEY = os.getenv("OPENAI_API_KEY")

# Model & hyperparams
LLM_ID = os.getenv("LLM_ID", "meta-llama/Llama-3.1-8B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
TOP_P = float(os.getenv("TOP_P", 1.0))
TEMPERATURE = float(os.getenv("TEMPERATURE", 1.0))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.0))
