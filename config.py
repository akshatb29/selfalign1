# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Load API Key
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_KEY:
    raise EnvironmentError("Please set the OPENROUTER_API_KEY environment variable in your .env file.")

# Define models to be used by the agents
# Using free models from OpenRouter for accessibility
FREE_MODELS = {
    "fast": "meta-llama/llama-3.3-8b-instruct:free",
    "reasoning": "deepseek/deepseek-r1-0528-qwen3-8b:free",
    "general": "meta-llama/llama-3.3-8b-instruct:free",
    "coding": "thudm/glm-4-32b:free",
    "chat": "qwen/qwen3-32b:free",
    "compact": "meta-llama/llama-3.3-8b-instruct:free"
}

# Function to create a configuration for a specific model
def make_llm_config(model_name):
    """Creates an LLM config dictionary for AutoGen."""
    return {
        "config_list": [{
            "model": model_name,
            "api_key": OPENROUTER_KEY,
            "base_url": "https://openrouter.ai/api/v1"
        }],
        "timeout": 120,
        "temperature": 0.7,
    }