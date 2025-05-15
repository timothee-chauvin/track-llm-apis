import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("track-llm-apis")
ROOT_DIR = Path(__file__).parent.parent


class Config:
    logger = logger
    db_path = ROOT_DIR / "db" / "llm_logprobs.db"
    prompt = "x " * 20  # Around 20 tokens
    max_completion_tokens = 1
    top_logprobs = {
        "openai": 20,
        "grok": 8,
        "openrouter": 20,
    }
