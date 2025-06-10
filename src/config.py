import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("track-llm-apis")
ROOT_DIR = Path(__file__).parent.parent


class Config:
    logger = logger
    root_dir = ROOT_DIR
    db_path = ROOT_DIR / "db" / "llm_logprobs.db"
    plots_dir = ROOT_DIR / "plots"
    prompts = [
        "x " * 20,  # Around 20 tokens
        "Let's generate random words! Only output the words, no other text. Continue the list: Underpay\nPolicy\nRisotto\nIdealist",
    ]
    max_completion_tokens = 1
    top_logprobs = {
        "openai": 20,
        "grok": 8,
        "openrouter": 20,
    }
    seed = 1  # note: the Grok API refuses a seed of 0, says it must be positive
