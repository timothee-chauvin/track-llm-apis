import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("track-llm-apis")
ROOT_DIR = Path(__file__).parent.parent.parent


class Config:
    logger = logger
    root_dir = ROOT_DIR
    db_path = ROOT_DIR / "db" / "llm_logprobs.db"
    plots_dir = ROOT_DIR / "plots"
    # Prompts to send to the smaller list of endpoints
    prompts = [
        "x ",
        "x " * 5,
        "x " * 20,
        "Let's generate random words! Only output the words, no other text. Continue the list: Underpay\nPolicy\nRisotto\nIdealist",
        "Let's generate random words! Only output the words, no other text. Continue the list: Sinuous\nCornbread\nStipulate\nOverreact",
        "reply in one token. 1+1=",
        # Random prompts
        # 2 characters
        "]\n",
        "HB",
        "e|",
        # 4 characters
        "xég",
        "\x04B\x02z",
        "\x1e·T",
        # 8 characters
        "\x06P\x1dz\x13ZTq",
        "ZZ\x17˚p|[",
        "\x14\x1ap88V?_",
    ]
    # Prompts to send to both the smaller and the extended lists of endpoints
    prompts_extended = [
        "x",
    ]
    max_completion_tokens = 1
    top_logprobs = {
        "openai": 20,
        "grok": 8,
        "openrouter": 20,
    }
    top_logprobs_openrouter = {
        # Default is 20, but these providers have a lower limit.
        "fireworks": 5,
        "azure": 5,
        "xai": 8,
    }
    api_seed = 1  # note: the Grok API refuses a seed of 0, says it must be positive
    max_retries = 15
    extended_endpoints_max_cost = 30  # sum of input and output costs per Mtok
