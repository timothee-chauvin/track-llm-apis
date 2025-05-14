import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import openai

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("track-llm-apis")

ROOT_DIR = Path(__file__).parent


@dataclass
class Endpoint:
    source: str
    name: str
    provider: str | None = None
    dtype: str | None = None


ENDPOINTS = [
    Endpoint("openai", "gpt-4o-mini"),
    Endpoint("openai", "gpt-4o"),
    Endpoint("openai", "gpt-4.1"),
    Endpoint("openai", "gpt-4.1-mini"),
    Endpoint("openai", "gpt-4.1-nano"),
    Endpoint("openai", "gpt-4-turbo"),
    Endpoint("openai", "gpt-4"),
    Endpoint("openai", "gpt-3.5-turbo-0125"),
    Endpoint("grok", "grok-3-beta"),
    Endpoint("grok", "grok-3-fast-beta"),
]

PROMPT = "x " * 20  # Around 20 tokens
MAX_COMPLETION_TOKENS = 1
TOP_LOGPROBS = {
    "openai": 20,
    "grok": 8,
}
DB_PATH = ROOT_DIR / "db" / "llm_logprobs.db"


class DatabaseManager:
    def __init__(self):
        if not DB_PATH.exists():
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(DB_PATH)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        for endpoint in ENDPOINTS:
            table_name = self._get_table_name(endpoint)
            cursor.execute(
                f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            prompt TEXT NOT NULL,
            top_tokens JSON NOT NULL,
            logprobs JSON NOT NULL
            )
            """
            )
        self.conn.commit()

    def _get_table_name(self, endpoint: Endpoint) -> str:
        return "#".join(
            [endpoint.source, endpoint.name, endpoint.provider or "", endpoint.dtype or ""]
        ).strip("#")

    def store_result(self, endpoint: Endpoint, tokens: list[str], logprobs: list[float]):
        table_name = self._get_table_name(endpoint)
        cursor = self.conn.cursor()
        date_str = datetime.now().isoformat()
        cursor.execute(
            f'INSERT INTO "{table_name}" (date, prompt, top_tokens, logprobs) VALUES (?, ?, ?, ?)',
            (
                date_str,
                PROMPT,
                json.dumps(tokens),
                json.dumps(logprobs),
            ),
        )
        self.conn.commit()
        logger.info(f"Stored results for {endpoint} at {date_str}")

    def close(self):
        self.conn.close()


class OpenAIClient:
    def __init__(self):
        self.client = openai.OpenAI()

    async def query(self, model: str) -> tuple[list[str], list[float]]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS["openai"],
                temperature=0,
            )

            # Extract logprobs for the first token
            if response.choices and response.choices[0].logprobs.content:
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                tokens = [logprob.token for logprob in logprobs]
                probs = [logprob.logprob for logprob in logprobs]
                return tokens, probs

            logger.error(f"No logprobs returned for {model}")
            return [], []
        except Exception as e:
            logger.error(f"Error querying OpenAI {model}: {e}")
            return [], []


class GrokClient:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("GROK_API_KEY"), base_url="https://api.x.ai/v1"
        )

    async def query(self, model: str) -> tuple[list[str], list[float]]:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS["grok"],
                temperature=0,
            )

            # Extract logprobs for the first token
            if response.choices and response.choices[0].logprobs.content:
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                tokens = [logprob.token for logprob in logprobs]
                probs = [logprob.logprob for logprob in logprobs]
                return tokens, probs

            logger.error(f"No logprobs returned for {model}")
            return [], []
        except Exception as e:
            logger.error(f"Error querying Grok {model}: {e}")
            return [], []


async def query_endpoint(endpoint: Endpoint, db_manager: DatabaseManager) -> None:
    if endpoint.source == "openai":
        client = OpenAIClient()
    elif endpoint.source == "grok":
        client = GrokClient()
    else:
        logger.error(f"Unsupported source: {endpoint.source}")
        return

    tokens, logprobs = await client.query(endpoint.name)

    if tokens and logprobs:
        db_manager.store_result(endpoint, tokens, logprobs)
    else:
        logger.error(f"Failed to get results for {endpoint}")


async def main_async():
    db_manager = DatabaseManager()

    try:
        tasks = [query_endpoint(endpoint, db_manager) for endpoint in ENDPOINTS]
        await asyncio.gather(*tasks)
    finally:
        db_manager.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
