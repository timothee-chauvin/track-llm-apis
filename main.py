import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import openai

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("track-llm-apis")

ROOT_DIR = Path(__file__).parent

MODELS = {
    "openai": [
        "gpt-4.1-nano",
    ],
    "grok": [
        "grok-3-beta",
    ],
}

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
        for provider, models in MODELS.items():
            for model in models:
                # Replace special characters in model names to make valid table names
                table_name = self._get_table_name(model)
                cursor.execute(
                    f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                prompt TEXT NOT NULL,
                top_tokens JSON NOT NULL,
                logprobs JSON NOT NULL
                )
                """
                )
        self.conn.commit()

    def _get_table_name(self, model: str) -> str:
        # Replace special characters in model names to make valid table names
        return f"model_{model.replace('-', '_').replace('.', '_')}"

    def store_result(self, model: str, tokens: list[str], logprobs: list[float]):
        table_name = self._get_table_name(model)
        cursor = self.conn.cursor()
        date_str = datetime.now().isoformat()
        cursor.execute(
            f"INSERT INTO {table_name} (date, prompt, top_tokens, logprobs) VALUES (?, ?, ?, ?)",
            (
                date_str,
                PROMPT,
                json.dumps(tokens),
                json.dumps(logprobs),
            ),
        )
        self.conn.commit()
        logger.info(f"Stored results for {model} at {date_str}")

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


async def query_model(provider: str, model: str, db_manager: DatabaseManager) -> None:
    if provider == "openai":
        client = OpenAIClient()
    elif provider == "grok":
        client = GrokClient()
    else:
        logger.error(f"Unsupported provider: {provider}")
        return

    tokens, logprobs = await client.query(model)

    if tokens and logprobs:
        db_manager.store_result(model, tokens, logprobs)
    else:
        logger.error(f"Failed to get results for {model}")


async def main_async():
    db_manager = DatabaseManager()

    try:
        tasks = [
            query_model(provider, model, db_manager)
            for provider, models in MODELS.items()
            for model in models
        ]
        await asyncio.gather(*tasks)
    finally:
        db_manager.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
