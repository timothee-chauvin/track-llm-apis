import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime

import openai
import requests

from config import Config

logger = Config.logger


@dataclass
class Endpoint:
    source: str
    name: str
    provider: str | None = None
    dtype: str | None = None
    max_logprobs: int | None = None

    def get_max_logprobs(self) -> int:
        if self.max_logprobs is None:
            return Config.top_logprobs[self.source]
        return self.max_logprobs


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
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "Kluster"),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "Lambda", "fp8"),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "Nebius", "fp8"),
]


class DatabaseManager:
    def __init__(self):
        if not Config.db_path.exists():
            Config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(Config.db_path)
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
                Config.prompt,
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

    async def query(self, endpoint: Endpoint) -> tuple[list[str], list[float]]:
        try:
            response = self.client.chat.completions.create(
                model=endpoint.name,
                messages=[{"role": "user", "content": Config.prompt}],
                max_completion_tokens=Config.max_completion_tokens,
                logprobs=True,
                top_logprobs=endpoint.get_max_logprobs(),
                temperature=0,
            )

            # Extract logprobs for the first token
            if response.choices and response.choices[0].logprobs.content:
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                tokens = [logprob.token for logprob in logprobs]
                probs = [logprob.logprob for logprob in logprobs]
                return tokens, probs

            logger.error(f"No logprobs returned for {endpoint}")
            return [], []
        except Exception as e:
            logger.error(f"Error querying {endpoint}: {e}")
            return [], []


class GrokClient:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("GROK_API_KEY"), base_url="https://api.x.ai/v1"
        )

    async def query(self, endpoint: Endpoint) -> tuple[list[str], list[float]]:
        try:
            response = self.client.chat.completions.create(
                model=endpoint.name,
                messages=[{"role": "user", "content": Config.prompt}],
                max_completion_tokens=Config.max_completion_tokens,
                logprobs=True,
                top_logprobs=endpoint.get_max_logprobs(),
                temperature=0,
            )

            # Extract logprobs for the first token
            if response.choices and response.choices[0].logprobs.content:
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                tokens = [logprob.token for logprob in logprobs]
                probs = [logprob.logprob for logprob in logprobs]
                return tokens, probs

            logger.error(f"No logprobs returned for {endpoint}")
            return [], []
        except Exception as e:
            logger.error(f"Error querying {endpoint}: {e}")
            return [], []


class OpenRouterClient:
    async def query(self, endpoint: Endpoint) -> tuple[list[str], list[float]]:
        request_data = {
            "model": endpoint.name,
            "messages": [{"role": "user", "content": Config.prompt}],
            "max_completion_tokens": Config.max_completion_tokens,
            "logprobs": True,
            "top_logprobs": endpoint.get_max_logprobs(),
            "temperature": 0,
            "provider": {
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        if endpoint.provider:
            request_data["provider"]["only"] = [endpoint.provider]
        if endpoint.dtype:
            request_data["provider"]["quantizations"] = [endpoint.dtype]

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                data=json.dumps(request_data),
            )

            # Extract logprobs for the first token
            response = response.json()
            if response["choices"] and response["choices"][0]["logprobs"]:
                logprobs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                tokens = [logprob["token"] for logprob in logprobs]
                probs = [logprob["logprob"] for logprob in logprobs]
                return tokens, probs

            logger.error(f"No logprobs returned for {endpoint}")
            return [], []
        except Exception as e:
            logger.error(f"Error querying {endpoint}: {e}")
            return [], []


async def query_endpoint(endpoint: Endpoint, db_manager: DatabaseManager) -> None:
    if endpoint.source == "openai":
        client = OpenAIClient()
    elif endpoint.source == "grok":
        client = GrokClient()
    elif endpoint.source == "openrouter":
        client = OpenRouterClient()
    else:
        logger.error(f"Unsupported source: {endpoint.source}")
        return

    tokens, logprobs = await client.query(endpoint)
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
