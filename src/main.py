import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime

import aiohttp
from openai import AsyncOpenAI

from config import Config

logger = Config.logger


@dataclass
class Endpoint:
    source: str
    name: str
    provider: str | None = None
    dtype: str | None = None
    max_logprobs: int | None = None
    cost: tuple[float, float] | None = None

    def get_max_logprobs(self) -> int:
        if self.max_logprobs is None:
            return Config.top_logprobs[self.source]
        return self.max_logprobs

    def __str__(self) -> str:
        return "#".join([self.source, self.name, self.provider or "", self.dtype or ""]).strip("#")


@dataclass
class Response:
    endpoint: Endpoint
    tokens: list[str]
    logprobs: list[float]
    cost: float


ENDPOINTS = [
    Endpoint("openai", "gpt-4o-mini", cost=(0.15, 0.60)),
    Endpoint("openai", "gpt-4o", cost=(2.5, 10)),
    Endpoint("openai", "gpt-4.1", cost=(2, 8)),
    Endpoint("openai", "gpt-4.1-mini", cost=(0.4, 1.6)),
    Endpoint("openai", "gpt-4.1-nano", cost=(0.1, 0.4)),
    # Endpoint("openai", "gpt-4-turbo", cost=(10, 30)),
    # Endpoint("openai", "gpt-4", cost=(30, 60)),
    Endpoint("openai", "gpt-3.5-turbo-0125", cost=(0.5, 1.5)),
    Endpoint("grok", "grok-3-beta", cost=(3, 15)),
    Endpoint("grok", "grok-3-fast-beta", cost=(5, 25)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "Kluster", cost=(0.33, 1.4)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "Lambda", "fp8", cost=(0.34, 0.88)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "Nebius", "fp8", cost=(0.5, 1.5)),
    Endpoint("openrouter", "qwen/qwen3-14b", "Nebius", "fp8", cost=(0.08, 0.24)),
    Endpoint("openrouter", "qwen/qwen3-32b", "Nebius", "fp8", cost=(0.1, 0.3)),
    Endpoint("openrouter", "microsoft/phi-3.5-mini-128k-instruct", "Nebius", cost=(0.03, 0.09)),
    Endpoint("openrouter", "meta-llama/llama-3.3-70b-instruct", "Nebius", "fp8", cost=(0.13, 0.4)),
    Endpoint(
        "openrouter", "meta-llama/llama-3.3-70b-instruct", "Kluster", "fp8", cost=(0.07, 0.33)
    ),
]


def compute_cost(usage: dict, endpoint: Endpoint) -> float:
    return (
        usage["prompt_tokens"] * endpoint.cost[0] / 1e6
        + usage["completion_tokens"] * endpoint.cost[1] / 1e6
    )


async def gather_with_concurrency(n, *coros):
    # Taken from https://stackoverflow.com/a/61478547
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


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
        return str(endpoint)

    def store_result(self, response: Response):
        table_name = self._get_table_name(response.endpoint)
        cursor = self.conn.cursor()
        date_str = datetime.now().isoformat()
        cursor.execute(
            f'INSERT INTO "{table_name}" (date, prompt, top_tokens, logprobs) VALUES (?, ?, ?, ?)',
            (
                date_str,
                Config.prompt,
                json.dumps(response.tokens),
                json.dumps(response.logprobs),
            ),
        )
        self.conn.commit()
        logger.info(f"Stored results for {response.endpoint}")

    def close(self):
        self.conn.close()


class OpenAIClient:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def query(self, endpoint: Endpoint) -> Response:
        try:
            response = await self.client.chat.completions.create(
                model=endpoint.name,
                messages=[{"role": "user", "content": Config.prompt}],
                max_completion_tokens=Config.max_completion_tokens,
                logprobs=True,
                top_logprobs=endpoint.get_max_logprobs(),
                temperature=0,
            )

            cost = compute_cost(response.usage.to_dict(), endpoint)

            # Extract logprobs for the first token
            if response.choices and response.choices[0].logprobs.content:
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                tokens = [logprob.token for logprob in logprobs]
                probs = [logprob.logprob for logprob in logprobs]
                return Response(endpoint, tokens, probs, cost)

            logger.error(f"No logprobs returned for {endpoint}")
            return Response(endpoint, [], [], cost)
        except Exception as e:
            logger.error(f"Error querying {endpoint}: {e}")
            return Response(endpoint, [], [], cost)


class GrokClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("GROK_API_KEY"), base_url="https://api.x.ai/v1")

    async def query(self, endpoint: Endpoint) -> Response:
        try:
            response = await self.client.chat.completions.create(
                model=endpoint.name,
                messages=[{"role": "user", "content": Config.prompt}],
                max_completion_tokens=Config.max_completion_tokens,
                logprobs=True,
                top_logprobs=endpoint.get_max_logprobs(),
                temperature=0,
            )

            cost = compute_cost(response.usage.to_dict(), endpoint)

            # Extract logprobs for the first token
            if response.choices and response.choices[0].logprobs.content:
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                tokens = [logprob.token for logprob in logprobs]
                probs = [logprob.logprob for logprob in logprobs]
                return Response(endpoint, tokens, probs, cost)

            logger.error(f"No logprobs returned for {endpoint}")
            return Response(endpoint, [], [], cost)
        except Exception as e:
            logger.error(f"Error querying {endpoint}: {e}")
            return Response(endpoint, [], [], cost)


class OpenRouterClient:
    async def query(self, endpoint: Endpoint) -> Response:
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
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                    json=request_data,
                ) as resp:
                    response = await resp.json()

            cost = compute_cost(response["usage"], endpoint)

            # Extract logprobs for the first token
            if response["choices"] and response["choices"][0]["logprobs"]:
                logprobs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                tokens = [logprob["token"] for logprob in logprobs]
                probs = [logprob["logprob"] for logprob in logprobs]
                return Response(endpoint, tokens, probs, cost)

            logger.error(f"No logprobs returned for {endpoint}")
            return Response(endpoint, [], [], cost)
        except Exception as e:
            logger.error(f"Error querying {endpoint}: {e}")
            return Response(endpoint, [], [], cost)


async def query_endpoint(endpoint: Endpoint, db_manager: DatabaseManager) -> Response:
    if endpoint.source == "openai":
        client = OpenAIClient()
    elif endpoint.source == "grok":
        client = GrokClient()
    elif endpoint.source == "openrouter":
        client = OpenRouterClient()
    else:
        raise ValueError(f"Unsupported source: {endpoint.source}")

    response = await client.query(endpoint)
    if response.tokens and response.logprobs:
        db_manager.store_result(response)
    return response


async def main_async():
    db_manager = DatabaseManager()

    try:
        tasks = [query_endpoint(endpoint, db_manager) for endpoint in ENDPOINTS]
        responses = await gather_with_concurrency(5, *tasks)
        costs = {str(response.endpoint): response.cost for response in responses}
        logger.info("Costs breakdown:")
        logger.info(json.dumps(costs, indent=2))
        total_cost = sum(costs.values())
        logger.info(f"Total cost: ${total_cost:.2e} ({(total_cost * 100):.2f} cents)")
    finally:
        db_manager.close()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
