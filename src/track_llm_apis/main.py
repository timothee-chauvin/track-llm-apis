import asyncio
import base64
import json
import os
import random
import sqlite3
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp
import fire
from dotenv import load_dotenv
from openai import AsyncOpenAI

from track_llm_apis.config import Config
from track_llm_apis.util import gather_with_concurrency

logger = Config.logger

load_dotenv()


@dataclass
class Endpoint:
    source: str
    name: str
    provider: str | None = None
    dtype: str | None = None
    max_logprobs: int | None = None
    cost: tuple[float, float] | None = None
    seed: int | None = None

    def get_max_logprobs(self) -> int:
        if self.max_logprobs is None:
            return Config.top_logprobs[self.source]
        return self.max_logprobs

    def __str__(self) -> str:
        seed_str = f"seed={self.seed}" if self.seed is not None else None
        potential_items = [self.source, self.name, self.provider, self.dtype, seed_str]
        items = [item for item in potential_items if item]
        return "#".join(items)


@dataclass
class Response:
    endpoint: Endpoint
    prompt: str
    tokens: list[str]
    logprobs: list[float]
    cost: float
    system_fingerprint: str | None = None
    error: str | None = None


ENDPOINTS = [
    Endpoint("openai", "gpt-4o-mini", cost=(0.15, 0.60)),
    Endpoint("openai", "gpt-4o", cost=(2.5, 10)),
    Endpoint("openai", "gpt-4.1", cost=(2, 8)),
    Endpoint("openai", "gpt-4.1-mini", cost=(0.4, 1.6)),
    Endpoint("openai", "gpt-4.1-nano", cost=(0.1, 0.4)),
    Endpoint("openai", "gpt-3.5-turbo-0125", cost=(0.5, 1.5)),
    Endpoint("grok", "grok-3-beta", cost=(3, 15)),
    Endpoint("grok", "grok-3-fast-beta", cost=(5, 25)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "Kluster", cost=(0.33, 1.4)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "Lambda", "fp8", cost=(0.34, 0.88)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "Nebius", "fp8", cost=(0.5, 1.5)),
    Endpoint("openrouter", "qwen/qwen3-14b", "Nebius", "fp8", cost=(0.08, 0.24)),
    Endpoint("openrouter", "qwen/qwen3-32b", "Nebius", "fp8", cost=(0.1, 0.3)),
    Endpoint("openrouter", "meta-llama/llama-3.3-70b-instruct", "Nebius", "fp8", cost=(0.13, 0.4)),
    Endpoint(
        "openrouter", "meta-llama/llama-3.3-70b-instruct", "Kluster", "fp8", cost=(0.07, 0.33)
    ),
    Endpoint("openrouter", "google/gemma-3-27b-it", "Kluster", cost=(0.1, 0.25)),
    Endpoint("openrouter", "google/gemma-3-27b-it", "Nebius", cost=(0.1, 0.3)),
    # Finetuned OpenAI models
    Endpoint("openai", "ft:gpt-4.1-nano-2025-04-14:personal:try-1:BZeUJpHW", cost=(0.2, 0.8)),
    Endpoint(
        "openai", "ft:gpt-4.1-nano-2025-04-14:personal:try-1-1epoch:BZebw08b", cost=(0.2, 0.8)
    ),
    Endpoint("openai", "ft:gpt-4.1-mini-2025-04-14:personal:try-1:BZefwmPw", cost=(0.8, 3.2)),
    Endpoint("openai", "ft:gpt-4.1-2025-04-14:personal:try-1:BZfWb0GC", cost=(3, 12)),
]

ENDPOINTS_WITH_SEED = []
for endpoint in ENDPOINTS:
    create_seed_version = any(
        (
            endpoint.source == "openai",
            endpoint.name == "grok-3-beta",  # not the fast-beta which is too costly
            # None of our openrouter models return a system fingerprint, so seed likely isn't taken into account
        )
    )
    if create_seed_version:
        endpoint_with_seed = deepcopy(endpoint)
        endpoint_with_seed.seed = Config.api_seed
        ENDPOINTS_WITH_SEED.append(endpoint_with_seed)


ENDPOINTS = ENDPOINTS + ENDPOINTS_WITH_SEED


def cost_per_year(n_input_tokens: int, queries_per_day: int):
    cost_by_endpoint = {}
    for endpoint in ENDPOINTS:
        cost_by_endpoint[str(endpoint)] = (
            (endpoint.cost[0] * n_input_tokens + endpoint.cost[1] * Config.max_completion_tokens)
            / 1e6
            * queries_per_day
            * 365
        )
    total_cost = sum(cost_by_endpoint.values())
    for endpoint, cost in sorted(cost_by_endpoint.items(), key=lambda x: x[1], reverse=True):
        print(f"{endpoint}: ${cost:.2f} ({cost / total_cost * 100:.2f}%)")

    print(f"\nTotal cost: ${total_cost:.2f}/year")


def compute_cost(usage: dict, endpoint: Endpoint) -> float:
    return (
        usage["prompt_tokens"] * endpoint.cost[0] / 1e6
        + usage["completion_tokens"] * endpoint.cost[1] / 1e6
    )


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
            logprobs JSON NOT NULL,
            system_fingerprint TEXT,
            seed INTEGER
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
            f'INSERT INTO "{table_name}" (date, prompt, top_tokens, logprobs, system_fingerprint, seed) VALUES (?, ?, ?, ?, ?, ?)',
            (
                date_str,
                base64.b64encode(response.prompt.encode()).decode(),
                json.dumps(response.tokens),
                json.dumps(response.logprobs),
                response.system_fingerprint,
                response.endpoint.seed,
            ),
        )
        self.conn.commit()
        logger.info(f"Stored results for {response.endpoint}")

    def close(self):
        self.conn.close()


class OpenAIClient:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def query(self, endpoint: Endpoint, prompt: str) -> Response:
        try:
            response = await self.client.chat.completions.create(
                model=endpoint.name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=Config.max_completion_tokens,
                logprobs=True,
                top_logprobs=endpoint.get_max_logprobs(),
                temperature=0,
                seed=endpoint.seed,
            )

            cost = compute_cost(response.usage.to_dict(), endpoint)

            # Extract logprobs for the first token
            if response.choices and response.choices[0].logprobs.content:
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                tokens = [logprob.token for logprob in logprobs]
                probs = [logprob.logprob for logprob in logprobs]
                return Response(endpoint, prompt, tokens, probs, cost, response.system_fingerprint)

            logger.error(f"No logprobs returned for {endpoint}")
            return Response(endpoint, prompt, [], [], cost, error="No logprobs returned")
        except Exception as e:
            logger.error(f"Error querying {endpoint}: {e}")
            return Response(endpoint, prompt, [], [], cost, error=str(e))


class GrokClient:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("GROK_API_KEY"), base_url="https://api.x.ai/v1")

    async def query(self, endpoint: Endpoint, prompt: str) -> Response:
        try:
            response = await self.client.chat.completions.create(
                model=endpoint.name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=Config.max_completion_tokens,
                logprobs=True,
                top_logprobs=endpoint.get_max_logprobs(),
                temperature=0,
                seed=endpoint.seed,
            )

            cost = compute_cost(response.usage.to_dict(), endpoint)

            # Extract logprobs for the first token
            if response.choices and response.choices[0].logprobs.content:
                logprobs = response.choices[0].logprobs.content[0].top_logprobs
                tokens = [logprob.token for logprob in logprobs]
                probs = [logprob.logprob for logprob in logprobs]
                return Response(endpoint, prompt, tokens, probs, cost, response.system_fingerprint)

            logger.error(f"No logprobs returned for {endpoint}")
            return Response(endpoint, prompt, [], [], cost, error="No logprobs returned")
        except Exception as e:
            logger.error(f"Error querying {endpoint}: {e}")
            return Response(endpoint, prompt, [], [], cost, error=str(e))


async def retry_with_exponential_backoff(
    func: Callable[..., Awaitable[Any]],
    *args,
    max_retries: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (aiohttp.ClientError, asyncio.TimeoutError),
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
    **kwargs,
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay between retries
        jitter: Whether to add random jitter to delay
        retryable_exceptions: Tuple of exception types that should trigger retries
        retryable_status_codes: HTTP status codes that should trigger retries
        **kwargs: Keyword arguments for the function
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            return result
        except aiohttp.ClientResponseError as e:
            # Check if this is a retryable HTTP status code
            if e.status in retryable_status_codes and attempt < max_retries:
                wait_time = min(max_delay, (base_delay * (2**attempt)))
                if jitter:
                    wait_time *= random.uniform(0.9, 1.1)

                logger.warning(
                    f"HTTP {e.status} error. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                await asyncio.sleep(wait_time)
                last_exception = e
                continue
            else:
                raise e
        except retryable_exceptions as e:
            if attempt < max_retries:
                wait_time = min(max_delay, (base_delay * (2**attempt)))
                if jitter:
                    wait_time *= random.uniform(0.9, 1.1)

                logger.warning(
                    f"Retryable error: {e}. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                await asyncio.sleep(wait_time)
                last_exception = e
                continue
            else:
                raise e
        except Exception as e:
            # Non-retryable exceptions are re-raised immediately
            logger.error(f"Non-retryable error: {e}")
            raise e

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected end of retry loop")


class OpenRouterClient:
    async def _make_request(self, endpoint: Endpoint, prompt: str) -> Response:
        request_data = {
            "model": endpoint.name,
            "messages": [{"role": "user", "content": prompt}],
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
        if endpoint.seed:
            request_data["seed"] = endpoint.seed

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"},
                json=request_data,
            ) as resp:
                if not resp.ok:
                    error_text = await resp.text()
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message=f"HTTP {resp.status}: {error_text}",
                    )
                response = await resp.json()

        cost = compute_cost(response["usage"], endpoint)

        # Extract logprobs for the first token
        if response["choices"] and response["choices"][0]["logprobs"]:
            logprobs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            tokens = [logprob["token"] for logprob in logprobs]
            probs = [logprob["logprob"] for logprob in logprobs]
            return Response(
                endpoint, prompt, tokens, probs, cost, response.get("system_fingerprint", None)
            )

        logger.error(f"No logprobs returned for {endpoint}")
        return Response(endpoint, prompt, [], [], cost, error="No logprobs returned")

    async def query(self, endpoint: Endpoint, prompt: str) -> Response:
        try:
            return await retry_with_exponential_backoff(
                self._make_request, endpoint, prompt, max_retries=Config.max_retries
            )
        except Exception as e:
            logger.error(f"Error querying {endpoint} after {Config.max_retries} retries: {e}")
            return Response(endpoint, prompt, [], [], 0.0, error=str(e))


async def query_endpoint(
    endpoint: Endpoint, prompt: str, db_manager: DatabaseManager | None = None
) -> Response:
    if endpoint.source == "openai":
        client = OpenAIClient()
    elif endpoint.source == "grok":
        client = GrokClient()
    elif endpoint.source == "openrouter":
        client = OpenRouterClient()
    else:
        raise ValueError(f"Unsupported source: {endpoint.source}")

    response = await client.query(endpoint, prompt)
    if response.tokens and response.logprobs:
        if db_manager:
            db_manager.store_result(response)
        else:
            print(f"INSERT INTO {str(endpoint)}")
            print(f"  date={datetime.now().isoformat()}")
            print(f"  prompt={base64.b64encode(prompt.encode()).decode()}")
            print(f"  top_tokens={json.dumps(response.tokens)}")
            print(f"  logprobs={json.dumps(response.logprobs)}")
            print(f"  system_fingerprint={response.system_fingerprint}")
            print(f"  seed={endpoint.seed}")
            print()

    return response


async def main_async(num_iterations: int, delay: float, no_db: bool = False):
    if not no_db:
        db_manager = DatabaseManager()
    else:
        db_manager = None

    # Query all endpoints every delay seconds
    max_workers = 10
    try:
        for i in range(num_iterations):
            logger.info(f"Query iteration {i + 1}/{num_iterations}")

            tasks = [
                query_endpoint(endpoint, prompt, db_manager)
                for prompt in Config.prompts
                for endpoint in ENDPOINTS
            ]
            logger.info(f"{len(tasks)} requests to send")
            responses = await gather_with_concurrency(max_workers, *tasks)
            costs = {str(response.endpoint): response.cost for response in responses}
            logger.info("Costs breakdown:")
            logger.info(json.dumps(costs, indent=2))
            total_cost = sum(costs.values())
            logger.info(f"Total cost: ${total_cost:.2e} ({(total_cost * 100):.2f} cents)")
            logger.info(f"Total cost per year assuming 1 every hour: ${total_cost * 24 * 365:.2e}")
            logger.info(
                f"Total errors: {sum(1 for response in responses if response.error)}/{len(responses)}"
            )

            if i < num_iterations - 1:  # Don't wait after the last iteration
                await asyncio.sleep(delay)
    finally:
        if db_manager:
            db_manager.close()


def main(num_iterations: int = 1, delay: float = 10, no_db: bool = False):
    asyncio.run(main_async(num_iterations, delay, no_db=no_db))


if __name__ == "__main__":
    fire.Fire(main)
    # cost_per_year(20, 34)
