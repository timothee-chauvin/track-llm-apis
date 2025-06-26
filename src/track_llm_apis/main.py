import asyncio
import base64
import json
import os
import sqlite3
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import fire
from dotenv import load_dotenv
from openai import AsyncOpenAI

from track_llm_apis.config import Config
from track_llm_apis.util import (
    gather_with_concurrency_streaming,
    retry_with_exponential_backoff,
    trim_to_length,
)

logger = Config.logger

load_dotenv()


@dataclass
class Endpoint:
    source: str
    name: str
    provider: str | None = None
    max_logprobs: int | None = None
    cost: tuple[float, float] | None = None
    seed: int | None = None

    def get_max_logprobs(self) -> int:
        if self.max_logprobs is None:
            if self.source == "openrouter":
                for provider_prefix in Config.top_logprobs_openrouter.keys():
                    if self.provider.lower().startswith(provider_prefix.lower()):
                        return Config.top_logprobs_openrouter[provider_prefix]
            return Config.top_logprobs[self.source]
        return self.max_logprobs

    def __str__(self) -> str:
        seed_str = f"seed={self.seed}" if self.seed is not None else None
        potential_items = [self.source, self.name, self.provider, seed_str]
        items = [item for item in potential_items if item]
        return "#".join(items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Endpoint):
            return False
        return (
            self.source == other.source
            and self.name == other.name
            and self.provider == other.provider
            and self.seed == other.seed
        )


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
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "klusterai", cost=(0.33, 1.4)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "lambda/fp8", cost=(0.34, 0.88)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "nebius/fp8", cost=(0.5, 1.5)),
    Endpoint("openrouter", "qwen/qwen3-14b", "nebius/fp8", cost=(0.08, 0.24)),
    Endpoint("openrouter", "qwen/qwen3-32b", "nebius/fp8", cost=(0.1, 0.3)),
    Endpoint("openrouter", "meta-llama/llama-3.3-70b-instruct", "nebius/fp8", cost=(0.13, 0.4)),
    Endpoint("openrouter", "meta-llama/llama-3.3-70b-instruct", "klusterai/fp8", cost=(0.07, 0.33)),
    Endpoint("openrouter", "google/gemma-3-27b-it", "klusterai", cost=(0.1, 0.25)),
    Endpoint("openrouter", "google/gemma-3-27b-it", "nebius/fp8", cost=(0.1, 0.3)),
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

# Output of generate_endpoints.py
ENDPOINTS_EXTENDED = [
    Endpoint("openrouter", "agentica-org/deepcoder-14b-preview:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "arliai/qwq-32b-arliai-rpr-v1:free", "chutes", cost=(0.0, 0.0)),
    Endpoint(
        "openrouter", "cognitivecomputations/dolphin3.0-mistral-24b:free", "chutes", cost=(0.0, 0.0)
    ),
    Endpoint(
        "openrouter",
        "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
        "chutes",
        cost=(0.0, 0.0),
    ),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324:free", "chutes/fp8", cost=(0.0, 0.0)),
    Endpoint("openrouter", "deepseek/deepseek-chat:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "deepseek/deepseek-r1-0528-qwen3-8b:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "deepseek/deepseek-r1-0528:free", "chutes", cost=(0.0, 0.0)),
    Endpoint(
        "openrouter", "deepseek/deepseek-r1-distill-llama-70b:free", "chutes", cost=(0.0, 0.0)
    ),
    Endpoint("openrouter", "google/gemma-2-9b-it:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "google/gemma-3-12b-it:free", "chutes/bf16", cost=(0.0, 0.0)),
    Endpoint("openrouter", "google/gemma-3-27b-it:free", "chutes/bf16", cost=(0.0, 0.0)),
    Endpoint("openrouter", "meta-llama/llama-4-maverick:free", "chutes/fp8", cost=(0.0, 0.0)),
    Endpoint("openrouter", "meta-llama/llama-4-scout:free", "chutes/bf16", cost=(0.0, 0.0)),
    Endpoint("openrouter", "minimax/minimax-m1:extended", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "mistralai/devstral-small:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "mistralai/mistral-nemo:free", "chutes", cost=(0.0, 0.0)),
    Endpoint(
        "openrouter", "mistralai/mistral-small-24b-instruct-2501:free", "chutes", cost=(0.0, 0.0)
    ),
    Endpoint(
        "openrouter", "mistralai/mistral-small-3.1-24b-instruct:free", "chutes", cost=(0.0, 0.0)
    ),
    Endpoint(
        "openrouter", "mistralai/mistral-small-3.2-24b-instruct:free", "chutes", cost=(0.0, 0.0)
    ),
    Endpoint("openrouter", "moonshotai/kimi-dev-72b:free", "chutes/bf16", cost=(0.0, 0.0)),
    Endpoint(
        "openrouter",
        "nousresearch/deephermes-3-llama-3-8b-preview:free",
        "chutes/bf16",
        cost=(0.0, 0.0),
    ),
    Endpoint(
        "openrouter", "nvidia/llama-3.3-nemotron-super-49b-v1:free", "chutes/bf16", cost=(0.0, 0.0)
    ),
    Endpoint("openrouter", "qwen/qwen-2.5-72b-instruct:free", "chutes/bf16", cost=(0.0, 0.0)),
    Endpoint("openrouter", "qwen/qwen-2.5-coder-32b-instruct:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "qwen/qwen3-14b:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "qwen/qwen3-235b-a22b:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "qwen/qwen3-30b-a3b:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "qwen/qwen3-32b:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "qwen/qwen3-8b:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "qwen/qwq-32b:free", "chutes/bf16", cost=(0.0, 0.0)),
    Endpoint("openrouter", "rekaai/reka-flash-3:free", "chutes/bf16", cost=(0.0, 0.0)),
    Endpoint("openrouter", "sarvamai/sarvam-m:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "shisa-ai/shisa-v2-llama3.3-70b:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "thudm/glm-4-32b:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "thudm/glm-z1-32b:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "tngtech/deepseek-r1t-chimera:free", "chutes", cost=(0.0, 0.0)),
    Endpoint("openrouter", "mistralai/mistral-nemo", "klusterai", cost=(0.01, 0.013)),
    Endpoint("openrouter", "meta-llama/llama-3.2-3b-instruct", "lambda/bf16", cost=(0.015, 0.025)),
    Endpoint(
        "openrouter", "meta-llama/llama-3.1-8b-instruct", "klusterai/fp8", cost=(0.016, 0.023)
    ),
    Endpoint("openrouter", "meta-llama/llama-3.1-8b-instruct", "nebius/fp8", cost=(0.02, 0.06)),
    Endpoint("openrouter", "meta-llama/llama-guard-3-8b", "nebius", cost=(0.02, 0.06)),
    Endpoint("openrouter", "meta-llama/llama-3.1-8b-instruct", "lambda/bf16", cost=(0.025, 0.04)),
    Endpoint(
        "openrouter", "nousresearch/hermes-2-pro-llama-3-8b", "lambda/bf16", cost=(0.025, 0.04)
    ),
    Endpoint("openrouter", "mistralai/mistral-nemo", "nebius/fp8", cost=(0.04, 0.12)),
    Endpoint(
        "openrouter", "meta-llama/llama-3.2-11b-vision-instruct", "lambda/bf16", cost=(0.05, 0.05)
    ),
    Endpoint(
        "openrouter", "mistralai/mistral-small-24b-instruct-2501", "klusterai", cost=(0.05, 0.09)
    ),
    Endpoint(
        "openrouter", "mistralai/mistral-small-3.1-24b-instruct", "nebius/fp8", cost=(0.05, 0.15)
    ),
    Endpoint("openrouter", "meta-llama/llama-3.3-70b-instruct", "klusterai/fp8", cost=(0.05, 0.19)),
    Endpoint("openrouter", "qwen/qwen-2.5-coder-32b-instruct", "nebius/fp8", cost=(0.06, 0.18)),
    Endpoint("openrouter", "qwen/qwen-2.5-coder-32b-instruct", "lambda/bf16", cost=(0.07, 0.16)),
    Endpoint("openrouter", "qwen/qwen3-14b", "nebius/fp8", cost=(0.08, 0.24)),
    Endpoint("openrouter", "meta-llama/llama-4-scout", "lambda/fp8", cost=(0.08, 0.3)),
    Endpoint("openrouter", "meta-llama/llama-4-scout", "klusterai", cost=(0.08, 0.45)),
    Endpoint("openrouter", "mistralai/pixtral-12b", "hyperbolic/bf16", cost=(0.1, 0.1)),
    Endpoint("openrouter", "google/gemma-3-27b-it", "klusterai", cost=(0.1, 0.18)),
    Endpoint("openrouter", "google/gemma-3-27b-it", "nebius/fp8", cost=(0.1, 0.3)),
    Endpoint("openrouter", "microsoft/phi-4", "nebius/fp8", cost=(0.1, 0.3)),
    Endpoint("openrouter", "qwen/qwen3-30b-a3b", "nebius/fp8", cost=(0.1, 0.3)),
    Endpoint("openrouter", "qwen/qwen3-32b", "lambda/fp8", cost=(0.1, 0.3)),
    Endpoint("openrouter", "qwen/qwen3-32b", "nebius/base", cost=(0.1, 0.3)),
    Endpoint("openrouter", "openai/gpt-4.1-nano", "openai", cost=(0.1, 0.4)),
    Endpoint("openrouter", "meta-llama/llama-3.1-70b-instruct", "lambda/fp8", cost=(0.12, 0.3)),
    Endpoint("openrouter", "meta-llama/llama-3.3-70b-instruct", "lambda/fp8", cost=(0.12, 0.3)),
    Endpoint("openrouter", "nousresearch/hermes-3-llama-3.1-70b", "lambda/fp8", cost=(0.12, 0.3)),
    Endpoint(
        "openrouter", "nvidia/llama-3.1-nemotron-70b-instruct", "lambda/fp8", cost=(0.12, 0.3)
    ),
    Endpoint("openrouter", "meta-llama/llama-3.1-70b-instruct", "nebius/fp8", cost=(0.13, 0.4)),
    Endpoint("openrouter", "meta-llama/llama-3.3-70b-instruct", "nebius/fp8", cost=(0.13, 0.4)),
    Endpoint(
        "openrouter", "nvidia/llama-3.3-nemotron-super-49b-v1", "nebius/fp8", cost=(0.13, 0.4)
    ),
    Endpoint("openrouter", "qwen/qwen-2.5-72b-instruct", "nebius/fp8", cost=(0.13, 0.4)),
    Endpoint("openrouter", "qwen/qwen3-235b-a22b", "klusterai/fp8", cost=(0.13, 2.0)),
    Endpoint("openrouter", "qwen/qwq-32b", "nebius/fp8", cost=(0.15, 0.45)),
    Endpoint("openrouter", "openai/gpt-4o-mini", "azure", cost=(0.15, 0.6)),
    Endpoint("openrouter", "openai/gpt-4o-mini", "openai", cost=(0.15, 0.6)),
    Endpoint("openrouter", "openai/gpt-4o-mini-2024-07-18", "openai", cost=(0.15, 0.6)),
    Endpoint("openrouter", "meta-llama/llama-4-maverick", "klusterai/fp8", cost=(0.15, 0.8)),
    Endpoint("openrouter", "meta-llama/llama-4-maverick", "lambda/fp8", cost=(0.18, 0.6)),
    Endpoint("openrouter", "meta-llama/llama-3.1-8b-instruct", "fireworks", cost=(0.2, 0.2)),
    Endpoint("openrouter", "meta-llama/llama-guard-3-8b", "fireworks", cost=(0.2, 0.2)),
    Endpoint("openrouter", "qwen/qwen-2.5-coder-32b-instruct", "hyperbolic/fp8", cost=(0.2, 0.2)),
    Endpoint("openrouter", "qwen/qwen-2.5-vl-7b-instruct", "hyperbolic/bf16", cost=(0.2, 0.2)),
    Endpoint(
        "openrouter",
        "sentientagi/dobby-mini-unhinged-plus-llama-3.1-8b",
        "fireworks",
        cost=(0.2, 0.2),
    ),
    Endpoint("openrouter", "deepseek/deepseek-r1-distill-llama-70b", "lambda/fp8", cost=(0.2, 0.6)),
    Endpoint("openrouter", "qwen/qwen3-235b-a22b", "nebius/fp8", cost=(0.2, 0.6)),
    Endpoint("openrouter", "qwen/qwen3-32b", "nebius/fast", cost=(0.2, 0.6)),
    Endpoint(
        "openrouter", "deepseek/deepseek-r1-distill-llama-70b", "nebius/fp8", cost=(0.25, 0.75)
    ),
    Endpoint("openrouter", "qwen/qwen2.5-vl-72b-instruct", "nebius/fp8", cost=(0.25, 0.75)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "klusterai", cost=(0.28, 1.15)),
    Endpoint("openrouter", "qwen/qwen-2.5-vl-7b-instruct", "klusterai", cost=(0.3, 0.3)),
    Endpoint("openrouter", "x-ai/grok-3-mini", "xai", cost=(0.3, 0.5)),
    Endpoint("openrouter", "x-ai/grok-3-mini-beta", "xai", cost=(0.3, 0.5)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "lambda/fp8", cost=(0.34, 0.88)),
    Endpoint(
        "openrouter", "nousresearch/hermes-3-llama-3.1-70b", "hyperbolic/fp8", cost=(0.4, 0.4)
    ),
    Endpoint("openrouter", "qwen/qwq-32b", "hyperbolic/bf16", cost=(0.4, 0.4)),
    Endpoint("openrouter", "openai/gpt-4.1-mini", "openai", cost=(0.4, 1.6)),
    Endpoint("openrouter", "deepseek/deepseek-chat", "nebius/fp8", cost=(0.5, 1.5)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "nebius/fp8", cost=(0.5, 1.5)),
    Endpoint("openrouter", "qwen/qwq-32b", "nebius/fp8", cost=(0.5, 1.5)),
    Endpoint("openrouter", "deepseek/deepseek-r1-0528", "lambda/fp8", cost=(0.5, 2.18)),
    Endpoint("openrouter", "deepseek/deepseek-r1", "lambda/fp8", cost=(0.54, 2.18)),
    Endpoint("openrouter", "qwen/qwen2.5-vl-72b-instruct", "hyperbolic", cost=(0.6, 0.6)),
    Endpoint(
        "openrouter", "nvidia/llama-3.1-nemotron-ultra-253b-v1", "nebius/fp8", cost=(0.6, 1.8)
    ),
    Endpoint("openrouter", "x-ai/grok-3-mini", "xai", cost=(0.6, 4.0)),
    Endpoint("openrouter", "x-ai/grok-3-mini-beta", "xai", cost=(0.6, 4.0)),
    Endpoint("openrouter", "meta-llama/llama-3.1-405b-instruct", "lambda/fp8", cost=(0.8, 0.8)),
    Endpoint("openrouter", "nousresearch/hermes-3-llama-3.1-405b", "lambda/fp8", cost=(0.8, 0.8)),
    Endpoint("openrouter", "deepseek/deepseek-r1", "nebius/base", cost=(0.8, 2.4)),
    Endpoint("openrouter", "deepseek/deepseek-r1-0528", "nebius/fp8", cost=(0.8, 2.4)),
    Endpoint("openrouter", "deepseek/deepseek-chat", "fireworks", cost=(0.9, 0.9)),
    Endpoint("openrouter", "deepseek/deepseek-chat-v3-0324", "fireworks", cost=(0.9, 0.9)),
    Endpoint("openrouter", "meta-llama/llama-3.1-70b-instruct", "fireworks", cost=(0.9, 0.9)),
    Endpoint("openrouter", "meta-llama/llama-3.3-70b-instruct", "fireworks/fp16", cost=(0.9, 0.9)),
    Endpoint("openrouter", "mistralai/mixtral-8x22b-instruct", "fireworks", cost=(0.9, 0.9)),
    Endpoint("openrouter", "qwen/qwen-2.5-72b-instruct", "fireworks", cost=(0.9, 0.9)),
    Endpoint("openrouter", "qwen/qwen2.5-vl-32b-instruct", "fireworks", cost=(0.9, 0.9)),
    Endpoint("openrouter", "qwen/qwq-32b", "fireworks", cost=(0.9, 0.9)),
    Endpoint("openrouter", "meta-llama/llama-3.1-405b-instruct", "nebius/fp8", cost=(1.0, 3.0)),
    Endpoint("openrouter", "nousresearch/hermes-3-llama-3.1-405b", "nebius/fp8", cost=(1.0, 3.0)),
    Endpoint("openrouter", "deepseek/deepseek-r1", "nebius/fast", cost=(2.0, 6.0)),
    Endpoint("openrouter", "openai/gpt-4.1", "openai", cost=(2.0, 8.0)),
    Endpoint("openrouter", "x-ai/grok-2-1212", "xai", cost=(2.0, 10.0)),
    Endpoint("openrouter", "x-ai/grok-2-vision-1212", "xai", cost=(2.0, 10.0)),
    Endpoint("openrouter", "deepseek/deepseek-r1-0528", "klusterai", cost=(2.5, 5.0)),
    Endpoint("openrouter", "openai/gpt-4o", "azure", cost=(2.5, 10.0)),
    Endpoint("openrouter", "openai/gpt-4o", "openai", cost=(2.5, 10.0)),
    Endpoint("openrouter", "openai/gpt-4o-2024-08-06", "azure", cost=(2.5, 10.0)),
    Endpoint("openrouter", "openai/gpt-4o-2024-08-06", "openai", cost=(2.5, 10.0)),
    Endpoint("openrouter", "openai/gpt-4o-2024-11-20", "openai", cost=(2.5, 10.0)),
    Endpoint("openrouter", "01-ai/yi-large", "fireworks", cost=(3.0, 3.0)),
    Endpoint("openrouter", "meta-llama/llama-3.1-405b-instruct", "fireworks/fp8", cost=(3.0, 3.0)),
    Endpoint("openrouter", "openai/gpt-3.5-turbo-16k", "openai", cost=(3.0, 4.0)),
    Endpoint("openrouter", "deepseek/deepseek-r1", "fireworks/fp8", cost=(3.0, 8.0)),
    Endpoint("openrouter", "deepseek/deepseek-r1-0528", "fireworks", cost=(3.0, 8.0)),
    Endpoint("openrouter", "x-ai/grok-3", "xai", cost=(3.0, 15.0)),
    Endpoint("openrouter", "x-ai/grok-3-beta", "xai", cost=(3.0, 15.0)),
    Endpoint("openrouter", "openai/chatgpt-4o-latest", "openai", cost=(5.0, 15.0)),
    Endpoint("openrouter", "openai/gpt-4o-2024-05-13", "azure", cost=(5.0, 15.0)),
    Endpoint("openrouter", "openai/gpt-4o-2024-05-13", "openai", cost=(5.0, 15.0)),
    Endpoint("openrouter", "x-ai/grok-beta", "xai", cost=(5.0, 15.0)),
    Endpoint("openrouter", "x-ai/grok-vision-beta", "xai", cost=(5.0, 15.0)),
    Endpoint("openrouter", "x-ai/grok-3", "xai/fast", cost=(5.0, 25.0)),
    Endpoint("openrouter", "x-ai/grok-3-beta", "xai", cost=(5.0, 25.0)),
    Endpoint("openrouter", "openai/gpt-4o:extended", "openai", cost=(6.0, 18.0)),
    Endpoint("openrouter", "openai/gpt-4-1106-preview", "openai", cost=(10.0, 30.0)),
    Endpoint("openrouter", "openai/gpt-4-turbo", "openai", cost=(10.0, 30.0)),
    Endpoint("openrouter", "openai/gpt-4-turbo-preview", "openai", cost=(10.0, 30.0)),
    Endpoint("openrouter", "openai/gpt-4", "azure", cost=(30.0, 60.0)),
    Endpoint("openrouter", "openai/gpt-4", "openai", cost=(30.0, 60.0)),
    Endpoint("openrouter", "openai/gpt-4-0314", "openai", cost=(30.0, 60.0)),
]

# filter out endpoints that were already in the original list
ENDPOINTS_EXTENDED = [endpoint for endpoint in ENDPOINTS_EXTENDED if endpoint not in ENDPOINTS]

# filter out endpoints that are too expensive
ENDPOINTS_EXTENDED = [
    endpoint
    for endpoint in ENDPOINTS_EXTENDED
    if endpoint.cost[0] + endpoint.cost[1] <= Config.extended_endpoints_max_cost
]


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
        for endpoint in ENDPOINTS + ENDPOINTS_EXTENDED:
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
        logger.debug(f"Stored results for {response.endpoint}")

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
            return Response(endpoint, prompt, [], [], 0.0, error=str(e))


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
            return Response(endpoint, prompt, [], [], 0.0, error=str(e))


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
    max_workers = 20
    try:
        for it in range(num_iterations):
            logger.info(f"Query iteration {it + 1}/{num_iterations}")

            tasks = [
                query_endpoint(endpoint, prompt, db_manager)
                for prompt in Config.prompts
                for endpoint in ENDPOINTS
            ]
            tasks.extend(
                [
                    query_endpoint(endpoint, prompt, db_manager)
                    for prompt in Config.prompts_extended
                    for endpoint in ENDPOINTS + ENDPOINTS_EXTENDED
                ]
            )
            logger.info(f"{len(tasks)} requests to send")
            responses = []
            i = 0
            async for response in gather_with_concurrency_streaming(max_workers, *tasks):
                responses.append(response)
                success_or_error = "SUCCESS" if not response.error else f"ERROR: {response.error}"
                logger.info(
                    f"{i + 1}/{len(tasks)}: {response.endpoint} ({repr(trim_to_length(response.prompt, 50))}) {success_or_error}"
                )
                i += 1

            costs = {
                str(response.endpoint): response.cost
                for response in sorted(responses, key=lambda x: x.cost, reverse=True)
            }
            logger.info("Costs breakdown:")
            logger.info(json.dumps(costs, indent=2))
            total_cost = sum(costs.values())
            logger.info(f"Total cost: ${total_cost:.2e} ({(total_cost * 100):.2f} cents)")
            logger.info(f"Total cost per year assuming 1 every hour: ${total_cost * 24 * 365:.2e}")
            logger.info(
                f"Total errors: {sum(1 for response in responses if response.error)}/{len(responses)}"
            )

            if it < num_iterations - 1:  # Don't wait after the last iteration
                await asyncio.sleep(delay)
    finally:
        if db_manager:
            db_manager.close()


def main(num_iterations: int = 1, delay: float = 10, no_db: bool = False):
    asyncio.run(main_async(num_iterations, delay, no_db=no_db))


if __name__ == "__main__":
    fire.Fire(main)
