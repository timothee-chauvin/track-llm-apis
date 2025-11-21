# from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import torch

from track_llm_apis.analyze import ResponseData, get_db_data
from track_llm_apis.sampling.analyze_logprobs import (
    logprob_time_series,
)

n_per_test = 10
pvalue_threshold = 0.005
pvalue_b = 2000


def lt_on_table(
    table_name: str, data: list[ResponseData]
) -> list[tuple[int, datetime, float, float]]:
    detections = []
    results = logprob_time_series(data, n_per_test, pvalue_b=pvalue_b)
    for i, result in results:
        if result.pvalue is not None and result.pvalue < pvalue_threshold:
            detections.append((i, data[i].date, result.statistic, result.pvalue))
    unique_days = set(date.date() for _, date, _, _ in detections)
    print(f"Detected {len(detections)} changes, on {len(unique_days)} unique days, in {table_name}")
    print(f"Unique days with detections: {len(unique_days)}")
    return detections


def lt_on_apis(tables: list[str] | None = None):
    prompt = "x"
    data = get_db_data(tables=tables, prompt=prompt, sort_by_date=True)
    for table_name, table_data in data.items():
        print(f"Table: {table_name} ({len(table_data)} samples)")
        lt_on_table(table_name, table_data)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # mp.set_start_method("spawn")
    tables = [
        # tables with known changes
        "openrouter#meta-llama/llama-3.1-70b-instruct#lambda/fp8",
        "openrouter#x-ai/grok-3-mini#xai",
        "openrouter#deepseek/deepseek-chat-v3-0324#nebius/fp8",
        # # 15 random tables
        "openrouter#deepseek/deepseek-chat-v3-0324#crusoe/fp8",
        "openrouter#openai/gpt-3.5-turbo#openai",
        "openrouter#deepseek/deepseek-r1-0528#crusoe/fp8",
        "openrouter#deepseek/deepseek-chat-v3-0324#hyperbolic/fp8",
        "openrouter#x-ai/grok-4#xai",
        "openrouter#cognitivecomputations/dolphin3.0-r1-mistral-24b#chutes",
        "openrouter#google/gemma-2-9b-it#chutes",
        "openrouter#arliai/qwq-32b-arliai-rpr-v1#chutes",
        "openrouter#mistralai/mistral-nemo#chutes",
        "openrouter#agentica-org/deepcoder-14b-preview#chutes",
        "openrouter#deepseek/deepseek-r1-0528-qwen3-8b#chutes",
        "openrouter#mistralai/mistral-small-3.1-24b-instruct#chutes",
        "openrouter#qwen/qwen3-32b#chutes",
        "openrouter#mistralai/devstral-small-2505#chutes",
        "openrouter#mistralai/mistral-small-24b-instruct-2501#chutes",
    ]
    lt_on_apis(tables=tables)
