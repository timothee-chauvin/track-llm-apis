# from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import torch

from track_llm_apis.analyze import ResponseData, get_db_data
from track_llm_apis.config import config
from track_llm_apis.sampling.analyze_logprobs import (
    logprob_time_series,
)
from track_llm_apis.sampling.common import TwoSampleTestResult

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
    torch.cuda.empty_cache()
    return detections


def lt_on_apis(tables: list[str] | None = None) -> dict[str, list[tuple[int, TwoSampleTestResult]]]:
    prompt = "x"
    data = get_db_data(tables=tables, prompt=prompt, sort_by_date=True)
    all_results = {}
    for table_name, table_data in data.items():
        if "ft:gpt" in table_name:
            continue
        all_results[table_name] = logprob_time_series(table_data, n_per_test, pvalue_b=pvalue_b)
    return all_results


def plot_threshold_analysis(tables: list[str] | None = None):
    """Create a plot showing the total number of detections and days with detections depending on the p-value threshold."""
    prompt = "x"
    data = get_db_data(tables=tables, prompt=prompt, sort_by_date=True)

    all_results = lt_on_apis(tables=tables)

    total_endpoints = len(all_results)
    total_tests = sum(len(r) for r in all_results.values())

    # Test different thresholds
    thresholds = np.array([10**exp * mult for exp in range(-10, -1) for mult in [1, 5]])
    n_detections = []
    n_unique_days = []

    for threshold in thresholds:
        total_detections = 0
        total_unique_days = 0
        for table_name, results in all_results.items():
            table_data = data[table_name]
            dates_with_detections = set()
            for i, result in results:
                if result.pvalue is not None and result.pvalue < threshold:
                    total_detections += 1
                    dates_with_detections.add(table_data[i].date.date())
            total_unique_days += len(dates_with_detections)
        n_detections.append(total_detections)
        n_unique_days.append(total_unique_days)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=n_detections,
            mode="lines+markers",
            name="Total Detections",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=n_unique_days,
            mode="lines+markers",
            name="Total Unique Days with Detections",
        )
    )
    fig.update_xaxes(type="log", exponentformat="e", title="p-value threshold", dtick=1)
    fig.update_yaxes(type="log", rangemode="nonnegative")
    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        title=f"Impact of p-value threshold on detections ({total_endpoints} endpoints analyzed, {total_tests} tests)",
    )
    fig.write_image(
        config.plots_dir / "paper" / "lt_on_apis_threshold_analysis.pdf", width=1200, height=800
    )


if __name__ == "__main__":
    # mp.set_start_method("spawn")
    tables = [
        # tables with known changes
        "openrouter#meta-llama/llama-3.1-70b-instruct#lambda/fp8",
        "openrouter#x-ai/grok-3-mini#xai",
        "openrouter#deepseek/deepseek-chat-v3-0324#nebius/fp8",
        # 15 random tables
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
    # lt_on_apis(tables=tables)
    plot_threshold_analysis(tables=tables)
