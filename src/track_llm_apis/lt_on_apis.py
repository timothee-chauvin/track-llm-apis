import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import orjson
import plotly.graph_objects as go
from pydantic import BaseModel
from scipy.signal import find_peaks
from tqdm import tqdm

from track_llm_apis.analyze import get_db_data, get_db_table_names, get_token_logprobs
from track_llm_apis.config import config, logger
from track_llm_apis.sampling.analyze_logprobs import (
    logprob_time_series,
)
from track_llm_apis.sampling.common import TwoSampleTestResultWithDate
from track_llm_apis.util import slugify, trim_to_length

n_per_test = 10
peak_threshold = 5e-4
# minimum distance between detected changes
peak_distance = n_per_test
# peaks must have at least this width
peak_width = 3
peak_rel_height = 1.0
pvalue_b = 100000
prompt = "x"


class LTOnAPIData(BaseModel):
    n_per_test: int
    pvalue_b: int
    # mapping from table name to list of hypothesis test results
    data: dict[str, list[TwoSampleTestResultWithDate]]


def _get_filename(n_per_test: int, pvalue_b: int) -> str:
    return f"lt_on_apis_n_per_test={n_per_test}_b={pvalue_b}.json"


def lt_on_apis(
    prompt: str, tables: list[str] | None = None, overwrite: bool = False
) -> dict[str, list[TwoSampleTestResultWithDate]]:
    """Compute pvalues on all supplied tables (or all tables if `tables` is None).
    Store them in a filename derived from n_per_test and pvalue_b. p-values for tables already in the cache file are used, unless `overwrite` is True."""
    cache_dir = config.plots_dir / "time_series_lt" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # TODO improve caching: no need for all the tables to be the same, should pick already cached tables and compute the others
    filepath = cache_dir / _get_filename(n_per_test, pvalue_b)
    if filepath.exists():
        logger.info(f"Loading hypothesis test results from cache file {filepath}...")
        cache_results = LTOnAPIData.model_validate(orjson.loads(filepath.read_bytes()))
    else:
        cache_results = LTOnAPIData(n_per_test=n_per_test, pvalue_b=pvalue_b, data={})
    if tables is None:
        tables = get_db_table_names(prompt=prompt)
        tables = [t for t in tables if "ft:gpt" not in t]

    if overwrite:
        tables_to_process = tables
        logger.info(f"{overwrite=}, recomputing all {len(tables_to_process)} tables.")
    else:
        tables_to_process = [t for t in tables if t not in cache_results.data.keys()]
        logger.info(
            f"{len(tables) - len(tables_to_process)}/{len(tables)} tables already in cache, processing {len(tables_to_process)} new tables."
        )

    if tables_to_process:
        data = get_db_data(tables=tables_to_process, prompt=prompt, sort_by_date=True)
        for i, (table_name, table_data) in tqdm(enumerate(data.items()), total=len(data)):
            if len(table_data) < 2 * n_per_test:
                logger.info(
                    f"Skipping table {table_name} with {len(table_data)} samples (less than 2 * n_per_test={2 * n_per_test})"
                )
                continue
            print(f"{i + 1}/{len(data)} {table_name} ({len(table_data)} samples)")
            cache_results.data[table_name] = logprob_time_series(
                table_data, n_per_test, pvalue_b=pvalue_b
            )

    logger.info(f"Saving hypothesis test results to cache file {filepath}...")
    with open(filepath, "wb") as f:
        f.write(orjson.dumps(cache_results.model_dump(mode="json")))

    return {k: v for k, v in cache_results.data.items() if k in tables}


def plot_threshold_analysis(tables: list[str] | None = None):
    """Create a plot showing the total number of detections and days with detections depending on the p-value threshold."""
    all_results = lt_on_apis(prompt=prompt, tables=tables)

    total_endpoints = len(all_results)
    total_tests = sum(len(r) for r in all_results.values())

    # Test different thresholds
    thresholds = np.array([10**exp * mult for exp in range(-10, -1) for mult in [1, 5]])
    n_detections = []
    n_unique_days = []

    for threshold in thresholds:
        total_detections = 0
        total_unique_days = 0
        for table_results in all_results.values():
            dates_with_detections = set()
            for result in table_results:
                if result.pvalue is not None and result.pvalue < threshold:
                    total_detections += 1
                    dates_with_detections.add(result.date.date())
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


def plot_pvalue_histogram(tables: list[str] | None = None):
    """Create a histogram of p-values across all endpoints."""
    all_results = lt_on_apis(prompt=prompt, tables=tables)
    total_endpoints = len(all_results)
    total_tests = sum(len(r) for r in all_results.values())

    all_pvalues = []
    for table_results in all_results.values():
        for result in table_results:
            if result.pvalue is not None:
                if result.pvalue == 0.0:
                    all_pvalues.append(10)
                else:
                    all_pvalues.append(-np.log10(result.pvalue))

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=all_pvalues,
            xbins=dict(start=0, end=11, size=0.1),
            name="P-value Histogram",
        )
    )
    # label each bar
    bin_edges = np.arange(0, 11.1, 0.1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _ = np.histogram([p for p in all_pvalues], bins=bin_edges)

    for mid, count in zip(bin_mids, counts):
        if count > 0:
            fig.add_annotation(
                x=mid,
                y=count,
                text=f"{10 ** (-mid):.1g}",
                showarrow=False,
                yshift=20,
                textangle=-90,
            )
    fig.update_xaxes(title="-log10(p-value)", dtick=1)
    fig.update_yaxes(title="Count")
    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        title=f"p-value histogram ({total_endpoints} endpoints analyzed, {total_tests} tests)",
    )
    fig.write_image(
        config.plots_dir / "paper" / "lt_on_apis_pvalue_histogram.pdf", width=1200, height=800
    )


def plot_top_token_logprobs_over_time(
    after: datetime | None = None,
    before: datetime | None = None,
    prompt: str | None = None,
    tables: list[str] | None = None,
    with_lt_results: bool = False,
    peak_threshold: float = peak_threshold,
):
    """Plot logprobs of top tokens over time for each prompt in each table.

    Args:
        after: Only plot data after this date.
        before: Only plot data before this date.
        prompt: Only plot data for this prompt.
        tables: Only plot data for these tables.
        with_lt_results: If True, include results of hypothesis tests.
        pvalue_threshold: Threshold below which pvalues are considered to indicate a change in the LLM API.
    """
    if with_lt_results:
        assert prompt is not None, "Prompt must be specified when with_lt_results is True"
        lt_results = lt_on_apis(prompt=prompt, tables=tables)

    data = get_db_data(after=after, before=before, tables=tables, prompt=prompt, sort_by_date=True)
    if with_lt_results:
        time_series_dir = config.plots_dir / "time_series_lt"
    else:
        time_series_dir = config.plots_dir / "time_series"
    os.makedirs(time_series_dir, exist_ok=True)

    n_plots = sum(len(set(row.prompt for row in table_rows)) for table_rows in data.values())

    pbar = tqdm(total=n_plots)
    for table_name in data.keys():
        rows = data[table_name]

        # Group rows by prompt
        prompt_groups = defaultdict(list)
        for row in rows:
            prompt_groups[row.prompt].append(row)

        # Create a plot for each prompt
        if prompt:
            if prompt not in prompt_groups:
                logger.info(f"Prompt {prompt} not found in {table_name}")
                continue
            prompt_groups = {prompt: prompt_groups[prompt]}
        for p, prompt_rows in prompt_groups.items():
            prompt_dir = time_series_dir / slugify(p, max_length=50, hash_length=8)
            os.makedirs(prompt_dir, exist_ok=True)

            all_token_logprobs = get_token_logprobs(prompt_rows, p, missing_policy="none")

            # Create the plot
            fig = go.Figure()

            # Get a fixed, sorted order of tokens
            sorted_tokens = sorted(all_token_logprobs.keys())

            # Add a line for each top token
            for token in sorted_tokens:
                token_logprobs = all_token_logprobs[token]
                fig.add_trace(
                    go.Scatter(
                        x=token_logprobs.dates,
                        y=token_logprobs.logprobs,
                        mode="lines+markers",
                        name=f'"{token}"',
                        line=dict(width=2),
                        marker=dict(size=4),
                    )
                )

            # Add p-values on secondary y-axis
            if with_lt_results and table_name in lt_results:
                pvalue_dates = [test_result.date for test_result in lt_results[table_name]]
                pvalues = np.array([test_result.pvalue for test_result in lt_results[table_name]])
                log_pvalues = -np.log10(pvalues + 1e-100)
                threshold_score = -np.log10(peak_threshold)
                scores_clipped = np.clip(log_pvalues - threshold_score, 0, None)

                # Calculate rolling average
                window_size = 1
                rolling_avg = []
                for i in range(len(pvalues)):
                    start_idx = max(0, i - window_size + 1)
                    rolling_avg.append(sum(pvalues[start_idx : i + 1]) / (i - start_idx + 1))

                fig.add_trace(
                    go.Scatter(
                        x=pvalue_dates,
                        y=rolling_avg,
                        mode="lines",
                        name="p-value",
                        line=dict(width=2, color="rgba(255, 0, 0, 0.5)"),
                        yaxis="y2",
                    )
                )

                pvalue_peaks = find_peaks(
                    scores_clipped,
                    height=1e-20,  # anything above 0
                    distance=peak_distance,
                    width=peak_width,
                    rel_height=peak_rel_height,
                )

                # Add vertical lines for detected changes
                for peak_idx in pvalue_peaks[0]:
                    peak_date = pvalue_dates[peak_idx]
                    fig.add_vline(
                        x=peak_date,
                        line_color="black",
                        line_width=1,
                        line_dash="dash",
                    )

            # Update layout
            title_suffix = f" (after {after.isoformat()})" if after else ""
            # Truncate prompt for title if it's too long
            prompt_preview = repr(trim_to_length(p, 50))
            fig.update_layout(
                title=f"Top Token Logprobs Over Time - {table_name}{title_suffix}<br>Prompt: {prompt_preview}",
                font_family=config.plotting.font_family,
                font_size=14,
                xaxis_title="Time",
                yaxis_title="Log Probability",
                yaxis2=dict(
                    type="log", exponentformat="e", title="p-value", overlaying="y", side="right"
                ),
                template=config.plotting.template,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.07),
            )

            # Save the plot
            stub = table_name.replace("/", "_").replace("#", "_")
            filename_suffix = f"_after_{after.strftime('%Y%m%d_%H%M%S')}" if after else ""
            fig_dir = prompt_dir
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = fig_dir / f"{stub}_logprobs_over_time{filename_suffix}.pdf"
            fig.write_image(fig_path, width=1200, height=800)
            fig.write_html(fig_path.with_suffix(".html"))
            logger.info(
                f"Saved logprobs over time for {table_name} (prompt start: {repr(p[:40])}) to {fig_path}"
            )
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
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
    # lt_on_apis(prompt=prompt, tables=tables, overwrite=False)
    # plot_threshold_analysis(tables=tables)
    # plot_pvalue_histogram(tables=tables)
    plot_top_token_logprobs_over_time(
        tables=tables,
        prompt=prompt,
        with_lt_results=True,
        peak_threshold=peak_threshold,
    )
