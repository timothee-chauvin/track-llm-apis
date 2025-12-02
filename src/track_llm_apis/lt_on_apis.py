import json
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import orjson
import plotly.express as px
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

n_per_test = 24
# minimum distance between detected changes
peak_distance = n_per_test
pvalue_b = 100000
stat_sigma_threshold = 12.0
stat_running_std_window = 100
stat_exclusion_zone = 2 * n_per_test
stat_absolute_threshold = 1.0
minimum_detectable_length_days = 14
prompt = "x"
max_points_per_token = 400

random.seed(0)


class LTOnAPIData(BaseModel):
    """Data as stored to the cache file."""

    n_per_test: int
    pvalue_b: int
    # mapping from table name to list of hypothesis test results
    data: dict[str, list[TwoSampleTestResultWithDate]]


class LTOnAPICache:
    _cache_dir = config.plots_dir / "time_series_lt" / "cache"
    data: dict[str, list[TwoSampleTestResultWithDate]]

    def __init__(self, n_per_test: int, pvalue_b: int):
        self.n_per_test = n_per_test
        self.pvalue_b = pvalue_b
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = self._cache_dir / self._get_filename()
        if self._filepath.exists():
            logger.info(f"Loading hypothesis test results from cache file {self._filepath}...")
            self.data = LTOnAPIData.model_validate(orjson.loads(self._filepath.read_bytes())).data
        else:
            self.data = LTOnAPIData(n_per_test=n_per_test, pvalue_b=pvalue_b, data={}).data

    def save(self):
        logger.info(f"Saving hypothesis test results to cache file {self._filepath}...")
        to_save = LTOnAPIData(n_per_test=self.n_per_test, pvalue_b=self.pvalue_b, data=self.data)
        with open(self._filepath, "wb") as f:
            f.write(orjson.dumps(to_save.model_dump(mode="json")))

    def _get_filename(self) -> str:
        return f"lt_on_apis_n_per_test={self.n_per_test}_b={self.pvalue_b}.json"


def lt_on_apis(
    prompt: str,
    tables: list[str] | None = None,
    overwrite: bool = False,
    after: datetime | None = None,
    before: datetime | None = None,
) -> dict[str, list[TwoSampleTestResultWithDate]]:
    """Compute pvalues on all supplied tables (or all tables if `tables` is None).
    Store them in a filename derived from n_per_test and pvalue_b. p-values for tables already in the cache file are used, unless `overwrite` is True."""
    cache = LTOnAPICache(n_per_test=n_per_test, pvalue_b=pvalue_b)
    if tables is None:
        tables = get_db_table_names(prompt=prompt)
        # Drop GPT-4* finetunes
        tables = [t for t in tables if "ft:gpt" not in t]
        # Drop endpoints with seed specified
        tables = [t for t in tables if "seed=" not in t]

    if overwrite:
        tables_to_process = tables
        logger.info(f"{overwrite=}, recomputing all {len(tables_to_process)} tables.")
    else:
        tables_to_process = [t for t in tables if t not in cache.data.keys()]
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
                cache.data[table_name] = []
                continue
            print(f"{i + 1}/{len(data)} {table_name} ({len(table_data)} samples)")
            cache.data[table_name] = logprob_time_series(table_data, n_per_test, pvalue_b=pvalue_b)
            if (i + 1) % 10 == 0:
                cache.save()
        cache.save()
    result = {k: v for k, v in cache.data.items() if k in tables}
    # filter by date
    if after is not None or before is not None:
        filtered_result = {}
        for k, trs in result.items():
            filtered_tests = [
                tr
                for tr in trs
                if (after is None or tr.date >= after) and (before is None or tr.date <= before)
            ]
            filtered_result[k] = filtered_tests
        result = filtered_result
    # filter out endpoints where there is less than a month between the first time
    # detection is possible (after stat_running_std_window + stat_exclusion_zone) and the last
    filtered_result = {}
    for k, trs in result.items():
        if len(trs) <= stat_running_std_window + stat_exclusion_zone:
            logger.info(
                f"Filtering out {k}: not enough statistics for even one change detection test."
            )
            continue
        first_usable_stat = trs[stat_running_std_window + stat_exclusion_zone].date
        last_usable_stat = trs[-1].date
        if last_usable_stat - first_usable_stat < timedelta(days=minimum_detectable_length_days):
            logger.info(
                f"Filtering out {k}: less than {minimum_detectable_length_days} days between {first_usable_stat} and {last_usable_stat}."
            )
            continue
        filtered_result[k] = trs
    return filtered_result


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
    downsample: bool = True,
):
    """Plot logprobs of top tokens over time for each prompt in each table.

    Args:
        after: Only plot data after this date.
        before: Only plot data before this date.
        prompt: Only plot data for this prompt.
        tables: Only plot data for these tables.
        with_lt_results: If True, include results of hypothesis tests.
        pvalue_threshold: Threshold below which pvalues are considered to indicate a change in the LLM API.
        downsample: If True, downsample the logprob time series to at most `max_points_per_token` points per top token.
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
            time_series_start = min(lp_data.dates[0] for lp_data in all_token_logprobs.values())
            time_series_end = max(lp_data.dates[-1] for lp_data in all_token_logprobs.values())

            # Create the plot
            fig = go.Figure()

            # Get a fixed, sorted order of tokens
            sorted_tokens = sorted(all_token_logprobs.keys())

            # Add a line for each top token
            for token in sorted_tokens:
                token_logprobs = all_token_logprobs[token]
                if downsample:
                    # Keep every n points so that there are at most max_points_per_token points
                    keep_every = max(1, len(token_logprobs.dates) // max_points_per_token)
                    token_logprobs.dates = token_logprobs.dates[::keep_every]
                    token_logprobs.logprobs = token_logprobs.logprobs[::keep_every]
                fig.add_trace(
                    go.Scattergl(
                        x=token_logprobs.dates,
                        y=token_logprobs.logprobs,
                        mode="lines+markers",
                        name=f'"{token}"',
                        line=dict(width=2),
                        marker=dict(size=4),
                    )
                )

            if with_lt_results and table_name in lt_results:
                # Add vertical lines for detected changes
                dates = [test_result.date for test_result in lt_results[table_name]]
                detected_change_indices, deviations = detect_changes(lt_results[table_name])
                print(detected_change_indices, deviations)
                for peak_idx, peak_deviation in zip(detected_change_indices, deviations):
                    peak_date = dates[peak_idx]
                    y_min = min(
                        lp
                        for tl in all_token_logprobs.values()
                        for lp in tl.logprobs
                        if lp is not None
                    )
                    fig.add_shape(
                        type="line",
                        x0=peak_date,
                        x1=peak_date,
                        y0=y_min,
                        y1=0,
                        line=dict(color="black", width=2, dash="dash"),
                    )
                    # print e.g. "12.3σ" to the right of the line
                    deviation_str = (
                        f"+{peak_deviation:.1f}σ" if peak_deviation != float("inf") else "+∞ σ"
                    )
                    fig.add_annotation(
                        x=peak_date,
                        y=0,
                        text=deviation_str,
                        showarrow=False,
                        yshift=20,
                        xshift=10,
                        textangle=-45,
                        font=dict(size=16),
                    )

            title = f"Endpoint: {get_model_name(table_name)}, Provider: {get_model_provider(table_name, capitalize=True)}"
            title += f"<br>Prompt: {repr(trim_to_length(p, 50))}"
            title += f"<br>{time_series_start.strftime('%B %d')} to {time_series_end.strftime('%B %d, %Y')}"
            if downsample and keep_every > 1:
                title += f" (downsampled to every {keep_every} points)"

            fig.update_layout(
                title=title,
                font_family=config.plotting.font_family,
                font_size=14,
                xaxis_title="Time",
                yaxis_title="Log Probability",
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
            logger.info(
                f"Saved logprobs over time for {table_name} (prompt start: {repr(p[:40])}) to {fig_path}"
            )
            pbar.update(1)
    pbar.close()


def detect_changes(
    test_results: list[TwoSampleTestResultWithDate],
) -> tuple[list[int], list[float]]:
    """Return the indices of the detected changes in the test_results using the statistics only, and the corresponding number of sigmas above the running mean."""

    extended_window_size = stat_running_std_window + stat_exclusion_zone
    if len(test_results) <= extended_window_size:
        logger.warning(
            f"Not enough test results ({len(test_results)}) to detect changes with running std window {stat_running_std_window} and exclusion zone {stat_exclusion_zone}."
        )
        return ([], [])

    statistics = np.array([tr.statistic for tr in test_results])
    # threshold exceedances, in absolute units of statistic
    exceedances = np.zeros(len(statistics))
    # deviations from the running mean, in multiple of standard deviations
    deviations = np.zeros(len(statistics))

    for i in range(extended_window_size, len(statistics)):
        # Calculate mean and std over the window ending just before the current point
        window_stats = statistics[i - extended_window_size : i - stat_exclusion_zone]
        mean = np.mean(window_stats)
        std = np.std(window_stats)

        absolute_threshold = mean + stat_sigma_threshold * std
        deviation = (statistics[i] - mean) / std if std > 0 else float("inf")
        deviations[i] = deviation
        if deviation > stat_sigma_threshold:
            exceedances[i] = statistics[i] - absolute_threshold

    # Use peak detection to return only the maximal statistic within each cluster
    peaks = find_peaks(
        exceedances,
        height=1e-20,  # anything above 0
        distance=peak_distance,
    )[0]

    peaks = [p for p in peaks if statistics[p] > stat_absolute_threshold]

    return ([int(idx) for idx in peaks], [float(deviations[idx]) for idx in peaks])


def benchmark(tables: dict[str, list[str]], prompt: str):
    lt_results = lt_on_apis(prompt=prompt, tables=list(tables.keys()))
    tp = 0
    fp = 0
    fn = 0
    tp_by_table = {}
    fp_by_table = {}
    fn_by_table = {}
    for table, real_change_dates in tables.items():
        pvalue_dates = [str(test_result.date.date()) for test_result in lt_results[table]]
        detection_indices, _ = detect_changes(lt_results[table])
        detection_dates = [pvalue_dates[idx] for idx in detection_indices]

        table_tp = len([d for d in detection_dates if d in real_change_dates])
        table_fp = len([d for d in detection_dates if d not in real_change_dates])
        if table_fp:
            print(
                f"Table {table}: False positives on dates {[d for d in detection_dates if d not in real_change_dates]}"
            )
        table_fn = len([d for d in real_change_dates if d not in detection_dates])
        if table_fn:
            print(
                f"Table {table}: Missed real changes on dates {[d for d in real_change_dates if d not in detection_dates]}"
            )
        tp += table_tp
        fp += table_fp
        fn += table_fn
        tp_by_table[table] = table_tp
        fp_by_table[table] = table_fp
        fn_by_table[table] = table_fn

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_by_table": tp_by_table,
        "fp_by_table": fp_by_table,
        "fn_by_table": fn_by_table,
    }


def get_model_provider(table_name: str, capitalize: bool = False) -> str:
    parts = table_name.split("#")
    if parts[0] == "openrouter":
        if parts[-1].startswith("seed="):
            last_part = parts[-2]
        else:
            last_part = parts[-1]
        provider = last_part.split("/")[0]
    elif parts[0] == "openai":
        provider = "openai"
    elif parts[0] == "grok":
        provider = "xai"
    else:
        raise ValueError(f"Unknown table name format: {table_name}")
    if capitalize:
        provider = {
            "xai": "xAI",
            "openai": "OpenAI",
            "fireworks": "Fireworks",
            "lambda": "Lambda",
            "azure": "Azure",
            "crusoe": "Crusoe",
            "nebius": "Nebius",
            "chutes": "Chutes",
            "hyperbolic": "Hyperbolic",
            "deepseek": "Deepseek",
        }[provider]
    return provider


def change_distribution_by_provider(
    prompt: str,
    tables: list[str] | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
) -> dict[str, float]:
    lt_results = lt_on_apis(prompt=prompt, tables=tables, after=after, before=before)
    # for each provider, total duration of monitoring across endpoints, in days
    provider_durations = defaultdict(float)
    # for each provider, total number of changes detected
    provider_changes = defaultdict(int)
    provider_num_endpoints = defaultdict(int)
    for table, results in lt_results.items():
        if len(results) == 0:
            continue
        prov = get_model_provider(table)
        changes, _ = detect_changes(results)
        # only count duration where changes can be detected, not initialization of the running windows
        duration = (
            results[-1].date - results[stat_running_std_window + stat_exclusion_zone].date
        ).days
        provider_durations[prov] += duration
        provider_changes[prov] += len(changes)
        provider_num_endpoints[prov] += 1

    total_endpoints = sum(provider_num_endpoints.values())
    total_changes = sum(provider_changes.values())
    total_duration = sum(provider_durations.values())
    total_change_rate = total_changes / total_duration * 365
    print(
        f"TOTAL ({total_endpoints} endpoints): {total_changes} changes over {total_duration / 365:.1f} years = {total_change_rate:.2f} changes/year"
    )

    provider_change_rates = {
        prov: provider_changes[prov] / provider_durations[prov] * 365 for prov in provider_durations
    }

    for prov in sorted(
        provider_change_rates.keys(), key=lambda p: provider_change_rates[p], reverse=True
    ):
        # print(
        #     f"{provider} ({provider_num_endpoints[provider]} endpoints): {provider_changes[provider]} changes over {provider_durations[provider] / 365:.1f} years = {change_rate:.2f} changes/year"
        # )

        print(
            f"{prov} & {provider_num_endpoints[prov]} & {provider_durations[prov] / 365:.1f} & {provider_changes[prov]} & {provider_change_rates[prov]:.1f} \\\\"
        )
    return provider_change_rates


def get_endpoints_with_changes(
    prompt: str,
    tables: list[str] | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
) -> list[str]:
    """Print info on which endpoints have or don't have detected changes Return the list of endpoints with at least one change."""
    lt_results = lt_on_apis(prompt=prompt, tables=tables, after=after, before=before)
    changes_per_endpoint = {}
    for table, results in lt_results.items():
        changes, _ = detect_changes(results)
        changes_per_endpoint[table] = len(changes)
    n_with_changes = sum(1 for n in changes_per_endpoint.values() if n > 0)
    percent_with_changes = n_with_changes / len(changes_per_endpoint) * 100
    n_changes = sum(n for n in changes_per_endpoint.values())
    total = len(changes_per_endpoint)
    for endpoint, endpoint_changes in sorted(changes_per_endpoint.items(), key=lambda x: x[1]):
        print(f"{endpoint}: {endpoint_changes} changes")
    print(
        f"{n_with_changes}/{total} endpoints with changes ({percent_with_changes:.1f}%), total {n_changes} changes"
    )
    return [endpoint for endpoint, changes in changes_per_endpoint.items() if changes > 0]


def change_dates(prompt: str, tables: list[str] | None = None) -> dict[str, dict[str, list[str]]]:
    lt_results = lt_on_apis(prompt=prompt, tables=tables)
    providers = set(get_model_provider(t) for t in lt_results.keys())
    result_dict = {provider: {} for provider in providers}
    assert all(d.tzinfo is not None for d in [tr.date for trs in lt_results.values() for tr in trs])
    for table, table_results in lt_results.items():
        provider = get_model_provider(table)
        pvalue_dates = [str(test_result.date) for test_result in table_results]
        detection_indices, _ = detect_changes(table_results)
        detection_dates = [pvalue_dates[idx] for idx in detection_indices]
        result_dict[provider][table] = [date for date in detection_dates]
    with open(config.plots_dir / "time_series_lt" / "change_dates_lt_on_apis.json", "w") as f:
        json.dump(result_dict, f, indent=2)
    return result_dict


def get_model_name(table: str) -> str:
    parts = table.split("#")
    if parts[0] == "openrouter":
        model_name = parts[1]
    else:
        model_name = parts[0]
    return model_name


def plot_change_dates(
    prompt: str,
    tables: list[str] | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
):
    """Plot detected change dates for each endpoint, with dots colored by provider."""
    lt_results = lt_on_apis(prompt=prompt, tables=tables, after=after, before=before)

    # Collect data for plotting
    plot_data = []
    endpoint_ranges = {}  # Store first and last sample dates for each endpoint

    for table, table_results in lt_results.items():
        provider = get_model_provider(table)
        model_name = get_model_name(table)
        pvalue_dates = [test_result.date for test_result in table_results]
        detection_indices, _ = detect_changes(table_results)
        if not detection_indices:
            continue
        detection_dates = [pvalue_dates[idx] for idx in detection_indices]

        # Store the date range for this endpoint
        if pvalue_dates:
            endpoint_ranges[table] = {
                "first": pvalue_dates[stat_running_std_window + stat_exclusion_zone],
                "last": max(pvalue_dates),
                "provider": provider,
            }

        for date in detection_dates:
            plot_data.append(
                {
                    "endpoint": table,
                    "model": model_name,
                    "date": date,
                    "provider": provider,
                }
            )

    providers = sorted(set(d["provider"] for d in endpoint_ranges.values()))

    # Sort endpoints: first by provider, then by first date, then by last date
    endpoints = sorted(
        endpoint_ranges.keys(),
        key=lambda m: (
            providers.index(endpoint_ranges[m]["provider"]),
            endpoint_ranges[m]["first"],
            endpoint_ranges[m]["last"],
        ),
    )

    # Create model to y-position mapping
    endpoint_to_y = {endpoint: i for i, endpoint in enumerate(endpoints)}

    # Get global date range
    all_dates = [d["date"] for d in plot_data]
    for ranges in endpoint_ranges.values():
        all_dates.extend([ranges["first"], ranges["last"]])
    global_min = min(all_dates)
    global_max = max(all_dates)

    fig = go.Figure()

    # Add grey rectangles for regions outside each endpoint's observation window
    for endpoint, ranges in endpoint_ranges.items():
        y = endpoint_to_y[endpoint]
        # Left grey zone
        fig.add_shape(
            type="rect",
            x0=global_min,
            x1=ranges["first"],
            y0=y - 0.55,
            y1=y + 0.55,
            fillcolor="lightgrey",
            line_width=0,
            layer="below",
        )
        # Right grey zone
        fig.add_shape(
            type="rect",
            x0=ranges["last"],
            x1=global_max,
            y0=y - 0.55,
            y1=y + 0.55,
            fillcolor="lightgrey",
            line_width=0,
            layer="below",
        )

    # Add traces for each provider
    plotly_colors = px.colors.qualitative.Plotly
    for i, provider in enumerate(providers):
        color = plotly_colors[i % len(plotly_colors)]
        provider_data = [d for d in plot_data if d["provider"] == provider]
        fig.add_trace(
            go.Scatter(
                x=[d["date"] for d in provider_data],
                y=[endpoint_to_y[d["endpoint"]] for d in provider_data],
                mode="markers",
                name=provider,
                marker=dict(
                    size=14,
                    color=color,
                    line=dict(width=1, color="dimgrey"),
                ),
            )
        )

    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        font_size=18,
        xaxis=dict(
            range=[global_min, global_max],
            gridcolor="lightgrey",
            tickfont=dict(size=20),
        ),
        yaxis=dict(
            tickmode="array",
            range=[len(endpoints) - 0.5, -0.5],
            tickvals=list(range(len(endpoints))),
            ticktext=[get_model_name(e) for e in endpoints],
            ticklabelstandoff=10,
            tickfont=dict(size=16),
            gridcolor="lightgrey",
        ),
        legend=dict(
            title="Provider",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=22),
        ),
        height=max(600, len(endpoints) * 12),
        margin=dict(l=300),  # Space for long model names
    )

    output_path = config.plots_dir / "time_series_lt" / "change_dates_plot.pdf"
    fig.write_image(output_path, width=1200, height=max(600, len(endpoints) * 10))
    fig.write_html(output_path.with_suffix(".html"))
    logger.info(f"Saved change dates plot to {output_path}")


def get_random_endpoint_per_provider(
    prompt: str,
    tables: list[str] | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
    with_changes_only: bool = False,
) -> list[str]:
    """Return a random endpoint for each provider."""
    lt_results = lt_on_apis(prompt=prompt, tables=tables, after=after, before=before)

    provider_endpoints: dict[str, list[str]] = defaultdict(list)

    for table in lt_results.keys():
        if not with_changes_only or detect_changes(lt_results[table])[0]:
            provider = get_model_provider(table)
            provider_endpoints[provider].append(table)

    random_endpoints = [random.choice(endpoints) for endpoints in provider_endpoints.values()]

    return random_endpoints


if __name__ == "__main__":
    tables = {
        # tables with known changes
        "openrouter#qwen/qwen3-32b#nebius/base": [],
        "openrouter#meta-llama/llama-3.1-70b-instruct#lambda/fp8": [
            "2025-07-15",
            "2025-07-30",
            "2025-07-31",
            "2025-08-11",
        ],
        # "openrouter#x-ai/grok-3-mini#xai",  # ambiguous
        "openrouter#deepseek/deepseek-chat-v3-0324#nebius/fp8": [],
        # 15 random tables
        "openrouter#deepseek/deepseek-chat-v3-0324#crusoe/fp8": [],
        # "openrouter#openai/gpt-3.5-turbo#openai",  # ambiguous
        # "openrouter#deepseek/deepseek-r1-0528#crusoe/fp8": ["2025-10-01", "2025-10-07"],  # too ambiguous for now
        "openrouter#x-ai/grok-4#xai": ["2025-09-02"],
        "openrouter#cognitivecomputations/dolphin3.0-r1-mistral-24b#chutes": [],
        # "openrouter#google/gemma-2-9b-it#chutes",  # ambiguous
        "openrouter#arliai/qwq-32b-arliai-rpr-v1#chutes": [],
        "openrouter#deepseek/deepseek-chat-v3-0324#hyperbolic/fp8": ["2025-08-01"],
        "openrouter#mistralai/mistral-nemo#chutes": [],
        "openrouter#agentica-org/deepcoder-14b-preview#chutes": [],
        # "openrouter#deepseek/deepseek-r1-0528-qwen3-8b#chutes":,  # nonsensical logprobs for one sample, creates false positives
        "openrouter#mistralai/mistral-small-3.1-24b-instruct#chutes": ["2025-10-09"],
        "openrouter#qwen/qwen3-32b#chutes": [],
        "openrouter#mistralai/devstral-small-2505#chutes": ["2025-09-06"],
        "openrouter#mistralai/mistral-small-24b-instruct-2501#chutes": [],
    }
    after = datetime(2025, 6, 27, tzinfo=ZoneInfo("Europe/Paris"))
    # after = datetime(2025, 8, 1, tzinfo=ZoneInfo("Europe/Paris"))
    before = datetime(2025, 11, 1, tzinfo=ZoneInfo("Europe/Paris"))
    # lt_on_apis(prompt=prompt, tables=tables, overwrite=False)
    # plot_threshold_analysis(tables=tables)
    # plot_pvalue_histogram(tables=None)
    # plot_top_token_logprobs_over_time(
    #     tables=list(tables.keys()),
    #     prompt=prompt,
    #     with_lt_results=True,
    #     after=after,
    # )
    # results = benchmark(tables=tables, prompt=prompt)
    # print(json.dumps(results, indent=2))
    # print(
    #     f"{n_per_test=}, {pvalue_b=}, {stat_sigma_threshold=} {stat_running_std_window=}:\nTP={results['tp']}, FP={results['fp']}, FN={results['fn']}"
    # )

    # print(
    #     json.dumps(
    #         change_distribution_by_provider(prompt=prompt, tables=None, after=after), indent=2
    #     )
    # )
    # print(
    #     json.dumps(
    #         change_distribution_by_provider(
    #             prompt=prompt, tables=list(tables.keys()), after=after, before=before
    #         ),
    #         indent=2,
    #     )
    # )
    # endpoints_with_changes = get_endpoints_with_changes(prompt=prompt, tables=None, after=after)
    endpoints_with_changes = get_endpoints_with_changes(prompt=prompt, tables=None, after=after)
    # change_dates_dict = change_dates(prompt=prompt, tables=None)
    # print(json.dumps(change_dates_dict, indent=2))
    # plot_change_dates(prompt=prompt, tables=None, after=after, before=before)
    # plot_change_dates(prompt=prompt, tables=None, after=after)

    # random_tables = get_random_endpoint_per_provider(prompt=prompt, tables=None, after=after)
    # endpoints_with_changes = get_random_endpoint_per_provider(
    #     prompt=prompt, tables=None, after=after, with_changes_only=True
    # )
    plot_top_token_logprobs_over_time(
        tables=endpoints_with_changes,
        prompt=prompt,
        with_lt_results=True,
        after=after,
    )
