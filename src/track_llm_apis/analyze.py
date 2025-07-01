import base64
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.offline
import plotly.tools as tls
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import linregress, shapiro
from tqdm import tqdm

from track_llm_apis.config import Config
from track_llm_apis.util import trim_to_length

logger = Config.logger

model_types = {
    "gpt-4o-mini": "?",
    "gpt-4o": "?",
    "gpt-4.1": "?",
    "gpt-4.1-mini": "?",
    "gpt-4.1-nano": "?",
    "gpt-4-turbo": "?",
    "gpt-4": "moe",
    "gpt-3.5-turbo-0125": "?",
    "grok-3-beta": "moe",
    "grok-3-fast-beta": "moe",
    "deepseek/deepseek-chat-v3-0324": "moe",
    "qwen/qwen3-14b": "dense",
    "qwen/qwen3-32b": "dense",
    "microsoft/phi-3.5-mini-128k-instruct": "dense",
    "meta-llama/llama-3.3-70b-instruct": "dense",
    "ft:gpt-4.1-nano-2025-04-14:personal:try-1:BZeUJpHW": "?",
    "ft:gpt-4.1-nano-2025-04-14:personal:try-1-1epoch:BZebw08b": "?",
    "ft:gpt-4.1-mini-2025-04-14:personal:try-1:BZefwmPw": "?",
    "ft:gpt-4.1-2025-04-14:personal:try-1:BZfWb0GC": "?",
}


@dataclass
class ResponseData:
    """Data for a single response from an LLM API."""

    date: datetime
    prompt: str
    top_tokens: list[str]
    logprobs: list[float]


@dataclass
class TokenLogprobs:
    """Logprobs for a single token and prompt. All dates are included. If the token isn't in the top_tokens for a given date, the logprob is None."""

    dates: list[datetime]
    logprobs: list[float | None]


def get_db_data(
    tables: list[str] | None = None, after: datetime | None = None
) -> dict[str, list[ResponseData]]:
    """Get data from the database.

    Args:
        tables: List of table names to get data from. If None, all tables are returned.
        after: Only return data after this date.

    Returns:
        A dict of table names to lists of ResponseData.
    """
    logger.info("Getting db data...")
    conn = sqlite3.connect(Config.db_path)
    cursor = conn.cursor()
    try:
        if tables is None:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            raw_table_names = [row[0] for row in cursor.fetchall()]
            # Filter out sqlite_sequence and other internal sqlite tables
            table_names = [name for name in raw_table_names if not name.startswith("sqlite_")]
        else:
            table_names = tables

        if not table_names:
            return {}

        select_parts = []
        for table_name in table_names:
            # Escape single quotes in table name for the string literal.
            escaped_table_name = table_name.replace("'", "''")
            select_part = f"SELECT '{escaped_table_name}' as table_name, date, prompt, top_tokens, logprobs FROM \"{table_name}\""
            if after:
                select_part += " WHERE date > ?"
            select_parts.append(select_part)

        query = " UNION ALL ".join(select_parts)

        params = []
        if after:
            params = [after.isoformat()] * len(table_names)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = defaultdict(list)
        for table_name_from_db, date, prompt, top_tokens, logprobs in tqdm(
            rows, desc="Processing DB data"
        ):
            results[table_name_from_db].append(
                ResponseData(
                    date=datetime.fromisoformat(date),
                    prompt=base64.b64decode(prompt).decode("utf-8"),
                    top_tokens=list(json.loads(top_tokens)),
                    logprobs=list(json.loads(logprobs)),
                )
            )
        return dict(results)
    except sqlite3.Error as e:
        Config.logger.error(f"An error occurred during database analysis: {e}")
        raise e
    finally:
        conn.close()


def get_token_logprobs(
    endpoint_data: list[ResponseData], prompt: str, missing_policy: Literal["min", "none"]
) -> dict[str, TokenLogprobs]:
    """
    Return a dict of all tokens found in `endpoint_data`, with the corresponding TokenLogprobs.
    For tokens that are not in the top_tokens for a given date, the logprob is set to either the
    minimum logprob for that date if missing_policy is "min", or None if missing_policy is "none".
    """
    endpoint_data = [row for row in endpoint_data if row.prompt == prompt]
    all_tokens = set(token for row in endpoint_data for token in row.top_tokens)
    token_logprob_dict = {token: TokenLogprobs(dates=[], logprobs=[]) for token in all_tokens}
    for row in endpoint_data:
        for token, logprob in zip(row.top_tokens, row.logprobs):
            token_logprob_dict[token].dates.append(row.date)
            token_logprob_dict[token].logprobs.append(logprob)
        tokens_absent_this_time = set(all_tokens) - set(row.top_tokens)
        for token in tokens_absent_this_time:
            token_logprob_dict[token].dates.append(row.date)
            match missing_policy:
                case "min":
                    token_logprob_dict[token].logprobs.append(min(row.logprobs))
                case "none":
                    token_logprob_dict[token].logprobs.append(None)
    return token_logprob_dict


def equivalence_classes(after: datetime | None = None):
    data = get_db_data(after=after)
    for table_name, rows in data.items():
        equivalence_classes = {}  # (top_tokens_tuple, logprobs_tuple) -> [dates]

        for date_str, _prompt, top_tokens, logprobs in rows:
            key = (tuple(top_tokens), tuple(logprobs))

            if key not in equivalence_classes:
                equivalence_classes[key] = []
            equivalence_classes[key].append(date_str)

        equivalence_classes = dict(
            sorted(equivalence_classes.items(), key=lambda x: len(x[1]), reverse=True)
        )

        print(f"\n# {table_name}")
        for (_, logprobs_key), dates in equivalence_classes.items():
            logprobs_display = list(logprobs_key)[:5] + ["..."]
            num_instances = len(dates)
            sorted_dates = sorted(dates)
            dates_str = ", ".join(sorted_dates)
            print(f"## {num_instances} times ({dates_str}):")
            print(f"  {logprobs_display}")


def get_top_token_logprobs(data, table_name, all_top_tokens: bool = False):
    # TODO this mixes the data from all tokens! Doesn't make any sense! Even the implementation is dumb
    # TODO and this also mixes the data from all prompts
    rows = data[table_name]
    if all_top_tokens:
        top_tokens = set(row[2][0] for row in rows)
    else:
        top_tokens = set([rows[0][2][0]])
    top_token_logprobs = []
    for top_token in top_tokens:
        for _date_str, _prompt, row_top_tokens, row_logprobs in rows:
            try:
                top_token_index = row_top_tokens.index(top_token)
                top_token_logprobs.append(row_logprobs[top_token_index])
            except ValueError:
                pass
    return top_token_logprobs


def top_logprob_variability(after: datetime | None = None):
    # TODO wrong (see get_top_token_logprobs)
    data = get_db_data(after=after)
    for table_name in data.keys():
        top_token_logprobs = get_top_token_logprobs(data, table_name, all_top_tokens=True)
        top_token_probs = [math.exp(logprob) for logprob in top_token_logprobs]

        print(f"\n# {table_name}")
        print(f"n_values: {len(top_token_logprobs)}")
        print("## logprobs")
        print(f"min: {min(top_token_logprobs)}")
        print(f"max: {max(top_token_logprobs)}")
        print(f"std: {statistics.stdev(top_token_logprobs)}")
        print("## probs")
        print(f"min: {min(top_token_probs)}")
        print(f"max: {max(top_token_probs)}")
        print(f"std: {statistics.stdev(top_token_probs)}")


def plot_prob_std(after: datetime | None = None):
    # TODO wrong (see get_top_token_logprobs)
    data = get_db_data(after=after)
    os.makedirs(Config.plots_dir, exist_ok=True)

    table_names = []
    stdevs = []
    stdev_cis = []
    model_colors = []

    # Color mapping for model types
    color_map = {"dense": "blue", "moe": "red", "?": "gray"}

    std_by_type = defaultdict(list)

    # Collect data for all tables first
    table_data = []
    for table_name in data.keys():
        print(f"\n# {table_name}")
        top_token_logprobs = get_top_token_logprobs(data, table_name, all_top_tokens=True)
        if len(top_token_logprobs) < 2:
            print("Skipped because it has less than 2 logprobs")
            continue
        top_token_probs = [math.exp(logprob) for logprob in top_token_logprobs]
        std_value = np.std(top_token_probs, ddof=1)
        std_ci = boostrap_std_ci(top_token_probs)
        print(f"std: {std_value} (CI: {std_ci})")

        model_name = table_name.split("#")[1]
        model_type = model_types[model_name]

        table_data.append((table_name, std_value, std_ci, model_type))
        std_by_type[model_type].append(std_value)

    # Sort by table name alphabetically
    table_data.sort(key=lambda x: x[0])

    # Extract sorted data for plotting
    for table_name, std_value, std_ci, model_type in table_data:
        table_names.append(table_name)
        stdevs.append(std_value)
        stdev_cis.append(std_ci)
        model_colors.append(color_map[model_type])

    print("# Average std by model type:")
    for model_type, stds in std_by_type.items():
        print(f"{model_type}: {sum(stds) / len(stds):.4f}")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=table_names,
            y=stdevs,
            marker_color=model_colors,
            text=stdevs,
            texttemplate="%{text:.4f}",
            textposition="outside",
            showlegend=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[stdev_cis[i][1] - stdevs[i] for i in range(len(stdevs))],
                arrayminus=[stdevs[i] - stdev_cis[i][0] for i in range(len(stdevs))],
                visible=True,
            ),
        )
    )

    for model_type, color in color_map.items():
        fig.add_trace(
            go.Bar(x=[None], y=[None], marker_color=color, name=model_type, showlegend=True)
        )

    title_suffix = f" (after {after.isoformat()})" if after else ""
    fig.update_layout(
        title=f"Standard Deviation of Top Token Probabilities by Model{title_suffix}",
        xaxis_title="Model",
        yaxis_title="Standard Deviation",
        template="plotly_white",
        xaxis_tickangle=-45,
        showlegend=True,
        barmode="group",
    )

    filename_suffix = f"_after_{after.strftime('%Y%m%d_%H%M%S')}" if after else ""
    fig_path = Config.plots_dir / f"model_prob_std_histogram{filename_suffix}.html"
    fig.write_html(fig_path)
    print(f"Saved std histogram to {fig_path}")


def boostrap_std_ci(data: list[float], n_samples: int = 1000) -> tuple[float, float]:
    """Use boostrapping to estimate the confidence interval for the standard deviation."""
    data = np.array(data)
    stds = []
    for i in range(n_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        stds.append(np.std(sample, ddof=1))
    stds = np.sort(stds)
    return np.percentile(stds, 2.5), np.percentile(stds, 97.5)


def plot_prob_histograms(after: datetime | None = None):
    # TODO wrong (see get_top_token_logprobs)
    data = get_db_data(after=after)
    histograms_dir = Config.plots_dir / "histograms"
    os.makedirs(histograms_dir, exist_ok=True)

    for table_name in data.keys():
        top_token_logprobs = get_top_token_logprobs(data, table_name, all_top_tokens=True)
        top_token_probs = [math.exp(logprob) for logprob in top_token_logprobs]

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Full Range [0, 1]", "Actual Range"],
            vertical_spacing=0.1,
        )

        nbins = 100
        fig.add_trace(go.Histogram(x=top_token_probs, nbinsx=nbins, name="Probs"), row=1, col=1)
        fig.add_trace(go.Histogram(x=top_token_probs, nbinsx=nbins, name="Probs"), row=2, col=1)
        fig.update_xaxes(range=[0, 1], row=1, col=1)

        title_suffix = f" (after {after.isoformat()})" if after else ""
        fig.update_layout(
            title=f"Probability Distribution for {table_name}{title_suffix}",
            template="plotly_white",
            showlegend=False,
        )

        fig.update_xaxes(title_text="Probability", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        stub = table_name.replace("/", "_").replace("#", "_")
        filename_suffix = f"_after_{after.strftime('%Y%m%d_%H%M%S')}" if after else ""
        fig_path = histograms_dir / f"{stub}_prob_histogram{filename_suffix}.html"
        fig.write_html(fig_path)
        print(f"Saved histogram for {table_name} to {fig_path}")


def plot_top_token_logprobs_over_time(after: datetime | None = None, prompt: str | None = None):
    """Plot logprobs of top tokens over time for each prompt in each table.

    Args:
        after: Only plot data after this date.
        prompt: Only plot data for this prompt.
    """
    data = get_db_data(after=after)
    time_series_dir = Config.plots_dir / "time_series"
    os.makedirs(time_series_dir, exist_ok=True)

    n_plots = sum(len(set(row.prompt for row in rows)) for rows in data.values())

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
        for prompt, prompt_rows in prompt_groups.items():
            # Create subdirectory based on first 16 characters of base64 of prompt, followed by 8 characters of MD5 hash of prompt
            prompt_base64 = (
                base64.b64encode(prompt.encode("utf-8")).decode("utf-8").replace("/", "-")[:16]
            )
            prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
            prompt_dir = time_series_dir / f"{prompt_base64}_{prompt_hash}"
            os.makedirs(prompt_dir, exist_ok=True)

            all_token_logprobs = get_token_logprobs(prompt_rows, prompt, missing_policy="none")

            # Create the plot
            fig = go.Figure()

            # Add a line for each top token
            for token, token_logprobs in all_token_logprobs.items():
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

            # Update layout
            title_suffix = f" (after {after.isoformat()})" if after else ""
            # Truncate prompt for title if it's too long
            prompt_preview = repr(trim_to_length(prompt, 50))
            fig.update_layout(
                title=f"Top Token Logprobs Over Time - {table_name}{title_suffix}<br>Prompt: {prompt_preview}",
                xaxis_title="Time",
                yaxis_title="Log Probability",
                template="plotly_white",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
            )

            # Save the plot
            stub = table_name.replace("/", "_").replace("#", "_")
            filename_suffix = f"_after_{after.strftime('%Y%m%d_%H%M%S')}" if after else ""
            fig_path = prompt_dir / f"{stub}_logprobs_over_time{filename_suffix}.html"
            fig.write_html(fig_path)
            logger.info(
                f"Saved logprobs over time for {table_name} (prompt hash: {prompt_hash}, prompt start: {repr(prompt[:40])}) to {fig_path}"
            )
            pbar.update(1)
    pbar.close()


def test_normality_shapiro():
    """
    For each endpoint, each prompt and each top token for which we have the logprob value at least 200 times,
    perform the Shapiro-Wilk test to test for normality.
    """
    data = get_db_data()
    min_statistic = float("inf")
    min_statistic_realization = None
    min_p_value = float("inf")
    min_p_value_realization = None
    max_statistic = float("-inf")
    max_statistic_realization = None
    max_p_value = float("-inf")
    max_p_value_realization = None
    for table_name in data.keys():
        rows = data[table_name]
        rows_by_prompt = defaultdict(list)
        for row in rows:
            rows_by_prompt[row[1]].append(row)
        for prompt, rows in rows_by_prompt.items():
            logprobs_by_token = defaultdict(list)
            for _, prompt, top_tokens, logprobs in rows:
                for i, token in enumerate(top_tokens):
                    logprobs_by_token[token].append(logprobs[i])
            for token, logprobs in logprobs_by_token.items():
                if len(logprobs) < 200:
                    continue
                statistic, p_value = shapiro(logprobs)
                logger.info(
                    f"{table_name}, prompt: {repr(trim_to_length(prompt, 20))}, token: {token}, ({len(logprobs)} occurrences): s={statistic:.4f}, p={p_value:.3e}"
                )
                realization = (table_name, prompt, token, len(logprobs), statistic, p_value)
                if p_value < min_p_value:
                    min_p_value = p_value
                    min_p_value_realization = realization
                if p_value > max_p_value:
                    max_p_value = p_value
                    max_p_value_realization = realization
                if statistic < min_statistic:
                    min_statistic = statistic
                    min_statistic_realization = realization
                if statistic > max_statistic:
                    max_statistic = statistic
                    max_statistic_realization = realization
    logger.info(f"Min statistic realization: {min_statistic_realization}")
    logger.info(f"Max statistic realization: {max_statistic_realization}")
    logger.info(f"Min p-value realization: {min_p_value_realization}")
    logger.info(f"Max p-value realization: {max_p_value_realization}")


def create_random_qq_plots(n_samples: int = 100):
    """
    Select n_samples random (endpoint, prompt, top token seen at least 200 times) tuples
    and create Q-Q plots with respect to a normal distribution.
    Convert matplotlib figures to plotly and save as HTML.
    Also creates combined subplot figures with readable plots.
    """
    random.seed(Config.seed)
    np.random.seed(Config.seed)

    data = get_db_data()

    # Collect all valid tuples (endpoint, prompt, token, logprobs)
    valid_tuples = []

    for table_name in data.keys():
        rows = data[table_name]
        rows_by_prompt = defaultdict(list)
        for row in rows:
            rows_by_prompt[row[1]].append(row)

        for prompt, rows in rows_by_prompt.items():
            logprobs_by_token = defaultdict(list)
            for _, prompt, top_tokens, logprobs in rows:
                for i, token in enumerate(top_tokens):
                    logprobs_by_token[token].append(logprobs[i])

            for token, logprobs in logprobs_by_token.items():
                if len(logprobs) >= 200:
                    valid_tuples.append((table_name, prompt, token, logprobs))

    logger.info(f"Found {len(valid_tuples)} valid tuples with at least 200 occurrences")

    if len(valid_tuples) == 0:
        logger.info("No valid tuples found!")
        return

    sample_size = min(n_samples, len(valid_tuples))
    sampled_tuples = random.sample(valid_tuples, sample_size)

    qq_plots_dir = Config.plots_dir / "qq_plots"
    os.makedirs(qq_plots_dir, exist_ok=True)

    logger.info(f"Creating Q-Q plots for {sample_size} randomly selected tuples...")

    # Store Q-Q plot data for reuse in combined plots
    qq_data = []
    r_squared_values = []

    # Create individual plots and collect Q-Q data
    for i, (table_name, prompt, token, logprobs) in enumerate(sampled_tuples):
        # Create individual plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sm.qqplot(np.array(logprobs), stats.norm, fit=True, line="45", ax=ax)

        # Extract Q-Q plot data from the matplotlib figure
        lines = ax.get_lines()
        # First line is the data points, second line is the reference line
        data_line = lines[0]  # The scatter plot data
        ref_line = lines[1]  # The reference line

        theoretical_quantiles = data_line.get_xdata()
        sample_quantiles = data_line.get_ydata()
        ref_x = ref_line.get_xdata()
        ref_y = ref_line.get_ydata()

        r_squared = linregress(theoretical_quantiles, sample_quantiles).rvalue ** 2
        r_squared_values.append(r_squared)

        qq_data.append((theoretical_quantiles, sample_quantiles, ref_x, ref_y))

        prompt_preview = repr(trim_to_length(prompt, 30))
        ax.set_title(
            f'Q-Q Plot vs Normal Distribution\n{table_name}\nToken: "{token}"\nPrompt: {prompt_preview}'
        )
        ax.set_xlabel("Theoretical Quantiles (Normal)")
        ax.set_ylabel("Sample Quantiles (Logprobs)")

        # Convert matplotlib figure to plotly
        plotly_fig = tls.mpl_to_plotly(fig)

        # Add R² annotation to top left
        plotly_fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"R² = {r_squared:.3f}",
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
        )

        plotly_fig.update_layout(
            template="plotly_white",
            showlegend=False,
        )

        # Create filename - use index and hash for uniqueness
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
        table_stub = table_name.replace("/", "_").replace("#", "_")
        token_stub = token.replace("/", "_").replace(" ", "_")[:20]
        filename = f"qqplot_{i:03d}_{table_stub}_{token_stub}_{prompt_hash}.html"

        fig_path = qq_plots_dir / filename
        plotly.offline.plot(plotly_fig, filename=str(fig_path), auto_open=False)

        plt.close(fig)

        logger.info(f"Saved Q-Q plot {i + 1}/{sample_size}: {filename}")
        logger.info(f"  - Table: {table_name}")
        logger.info(f"  - Token: '{token}' ({len(logprobs)} occurrences)")
        logger.info(f"  - R² = {r_squared:.3f}")
        logger.info(f"  - Prompt start: {repr(prompt[:50])}")

    # Create combined plots with 10x10 subplots (100 per page)
    plots_per_page = 100
    rows_per_page = 10
    cols_per_page = 10
    num_pages = math.ceil(sample_size / plots_per_page)

    for page in range(num_pages):
        start_idx = page * plots_per_page
        end_idx = min(start_idx + plots_per_page, sample_size)

        subplot_titles = []
        for i in range(end_idx - start_idx):
            plot_idx = start_idx + i
            r_squared = r_squared_values[plot_idx]
            subplot_titles.append(f"{plot_idx} (R²: {r_squared:.2f})")

        # Pad with empty titles if needed
        while len(subplot_titles) < plots_per_page:
            subplot_titles.append("")

        combined_fig = make_subplots(
            rows=rows_per_page,
            cols=cols_per_page,
            subplot_titles=subplot_titles,
            vertical_spacing=0.04,
            horizontal_spacing=0.03,
        )

        for i in range(end_idx - start_idx):
            plot_idx = start_idx + i
            theoretical_quantiles, sample_quantiles, ref_x, ref_y = qq_data[plot_idx]

            # Add to combined subplot
            row = i // cols_per_page + 1
            col = i % cols_per_page + 1

            # Add scatter plot to subplot using pre-computed data
            combined_fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode="markers",
                    marker=dict(size=2, color="blue", opacity=0.6),
                    showlegend=False,
                    name=f"Data {plot_idx}",
                ),
                row=row,
                col=col,
            )

            # Add reference line using pre-computed data
            combined_fig.add_trace(
                go.Scatter(
                    x=ref_x,
                    y=ref_y,
                    mode="lines",
                    line=dict(color="red", width=1.5),
                    showlegend=False,
                    name=f"Reference {plot_idx}",
                ),
                row=row,
                col=col,
            )

        # Update combined figure layout
        page_title = f"Combined Q-Q Plots vs Normal Distribution (Page {page + 1}/{num_pages}, seed={Config.seed})"
        combined_fig.update_layout(
            title=page_title,
            template="plotly_white",
            showlegend=False,
            height=1600,
            width=1600,
        )

        for annotation in combined_fig["layout"]["annotations"]:
            annotation.font = dict(size=11)

        # Add axis labels only to bottom row and left column
        for i in range(1, rows_per_page + 1):
            for j in range(1, cols_per_page + 1):
                if i == rows_per_page:  # Bottom row
                    combined_fig.update_xaxes(title_text="Theoretical", row=i, col=j)
                if j == 1:  # Left column
                    combined_fig.update_yaxes(title_text="Sample", row=i, col=j)

        # Save combined figure for this page
        combined_fig_path = (
            qq_plots_dir / f"combined_qq_plots_page_{page + 1}_seed_{Config.seed}.html"
        )
        combined_fig.write_html(combined_fig_path)
        logger.info(f"Saved combined Q-Q plot page {page + 1}/{num_pages} to {combined_fig_path}")

    logger.info(f"\nAll individual Q-Q plots saved to {qq_plots_dir}")
    logger.info(f"Combined Q-Q plots saved as {num_pages} pages to {qq_plots_dir}")


if __name__ == "__main__":
    # equivalence_classes()
    # top_logprob_variability()
    # plot_prob_histograms()
    # 2025-05-29 at 16:59: started querying endpoints with seed 1, so can be compared with the ones without a seed.
    # plot_prob_std(after=datetime(2025, 5, 29, 16, 59))
    # plot_top_token_logprobs_over_time()
    # test_normality_shapiro()
    create_random_qq_plots()
