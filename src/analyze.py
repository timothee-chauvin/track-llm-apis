import json
import math
import os
import sqlite3
import statistics
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config

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


def get_db_data() -> dict[str, list[tuple[str, str, list, list]]]:
    conn = sqlite3.connect(Config.db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        raw_table_names = [row[0] for row in cursor.fetchall()]
        # Filter out sqlite_sequence and other internal sqlite tables
        table_names = [name for name in raw_table_names if not name.startswith("sqlite_")]
        results = {}
        for table_name in table_names:
            cursor.execute(f'SELECT date, prompt, top_tokens, logprobs FROM "{table_name}"')
            rows = cursor.fetchall()
            results[table_name] = []
            for date, prompt, top_tokens, logprobs in rows:
                results[table_name].append(
                    (date, prompt, list(json.loads(top_tokens)), list(json.loads(logprobs)))
                )
        return results
    except sqlite3.Error as e:
        Config.logger.error(f"An error occurred during database analysis: {e}")
    finally:
        conn.close()


def equivalence_classes():
    data = get_db_data()
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


def top_logprob_variability():
    data = get_db_data()
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


def plot_prob_std():
    data = get_db_data()
    os.makedirs(Config.plots_dir, exist_ok=True)

    table_names = []
    stdevs = []
    model_colors = []

    # Color mapping for model types
    color_map = {"dense": "blue", "moe": "red", "?": "gray"}

    std_by_type = defaultdict(list)

    # Collect data for all tables first
    table_data = []
    for table_name in data.keys():
        top_token_logprobs = get_top_token_logprobs(data, table_name, all_top_tokens=True)
        top_token_probs = [math.exp(logprob) for logprob in top_token_logprobs]
        std_value = statistics.stdev(top_token_probs)
        print(f"\n# {table_name}")
        print(f"std: {std_value}")

        model_name = table_name.split("#")[1]
        model_type = model_types[model_name]

        table_data.append((table_name, std_value, model_type))
        std_by_type[model_type].append(std_value)

    # Sort by table name alphabetically
    table_data.sort(key=lambda x: x[0])

    # Extract sorted data for plotting
    for table_name, std_value, model_type in table_data:
        table_names.append(table_name)
        stdevs.append(std_value)
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
        )
    )

    for model_type, color in color_map.items():
        fig.add_trace(
            go.Bar(x=[None], y=[None], marker_color=color, name=model_type, showlegend=True)
        )

    fig.update_layout(
        title="Standard Deviation of Top Token Probabilities by Model",
        xaxis_title="Model",
        yaxis_title="Standard Deviation",
        template="plotly_white",
        xaxis_tickangle=-45,
        showlegend=True,
        barmode="group",
    )

    fig_path = Config.plots_dir / "model_prob_std_histogram.html"
    fig.write_html(fig_path)
    print(f"Saved std histogram to {fig_path}")


def plot_prob_histograms():
    data = get_db_data()
    os.makedirs(Config.plots_dir, exist_ok=True)

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

        fig.update_layout(
            title=f"Probability Distribution for {table_name}",
            template="plotly_white",
            showlegend=False,
        )

        fig.update_xaxes(title_text="Probability", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        stub = table_name.replace("/", "_").replace("#", "_")
        fig_path = Config.plots_dir / f"{stub}_prob_histogram.html"
        fig.write_html(fig_path)
        print(f"Saved histogram for {table_name} to {fig_path}")


if __name__ == "__main__":
    # equivalence_classes()
    # top_logprob_variability()
    # plot_prob_histograms()
    plot_prob_std()
