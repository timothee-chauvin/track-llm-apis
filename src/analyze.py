import json
import math
import os
import sqlite3
import statistics

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config


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


def top_logprob_variability():
    data = get_db_data()
    for table_name, rows in data.items():
        _, _, first_top_tokens, _ = rows[0]
        top_token = first_top_tokens[0]
        top_token_logprobs = []
        for _date_str, _prompt, top_tokens, top_logprobs in rows:
            try:
                top_token_index = top_tokens.index(top_token)
                top_token_logprobs.append(top_logprobs[top_token_index])
            except ValueError:
                pass

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


def plot_prob_histograms():
    data = get_db_data()
    os.makedirs(Config.plots_dir, exist_ok=True)

    for table_name, rows in data.items():
        _, _, first_top_tokens, _ = rows[0]
        top_token = first_top_tokens[0]
        top_token_logprobs = []
        for _date_str, _prompt, top_tokens, top_logprobs in rows:
            try:
                top_token_index = top_tokens.index(top_token)
                top_token_logprobs.append(top_logprobs[top_token_index])
            except ValueError:
                pass
        all_probs = [math.exp(logprob) for logprob in top_token_logprobs]

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Full Range [0, 1]", "Actual Range"],
            vertical_spacing=0.1,
        )

        nbins = 100
        fig.add_trace(go.Histogram(x=all_probs, nbinsx=nbins, name="Probs"), row=1, col=1)
        fig.add_trace(go.Histogram(x=all_probs, nbinsx=nbins, name="Probs"), row=2, col=1)
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
    plot_prob_histograms()
