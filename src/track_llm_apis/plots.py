"""plots for the paper"""

import os
from datetime import datetime

import plotly.graph_objects as go

from track_llm_apis.analyze import get_db_data, get_token_logprobs
from track_llm_apis.config import config
from track_llm_apis.main import Endpoint

logger = config.logger
PLOTS_DIR = config.plots_dir / "paper"
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_logprob_time_series():
    endpoint = Endpoint("openai", "gpt-4.1", cost=(2, 8))
    prompt = "x"
    start_date = datetime(2025, 8, 1, 0, 0, 0)
    end_date = datetime(2025, 8, 14, 23, 59, 59)
    data = get_db_data(tables=[str(endpoint)], after=start_date, before=end_date, prompt=prompt)
    data = data[str(endpoint)]
    all_token_logprobs = get_token_logprobs(data, prompt, missing_policy="none")
    fig = go.Figure()
    # Sort tokens by their average logprob value (descending order)
    sorted_tokens = sorted(
        all_token_logprobs.keys(),
        key=lambda token: sum(lp for lp in all_token_logprobs[token].logprobs if lp is not None)
        / len([lp for lp in all_token_logprobs[token].logprobs if lp is not None]),
        reverse=True,
    )
    for token in sorted_tokens:
        token_logprobs = all_token_logprobs[token]
        fig.add_trace(
            go.Scatter(
                x=token_logprobs.dates,
                y=token_logprobs.logprobs,
                mode="lines+markers",
                name=f'"{token}"',
                line=dict(width=1),
                marker=dict(size=2),
            )
        )

    fig.update_layout(
        font_family="Spectral",
        template="plotly_white",
        margin=dict(l=0, r=0, t=100, b=50),
        font=dict(size=16),
        showlegend=True,
        xaxis=dict(
            tickformat="%b %d",  # Format as "Aug 01" instead of showing the year
        ),
        yaxis=dict(
            title="Logprobs",
            range=[-15, 0],
            dtick=2,
        ),
        title={"text": "Logprobs over time for GPT-4.1 with prompt: 'x'", "x": 0.5},
    )

    fig_path = PLOTS_DIR / "example_logprob_time_series.pdf"
    fig.write_image(fig_path)


if __name__ == "__main__":
    plot_logprob_time_series()
