"""plots for the paper"""

import json
import math
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


def plot_difficulty_scale_auc():
    dirnames = [
        "2025-09-14_20-54-37_google2fgemma-3-270m-it",
        "2025-09-14_20-54-44_Qwen2fQwen2.5-0.5B-Instruct",
        "2025-09-14_20-54-45_allenai2fOLMo-2-1124-7B-Instruct",
        "2025-09-14_20-54-45_mistralai2fMistral-7B-Instruct-v0.3",
        "2025-09-14_20-54-46_google2fgemma-3-1b-it",
        "2025-09-14_20-54-46_microsoft2fPhi-3-mini-4k-instruct",
        "2025-09-14_20-54-47_meta-llama2fLlama-3.1-8B-Instruct",
    ]
    analysis_dirs = [config.sampling_data_dir / "keep" / d for d in dirnames]
    difficulty_scales = {
        "finetune_no_lora": {
            "title": "finetuning, no LoRA",
            "match_fn": lambda v: v["type"] == "finetune" and v["lora"] is False,
            "scale_attr": "n_samples",
        },
        "finetune_lora": {
            "title": "finetuning, LoRA",
            "match_fn": lambda v: v["type"] == "finetune" and v["lora"] is True,
            "scale_attr": "n_samples",
        },
        "random_noise": {
            "title": "random noise",
            "match_fn": lambda v: v["type"] == "random_noise",
            "scale_attr": "scale",
        },
        "weight_pruning_magnitude": {
            "title": "weight pruning, magnitude",
            "match_fn": lambda v: v["type"] == "weight_pruning" and v["method"] == "magnitude",
            "scale_attr": "scale",
        },
        "weight_pruning_random": {
            "title": "weight pruning, random",
            "match_fn": lambda v: v["type"] == "weight_pruning" and v["method"] == "random",
            "scale_attr": "scale",
        },
    }

    for analysis_dir in analysis_dirs:
        with open(analysis_dir / "baseline_analysis.json") as f:
            analysis_data = json.load(f)
            sources = analysis_data["metadata"]["sources"]
        variant_descriptions = [k for k in analysis_data.keys() if k not in ["metadata", "all"]]
        for scale_name, scale_info in difficulty_scales.items():
            variant_description_subset = []
            for k in variant_descriptions:
                variant = json.loads(k)
                if scale_info["match_fn"](variant):
                    variant_description_subset.append(k)
            scale_attr = scale_info["scale_attr"]
            variant_description_subset = sorted(
                variant_description_subset, key=lambda k: json.loads(k)[scale_attr], reverse=True
            )
            xaxis_values = [json.loads(k)[scale_attr] for k in variant_description_subset]
            print(xaxis_values)
            roc_auc_avg = {source: [] for source in sources}
            roc_auc_err_high = {source: [] for source in sources}
            roc_auc_err_low = {source: [] for source in sources}
            for k in variant_description_subset:
                results = analysis_data[k]
                for source in sources:
                    lower = results[source]["roc_auc_ci"]["lower"]
                    avg = results[source]["roc_auc_ci"]["avg"]
                    upper = results[source]["roc_auc_ci"]["upper"]
                    roc_auc_avg[source].append(avg)
                    roc_auc_err_high[source].append(upper - avg)
                    roc_auc_err_low[source].append(avg - lower)

            fig = go.Figure()
            for source in sources:
                fig.add_trace(
                    go.Scatter(
                        x=xaxis_values,
                        y=roc_auc_avg[source],
                        name=source,
                        error_y=dict(
                            type="data",
                            array=roc_auc_err_high[source],
                            arrayminus=roc_auc_err_low[source],
                        ),
                    )
                )

            # Add horizontal line for random guessing
            fig.add_hline(
                y=0.5,
                line_dash="dash",
                line_color="gray",
                annotation_text="Random guessing",
                annotation_position="bottom right",
            )

            if xaxis_values[-1] < 1:
                xaxis_ticktext = [f"2<sup>{round(math.log(x, 2))}</sup>" for x in xaxis_values]
            else:
                xaxis_ticktext = xaxis_values

            fig.update_layout(
                font_family="Spectral",
                template="plotly_white",
                title=f"{analysis_data['metadata']['model_name']}, {scale_info['title']}",
                xaxis=dict(
                    type="log",
                    autorange="reversed",
                    tickmode="array",
                    tickvals=xaxis_values,
                    ticktext=xaxis_ticktext,
                ),
            )
            fig_path = PLOTS_DIR / f"{scale_name}_{analysis_dir.name}.pdf"
            fig.write_image(fig_path)
            logger.info(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    # plot_logprob_time_series()
    plot_difficulty_scale_auc()
