import json
import os
import random
import time
from pathlib import Path
from typing import Any, Literal

import orjson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from pydantic import BaseModel, field_validator
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from track_llm_apis.config import config
from track_llm_apis.sampling.analyze_gao2025 import CompletionSample, run_two_sample_test_torch
from track_llm_apis.sampling.analyze_logprobs import logprob_two_sample_test
from track_llm_apis.sampling.analyze_mmlu import mmlu_two_sample_test
from track_llm_apis.sampling.common import (
    AnalysisResult,
    CIResult,
    CompressedOutput,
    CompressedOutputRow,
    DataSource,
    TwoSampleMultiTestResult,
)
from track_llm_apis.tinychange import TinyChange
from track_llm_apis.util import slugify, trim_to_length

logger = config.logger


def get_logprobs_transformers(model, tokenizer, prompt, model_device):
    """Get log probabilities for the first generated token using model.generate() from transformers."""
    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )
        first_token_logits = outputs.scores[0]  # Shape: [batch_size, vocab_size]
        first_token_logprobs = torch.log_softmax(first_token_logits, dim=-1)
        return first_token_logprobs[0]


def compute_kl_divergence(original_logprobs, variant_logprobs):
    """Compute KL divergence between two log probability distributions."""
    return F.kl_div(variant_logprobs, original_logprobs, log_target=True, reduction="mean")


def print_logprobs_summary(logprobs, tokenizer, model_name):
    """logger.info summary of top logprobs for a model."""
    top_10_logprobs, top_10_indices = torch.topk(logprobs, 10)

    logger.info(f"\nTop 10 tokens for {model_name}:")
    for i in range(10):
        token_id = top_10_indices[i].item()
        token = tokenizer.decode(token_id)
        logprob = top_10_logprobs[i].item()
        logger.info(f"Rank {i + 1}: '{token}' (ID: {token_id}) - Log prob: {logprob:g}")


def print_logprobs(model, tokenizer, prompt, model_device):
    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model_device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
        )

        first_token_logits = outputs.scores[0]  # Shape: [batch_size, vocab_size]
        first_token_logprobs = torch.log_softmax(first_token_logits, dim=-1)
        top_10_logprobs, top_10_indices = torch.topk(first_token_logprobs[0], 10)

        logger.info("Top 10 tokens and their log probabilities:")
        for i in range(10):
            token_id = top_10_indices[i].item()
            token = tokenizer.decode(token_id)
            logprob = top_10_logprobs[i].item()
            logger.info(f"Rank {i + 1}: '{token}' (ID: {token_id}) - Log prob: {logprob:g}")

        # Also logger.info the actually generated token for reference
        generated_token_id = outputs.sequences[0, -1].item()
        generated_token = tokenizer.decode(generated_token_id)
        logger.info(f"\nActually generated token: '{generated_token}' (ID: {generated_token_id})")


def plot_logprobs_over_time(
    all_logprobs,
    prompt: str,
    base_model_name: str,
    variant_description: dict[str, Any],
    batch_size: int,
):
    """Plot logprobs over time for all tokens that appear in the series."""
    prompt_slug = slugify(prompt, max_length=50, hash_length=8)
    model_name_slug = slugify(base_model_name, hash_length=0)
    prompt_dir = (
        config.plots_dir
        / "time_series_local"
        / prompt_slug
        / model_name_slug
        / f"batch_size={batch_size}"
    )
    os.makedirs(prompt_dir, exist_ok=True)
    description_str = "_".join(str(v) for v in variant_description.values())
    description_slug = slugify(description_str, max_length=100, hash_length=8)
    filename = f"{description_slug}.html"

    # Collect all unique tokens
    all_tokens = set()
    for logprob_dict in all_logprobs:
        all_tokens.update(logprob_dict.keys())

    fig = go.Figure()  # pyright: ignore[reportCallIssue]

    # For each token, create its time series
    for token_id in sorted(all_tokens):
        logprob_series = []

        for logprob_dict in all_logprobs:
            if token_id in logprob_dict:
                logprob_obj = logprob_dict[token_id]
                decoded_token = logprob_obj.decoded_token
                logprob_series.append(logprob_obj.logprob)
            else:
                logprob_series.append(None)

        fig.add_trace(
            go.Scatter(  # pyright: ignore[reportCallIssue]
                x=list(range(len(all_logprobs))),
                y=logprob_series,
                mode="lines+markers",
                name=decoded_token,
                connectgaps=False,
            )
        )

    prompt_preview = repr(trim_to_length(prompt, 50))
    fig.update_layout(
        title=f"Top Token Logprobs Over Time - {variant_description}<br>Prompt: {prompt_preview}",
        xaxis_title="Iteration",
        yaxis_title="Log Probability",
        template="plotly_white",
    )

    fig.write_html(prompt_dir / filename)


def prompt_rows_dict_to_completion_sample(
    prompt_rows_dict: dict[str, list[CompressedOutputRow]], tokenizer: PreTrainedTokenizerBase
) -> CompletionSample:
    prompts_list = []
    completions_list = []
    for i, prompt in enumerate(prompt_rows_dict.keys()):
        rows = prompt_rows_dict[prompt]
        for row in rows:
            prompts_list.append(i)
            completions_list.append(row.text[0])
    prompts = torch.tensor(prompts_list).to(device=config.analysis.device, dtype=torch.int32)
    completions = tokenizer(
        completions_list,
        padding="max_length",
        truncation=True,
        max_length=config.sampling.gao2025.max_tokens,
        return_tensors="pt",
    )["input_ids"].to(device=config.analysis.device, dtype=torch.int32)
    return CompletionSample(prompts=prompts, completions=completions)


def gao2025_two_sample_test(
    rows_subset: dict[str, list[CompressedOutputRow]],
    unchanged_rows_subset: dict[str, list[CompressedOutputRow]],
    tokenizer: PreTrainedTokenizerBase,
    pvalue_b: int = 1000,
):
    sample1 = prompt_rows_dict_to_completion_sample(rows_subset, tokenizer)
    sample2 = prompt_rows_dict_to_completion_sample(unchanged_rows_subset, tokenizer)
    return run_two_sample_test_torch(sample1, sample2, b=pvalue_b)


def evaluate_detector_on_variant(
    source: DataSource,
    rows1: dict[str, list[CompressedOutputRow]],
    rows2: dict[str, list[CompressedOutputRow]],
    same: bool,
    prompt_length: dict[str, int],
    tokenizer: PreTrainedTokenizerBase | None = None,
    logprob_prompt: str | None = None,
    n_tests: int = 2000,
    pvalue_b: int = 1000,
) -> TwoSampleMultiTestResult:
    """
    Args:
      rows1, rows2: dictionary of prompts to lists of rows
    """
    if logprob_prompt:
        logger.info(f"Evaluating prompt {repr(logprob_prompt)}...")
    else:
        logger.info(f"Evaluating detector {source}...")
    match source:
        case DataSource.GAO2025:
            samples_per_prompt = config.sampling.gao2025.n_wikipedia_samples_per_prompt
            two_sample_test_fn = gao2025_two_sample_test
        case DataSource.MMLU:
            samples_per_prompt = config.sampling.mmlu.n_samples_per_prompt
            two_sample_test_fn = mmlu_two_sample_test
        case DataSource.US:
            samples_per_prompt = config.sampling.logprob.n_samples_per_prompt
            two_sample_test_fn = logprob_two_sample_test
            # Only keep the provided or default prompt, discarding the others
            if logprob_prompt is None:
                logprob_prompt = config.sampling.logprob.default_prompt
            rows1 = {logprob_prompt: rows1[logprob_prompt]}
            rows2 = {logprob_prompt: rows2[logprob_prompt]}

    token_count = {"i": [], "o": []}
    stats = []
    pvalues = []
    for _ in range(n_tests):
        if same:
            assert rows1 == rows2
            subset = {p: random.sample(rows, 2 * samples_per_prompt) for p, rows in rows1.items()}
            sample1 = {p: subset[p][:samples_per_prompt] for p in rows1.keys()}
            sample2 = {p: subset[p][samples_per_prompt:] for p in rows2.keys()}
        else:
            sample1 = {p: random.sample(rows, samples_per_prompt) for p, rows in rows1.items()}
            sample2 = {p: random.sample(rows, samples_per_prompt) for p, rows in rows2.items()}

        result = two_sample_test_fn(sample1, sample2, pvalue_b=pvalue_b, tokenizer=tokenizer)
        stats.append(result.statistic)
        if pvalue_b > 0:
            pvalues.append(result.pvalue)
        token_count["i"].append(
            sum(prompt_length[p] * len(r) for p, r in sample1.items())
            + sum(prompt_length[p] * len(r) for p, r in sample2.items())
        )
        token_count["o"].append(
            sum(r.text[1] for rows in sample1.values() for r in rows)
            + sum(r.text[1] for rows in sample2.values() for r in rows)
        )

    return TwoSampleMultiTestResult(
        stats=stats,
        pvalues=pvalues if pvalue_b > 0 else None,
        input_token_avg=sum(token_count["i"]) / len(token_count["i"]),
        output_token_avg=sum(token_count["o"]) / len(token_count["o"]),
    )


class PlotData(BaseModel):
    experiment: Literal["baseline", "ablation_prompt"]
    model_name: str
    variant: str | None
    roc_curves: dict[DataSource | str, list[tuple[list[float], list[float]]]]
    roc_auc_ci: dict[DataSource | str, CIResult]

    @field_validator("roc_curves", "roc_auc_ci", mode="before")
    @classmethod
    def validate_roc_curves(cls, v, info):
        if info.data.get("experiment") == "baseline":
            if isinstance(v, dict) and v:
                first_key = next(iter(v))
                if isinstance(first_key, str):
                    return {DataSource(int(k)): v for k, v in v.items()}
        return v


def get_plot_dir(sampling_directory: Path) -> Path:
    directory = config.plots_dir / "roc_curves" / sampling_directory.name
    os.makedirs(directory, exist_ok=True)
    return directory


def get_ablation_prompt_data_dir(sampling_directory: Path) -> Path:
    directory = get_plot_dir(sampling_directory) / "ablation_prompt" / "data"
    os.makedirs(directory, exist_ok=True)
    return directory


def get_baselines_data_dir(sampling_directory: Path) -> Path:
    directory = get_plot_dir(sampling_directory) / "baselines" / "data"
    os.makedirs(directory, exist_ok=True)
    return directory


def plot_roc_curve_with_fs_cache(plot_data: PlotData, out_dir: Path):
    os.makedirs(out_dir, exist_ok=True)
    variant_slug = slugify(_variant_preslug(plot_data.variant), max_length=200, hash_length=0)
    data_path = out_dir / f"{variant_slug}.json"
    with open(data_path, "w") as f:
        json.dump(plot_data.model_dump(mode="json"), f, indent=2)
    plot_roc_curve(data_path)


def _variant_preslug(variant: str | None) -> str:
    if variant is None:
        return "all"
    variant_metadata = orjson.loads(variant)
    variant_preslug = (
        variant_metadata["type"]
        + ","
        + ",".join(f"{k}={v}" for k, v in variant_metadata.items() if k != "type")
    )
    return variant_preslug


def plot_roc_curve(
    plot_data_path: Path,
):
    """From the path of a data file containing the necessary data, plot the ROC curves in the parent directory,
    with the same name except for the extension."""
    plot_data = PlotData.model_validate_json(plot_data_path.read_text())
    conditions = list(plot_data.roc_curves.keys())
    fig = go.Figure()

    # Define colors for each data source
    colors = px.colors.qualitative.Plotly

    for i, condition in enumerate(conditions):
        match plot_data.experiment:
            case "baseline":
                # Conditions are DataSource objects
                condition_name = condition.name
            case "ablation_prompt":
                # Conditions are prompts
                condition_name = repr(condition)

        color = colors[i % len(colors)]
        display_name = f"{condition_name} (AUC: {plot_data.roc_auc_ci[condition].avg:.4f})"

        for j, (fpr, tpr) in enumerate(plot_data.roc_curves[condition]):
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=display_name,
                    line=dict(color=color),
                    showlegend=(j == 0),  # Only show legend for first curve of each condition
                    legendgroup=condition_name,  # Group all curves from same condition
                )
            )

    # Random chance diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random chance",
            line=dict(dash="dash", color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    variant = plot_data.variant
    title = f"ROC Curves on model {plot_data.model_name}"
    if variant:
        title += f", on variant: {_variant_preslug(variant)}"
    else:
        title += " across all variants"
    fig.update_layout(
        font_family="Spectral",
        template="plotly_white",
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    plot_dir = plot_data_path.parent.parent
    filename_base = plot_data_path.stem
    fig.write_html(plot_dir / f"{filename_base}.html")
    logger.info(f"Saved ROC curve to {plot_dir / f'{filename_base}.html'}")


def baseline_analysis(
    directory: Path,
    data: CompressedOutput,
    sources: list[DataSource],
    n_tests: int = 1000,
    pvalue_b: int = 1000,
):
    if DataSource.GAO2025 in sources:
        tokenizer = AutoTokenizer.from_pretrained(data.model_name)
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            logger.info(f"Setting pad token to eos token for {data.model_name}")
            tokenizer.pad_token = tokenizer.eos_token

    analysis_results = AnalysisResult(
        experiment="baseline",
        model_name=data.model_name,
        n_tests=n_tests,
        pvalue_b=pvalue_b,
    )

    logger.info(f"{sources=}")
    logger.info(f"{data.model_name=}")
    variants = [v for v in data.references.variants.keys() if v != TinyChange.unchanged_str()]
    prompts = list(data.references.prompts.keys())
    prompt_length = {prompt: tokens for prompt, tokens in prompts}

    logger.info("Splitting data by source...")
    data_by_source = {source: data.filter(datasource=source) for source in sources}

    logger.info("Getting rows by variant...")
    rows_by_variant = {source: data_by_source[source].get_rows_by_variant() for source in sources}
    unchanged_rows = {
        source: rows_by_variant[source][TinyChange.unchanged_str()] for source in sources
    }
    unchanged_rows_by_prompt = {
        source: data_by_source[source].get_rows_by_prompt(unchanged_rows[source])
        for source in sources
    }

    start_time = time.time()
    # Get data on the false positive rate
    analysis_results.original = {
        str(source.value): evaluate_detector_on_variant(
            source,
            unchanged_rows_by_prompt[source],
            unchanged_rows_by_prompt[source],
            same=True,
            prompt_length=prompt_length,
            tokenizer=tokenizer,
            n_tests=n_tests,
            pvalue_b=pvalue_b,
        )
        for source in sources
    }
    for variant_idx, variant in enumerate(variants):
        logger.info(f"Variant {variant_idx + 1}/{len(variants)}: {variant}")
        analysis_results.variants[variant] = {}
        for source in sources:
            variant_rows = rows_by_variant[source][variant]
            rows_by_prompt = data_by_source[source].get_rows_by_prompt(variant_rows)
            analysis_results.variants[variant][str(source.value)] = evaluate_detector_on_variant(
                source,
                unchanged_rows_by_prompt[source],
                rows_by_prompt,
                same=False,
                prompt_length=prompt_length,
                tokenizer=tokenizer,
                n_tests=n_tests,
                pvalue_b=pvalue_b,
            )

        with open(directory / "baseline_analysis.json", "wb") as f:
            f.write(orjson.dumps(analysis_results.model_dump(mode="json")))

    # multivariant_roc_curves = {
    #     source: TwoSampleMultiTestResultMultiROC.multivariant_rocs(
    #         results_original[source], [results[source] for results in variant_results.values()]
    #     )
    #     for source in sources
    # }
    # multivariant_roc_auc_ci = {
    #     source: TwoSampleMultiTestResultMultiROC.multivariant_roc_auc_ci(
    #         results_original[source],
    #         [results[source] for results in variant_results.values()],
    #         results_alpha,
    #     )
    #     for source in sources
    # }
    # avg_roc_auc_ci = {
    #     source: TwoSampleMultiTestResultMultiROC.roc_auc_avg_ci(
    #         results_original[source],
    #         [results[source] for results in variant_results.values()],
    #         results_alpha,
    #     )
    #     for source in sources
    # }
    # analysis_results["all"] = {}
    # for source in sources:
    #     analysis_results["all"][source.name] = {
    #         "multivariant_roc_auc_ci": multivariant_roc_auc_ci[source].model_dump(mode="json"),
    #         "avg_roc_auc_ci": avg_roc_auc_ci[source].model_dump(mode="json"),
    #     }
    #     logger.info(f"Multivariant ROC AUC for {source}: {multivariant_roc_auc_ci[source]}")
    #     logger.info(f"Average ROC AUC for {source}: {avg_roc_auc_ci[source]}")
    # if plot_roc:
    #     plot_roc_curve_with_fs_cache(
    #         PlotData(
    #             experiment="baseline",
    #             model_name=data.model_name,
    #             variant=None,
    #             roc_curves=multivariant_roc_curves,
    #             roc_auc_ci=multivariant_roc_auc_ci,
    #         ),
    #         get_baselines_data_dir(directory),
    #     )
    # with open(directory / "baseline_analysis.json", "w") as f:
    #     json.dump(analysis_results, f, indent=2)
    end_time = time.time()
    logger.info(f"Time taken: {(end_time - start_time):.2f} seconds")


def ablation_influence_of_prompt(
    directory: Path,
    data: CompressedOutput,
    n_tests: int = 1000,
    pvalue_b: int = 1000,
):
    """Test the influence of the prompt choice on detection performance for the logprob method."""
    source = DataSource.US
    filtered_data = data.filter(datasource=source)
    rows_by_variant = filtered_data.get_rows_by_variant()
    unchanged_rows = rows_by_variant[TinyChange.unchanged_str()]
    unchanged_rows_by_prompt = filtered_data.get_rows_by_prompt(unchanged_rows)

    prompt_length = {
        prompt: tokens
        for prompt, tokens in filtered_data.references.prompts.keys()
        if prompt in unchanged_rows_by_prompt
    }
    prompts = list(prompt_length.keys())
    analysis_results = AnalysisResult(
        experiment="ablation_prompt",
        model_name=data.model_name,
        n_tests=n_tests,
        pvalue_b=pvalue_b,
    )
    variants = [v for v in data.references.variants.keys() if v != TinyChange.unchanged_str()]
    analysis_results.original = {
        prompt: evaluate_detector_on_variant(
            source,
            unchanged_rows_by_prompt,
            unchanged_rows_by_prompt,
            same=True,
            prompt_length=prompt_length,
            logprob_prompt=prompt,
            n_tests=n_tests,
            pvalue_b=pvalue_b,
        )
        for prompt in prompts
    }
    for variant_idx, variant in enumerate(variants):
        logger.info(f"Variant {variant_idx + 1}/{len(variants)}: {variant}")
        rows_by_prompt = filtered_data.get_rows_by_prompt(rows_by_variant[variant])
        analysis_results.variants[variant] = {
            prompt: evaluate_detector_on_variant(
                source,
                unchanged_rows_by_prompt,
                rows_by_prompt,
                same=False,
                prompt_length=prompt_length,
                logprob_prompt=prompt,
                n_tests=n_tests,
                pvalue_b=pvalue_b,
            )
            for prompt in prompts
        }
        with open(directory / "prompt_ablation_analysis.json", "wb") as f:
            f.write(orjson.dumps(analysis_results.model_dump(mode="json")))

    # Maybe reintroduce later if needed
    # if plot_roc:
    #     plot_roc_curve_with_fs_cache(
    #         PlotData(
    #             experiment="ablation_prompt",
    #             model_name=data.model_name,
    #             variant=None,
    #             roc_curves=multivariant_roc_curves,
    #             roc_auc_ci=multivariant_roc_auc_ci,
    #         ),
    #         get_ablation_prompt_data_dir(directory),
    #     )
    # with open(directory / "prompt_ablation_analysis.json", "wb") as f:
    #     f.write(orjson.dumps(analysis_results.model_dump(mode="json")))


def ablation_influence_of_prompt_plot():
    """Expects one `prompt_ablation_analysis.json` per sampling dir."""
    rows = []
    sampling_dirs = [
        config.sampling_data_dir / "keep" / dirname for dirname in config.analysis.sampling_dirnames
    ]

    for sampling_dir in sampling_dirs:
        p = Path(sampling_dir) / "prompt_ablation_analysis.json"
        with open(p) as f:
            analysis = orjson.loads(f.read())

        model = analysis["metadata"]["model_name"]
        for prompt, pa in analysis["all"].items():
            avg = pa["avg_roc_auc_ci"]["avg"]
            lower = pa["avg_roc_auc_ci"]["lower"]
            upper = pa["avg_roc_auc_ci"]["upper"]

            rows.append(
                {
                    "prompt": prompt,
                    "model": model,
                    "auc": avg,
                    "err_low": avg - lower,
                    "err_high": upper - avg,
                }
            )

    df = pd.DataFrame(rows)

    fig = px.line(
        df,
        x="model",
        y="auc",
        color="prompt",  # one line per prompt
        markers=True,
        error_y="err_high",
        error_y_minus="err_low",
        labels={"model": "Model", "auc": "Average ROC AUC", "prompt": "Prompt"},
    )

    fig.update_layout(
        title="Average ROC AUC per Model across Prompts",
        legend_title="Prompt",
        margin=dict(l=10, r=10, t=40, b=40),
    )

    plot_path = config.plots_dir / "paper" / "prompt_ablation_analysis.html"
    fig.write_html(plot_path)
    logger.info(f"Prompt ablation analysis plot saved to {plot_path}")


if __name__ == "__main__":
    analysis_config = config.analysis

    if analysis_config.experiment in ["baseline", "ablation_prompt"]:
        output_dir = config.sampling_data_dir / "keep" / analysis_config.sampling_dirname
        compressed_output = CompressedOutput.from_json_dir(output_dir)
        logger.info(compressed_output.model_name)
        logger.info(f"number of rows: {len(compressed_output.rows)}")
        for ref_attr in compressed_output.references.__dict__.keys():
            ref = getattr(compressed_output.references, ref_attr)
            logger.info(f"length of field '{ref_attr}': {len(ref)}")
    if analysis_config.experiment == "baseline":
        baseline_analysis(
            directory=output_dir,
            data=compressed_output,
            sources=[DataSource.US, DataSource.MMLU, DataSource.GAO2025],
            n_tests=analysis_config.n_tests,
            pvalue_b=analysis_config.pvalue_b,
        )
    elif analysis_config.experiment == "ablation_prompt":
        ablation_influence_of_prompt(
            directory=output_dir,
            data=compressed_output,
            n_tests=analysis_config.n_tests,
            pvalue_b=analysis_config.pvalue_b,
        )
    elif analysis_config.experiment == "ablation_prompt_plot":
        ablation_influence_of_prompt_plot()
