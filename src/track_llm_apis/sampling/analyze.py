import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from track_llm_apis.config import config
from track_llm_apis.sampling.analyze_gao2025 import CompletionSample, run_two_sample_test_torch
from track_llm_apis.sampling.analyze_logprobs import logprob_two_sample_test
from track_llm_apis.sampling.analyze_mmlu import mmlu_two_sample_test
from track_llm_apis.sampling.common import (
    CompressedOutput,
    DataSource,
    OutputRow,
    TwoSampleTestResults,
    UncompressedOutput,
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
    """Print summary of top logprobs for a model."""
    top_10_logprobs, top_10_indices = torch.topk(logprobs, 10)

    print(f"\nTop 10 tokens for {model_name}:")
    for i in range(10):
        token_id = top_10_indices[i].item()
        token = tokenizer.decode(token_id)
        logprob = top_10_logprobs[i].item()
        print(f"Rank {i + 1}: '{token}' (ID: {token_id}) - Log prob: {logprob:g}")


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

        print("Top 10 tokens and their log probabilities:")
        for i in range(10):
            token_id = top_10_indices[i].item()
            token = tokenizer.decode(token_id)
            logprob = top_10_logprobs[i].item()
            print(f"Rank {i + 1}: '{token}' (ID: {token_id}) - Log prob: {logprob:g}")

        # Also print the actually generated token for reference
        generated_token_id = outputs.sequences[0, -1].item()
        generated_token = tokenizer.decode(generated_token_id)
        print(f"\nActually generated token: '{generated_token}' (ID: {generated_token_id})")


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
    prompt_rows_dict: dict[str, list[OutputRow]], tokenizer: PreTrainedTokenizerBase
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
    rows_subset: dict[str, list[OutputRow]],
    unchanged_rows_subset: dict[str, list[OutputRow]],
    tokenizer: PreTrainedTokenizerBase,
    compute_pvalue: bool = True,
    b: int = 1000,
):
    sample1 = prompt_rows_dict_to_completion_sample(rows_subset, tokenizer)
    sample2 = prompt_rows_dict_to_completion_sample(unchanged_rows_subset, tokenizer)
    return run_two_sample_test_torch(sample1, sample2, b=b, compute_pvalue=compute_pvalue)


def evaluate_detectors_on_variant(
    source: DataSource,
    rows1: dict[str, list[OutputRow]],
    rows2: dict[str, list[OutputRow]],
    same: bool,
    prompt_length: dict[str, int],
    tokenizer: PreTrainedTokenizerBase | None = None,
    logprob_prompt: str | None = None,
    compute_pvalue: bool = True,
    n_subsets_with_replacement: int = 200,
    b: int = 1000,
) -> TwoSampleTestResults:
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

    n_subsets = n_subsets_with_replacement
    stats = []
    pvalues = []
    n_input_tokens = []
    n_output_tokens = []
    sample_pairs = []
    if source == DataSource.US:
        # Only keep the provided or default prompt, discarding the others
        if logprob_prompt is None:
            logprob_prompt = config.sampling.logprob.default_prompt
        rows1 = {logprob_prompt: rows1[logprob_prompt]}
        rows2 = {logprob_prompt: rows2[logprob_prompt]}
    for _ in range(n_subsets):
        if same:
            assert rows1 == rows2
            subset = {p: random.sample(rows, 2 * samples_per_prompt) for p, rows in rows1.items()}
            rows1_subset = {p: subset[p][:samples_per_prompt] for p in rows1.keys()}
            rows2_subset = {p: subset[p][samples_per_prompt:] for p in rows2.keys()}
        else:
            rows1_subset = {p: random.sample(rows, samples_per_prompt) for p, rows in rows1.items()}
            rows2_subset = {p: random.sample(rows, samples_per_prompt) for p, rows in rows2.items()}
        sample_pairs.append((rows1_subset, rows2_subset))

    for sample1, sample2 in sample_pairs:
        result = two_sample_test_fn(
            sample1, sample2, b=b, compute_pvalue=compute_pvalue, tokenizer=tokenizer
        )
        stats.append(result.statistic)
        if compute_pvalue:
            pvalues.append(result.pvalue)
        n_input_tokens.append(
            sum(prompt_length[p] * len(r) for p, r in sample1.items())
            + sum(prompt_length[p] * len(r) for p, r in sample2.items())
        )
        n_output_tokens.append(
            sum(r.text[1] for rows in sample1.values() for r in rows)
            + sum(r.text[1] for rows in sample2.values() for r in rows)
        )

    return TwoSampleTestResults(
        stats=stats,
        pvalues=pvalues if compute_pvalue else None,
        n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens,
    )


def plot_roc_curve(
    roc_curves: dict[DataSource, tuple[np.ndarray, np.ndarray, np.ndarray]],
    aucs: dict[DataSource, float],
    model_name: str,
    variant: str | None,
):
    sources = list(roc_curves.keys())
    fig = go.Figure()
    for source in sources:
        fpr, tpr, _ = roc_curves[source]
        auc = aucs[source]
        display_name = f"{source.name} (AUC: {auc:.4f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=display_name))

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
    if variant:
        variant_metadata = json.loads(variant)
        variant_preslug = (
            variant_metadata["type"]
            + ","
            + ",".join(f"{k}={v}" for k, v in variant_metadata.items() if k != "type")
        )
    else:
        variant_preslug = "all"
    title = f"ROC Curves on model {model_name}"
    if variant:
        title += f", on variant: {variant_preslug}"
    else:
        title += " across all variants"
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    model_slug = slugify(model_name, max_length=100, hash_length=0)
    variant_slug = slugify(variant_preslug, max_length=200, hash_length=0)
    out_dir = config.plots_dir / "roc_curves" / model_slug
    os.makedirs(out_dir, exist_ok=True)
    fig.write_html(out_dir / f"{variant_slug}.html")


def evaluate_detectors(
    directory: Path,
    data: CompressedOutput,
    sources: list[DataSource],
    alpha: float,
    compute_pvalue: bool = False,
    plot_roc: bool = False,
    n_subsets_with_replacement: int = 200,
    b: int = 1000,
):
    analysis_results = {
        "metadata": {
            "model_name": data.model_name,
            "sources": [source.name for source in sources],
            "alpha": alpha,
            "compute_pvalue": compute_pvalue,
            "n_subsets_with_replacement": n_subsets_with_replacement,
            "b": b,
            "used_tokens": {source.name: None for source in sources},
        }
    }

    print(f"{sources=}")
    print(f"{data.model_name=}")
    variants = [v for v in data.references_dict["variant"] if v != TinyChange.unchanged_str()]
    prompts = data.references_dict["prompt"]
    prompt_length = {prompt: tokens for prompt, tokens in prompts}

    rows_by_variant = {
        source: UncompressedOutput.from_compressed_output(
            data, keep_datasource=source
        ).rows_by_variant()
        for source in sources
    }
    unchanged_rows = {
        source: rows_by_variant[source][TinyChange.unchanged_str()] for source in sources
    }
    unchanged_rows_by_prompt = {
        source: UncompressedOutput.rows_by_prompt(unchanged_rows[source]) for source in sources
    }

    tokenizer = AutoTokenizer.from_pretrained(data.model_name)
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        logger.info(f"Setting pad token to eos token for {data.model_name}")
        tokenizer.pad_token = tokenizer.eos_token

    start_time = time.time()
    # Get data on the false positive rate
    results_original = {
        source: evaluate_detectors_on_variant(
            source,
            unchanged_rows_by_prompt[source],
            unchanged_rows_by_prompt[source],
            same=True,
            prompt_length=prompt_length,
            tokenizer=tokenizer,
            compute_pvalue=compute_pvalue,
            n_subsets_with_replacement=n_subsets_with_replacement,
            b=b,
        )
        for source in sources
    }
    y_true_orig = {source: [0] * len(results_original[source].stats) for source in sources}
    y_pred_orig = {source: results_original[source].stats for source in sources}
    y_true = copy.deepcopy(y_true_orig)
    y_pred = copy.deepcopy(y_pred_orig)
    for variant_idx, variant in enumerate(variants):
        print(f"Variant {variant_idx + 1}/{len(variants)}: {variant}")
        roc_curves = {}
        roc_aucs = {}
        analysis_results[variant] = {}
        for source in sources:
            variant_rows = rows_by_variant[source][variant]
            rows_by_prompt = UncompressedOutput.rows_by_prompt(variant_rows)
            assert list(rows_by_prompt.keys()) == list(unchanged_rows_by_prompt[source].keys())
            results = evaluate_detectors_on_variant(
                source,
                unchanged_rows_by_prompt[source],
                rows_by_prompt,
                same=False,
                prompt_length=prompt_length,
                tokenizer=tokenizer,
                compute_pvalue=compute_pvalue,
                n_subsets_with_replacement=n_subsets_with_replacement,
                b=b,
            )
            variant_true = [1] * len(results.stats)
            variant_pred = results.stats
            y_true[source].extend(variant_true)
            y_pred[source].extend(variant_pred)
            if plot_roc:
                roc_curves[source] = roc_curve(
                    y_true_orig[source] + variant_true, y_pred_orig[source] + variant_pred
                )
            roc_aucs[source] = roc_auc_score(
                y_true_orig[source] + variant_true, y_pred_orig[source] + variant_pred
            )
            stat_avg = sum(results.stats) / len(results.stats)
            pvalue_avg = sum(results.pvalues) / len(results.pvalues) if results.pvalues else None
            input_tokens_avg = sum(results.n_input_tokens) / len(results.n_input_tokens)
            output_tokens_avg = sum(results.n_output_tokens) / len(results.n_output_tokens)
            power = (
                sum(pvalue < alpha for pvalue in results.pvalues) / len(results.pvalues)
                if results.pvalues
                else None
            )
            print(
                f"  * {source}: {pvalue_avg=}, {stat_avg=}, {power=}, {input_tokens_avg=}, {output_tokens_avg=}, roc_auc={roc_aucs[source]}"
            )
            stats_original = sorted(results_original[source].stats)
            stats_variant = sorted(results.stats)
            if analysis_results["metadata"]["used_tokens"][source.name] is None:
                analysis_results["metadata"]["used_tokens"][source.name] = {
                    "input": input_tokens_avg,
                    "output": output_tokens_avg,
                }
            analysis_results[variant][source.name] = {
                "pvalue_avg": pvalue_avg,
                "stat_avg": stat_avg,
                "power": power,
                "roc_auc": roc_aucs[source],
                "stats_original": {
                    "min": stats_original[0],
                    "max": stats_original[-1],
                },
                "stats_variant": {
                    "min": stats_variant[0],
                    "max": stats_variant[-1],
                },
            }
        if plot_roc:
            plot_roc_curve(roc_curves, roc_aucs, data.model_name, variant)
        with open(directory / "analysis.json", "w") as f:
            json.dump(analysis_results, f, indent=2)

    overall_roc_curves = {source: roc_curve(y_true[source], y_pred[source]) for source in sources}
    overall_roc_aucs = {source: roc_auc_score(y_true[source], y_pred[source]) for source in sources}
    for source in sources:
        print(f"Overall ROC AUC for {source}: {overall_roc_aucs[source]}")
    if plot_roc:
        plot_roc_curve(overall_roc_curves, overall_roc_aucs, data.model_name, None)
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.2f} seconds")


if __name__ == "__main__":
    output_dirs = [
        config.sampling_data_dir / "keep" / "2025-09-07_13-15-56",
        config.sampling_data_dir / "keep" / "2025-09-07_15-28-14",
        config.sampling_data_dir / "keep" / "2025-09-08_14-14-47",
    ]

    for output_dir in output_dirs:
        compressed_output = CompressedOutput.from_json(output_dir)
        print(compressed_output.model_name)
        print(f"number of rows: {len(compressed_output.rows)}")
        for ref in compressed_output.references:
            print(f"length of field '{ref.row_attr}': {len(ref.elems)}")
        evaluate_detectors(
            directory=output_dir,
            data=compressed_output,
            sources=[DataSource.US, DataSource.MMLU, DataSource.GAO2025],
            alpha=0.05,
            compute_pvalue=False,
            plot_roc=False,
            n_subsets_with_replacement=20,
            b=1000,
        )
