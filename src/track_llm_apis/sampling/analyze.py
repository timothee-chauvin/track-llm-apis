import os
import time
from typing import Any

import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from track_llm_apis.config import config
from track_llm_apis.sampling.analyze_gao2025 import CompletionSample, run_two_sample_test_torch
from track_llm_apis.sampling.analyze_logprobs import logprob_two_sample_test
from track_llm_apis.sampling.analyze_mmlu import mmlu_two_sample_test
from track_llm_apis.sampling.common import (
    CompressedOutput,
    DataSource,
    OutputRow,
    UncompressedOutput,
)
from track_llm_apis.tinychange import TinyChange
from track_llm_apis.util import slugify, trim_to_length


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
    prompt_rows_dict: dict[str, list[OutputRow]], tokenizer: AutoTokenizer
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
        padding=True,
        truncation=True,
        max_length=config.sampling.gao2025.max_tokens,
        return_tensors="pt",
    )["input_ids"].to(device=config.analysis.device, dtype=torch.int32)
    return CompletionSample(prompts=prompts, completions=completions)


def gao2025_two_sample_test(
    rows_subset: dict[str, list[OutputRow]],
    unchanged_rows_subset: dict[str, list[OutputRow]],
    tokenizer: AutoTokenizer,
    b: int = 1000,
):
    sample1 = prompt_rows_dict_to_completion_sample(rows_subset, tokenizer)
    sample2 = prompt_rows_dict_to_completion_sample(unchanged_rows_subset, tokenizer)
    return run_two_sample_test_torch(sample1, sample2, b=b)


def gen_sample_pairs(
    source: DataSource,
    rows_subset: dict[str, list[OutputRow]],
    unchanged_rows_subset: dict[str, list[OutputRow]],
) -> list[tuple[dict[str, list[OutputRow]], dict[str, list[OutputRow]]]]:
    match source:
        case DataSource.GAO2025 | DataSource.MMLU:
            return [(rows_subset, unchanged_rows_subset)]
        case DataSource.US:
            return [
                ({prompt: rows_subset[prompt]}, {prompt: unchanged_rows_subset[prompt]})
                for prompt in rows_subset.keys()
            ]


def evaluate_detectors_on_variant(
    source: DataSource,
    unchanged_rows_by_prompt: dict[str, list[OutputRow]],
    rows_by_prompt: dict[str, list[OutputRow]],
    prompt_length: dict[str, int],
    tokenizer: AutoTokenizer,
    power: float,
    alpha: float,
):
    pvalue_sum = 0
    stat_sum = 0
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

    n_subsets = config.sampling.variants_n_samples // samples_per_prompt
    pvalues = []
    n_input_tokens = []
    n_output_tokens = []
    sample_pairs = []
    for i in range(n_subsets):
        start = i * samples_per_prompt
        end = (i + 1) * samples_per_prompt
        rows_subset = {p: r[start:end] for p, r in rows_by_prompt.items()}
        unchanged_rows_subset = {p: r[start:end] for p, r in unchanged_rows_by_prompt.items()}
        sample_pairs.extend(gen_sample_pairs(source, rows_subset, unchanged_rows_subset))

    for sample1, sample2 in sample_pairs:
        pvalue, stat = two_sample_test_fn(sample1, sample2, b=1000, tokenizer=tokenizer)
        pvalues.append(pvalue)
        pvalue_sum += pvalue
        stat_sum += stat
        n_input_tokens.append(
            sum(prompt_length[p] * len(r) for p, r in sample1.items())
            + sum(prompt_length[p] * len(r) for p, r in sample2.items())
        )
        n_output_tokens.append(
            sum(r.text[1] for rows in sample1.values() for r in rows)
            + sum(r.text[1] for rows in sample2.values() for r in rows)
        )

    n_tests = len(sample_pairs)
    pvalue_avg = pvalue_sum / n_tests
    stat_avg = stat_sum / n_tests
    input_tokens_avg = sum(n_input_tokens) / n_tests
    output_tokens_avg = sum(n_output_tokens) / n_tests
    power = sum(pvalue < alpha for pvalue in pvalues) / n_tests
    return pvalue_avg, stat_avg, power, input_tokens_avg, output_tokens_avg


def evaluate_detectors(data: CompressedOutput, source: DataSource, power: float, alpha: float):
    variants = [v for v in data.references_dict["variant"] if v != TinyChange.unchanged_str()]
    prompts = data.references_dict["prompt"]
    prompt_length = {prompt: tokens for prompt, tokens in prompts}
    uncompressed_output = UncompressedOutput.from_compressed_output(data, keep_datasource=source)
    tokenizer = AutoTokenizer.from_pretrained(data.model_name)
    start_time = time.time()
    rows_by_variant = uncompressed_output.rows_by_variant()
    unchanged_rows = rows_by_variant[TinyChange.unchanged_str()]
    unchanged_rows_by_prompt = UncompressedOutput.rows_by_prompt(unchanged_rows)
    for variant_idx, variant in enumerate(variants):
        variant_rows = rows_by_variant[variant]
        rows_by_prompt = UncompressedOutput.rows_by_prompt(variant_rows)
        assert list(rows_by_prompt.keys()) == list(unchanged_rows_by_prompt.keys())
        pvalue_avg, stat_avg, power, input_tokens_avg, output_tokens_avg = (
            evaluate_detectors_on_variant(
                source,
                unchanged_rows_by_prompt,
                rows_by_prompt,
                prompt_length,
                tokenizer,
                power,
                alpha,
            )
        )
        print(
            f"Variant {variant_idx + 1}/{len(rows_by_variant)}: {variant}, P-value average: {pvalue_avg}, Statistic average: {stat_avg}, Power: {power:.2%}, Input tokens average: {input_tokens_avg}, Output tokens average: {output_tokens_avg}"
        )
    end_time = time.time()
    print(f"Time taken: {(end_time - start_time):.2f} seconds")


if __name__ == "__main__":
    output_dirs = [
        config.sampling_data_dir / "keep" / "2025-08-14_12-35-09",
    ]

    for output_dir in output_dirs:
        compressed_output = CompressedOutput.from_json(output_dir)
        print(compressed_output.model_name)
        print(f"number of rows: {len(compressed_output.rows)}")
        for ref in compressed_output.references:
            print(f"length of field '{ref.row_attr}': {len(ref.elems)}")
        for alpha in [0.05]:
            for source in [DataSource.MMLU, DataSource.US, DataSource.GAO2025]:
                evaluate_detectors(data=compressed_output, source=source, power=0.8, alpha=alpha)
