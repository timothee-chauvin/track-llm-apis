import asyncio
import json
import os
import random
import tempfile
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from track_llm_apis.config import Config
from track_llm_apis.tinychange import TinyChange, TinyChangeConfig, load_lmsys_chat_1m
from track_llm_apis.util import (
    available_gpu_memory_fraction,
    format_mmlu_prompt,
    slugify,
    trim_to_length,
)

logger = Config.logger

DEVICE = "cuda"


random.seed(Config.seed)
np.random.seed(Config.seed)


def get_logprobs_transformers(model, tokenizer, prompt):
    """Get log probabilities for the first generated token using model.generate() from transformers."""
    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
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


def print_logprobs(model, tokenizer, prompt):
    full_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
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


def vllm_batch_inference(
    llm: LLM,
    batch_prompts: list[str],
    target_position: int,
    max_tokens: int = 1,
    temperature: float = 0.0,
    logprobs_topk: int = 20,
) -> dict:
    """
    Perform vLLM inference on a target prompt mixed with simulated traffic within the batch.

    Args:
        llm: already initialized vLLM model
        batch_prompts: List of prompts in the batch
        target_position: Position of the target prompt in the batch
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        logprobs_topk: Number of logprobs to return per token
    Returns:
        Dictionary of the form {text: str, logprobs: list[dict]} (one logprob dict per output token)
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs_topk,
    )
    outputs = llm.generate(batch_prompts, sampling_params)
    target_output = outputs[target_position]
    return {
        "text": target_output.outputs[0].text,
        "logprobs": target_output.outputs[0].logprobs,
        # todo n_input_tokens, n_output_tokens
    }


def vllm_inference(
    llm: LLM,
    prompts: list[str],
    n_samples: int,
    max_tokens: int,
    temperature: float,
):
    sampling_params = SamplingParams(
        n=n_samples,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    outputs = llm.generate(prompts, sampling_params)
    return {
        output.prompt: {
            "input_tokens": len(output.prompt_token_ids),
            "outputs": [
                {
                    "text": c.text,
                    "output_tokens": len(c["token_ids"]),
                }
                for c in output.outputs
            ],
        }
        for output in outputs
    }


def vllm_inference_random_traffic(
    llm: LLM,
    prompt: str,
    other_prompts: list[str],
    batch_size: int,
    n_samples: int,
    max_tokens: int,
    temperature: float,
):
    """
    Return `n_samples` completions for the first `max_tokens` inference tokens of a target prompt mixed with random traffic.

    Args:
        llm: initialized vLLM model
        prompt: The target prompt to track
        other_prompts: List of other prompts to mix with the target prompt
        batch_size: Number of prompts to generate in each batch
        n_samples: Number of times to run the inference
        max_tokens: Number of output tokens to generate
    Returns:
        List of dictionaries of the form {text: str, logprobs: list[dict]}, one for each sample
    """
    all_results = []

    for _ in range(n_samples):
        traffic_prompts = random.sample(other_prompts, batch_size - 1)
        target_position = random.randint(0, batch_size - 1)
        batch_prompts = (
            traffic_prompts[:target_position] + [prompt] + traffic_prompts[target_position:]
        )
        result = vllm_batch_inference(
            llm=llm,
            batch_prompts=batch_prompts,
            target_position=target_position,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        all_results.append(result)

    return all_results


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
        Config.plots_dir
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

    fig = go.Figure()

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
            go.Scatter(
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


async def main():
    max_tokens = 50
    DEBUG = True

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompts = Config.prompts + Config.prompts_extended
    output_dir = Config.sampling_data_dir
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = TinyChangeConfig()
    config.finetuning_dataset = load_lmsys_chat_1m()
    other_prompts = [item["conversation"][0]["content"] for item in config.finetuning_dataset]
    if DEBUG:
        max_tokens = 5
        config.enable_finetuning = False
        config.enable_lora_finetuning = False
        config.enable_weight_pruning = False
        config.enable_quantization = False
        config.random_noise_scale = [0.1]
    tiny_change = TinyChange(model, tokenizer, config)
    all_data = {}

    # Load the MMLU prompts
    mmlu = load_dataset("cais/mmlu", "abstract_algebra", split="test")

    # Synchronous iteration for testing
    async_iter = tiny_change.__aiter__()
    try:
        while True:
            variant = await async_iter.__anext__()
            if variant.description["type"] == "unchanged":
                n_samples = 1000
            else:
                n_samples = 100
            if DEBUG:
                n_samples //= 100
            logger.info(f"Generated variant: ({variant.model_hash})")
            logger.info(json.dumps(variant.description, indent=2))
            variant_name = variant.name()
            all_data[variant_name] = {
                "us": {},
                "mmlu": {},
                "gao2025": {},
            }

            # Export the model to a temporary directory in huggingface format for use by vLLM
            with tempfile.TemporaryDirectory() as temp_dir:
                variant.model.save_pretrained(temp_dir)
                tokenizer.save_pretrained(temp_dir)

                # enforce_eager=True just to run faster without the CUDA graph optimization step
                available_memory_fraction = available_gpu_memory_fraction()
                llm = LLM(
                    model=temp_dir,
                    enforce_eager=True,
                    gpu_memory_utilization=0.95 * available_memory_fraction,
                )

                # MMLU
                results = vllm_inference(
                    llm=llm,
                    prompts=[format_mmlu_prompt(item) for item in mmlu],
                    n_samples=n_samples,
                    max_tokens=5,
                    temperature=0.1,
                )

                all_data[variant_name]["mmlu"] = results

                # Our prompts
                for prompt in prompts:
                    all_data[variant_name]["us"][prompt] = []
                    results = vllm_inference_random_traffic(
                        llm=llm,
                        prompt=prompt,
                        other_prompts=other_prompts,
                        batch_size=16,
                        n_samples=n_samples,
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                    for result in results:
                        all_data[variant_name]["us"][prompt].append(
                            {
                                "text": result["text"],
                                "logprobs": result["logprobs"],
                            }
                        )

    except StopAsyncIteration:
        logger.info("All variants processed")

    with open(output_dir / "all_data.json", "w") as f:
        json.dump(all_data, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
