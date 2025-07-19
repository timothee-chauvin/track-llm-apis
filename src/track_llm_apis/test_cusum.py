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
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from track_llm_apis.config import Config
from track_llm_apis.tinychange import TinyChange, TinyChangeConfig, load_lmsys_chat_1m
from track_llm_apis.util import slugify

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
    logprobs: int = 20,
) -> dict:
    """
    Perform vLLM inference on a target prompt mixed with simulated traffic within the batch.

    Args:
        llm: already initialized vLLM model
        batch_prompts: List of prompts in the batch
        target_position: Position of the target prompt in the batch
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        logprobs: Number of logprobs to return per token
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs,
    )
    outputs = llm.generate(batch_prompts, sampling_params)
    target_output = outputs[target_position]
    return target_output.outputs[0]


def vllm_time_series_random_traffic(
    prompt: str,
    model_path: str,
    other_prompts: list[str],
    batch_size: int = 16,
    n_inferences: int = 200,
):
    """
    Return logprobs over time for the first inference token of a target prompt mixed with random traffic.

    Args:
        prompt: The target prompt to track
        model_path: Path or name of a vLLM-compatible model
        other_prompts: List of other prompts to mix with the target prompt
        batch_size: Number of prompts to generate in each batch
        n_inferences: Number of inferences to run
    Returns:
        List of logprobs over time for the first inference token of the target prompt
    """
    # enforce_eager=True just to run faster without the CUDA graph optimization step
    llm = LLM(model=model_path, enforce_eager=True)

    all_logprobs = []

    for _ in range(n_inferences):
        traffic_prompts = random.sample(other_prompts, batch_size - 1)
        target_position = random.randint(0, batch_size - 1)
        batch_prompts = (
            traffic_prompts[:target_position] + [prompt] + traffic_prompts[target_position:]
        )
        result = vllm_batch_inference(llm, batch_prompts, target_position)
        all_logprobs.append(result.logprobs[0])

    return all_logprobs


def plot_logprobs_over_time(
    all_logprobs, prompt: str, base_model_name: str, variant_description: dict[str, Any]
):
    """Plot logprobs over time for all tokens that appear in the series."""
    prompt_slug = slugify(prompt, max_length=50, hash_length=8)
    prompt_dir = Config.plots_dir / "time_series_local" / prompt_slug / base_model_name
    os.makedirs(prompt_dir, exist_ok=True)
    description_slug = slugify(
        json.dumps(variant_description, separators=(",", ":")), max_length=50, hash_length=8
    )
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

    fig.update_layout(
        title=f"Logprobs Over Time - {prompt_slug}",
        xaxis_title="Iteration",
        yaxis_title="Log Probability",
        template="plotly_white",
    )

    fig.write_html(prompt_dir / filename)


async def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "x"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = TinyChangeConfig()
    config.finetuning_dataset = load_lmsys_chat_1m()
    other_prompts = [item["conversation"][0]["content"] for item in config.finetuning_dataset]
    tiny_change = TinyChange(model, tokenizer, config)

    # Synchronous iteration for testing
    async_iter = tiny_change.__aiter__()
    try:
        while True:
            variant = await async_iter.__anext__()
            logger.info(f"Generated variant: ({variant.model_hash})")
            logger.info(json.dumps(variant.description, indent=2))

            # Export the model to a temporary directory in huggingface format for use by vLLM
            with tempfile.TemporaryDirectory() as temp_dir:
                variant.model.save_pretrained(temp_dir)
                logprobs = vllm_time_series_random_traffic(
                    prompt, temp_dir, other_prompts, batch_size=16, n_inferences=200
                )
                plot_logprobs_over_time(logprobs, prompt, model_name, variant.description)

    except StopAsyncIteration:
        logger.info("All variants processed")


if __name__ == "__main__":
    asyncio.run(main())
