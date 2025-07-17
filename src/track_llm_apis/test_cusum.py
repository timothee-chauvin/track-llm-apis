import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from track_llm_apis.config import Config
from track_llm_apis.tinychange import TinyChange, TinyChangeConfig, load_lmsys_chat_1m

logger = Config.logger

DEVICE = "cuda"


random.seed(Config.seed)
np.random.seed(Config.seed)


async def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = TinyChangeConfig()
    if config.enable_finetuning or config.enable_lora_finetuning:
        config.finetuning_dataset = load_lmsys_chat_1m()
    tiny_change = TinyChange(model, tokenizer, config)

    # Get original model logprobs for comparison
    original_logprobs = get_logprobs(model, tokenizer, "x")
    print_logprobs_summary(original_logprobs, tokenizer, "Original model")

    kl_divergences = {}

    # Synchronous iteration for testing
    async_iter = tiny_change.__aiter__()
    try:
        while True:
            variant = await async_iter.__anext__()
            logger.info(f"Generated variant: ({variant.model_hash})")
            logger.info(json.dumps(variant.description, indent=2))

            variant_logprobs = get_logprobs(variant.model, tokenizer, "x")
            print_logprobs_summary(variant_logprobs, tokenizer, f"Variant {variant.model_hash}")

            kl_div = compute_kl_divergence(original_logprobs, variant_logprobs)

            description_key = json.dumps(variant.description, separators=(",", ":"))
            kl_divergences[description_key] = kl_div.item()

            logger.info(f"KL divergence: {kl_div.item():g}")

    except StopAsyncIteration:
        logger.info("All variants processed")
        logger.info("KL Divergences summary:")
        for desc, kl in kl_divergences.items():
            logger.info(f"  {desc}: {kl:g}")


def get_logprobs(model, tokenizer, prompt):
    """Get log probabilities for the first generated token."""
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


def variance_with_random_traffic():
    batch_size = 16
    target_prompt = "Hello, how are you?"
    n_inferences = 10
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    dataset = load_lmsys_chat_1m(
        gpt4_filter=True, redacted_filter=True, flagged_filter=True, first_turn_only=True
    )
    # enforce_eager=True just to run faster without the CUDA graph optimization step
    llm = LLM(model=model_name, enforce_eager=True)
    user_prompts = [item["conversation"][0]["content"] for item in dataset]
    for i in range(n_inferences):
        traffic_prompts = random.sample(user_prompts, batch_size - 1)
        target_position = random.randint(0, batch_size - 1)
        batch_prompts = (
            traffic_prompts[:target_position] + [target_prompt] + traffic_prompts[target_position:]
        )
        result = vllm_batch_inference(llm, batch_prompts, target_position)
        for token_id, logprobs in result.logprobs[0].items():
            print(token_id, logprobs.logprob, logprobs.decoded_token)


if __name__ == "__main__":
    # asyncio.run(main())
    variance_with_random_traffic()
