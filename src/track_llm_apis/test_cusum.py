import asyncio
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, CompletionOutput, RequestOutput, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

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

# In order to be able to pass functions as args in LLM.collective_rpc()
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

random.seed(Config.seed)
np.random.seed(Config.seed)


@dataclass
class SamplingOutput:
    text: str
    output_tokens: int
    logprobs: list[dict] | None

    @classmethod
    def from_completion_output(cls, completion_output: CompletionOutput):
        if completion_output.logprobs is None:
            logprobs_dict = None
        else:
            logprobs_dict = [
                {k: v.__dict__ for k, v in logprobs.items()}
                for logprobs in completion_output.logprobs
            ]
        return cls(
            text=completion_output.text,
            output_tokens=len(completion_output.token_ids),
            logprobs=logprobs_dict,
        )


@dataclass
class PromptAndOutputs:
    prompt: str
    input_tokens: int
    outputs: list[SamplingOutput]

    @classmethod
    def from_request_output(cls, request_output: RequestOutput):
        return cls(
            prompt=request_output.prompt,
            input_tokens=len(request_output.prompt_token_ids),
            outputs=[
                SamplingOutput.from_completion_output(output) for output in request_output.outputs
            ],
        )

    @classmethod
    def from_multiple(cls, prompts_and_outputs: list["PromptAndOutputs"]):
        assert all(p.prompt == prompts_and_outputs[0].prompt for p in prompts_and_outputs)
        assert all(
            p.input_tokens == prompts_and_outputs[0].input_tokens for p in prompts_and_outputs
        )
        return cls(
            prompt=prompts_and_outputs[0].prompt,
            input_tokens=prompts_and_outputs[0].input_tokens,
            outputs=[output for p in prompts_and_outputs for output in p.outputs],
        )


class WorkerExtension:
    def debug(self):
        return (
            repr(self.model_runner.model),
            repr(dir(self.model_runner.model)),
        )

    def update_weights_from_ipc_handles(self, ipc_handles):
        """Update model weights from IPC handles."""
        weights = []
        device_id = self.device.index

        for name, handle in ipc_handles.items():
            func, args = handle
            list_args = list(args)
            # Update device ID to current device
            list_args[6] = device_id
            tensor = func(*list_args)
            weights.append((name, tensor))

        # Load the weights into the model
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()
        return f"Updated {len(weights)} weight tensors"


def create_ipc_handles(model: torch.nn.Module):
    """Create IPC handles for all model parameters."""
    ipc_handles = {}
    for name, param in model.named_parameters():
        # Ensure tensor is contiguous and on GPU
        if not param.is_contiguous():
            param = param.contiguous()
        ipc_handles[name] = reduce_tensor(param.detach())
    return ipc_handles


def cleanup_vllm(llm):
    """Clean up vLLM instance and free GPU memory."""
    destroy_model_parallel()
    # https://github.com/vllm-project/vllm/issues/1908#issuecomment-2975218097
    llm.llm_engine.engine_core.shutdown()
    del llm
    torch.cuda.empty_cache()


def load_model_to_vllm(llm: LLM | None, model, tokenizer):
    """Load a model into vLLM using temporary directory or IPC handles."""
    if llm is None:
        # First time: create vLLM instance
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model and tokenizer to temporary directory
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)

            # Load into vLLM
            available_memory_fraction = available_gpu_memory_fraction()
            vllm_memory = 0.2 * available_memory_fraction
            while True:
                try:
                    llm = LLM(
                        model=temp_dir,
                        enforce_eager=True,
                        gpu_memory_utilization=vllm_memory,
                        worker_extension_cls="__main__.WorkerExtension",
                    )
                    return llm
                except RuntimeError:
                    vllm_memory += 0.1 * available_memory_fraction
                    if vllm_memory > available_memory_fraction:
                        raise RuntimeError("Failed to load model into vLLM")

    else:
        # Subsequent times: update weights via IPC
        logger.info("Creating IPC handles from model weights...")
        ipc_handles = create_ipc_handles(model)

        logger.info("Updating vLLM weights via IPC...")
        result = llm.collective_rpc("update_weights_from_ipc_handles", args=(ipc_handles,))
        logger.info(f"IPC update result: {result}")

        return llm


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
    return [PromptAndOutputs.from_request_output(output) for output in outputs]


def vllm_inference_random_traffic_one_prompt(
    llm: LLM,
    prompt: str,
    other_prompts: list[str],
    batch_size: int,
    n_samples: int,
    max_tokens: int,
    temperature: float,
    logprobs_topk: int,
) -> PromptAndOutputs:
    """
    Return `n_samples` completions for the first `max_tokens` inference tokens of a target prompt mixed with random traffic.

    Args: see vllm_inference_random_traffic
    """
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs_topk,
    )
    all_results = []

    for _ in range(n_samples):
        traffic_prompts = random.sample(other_prompts, batch_size - 1)
        target_position = random.randint(0, batch_size - 1)
        batch_prompts = (
            traffic_prompts[:target_position] + [prompt] + traffic_prompts[target_position:]
        )

        outputs = llm.generate(batch_prompts, sampling_params)
        all_results.append(PromptAndOutputs.from_request_output(outputs[target_position]))
    return PromptAndOutputs.from_multiple(all_results)


def vllm_inference_random_traffic(
    llm: LLM,
    prompts: list[str],
    other_prompts: list[str],
    batch_size: int,
    n_samples: int,
    max_tokens: int,
    temperature: float,
    logprobs_topk: int,
) -> list[PromptAndOutputs]:
    """
    Return `n_samples` completions for the first `max_tokens` inference tokens of a list of prompts mixed with random traffic.

    Args:
        llm: initialized vLLM model
        prompts: List of prompts to track
        other_prompts: List of other prompts to mix with the prompts
        batch_size: Number of prompts to generate in each batch
        n_samples: Number of times to run the inference
        max_tokens: Number of output tokens to generate
        temperature: Sampling temperature
        logprobs_topk: Number of logprobs to return per token position
    """
    return [
        vllm_inference_random_traffic_one_prompt(
            llm=llm,
            prompt=prompt,
            other_prompts=other_prompts,
            batch_size=batch_size,
            n_samples=n_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs_topk=logprobs_topk,
        )
        for prompt in prompts
    ]


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
        config.enable_finetuning = False
        config.enable_lora_finetuning = False
        config.enable_weight_pruning = False
        config.enable_quantization = False
        config.random_noise_scale = [0.1]
        prompts = prompts[:5]
    tiny_change = TinyChange(model, tokenizer, config)
    all_data = {}

    # Load the MMLU prompts
    mmlu = load_dataset("cais/mmlu", "abstract_algebra", split="test")

    # Initialize vLLM instance once
    llm = None

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
                "us": [],
                "mmlu": [],
                "gao2025": [],
            }

            if llm is not None:
                llm.wake_up()

            # Load model into vLLM (first time creates instance, subsequent times update weights)
            llm = load_model_to_vllm(llm, variant.model, tokenizer)

            # MMLU
            results = vllm_inference(
                llm=llm,
                prompts=[format_mmlu_prompt(item) for item in mmlu],
                n_samples=n_samples,
                max_tokens=5,
                temperature=0.1,
            )

            all_data[variant_name]["mmlu"] = [asdict(r) for r in results]

            # Our prompts
            results = vllm_inference_random_traffic(
                llm=llm,
                prompts=prompts,
                other_prompts=other_prompts,
                batch_size=16,
                n_samples=n_samples,
                max_tokens=1,
                temperature=0.0,
                logprobs_topk=20,
            )
            all_data[variant_name]["us"] = [asdict(r) for r in results]

            # Free up model weights and KV cache from vLLM memory
            llm.sleep(level=2)

    except StopAsyncIteration:
        logger.info("All variants processed")

    with open(output_dir / "all_data.json", "w") as f:
        json.dump(all_data, f, indent=2)

    if llm is not None:
        cleanup_vllm(llm)


if __name__ == "__main__":
    asyncio.run(main())
