import asyncio
import json
import os
import pickle
import random
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from track_llm_apis.config import DeviceConfig, config
from track_llm_apis.tinychange import TinyChange, TinyChangeConfig
from track_llm_apis.util import (
    available_gpu_memory_fraction,
    fast_hash,
    format_mmlu_prompt,
    format_wikipedia_prompt,
    get_model_hash,
    patch_chat_template,
    slugify,
    temporary_env,
    trim_to_length,
    used_gpu_memory,
)
from track_llm_apis.wikipedia import get_wikipedia_samples

logger = config.logger

# In order to be able to pass functions as args in LLM.collective_rpc()
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

random.seed(config.seed)
np.random.seed(config.seed)

# target_memory_gb = 20
# total_memory_gb = torch.cuda.mem_get_info()[1] / 1024**3
# torch.cuda.memory.set_per_process_memory_fraction(target_memory_gb / total_memory_gb)


class WorkerExtension:
    """
    Class for vLLM's worker to inherit from.
    """

    def debug(self):
        return (
            repr(self.model_runner.model),  # pyright: ignore[reportAttributeAccessIssue]
            repr(dir(self.model_runner.model)),  # pyright: ignore[reportAttributeAccessIssue]
        )

    def update_weights_from_ipc_handles(self, ipc_handles):
        """Update model weights from IPC handles."""
        weights = []
        device_id = self.device.index  # pyright: ignore[reportAttributeAccessIssue]

        for name, handle in ipc_handles.items():
            func, args = handle
            list_args = list(args)
            # Update device ID to current device
            list_args[6] = device_id
            tensor = func(*list_args)
            weights.append((name, tensor))

        # Load the weights into the model
        self.model_runner.model.load_weights(weights=weights)  # pyright: ignore[reportAttributeAccessIssue]
        torch.cuda.synchronize()
        return f"Updated {len(weights)} weight tensors"


class DataSource(Enum):
    US = 0
    MMLU = 1
    GAO2025 = 2


@dataclass
class OutputRow:
    variant: str
    source: DataSource
    # (prompt, input_tokens)
    prompt: tuple[str, int]
    # (text, output_tokens)
    text: tuple[str, int]
    logprobs: list[dict[int, float]] | None = None

    @classmethod
    def from_request_output(
        cls, request_output: RequestOutput, variant: str, source: DataSource
    ) -> list["OutputRow"]:
        prompt_length = (
            len(request_output.prompt_token_ids)
            if request_output.prompt_token_ids is not None
            else 0
        )
        prompt = (request_output.prompt or "", prompt_length)
        rows = []
        for output in request_output.outputs:
            if output.logprobs is None:
                logprobs_dicts = None
            else:
                logprobs_dicts = [
                    {int(k): v.logprob for k, v in logprobs.items()} for logprobs in output.logprobs
                ]
            rows.append(
                cls(
                    variant=variant,
                    source=source,
                    prompt=prompt,
                    text=(output.text, len(output.token_ids)),
                    logprobs=logprobs_dicts,
                )
            )
        return rows


@dataclass
class CompressedOutputRow:
    source: DataSource
    variant_ref: str
    prompt_ref: str
    text_ref: str | None = None
    logprobs_ref: str | None = None


class CompressedOutput:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.variants: dict[str, str] = {}
        self.prompts: dict[str, tuple[str, int]] = {}
        self.texts: dict[str, tuple[str, int]] = {}
        self.logprobs: dict[str, list[dict[int, float]]] = {}
        self.rows: list[CompressedOutputRow] = []

    def add_row(self, row: OutputRow):
        variant_hash = fast_hash(row.variant)
        if variant_hash not in self.variants:
            self.variants[variant_hash] = row.variant

        prompt_hash = fast_hash(str(row.prompt))
        if prompt_hash not in self.prompts:
            self.prompts[prompt_hash] = row.prompt

        text_hash = fast_hash(str(row.text))
        if text_hash not in self.texts:
            self.texts[text_hash] = row.text

        logprobs_hash = None
        if row.logprobs is not None:
            logprobs_hash = fast_hash(json.dumps(row.logprobs))
            if logprobs_hash not in self.logprobs:
                self.logprobs[logprobs_hash] = row.logprobs

        self.rows.append(
            CompressedOutputRow(
                source=row.source,
                variant_ref=variant_hash,
                prompt_ref=prompt_hash,
                text_ref=text_hash,
                logprobs_ref=logprobs_hash,
            )
        )

    def dump_pkl(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        pkl_filename_basic = f"{slugify(self.model_name, max_length=200, hash_length=0)}_basic.pkl"
        pkl_path_basic = output_dir / pkl_filename_basic
        with open(pkl_path_basic, "wb") as f:
            pickle.dump(self, f)

        pkl_filename_compressed = (
            f"{slugify(self.model_name, max_length=200, hash_length=0)}_compressed.pkl"
        )
        pkl_path_compressed = output_dir / pkl_filename_compressed
        to_save = {}
        hashes_to_indices = {}
        for mapping in [self.variants, self.prompts, self.texts, self.logprobs]:
            for i, hash in enumerate(mapping.keys()):
                hashes_to_indices[hash] = i
        to_save["rows"] = [
            (
                row.source.value,
                hashes_to_indices[row.variant_ref],
                hashes_to_indices[row.prompt_ref],
                hashes_to_indices[row.text_ref],
                hashes_to_indices[row.logprobs_ref] if row.logprobs_ref is not None else None,
            )
            for row in self.rows
        ]
        to_save["variants"] = [(i, v) for i, v in enumerate(self.variants.values())]
        to_save["prompts"] = [(i, p) for i, p in enumerate(self.prompts.values())]
        to_save["texts"] = [(i, t) for i, t in enumerate(self.texts.values())]
        to_save["logprobs"] = [(i, lp) for i, lp in enumerate(self.logprobs.values())]
        with open(pkl_path_compressed, "wb") as f:
            pickle.dump(to_save, f)

    def dump_db(self, output_dir: Path):
        # Convert all references from hashes to integers
        hashes_to_indices = {}
        for mapping in [self.variants, self.prompts, self.texts, self.logprobs]:
            for i, hash in enumerate(mapping.keys()):
                hashes_to_indices[hash] = i

        output_dir.mkdir(parents=True, exist_ok=True)
        db_filename = f"{slugify(self.model_name, max_length=200, hash_length=0)}.db"
        db_path = output_dir / db_filename
        if db_path.exists():
            os.remove(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rows (
            source INTEGER NOT NULL,
            variant_ref INTEGER NOT NULL,
            prompt_ref INTEGER NOT NULL,
            text_ref INTEGER NOT NULL,
            logprobs_ref INTEGER
            )"""
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS variants (ref INTEGER PRIMARY KEY, variant TEXT NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS prompts (ref INTEGER PRIMARY KEY, prompt TEXT NOT NULL, prompt_length INTEGER NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS texts (ref INTEGER PRIMARY KEY, text TEXT NOT NULL, text_length INTEGER NOT NULL)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS logprobs (ref INTEGER PRIMARY KEY, logprobs TEXT)"
        )
        cursor.executemany(
            "INSERT INTO rows (source, variant_ref, prompt_ref, text_ref, logprobs_ref) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    row.source.value,
                    hashes_to_indices[row.variant_ref],
                    hashes_to_indices[row.prompt_ref],
                    hashes_to_indices[row.text_ref],
                    hashes_to_indices[row.logprobs_ref] if row.logprobs_ref is not None else None,
                )
                for row in self.rows
            ],
        )
        cursor.executemany(
            "INSERT INTO variants (ref, variant) VALUES (?, ?)",
            [(i, v) for i, v in enumerate(self.variants.values())],
        )
        cursor.executemany(
            "INSERT INTO prompts (ref, prompt, prompt_length) VALUES (?, ?, ?)",
            [(i, p[0], p[1]) for i, p in enumerate(self.prompts.values())],
        )
        cursor.executemany(
            "INSERT INTO texts (ref, text, text_length) VALUES (?, ?, ?)",
            [(i, t[0], t[1]) for i, t in enumerate(self.texts.values())],
        )
        cursor.executemany(
            "INSERT INTO logprobs (ref, logprobs) VALUES (?, ?)",
            [(i, str(lp)) for i, lp in enumerate(self.logprobs.values())],
        )
        conn.commit()
        conn.close()


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


def init_vllm(model, tokenizer, vllm_device: str) -> LLM:
    assert vllm_device == "cuda" or (vllm_device.startswith("cuda:") and vllm_device[5:].isdigit())
    if vllm_device == "cuda":
        visible_devices = "0"
    else:
        visible_devices = vllm_device[5:]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model and tokenizer to temporary directory
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)

        with temporary_env("CUDA_VISIBLE_DEVICES", visible_devices):
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


def load_model_to_vllm(llm: LLM, model) -> None:
    """Load a model into a running instance of vLLM in-place, using IPC handles."""
    logger.info("Creating IPC handles from model weights...")
    ipc_handles = create_ipc_handles(model)

    logger.info("Updating vLLM weights via IPC...")
    result = llm.collective_rpc("update_weights_from_ipc_handles", args=(ipc_handles,))
    logger.info(f"IPC update result: {result}")


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


def vllm_inference(
    llm: LLM,
    prompts: list[str],
    n_samples: int,
    max_tokens: int,
    temperature: float | int,
    variant: str,
    source: DataSource,
) -> list[OutputRow]:
    sampling_params = SamplingParams(
        n=n_samples,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [
        row
        for output in outputs
        for row in OutputRow.from_request_output(output, variant=variant, source=source)
    ]


def vllm_inference_random_traffic(
    llm: LLM,
    prompts: list[str],
    other_prompts: list[str],
    batch_size: int,
    n_samples: int,
    max_tokens: int,
    temperature: float | int,
    logprobs_topk: int,
    variant: str,
    source: DataSource,
) -> list[OutputRow]:
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
    if max_tokens > 1:
        raise NotImplementedError("max_tokens > 1 not implemented wrt saving (see OutputRow)")
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs_topk,
    )
    results = []
    # Generate a new random batch for each sample.
    for _ in range(n_samples):
        # choices: with replacement (traffic prompts can be repeated).
        # sample: without replacement (target positions must be unique).
        traffic_prompts = random.choices(other_prompts, k=batch_size - len(prompts))
        # Position in the batch of the first prompt, second prompt, etc.
        prompt_positions = random.sample(range(batch_size), k=len(prompts))
        other_positions = [i for i in range(batch_size) if i not in prompt_positions]
        batch_prompts = [""] * batch_size
        for i, prompt in enumerate(prompts):
            batch_prompts[prompt_positions[i]] = prompt
        for i in range(batch_size - len(prompts)):
            batch_prompts[other_positions[i]] = traffic_prompts[i]
        outputs = llm.generate(batch_prompts, sampling_params)
        for i, prompt in enumerate(prompts):
            prompt_position = prompt_positions[i]
            results.extend(
                OutputRow.from_request_output(
                    outputs[prompt_position], variant=variant, source=source
                )
            )
    return results


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


async def main():
    DEBUG = False
    if DEBUG:
        config.sampling.original_model_n_samples = 5
        config.sampling.variants_n_samples = 5
        # config.sampling.model_name = "microsoft/Phi-3-mini-4k-instruct"

    config.sampling.device_config = DeviceConfig(
        vllm_device="cuda:0",
        original_model_device="cuda:0",
        variants_device="cuda:1",
    )
    tc_config = TinyChangeConfig(variants_device=config.sampling.device_config.variants_device)
    if DEBUG:
        # tc_config.enable_finetuning = False
        # tc_config.finetuning_samples = [1, 16]
        # tc_config.enable_lora_finetuning = False
        tc_config.enable_weight_pruning = False
        tc_config.finetuning_samples = [1, 16, 32]
        tc_config.weight_pruning_random_scale = []
        tc_config.weight_pruning_magnitude_scale = [0.1]
        tc_config.enable_quantization = False
        tc_config.enable_random_noise = False

    model_name = config.sampling.model_name
    prompts = config.prompts + config.prompts_extended
    output_dir = config.sampling_data_dir / config.date
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(
        config.sampling.device_config.original_model_device
    )
    # if model.dtype.itemsize > 2:
    #     logger.info(f"Converting model from {model.dtype} to bfloat16")
    #     model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    patch_chat_template(tokenizer, config.chat_templates)
    assert isinstance(tc_config.finetuning_dataset, Dataset)
    tiny_change = TinyChange(model, tokenizer, tc_config)
    n_variants = tiny_change.n_variants
    compressed_output = CompressedOutput(model_name)

    gao2025_config = config.sampling.gao2025
    mmlu_config = config.sampling.mmlu
    logprob_config = config.sampling.logprob

    # Load the MMLU prompts
    mmlu = load_dataset("cais/mmlu", mmlu_config.subset_name, split="test")

    wikipedia = get_wikipedia_samples(
        n=gao2025_config.n_wikipedia_samples, seed=gao2025_config.wikipedia_seed
    )

    metadata = {
        "config": config.model_dump(
            mode="json",
            exclude={"api", "analysis"},
        ),
        "tinychange_config": tc_config.model_dump(mode="json"),
        "dtype": str(model.dtype),
        "model_hash": get_model_hash(model),
        "chat_template_hash": fast_hash(tokenizer.chat_template),
        "n_processed_variants": 0,
        "n_total_variants": n_variants,
        "processed_variants": [],
    }
    # Initialize vLLM instance
    llm = init_vllm(model, tokenizer, config.sampling.device_config.vllm_device)

    # Synchronous iteration for testing
    async_iter = tiny_change.__aiter__()
    i = 0
    try:
        while True:
            gen_start = time.time()
            variant = await async_iter.__anext__()
            gen_time = time.time() - gen_start
            gen_time_str = str(timedelta(seconds=gen_time))
            i += 1
            if variant.description["type"] == "unchanged":
                n_samples = config.sampling.original_model_n_samples
            else:
                n_samples = config.sampling.variants_n_samples
            logger.info(f"Generated variant {i}/{n_variants}: ({variant.model_hash})")
            logger.info(json.dumps(variant.description))
            logger.info(f"Generation time: {gen_time_str}")
            logger.info(used_gpu_memory(cleanup=True, as_str=True))
            variant_name = variant.name()

            inference_start = time.time()
            if llm is not None:
                llm.wake_up()

            load_model_to_vllm(llm, variant.model)

            # Model Equality Testing: Which Model Is This API Serving?
            results = vllm_inference(
                llm=llm,
                prompts=[format_wikipedia_prompt(item) for item in wikipedia],
                n_samples=n_samples,
                max_tokens=gao2025_config.max_tokens,
                temperature=gao2025_config.temperature,
                variant=variant_name,
                source=DataSource.GAO2025,
            )

            for row in results:
                compressed_output.add_row(row)

            # MMLU
            results = vllm_inference(
                llm=llm,
                prompts=[format_mmlu_prompt(cast(dict, item)) for item in mmlu],
                n_samples=n_samples,
                max_tokens=mmlu_config.max_tokens,
                temperature=mmlu_config.temperature,
                variant=variant_name,
                source=DataSource.MMLU,
            )

            for row in results:
                compressed_output.add_row(row)

            # Our prompts
            results = vllm_inference_random_traffic(
                llm=llm,
                prompts=prompts,
                other_prompts=logprob_config.other_prompts,
                batch_size=logprob_config.batch_size,
                n_samples=n_samples,
                max_tokens=config.max_completion_tokens,
                temperature=logprob_config.temperature,
                logprobs_topk=logprob_config.topk,
                variant=variant_name,
                source=DataSource.US,
            )
            inference_time = time.time() - inference_start
            inference_time_str = str(timedelta(seconds=inference_time))
            logger.info(f"Inference time: {inference_time_str}")
            for row in results:
                compressed_output.add_row(row)
            metadata["processed_variants"].append(
                {
                    variant_name: {
                        "description": variant.description,
                        "model_hash": variant.model_hash,
                        "generation_time": gen_time,
                        "generation_time_str": gen_time_str,
                        "inference_time": inference_time,
                        "inference_time_str": inference_time_str,
                    }
                }
            )
            metadata["n_processed_variants"] = len(metadata["processed_variants"])

            # Free up model weights and KV cache from vLLM memory
            llm.sleep(level=2)
            del variant.model
            del variant
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            compressed_output.dump_pkl(output_dir)
            compressed_output.dump_db(output_dir)

    except StopAsyncIteration:
        logger.info("All variants processed")

    if llm is not None:
        cleanup_vllm(llm)


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()
