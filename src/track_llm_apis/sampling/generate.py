import asyncio
import gzip
import json
import os
import pickle
import random
import tempfile
import time
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field
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


class OutputRow(BaseModel):
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


class CompressedOutputRow(BaseModel):
    source: int
    variant_idx: int
    prompt_idx: int
    text_idx: int
    logprobs_idx: int | None = None

    def to_json(self) -> tuple[int, int, int, int, int | None]:
        return (self.source, self.variant_idx, self.prompt_idx, self.text_idx, self.logprobs_idx)

    @classmethod
    def from_json(cls, json_tuple: tuple[int, int, int, int, int | None]) -> Self:
        return cls(**dict(zip(cls.__annotations__, json_tuple)))


class Reference(BaseModel):
    row_attr: str
    elems: list[Any] = Field(default_factory=list)
    hash_to_idx: dict[str, int] = Field(default_factory=dict)

    _ROW_ATTR_TO_COMPRESSED_ROW_ATTR = {
        "variant": "variant_idx",
        "prompt": "prompt_idx",
        "text": "text_idx",
        "logprobs": "logprobs_idx",
    }

    @property
    def compressed_row_attr(self) -> str:
        return self._ROW_ATTR_TO_COMPRESSED_ROW_ATTR[self.row_attr]


class CompressedOutput(BaseModel):
    model_name: str
    rows: list[CompressedOutputRow] = Field(default_factory=list)
    references: list[Reference] = Field(
        default_factory=lambda: [
            Reference(row_attr="variant"),
            Reference(row_attr="prompt"),
            Reference(row_attr="text"),
            Reference(row_attr="logprobs"),
        ]
    )

    def add_row(self, row: OutputRow):
        compressed_row_kwargs = {"source": row.source.value}
        for ref in self.references:
            elem = row.__getattribute__(ref.row_attr)
            if elem is None:
                assert ref.row_attr == "logprobs"  # only logprobs can be None
                compressed_row_kwargs[ref.compressed_row_attr] = None
                continue

            if isinstance(elem, str):
                elem_hash = fast_hash(elem)
            else:
                elem_hash = fast_hash(json.dumps(elem))
            elem_hash = fast_hash(str(elem))
            elem_idx = ref.hash_to_idx.get(elem_hash, None)
            if elem_idx is None:
                # This element isn't stored yet
                elem_idx = len(ref.elems)
                ref.elems.append(elem)
                ref.hash_to_idx[elem_hash] = elem_idx
            compressed_row_kwargs[ref.compressed_row_attr] = elem_idx

        self.rows.append(CompressedOutputRow(**compressed_row_kwargs))

    def dump_pkl(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        pkl_filename = f"{slugify(self.model_name, max_length=200, hash_length=0)}.pkl.gz"
        pkl_path = output_dir / pkl_filename
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pkl(pkl_dir: Path) -> Self:
        # Find the only .pkl.gz file in the directory
        pkl_paths = list(pkl_dir.glob("*.pkl.gz"))
        if len(pkl_paths) != 1:
            raise ValueError(f"Expected 1 .pkl.gz file in {pkl_dir}, got {len(pkl_paths)}")
        pkl_path = pkl_paths[0]
        with gzip.open(pkl_path, "rb") as f:
            return pickle.load(f)

    def dump_json(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        json_filename = f"{slugify(self.model_name, max_length=200, hash_length=0)}.json.gz"
        json_path = output_dir / json_filename
        json_dict = {
            "model_name": self.model_name,
            "rows": [row.to_json() for row in self.rows],
            "references": {ref.row_attr: ref.elems for ref in self.references},
        }
        with gzip.open(json_path, "wt") as f:
            json.dump(json_dict, f, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_dir: Path) -> Self:
        # Find the only .json.gz file in the directory
        json_paths = list(json_dir.glob("*.json.gz"))
        if len(json_paths) != 1:
            raise ValueError(f"Expected 1 .json.gz file in {json_dir}, got {len(json_paths)}")
        json_path = json_paths[0]
        with gzip.open(json_path, "rt") as f:
            json_dict = json.load(f)
        return cls(
            model_name=json_dict["model_name"],
            rows=[CompressedOutputRow.from_json(row) for row in json_dict["rows"]],
            references=[
                Reference(row_attr=name, elems=elems)
                for name, elems in json_dict["references"].items()
            ],
        )


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
    start_time = time.time()
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
        tc_config.finetuning_samples = [1]
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
    compressed_output = CompressedOutput(model_name=model_name)

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
    logger.info(f"Initial metadata:\n{json.dumps(metadata, indent=2)}")
    # Initialize vLLM instance
    llm = init_vllm(model, tokenizer, config.sampling.device_config.vllm_device)

    # Synchronous iteration for testing
    async_iter = tiny_change.__aiter__()
    i = 0
    total_gen_time = 0.0
    total_inference_time = 0.0
    try:
        while True:
            gen_start = time.time()
            variant = await async_iter.__anext__()
            gen_time = time.time() - gen_start
            gen_time_str = str(timedelta(seconds=gen_time))
            total_gen_time += gen_time
            total_gen_time_str = str(timedelta(seconds=total_gen_time))
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
            total_inference_time += inference_time
            total_inference_time_str = str(timedelta(seconds=total_inference_time))
            logger.info(f"Inference time: {inference_time_str}")
            for row in results:
                compressed_output.add_row(row)

            # Free up model weights and KV cache from vLLM memory
            llm.sleep(level=2)

            total_time = time.time() - start_time
            total_time_str = str(timedelta(seconds=total_time))

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
            metadata["total_gen_time"] = total_gen_time
            metadata["total_gen_time_str"] = total_gen_time_str
            metadata["total_inference_time"] = total_inference_time
            metadata["total_inference_time_str"] = total_inference_time_str
            metadata["total_time"] = total_time
            metadata["total_time_str"] = total_time_str

            with open(output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            compressed_output.dump_pkl(output_dir)
            compressed_output.dump_json(output_dir)
            del variant.model
            del variant

    except StopAsyncIteration:
        logger.info("All variants processed")

    if llm is not None:
        cleanup_vllm(llm)


def entrypoint():
    asyncio.run(main())


if __name__ == "__main__":
    entrypoint()
