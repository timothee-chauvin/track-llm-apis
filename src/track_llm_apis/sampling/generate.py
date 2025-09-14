import asyncio
import json
import os
import random
import tempfile
import time
from datetime import timedelta
from typing import cast

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from track_llm_apis.config import DeviceConfig, config
from track_llm_apis.sampling.common import CompressedOutput, DataSource, OutputRow
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
            vllm_memory = 0.5 * available_memory_fraction
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
    outputs = llm.generate(prompts, sampling_params, use_tqdm=config.sampling.vllm_use_tqdm)
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


async def main():
    start_time = time.time()
    DEBUG = False
    if DEBUG:
        config.sampling.original_model_n_samples = 5
        config.sampling.variants_n_samples = 5

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
    model_slug = slugify(model_name, max_length=100, hash_length=0)
    prompts = config.prompts + config.prompts_extended
    output_dir = config.sampling_data_dir / f"{config.date}_{model_slug}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading model {model_name}...")
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
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    compressed_output = CompressedOutput(model_name=model_name, gpus=gpus)

    gao2025_config = config.sampling.gao2025
    mmlu_config = config.sampling.mmlu
    logprob_config = config.sampling.logprob

    # Load the MMLU prompts
    mmlu = load_dataset("cais/mmlu", mmlu_config.subset_name, split="test")

    wikipedia = get_wikipedia_samples(
        n=gao2025_config.n_wikipedia_prompts, seed=gao2025_config.wikipedia_seed
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
            if variant.description == TinyChange.UNCHANGED_DESCRIPTION:
                n_samples = config.sampling.original_model_n_samples
            else:
                n_samples = config.sampling.variants_n_samples
            logger.info(f"Generated variant {i}/{n_variants}: ({variant.model_hash})")
            logger.info(json.dumps(variant.description))
            logger.info(f"Generation time: {gen_time_str}")
            logger.info(used_gpu_memory(cleanup=True, as_str=True))
            variant_name = variant.name()

            inference_start = time.time()
            if llm is not None and config.sampling.vllm_enable_sleep_mode:
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
            if config.sampling.vllm_enable_sleep_mode:
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
