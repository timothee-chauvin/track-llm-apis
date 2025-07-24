import copy
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from track_llm_apis.util import available_gpu_memory_fraction

DEVICE = "cuda"


def random_noise(scale: float, model: torch.nn.Module):
    new_model = copy.deepcopy(model)
    for _, param in new_model.named_parameters():
        if param.requires_grad:
            param.data += torch.normal(
                mean=0.0,
                std=scale,
                size=param.data.shape,
                device=param.data.device,
                dtype=param.data.dtype,
            )
    return new_model


def cleanup_vllm(llm):
    """Clean up vLLM instance and free GPU memory."""
    destroy_model_parallel()
    # https://github.com/vllm-project/vllm/issues/1908#issuecomment-2975218097
    llm.llm_engine.engine_core.shutdown()
    del llm
    torch.cuda.empty_cache()


def load_model_to_vllm(llm: LLM | None, model, tokenizer):
    """Load a model into vLLM using temporary directory."""
    if llm is not None:
        cleanup_vllm(llm)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model and tokenizer to temporary directory
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)

        # Load into vLLM
        available_memory_fraction = available_gpu_memory_fraction()
        llm = LLM(
            model=temp_dir,
            enforce_eager=True,
            gpu_memory_utilization=0.95 * available_memory_fraction,
        )
        return llm


def run_inference(llm, prompt="Hi"):
    """Run inference on the given prompt and return the result."""
    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    second_model = random_noise(scale=0.5, model=model)

    prompt = "Hi"
    num_switches = 5  # Number of times to switch between models

    print(f"Starting model switching benchmark with prompt: '{prompt}'")
    print(f"Will switch between models {num_switches} times\n")

    print("Loading original model into vLLM for the first time...")
    start_time = time.time()
    llm = load_model_to_vllm(None, model, tokenizer)
    load_time = time.time() - start_time
    print(f"Original model loaded in {load_time:.2f}s")
    result1 = run_inference(llm, prompt)
    print(f"Original model output: '{result1}'")

    for i in range(num_switches):
        print(f"=== Switch {i + 1} ===")

        print("Loading noisy model into vLLM...")
        start_time = time.time()
        llm = load_model_to_vllm(llm, second_model, tokenizer)
        load_time = time.time() - start_time
        print(f"Noisy model loaded in {load_time:.2f}s")

        result2 = run_inference(llm, prompt)
        print(f"Noisy model output: '{result2}'")

        print("Loading original model into vLLM...")
        start_time = time.time()
        llm = load_model_to_vllm(llm, model, tokenizer)
        print(f"Original model loaded in {load_time:.2f}s")
        result1 = run_inference(llm, prompt)
        print(f"Original model output: '{result1}'")

    print("Benchmark complete!")
